#!/usr/bin/env python3
"""Run a configured DSGE estimation pipeline from YAML.

The model module must expose a factory function returning a dictionary with:
  model: DSGE instance
  registry: ParamRegistry
  theta_econ or theta_work: starting parameters
  data optional: T x n_obs array, unless `data.path` is configured
  measurement optional
  map_bounds optional
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_builders.steady import SteadyConfig
from src.config import build_bundle_from_config


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_config(path: Path) -> dict[str, Any]:
    default_path = ROOT / "configs/default.yaml"
    default = _load_yaml(default_path) if default_path.exists() else {}
    return _deep_merge(default, _load_yaml(path))


def _load_factory(module_name: str, factory_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, factory_name)


def _load_data(cfg: dict[str, Any], bundle: dict[str, Any]) -> np.ndarray:
    if "data" in bundle and bundle["data"] is not None:
        return np.asarray(bundle["data"], dtype=float)

    data_cfg = cfg.get("data") or {}
    if "values" in data_cfg:
        return np.asarray(data_cfg["values"], dtype=float)

    path = data_cfg.get("path")
    columns = data_cfg.get("columns")
    if not path:
        raise ValueError("No data provided. Set bundle['data'] or data.path in the YAML.")
    if not columns:
        raise ValueError("data.columns is required when loading data.path.")

    data_path = Path(path)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    df = pd.read_csv(data_path)
    return df.loc[:, columns].to_numpy(dtype=float)


def run_from_config(config_path: Path, *, dry_run: bool = False) -> dict[str, Any]:
    cfg = _load_config(config_path)
    model_cfg = cfg.get("model") or {}
    module_name = model_cfg.get("module")
    factory_name = model_cfg.get("factory", "build_model")

    if dry_run:
        return {"status": "ok", "config": cfg}

    if module_name:
        factory = _load_factory(module_name, factory_name)
        bundle = factory(cfg)
    else:
        bundle = build_bundle_from_config(cfg)

    model = bundle["model"]
    registry = bundle["registry"]
    theta = bundle.get("theta_econ", bundle.get("theta_work", cfg.get("parameters", {}).get("theta_econ")))
    if theta is None:
        raise ValueError("Provide theta_econ/theta_work in the factory bundle or parameters.theta_econ.")

    data = _load_data(cfg, bundle)
    steady_cfg_raw = cfg.get("steady") or {}
    steady_cfg = SteadyConfig(
        max_iter=int(steady_cfg_raw.get("max_iter", 200)),
        tol_f=float(steady_cfg_raw.get("tol_f", 1e-12)),
    )

    map_cfg = cfg.get("map") or {}
    mcmc_cfg = cfg.get("mcmc") or {}
    solver_cfg = cfg.get("solver") or {}
    likelihood_cfg = cfg.get("likelihood") or {}

    result = model.compute(
        registry=registry,
        theta_struct=theta,
        data=data,
        compute_steady=bool(steady_cfg_raw.get("enabled", True)),
        steady_cfg=steady_cfg,
        measurement=bundle.get("measurement"),
        div=float(solver_cfg.get("div", 1.0 + 1e-6)),
        likelihood_kwargs={
            "initial_covariance": likelihood_cfg.get("initial_covariance", "stationary"),
            "diffuse_scale": float(likelihood_cfg.get("diffuse_scale", 1e6)),
        },
        map=bool(map_cfg.get("enabled", True)),
        map_bounds=bundle.get("map_bounds", map_cfg.get("bounds")),
        map_kwargs={
            "method": map_cfg.get("method", "L-BFGS-B"),
            "hess_step": float(map_cfg.get("hess_step", 1e-4)),
            "tau_scale": float(map_cfg.get("tau_scale", 0.5)),
            "hessian_strategy": map_cfg.get("hessian_strategy", "auto"),
            "include_jacobian_prior": bool(map_cfg.get("include_jacobian_prior", False)),
        },
        run_mcmc=bool(mcmc_cfg.get("enabled", False)),
        mcmc_draws=int(mcmc_cfg.get("draws", 4000)),
        mcmc_kwargs={k: v for k, v in mcmc_cfg.items() if k != "enabled"},
        log_summary=True,
    )

    irf_cfg = cfg.get("irf") or {}
    if bool(irf_cfg.get("enabled", False)) and result.get("mcmc") is not None:
        result["irf"] = model.impulse_responses(
            horizon=int(irf_cfg.get("horizon", 20)),
            quantiles=irf_cfg.get("quantiles", [0.16, 0.5, 0.84]),
            shock_scale=irf_cfg.get("shock_scale", "std"),
            plot=False,
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a PyDSGEforge YAML pipeline.")
    parser.add_argument("--config", required=True, type=Path, help="Path to a YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print merged config only.")
    args = parser.parse_args()

    result = run_from_config(args.config, dry_run=args.dry_run)
    if args.dry_run:
        print(yaml.safe_dump(result["config"], sort_keys=False))
    else:
        printable = {
            "steady": bool(result.get("steady")),
            "map_success": None if result.get("map") is None else result["map"].get("success"),
            "mcmc": bool(result.get("mcmc")),
            "irf": bool(result.get("irf")),
        }
        print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
