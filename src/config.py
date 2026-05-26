from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import sympy as sp
import yaml

from src.dgse import DSGE
from src.specification.param_registry_class import ParamRegistry
from src.specification.param_specifications import HSpec, ParamSpec, PriorSpec, QSpec


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _symbols(names: Optional[Iterable[str]]) -> Tuple[sp.Symbol, ...]:
    return tuple(sp.Symbol(str(name)) for name in (names or ()))


def _parse_equation(raw: str, namespace: Dict[str, sp.Symbol]):
    text = str(raw).strip()
    if "=" in text:
        lhs, rhs = text.split("=", 1)
        return sp.Eq(
            sp.sympify(lhs.strip(), locals=namespace),
            sp.sympify(rhs.strip(), locals=namespace),
        )
    return sp.sympify(text, locals=namespace)


def _build_prior(raw: Optional[Dict[str, Any]]) -> Optional[PriorSpec]:
    if not raw:
        return None
    family = raw["family"]
    params = raw.get("params", {})
    return PriorSpec(family, {key: float(value) for key, value in params.items()})


def _param_specs(parameters_cfg: Dict[str, Any], namespace: Dict[str, sp.Symbol]):
    specs_cfg = parameters_cfg.get("specs", parameters_cfg)
    reserved = {"theta_econ", "theta_work"}
    specs = []

    for name, raw in specs_cfg.items():
        if name in reserved:
            continue
        raw = raw or {}
        symbol_name = raw.get("symbol", name)
        symbol = namespace.setdefault(symbol_name, sp.Symbol(symbol_name))
        meta = dict(raw.get("meta", {}))
        bounds = raw.get("bounds", {})
        if "lower" in bounds:
            meta["lower"] = float(bounds["lower"])
        if "upper" in bounds:
            meta["upper"] = float(bounds["upper"])

        specs.append(
            ParamSpec(
                name=name,
                symbol=symbol,
                transform=raw.get("transform", "id"),
                prior=_build_prior(raw.get("prior")),
                role=raw.get("role", "struct"),
                meta=meta,
            )
        )
    return specs


def _build_hspec(raw: Optional[Dict[str, Any]]) -> Optional[HSpec]:
    if not raw:
        return None
    if "fixed" in raw and raw["fixed"] is not None:
        return HSpec(fixed=np.asarray(raw["fixed"], dtype=float))
    if "diag_params" in raw and raw["diag_params"] is not None:
        return HSpec(diag_params=list(raw["diag_params"]))
    return HSpec()


def build_bundle_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = cfg.get("model") or {}
    variables = model_cfg.get("variables") or {}
    parameters_cfg = cfg.get("parameters") or {}

    y_t = _symbols(variables.get("states"))
    y_tp1 = _symbols(variables.get("leads"))
    y_tm1 = _symbols(variables.get("lags"))
    eps_t = _symbols(variables.get("shocks"))
    eta_t = _symbols(variables.get("expectation_shocks"))

    namespace: Dict[str, sp.Symbol] = {
        str(sym): sym
        for group in (y_t, y_tp1, y_tm1, eps_t, eta_t)
        for sym in group
    }
    params = _param_specs(parameters_cfg, namespace)
    equations = [
        _parse_equation(raw, namespace)
        for raw in model_cfg.get("equations", [])
    ]
    if not equations:
        raise ValueError("model.equations no puede estar vacío en el modo YAML.")

    registry = ParamRegistry(
        params=params,
        qspec=QSpec(diag_params=list((cfg.get("q") or {}).get("diag_params", [])))
        if cfg.get("q") is not None
        else None,
        hspec=_build_hspec(cfg.get("h")),
    )
    model = DSGE(
        equations=equations,
        y_t=y_t,
        y_tp1=y_tp1,
        y_tm1=y_tm1,
        eps_t=eps_t,
        eta_t=eta_t,
        metadata={
            "name": model_cfg.get("name", "yaml_model"),
            "source": model_cfg.get("source", "yaml"),
        },
    )

    theta_econ = parameters_cfg.get("theta_econ")
    theta_work = parameters_cfg.get("theta_work")
    bundle = {"model": model, "registry": registry}
    if theta_econ is not None:
        bundle["theta_econ"] = theta_econ
    if theta_work is not None:
        bundle["theta_work"] = theta_work
    return bundle
