#!/usr/bin/env python3
"""Re-evaluate the committed Dynare reference with PyDSGEforge."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import build_bundle_from_config, load_yaml_config
from src.inference.likelihoods import log_like, st_sp
from src.inference.posteriors import log_posterior


def main() -> int:
    cfg = load_yaml_config(ROOT / "configs/nk_full_yaml.yaml")
    bundle = build_bundle_from_config(cfg)
    model = bundle["model"]
    registry = bundle["registry"]
    mode_record = json.loads((HERE / "reference/dynare_mode.json").read_text(encoding="utf-8"))
    metrics = json.loads((HERE / "reference/parity_metrics.json").read_text(encoding="utf-8"))
    theta = registry.from_econ_dict(mode_record["parameters"])
    data = pd.read_csv(HERE / "data/colombia_nk_quarterly.csv")[["x", "pi", "i"]].to_numpy()

    common = dict(
        equations=model._equations,
        y_t=model._y_t,
        y_tp1=model._y_tp1,
        eps_t=model._eps_t,
        registry=registry,
        y_tm1=model._y_tm1,
        eta_t=model._eta_t,
        div=float(cfg["solver"]["div"]),
    )
    py_loglike = log_like(theta, data, **common, kalman_backend="python")
    py_logposterior = log_posterior(
        theta,
        data,
        **common,
        include_jacobian=False,
        likelihood_kwargs={"initial_covariance": "stationary"},
    )
    _, _, impact, eu, _, measurement = st_sp(theta, **common)
    p = mode_record["parameters"]
    structural = np.array(
        [
            [1.0, 0.0, 1.0 / p["sigma"]],
            [-p["kappa"], 1.0, 0.0],
            [-p["phi_x"], -p["phi_pi"], 1.0],
        ]
    )
    irf_difference = float(np.max(np.abs(measurement @ impact - np.linalg.inv(structural))))

    checks = {
        "BK conditions": tuple(eu) == (1, 1),
        "log likelihood": abs(py_loglike - metrics["dynare_log_likelihood"]) < 1e-8,
        "log posterior": abs(py_logposterior - metrics["dynare_log_posterior"]) < 1e-8,
        "impact IRFs": irf_difference < 1e-12,
    }
    print(f"Dynare log likelihood : {metrics['dynare_log_likelihood']:.12f}")
    print(f"Python log likelihood : {py_loglike:.12f}")
    print(f"Dynare log posterior  : {metrics['dynare_log_posterior']:.12f}")
    print(f"Python log posterior  : {py_logposterior:.12f}")
    print(f"Maximum IRF difference: {irf_difference:.3e}")
    for label, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {label}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
