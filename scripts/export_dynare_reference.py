#!/usr/bin/env python3
"""Extract the small, reviewable Dynare fixture used by the parity checks.

The legacy ``nk_colombia`` directory contains a full Dynare working tree.  This
script deliberately exports only observations, posterior draws, the posterior
mode, and the numerical comparison metrics needed for reproducibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
PARAMETER_ORDER = [
    "sig_d",
    "sig_s",
    "sig_m",
    "beta",
    "sigma",
    "kappa",
    "phi_pi",
    "phi_x",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "nk_colombia")
    parser.add_argument("--target", type=Path, default=ROOT / "dynare_comprobation")
    args = parser.parse_args()

    source = args.source.resolve()
    target = args.target.resolve()
    data_dir = target / "data"
    reference_dir = target / "reference"
    data_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    data_mat = loadmat(source / "metropolis/nk_colombia_data.mat", squeeze_me=True)
    observations = np.asarray(data_mat["stock_data"], dtype=float).T
    pd.DataFrame(observations, columns=["x", "pi", "i"]).to_csv(
        data_dir / "colombia_nk_quarterly.csv", index=False
    )

    chain_mat = loadmat(source / "metropolis/nk_colombia_param1.mat", squeeze_me=True)
    draws = pd.DataFrame(np.asarray(chain_mat["stock"], dtype=float), columns=PARAMETER_ORDER)
    draws["log_posterior"] = np.asarray(chain_mat["stock_logpo"], dtype=float)
    draws.to_csv(reference_dir / "dynare_posterior_draws.csv", index=False)

    mode_mat = loadmat(source / "Output/nk_colombia_mode.mat", squeeze_me=True)
    mode_values = np.asarray(mode_mat["xparam1"], dtype=float)
    mode = {
        "parameter_order": PARAMETER_ORDER,
        "parameters": dict(zip(PARAMETER_ORDER, mode_values.tolist())),
        "negative_log_posterior": float(mode_mat["fval"]),
    }
    (reference_dir / "dynare_mode.json").write_text(
        json.dumps(mode, indent=2) + "\n", encoding="utf-8"
    )

    metrics_source = ROOT / "outputs/dynare_comparison/metrics.json"
    if metrics_source.exists():
        metrics = json.loads(metrics_source.read_text(encoding="utf-8"))
        (reference_dir / "parity_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )

    print(f"Exported clean Dynare fixture to {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
