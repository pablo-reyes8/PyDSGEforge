#!/usr/bin/env python3
"""Build the reproducible figures and metrics shown in the README."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import gaussian_kde, norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import build_bundle_from_config
from src.inference.likelihoods import log_like, st_sp
from src.inference.posteriors import log_posterior


YAML_SHOWCASES = (
    ("configs/hybrid_nk_medium.yaml", 20260714),
    ("configs/open_economy_nk.yaml", 20260715),
)
OUTPUT_ROOT = ROOT / "outputs"


def _load_yaml(relative_path: str) -> dict:
    with (ROOT / relative_path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _state_space(bundle: dict, cfg: dict, theta_work: np.ndarray):
    model = bundle["model"]
    registry = bundle["registry"]
    return st_sp(
        theta_work,
        model._equations,
        model._y_t,
        model._y_tp1,
        model._eps_t,
        registry,
        y_tm1=model._y_tm1,
        eta_t=model._eta_t,
        div=float((cfg.get("solver") or {}).get("div", 1.0 + 1e-6)),
    )


def generate_synthetic_data(relative_path: str, seed: int, periods: int = 160) -> Path:
    cfg = _load_yaml(relative_path)
    bundle = build_bundle_from_config(cfg)
    registry = bundle["registry"]
    theta = registry.from_econ_dict(bundle["theta_econ"])
    transition, constant, impact, eu, psi0, psi2 = _state_space(bundle, cfg, theta)
    if tuple(eu) != (1, 1):
        raise RuntimeError(f"{relative_path} does not satisfy BK at the data-generating values: {eu}")

    rng = np.random.default_rng(seed)
    q = registry.build_Q(theta, impact.shape[1])
    h = registry.build_H(theta, psi2.shape[0])
    shock_chol = np.linalg.cholesky(q)
    measurement_chol = np.linalg.cholesky(h)

    burn = 240
    state = np.zeros(transition.shape[0])
    observations = []
    for t in range(burn + periods):
        structural = shock_chol @ rng.standard_normal(impact.shape[1])
        state = constant + transition @ state + impact @ structural
        measurement = measurement_chol @ rng.standard_normal(psi2.shape[0])
        if t >= burn:
            observations.append(psi0 + psi2 @ state + measurement)

    data_cfg = cfg["data"]
    output_path = ROOT / data_cfg["path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(observations, columns=data_cfg["columns"]).to_csv(output_path, index=False)
    return output_path


def _plot_irf_grid(irfs: dict, title: str, output_stem: Path) -> None:
    shock_names = list(irfs)
    first = irfs[shock_names[0]]
    observable_names = first["observables"][:3]
    horizon = first["horizon"]
    xgrid = np.arange(horizon)

    fig, axes = plt.subplots(
        len(observable_names),
        len(shock_names),
        figsize=(3.35 * len(shock_names), 7.0),
        sharex=True,
        squeeze=False,
    )
    for col, shock in enumerate(shock_names):
        info = irfs[shock]
        summary = info["summary"]
        low, median, high = summary[0], summary[len(summary) // 2], summary[-1]
        for row, observable in enumerate(observable_names):
            ax = axes[row, col]
            ax.fill_between(xgrid, low[row], high[row], color="#9EC5E5", alpha=0.45, linewidth=0)
            ax.plot(xgrid, median[row], color="#114B7A", linewidth=2.0)
            ax.axhline(0.0, color="#6B7280", linewidth=0.8, linestyle="--")
            ax.grid(alpha=0.18)
            if row == 0:
                ax.set_title(shock.replace("eps_", "Shock "), fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(observable.replace("_t", ""), fontweight="bold")
            if row == len(observable_names) - 1:
                ax.set_xlabel("Quarter")
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_figure(fig, output_stem)


def estimate_yaml_showcase(relative_path: str, seed: int, draws_override: int | None = None) -> dict:
    cfg = _load_yaml(relative_path)
    bundle = build_bundle_from_config(cfg)
    model = bundle["model"]
    registry = bundle["registry"]
    data_cfg = cfg["data"]
    data = pd.read_csv(ROOT / data_cfg["path"])[data_cfg["columns"]].to_numpy(dtype=float)
    map_cfg = cfg["map"]
    mcmc_cfg = dict(cfg["mcmc"])
    draws = int(draws_override or mcmc_cfg.pop("draws"))
    mcmc_cfg.pop("enabled", None)
    if draws_override is not None:
        mcmc_cfg["warmup"] = min(int(mcmc_cfg.get("warmup", 0)), draws // 2)
        mcmc_cfg["adapt_block"] = min(
            int(mcmc_cfg.get("adapt_block", 100)),
            max(25, int(mcmc_cfg["warmup"]) // 3),
        )

    result = model.compute(
        registry=registry,
        theta_struct=bundle["theta_econ"],
        data=data,
        compute_steady=False,
        div=float(cfg["solver"]["div"]),
        likelihood_kwargs={
            "initial_covariance": cfg["likelihood"]["initial_covariance"],
            "diffuse_scale": float(cfg["likelihood"]["diffuse_scale"]),
        },
        map=True,
        map_bounds="auto",
        map_kwargs={
            "method": map_cfg["method"],
            "hess_step": float(map_cfg["hess_step"]),
            "tau_scale": float(map_cfg["tau_scale"]),
            "hessian_strategy": map_cfg["hessian_strategy"],
            "include_jacobian_prior": bool(map_cfg["include_jacobian_prior"]),
        },
        run_mcmc=True,
        mcmc_draws=draws,
        mcmc_cov=np.eye(len(registry.params)) * 2.5e-4,
        mcmc_rng=np.random.default_rng(seed + 1000),
        mcmc_kwargs=mcmc_cfg,
        log_summary=True,
    )
    plt.close("all")

    irf_cfg = cfg["irf"]
    irfs = model.impulse_responses(
        horizon=int(irf_cfg["horizon"]),
        quantiles=irf_cfg["quantiles"],
        shock_scale=irf_cfg["shock_scale"],
        div=float(cfg["solver"]["div"]),
        plot=False,
    )

    slug = Path(relative_path).stem
    output_dir = OUTPUT_ROOT / slug
    _plot_irf_grid(irfs, cfg["model"]["name"], output_dir / "irfs")

    posterior = registry.to_econ_dict_subset(model._mcmc_draws_after_burn, names=registry.names)
    summary = {
        "model": cfg["model"]["name"],
        "observations": int(data.shape[0]),
        "map_success": bool(result["map"]["success"]),
        "negative_log_posterior_map": float(result["map"]["neglogpost_map"]),
        "acceptance_rate": float(result["mcmc"]["acceptance_rate"]),
        "posterior_draws": int(len(posterior)),
        "map_economic": registry.to_econ_dict(result["map"]["theta_map"]),
        "posterior_mean": posterior.mean().to_dict(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    posterior.to_csv(output_dir / "posterior_draws.csv", index=False)
    return summary


def _dynare_bundle_and_mode():
    cfg = _load_yaml("configs/nk_full_yaml.yaml")
    bundle = build_bundle_from_config(cfg)
    mode_record = json.loads(
        (ROOT / "dynare_comprobation/reference/dynare_mode.json").read_text(encoding="utf-8")
    )
    return cfg, bundle, mode_record["parameters"], mode_record


def _plot_posterior_comparison(py_draws: pd.DataFrame, dynare_draws: pd.DataFrame, output_stem: Path) -> None:
    names = ["beta", "sigma", "kappa", "phi_pi", "phi_x", "sig_d", "sig_s", "sig_m"]
    labels = {
        "beta": r"$\beta$",
        "sigma": r"$\sigma$",
        "kappa": r"$\kappa$",
        "phi_pi": r"$\phi_\pi$",
        "phi_x": r"$\phi_x$",
        "sig_d": r"$\sigma_d$",
        "sig_s": r"$\sigma_s$",
        "sig_m": r"$\sigma_m$",
    }
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.2))

    def density(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
        scale = float(np.std(values))
        if values.size < 2 or not np.isfinite(scale) or scale < 1e-10:
            center = float(np.mean(values))
            return norm.pdf(grid, loc=center, scale=max(abs(center) * 0.01, 1e-4))
        return gaussian_kde(values)(grid)

    for ax, name in zip(axes.ravel(), names):
        py = py_draws[name].to_numpy(dtype=float)
        dy = dynare_draws[name].to_numpy(dtype=float)
        pooled = np.concatenate([py, dy])
        lo, hi = np.quantile(pooled, [0.005, 0.995])
        pad = 0.08 * max(hi - lo, 1e-6)
        grid = np.linspace(lo - pad, hi + pad, 400)
        ax.plot(grid, density(dy, grid), color="#16697A", linewidth=2.1, label="Dynare")
        ax.plot(grid, density(py, grid), color="#D95F02", linewidth=2.1, linestyle="--", label="PyDSGEforge")
        ax.axvline(dy.mean(), color="#16697A", alpha=0.55, linewidth=1.0)
        ax.axvline(py.mean(), color="#D95F02", alpha=0.55, linewidth=1.0, linestyle="--")
        ax.set_title(labels[name], fontsize=13)
        ax.grid(alpha=0.18)
        ax.set_yticks([])
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.suptitle("Posterior comparison on the same NK model and data", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, output_stem)


def _plot_dynare_irf_comparison(py_impact: np.ndarray, dynare_impact: np.ndarray, output_stem: Path) -> float:
    observables = ["Output gap", "Inflation", "Policy rate"]
    shocks = ["Demand", "Cost-push", "Monetary"]
    horizon = 12
    py_paths = np.zeros((3, 3, horizon))
    dyn_paths = np.zeros_like(py_paths)
    py_paths[:, :, 0] = py_impact
    dyn_paths[:, :, 0] = dynare_impact

    fig, axes = plt.subplots(3, 3, figsize=(11.2, 7.2), sharex=True)
    xgrid = np.arange(horizon)
    for row, observable in enumerate(observables):
        for col, shock in enumerate(shocks):
            ax = axes[row, col]
            ax.plot(xgrid, dyn_paths[row, col], color="#16697A", linewidth=2.7, label="Dynare")
            ax.plot(xgrid, py_paths[row, col], color="#F28E2B", linewidth=1.8, linestyle="--", label="PyDSGEforge")
            ax.axhline(0.0, color="#6B7280", linewidth=0.8)
            ax.grid(alpha=0.18)
            if row == 0:
                ax.set_title(shock, fontweight="bold")
            if col == 0:
                ax.set_ylabel(observable)
            if row == 2:
                ax.set_xlabel("Quarter")
    axes[0, 0].legend(frameon=False, loc="best")
    fig.suptitle("Unit-shock IRFs at the Dynare posterior mode", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, output_stem)
    return float(np.max(np.abs(py_paths - dyn_paths)))


def build_dynare_comparison(draws: int = 4000) -> dict:
    cfg, bundle, mode, mode_record = _dynare_bundle_and_mode()
    model = bundle["model"]
    registry = bundle["registry"]
    theta_mode = registry.from_econ_dict(mode)
    data = pd.read_csv(ROOT / "dynare_comprobation/data/colombia_nk_quarterly.csv")[
        ["x", "pi", "i"]
    ].to_numpy(dtype=float)
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
    py_loglike = log_like(theta_mode, data, **common, kalman_backend="python")
    logprior = registry.log_prior(theta_mode, include_jacobian=False)
    dynare_logposterior = -float(mode_record["negative_log_posterior"])
    dynare_loglike = dynare_logposterior - logprior

    result = model.compute(
        registry=registry,
        theta_struct=mode,
        data=data,
        compute_steady=False,
        div=float(cfg["solver"]["div"]),
        likelihood_kwargs={"initial_covariance": "stationary"},
        map=False,
        run_mcmc=True,
        mcmc_draws=draws,
        mcmc_cov=np.eye(len(registry.params)) * 1.0e-3,
        mcmc_rng=np.random.default_rng(20260716),
        mcmc_kwargs={
            "include_jacobian_prior": True,
            "adapt": True,
            "warmup": min(1500, draws // 2),
            "adapt_block": min(250, max(25, min(1500, draws // 2) // 3)),
            "low": 0.15,
            "high": 0.35,
            "shrink": 0.5,
            "expand": 1.5,
            "logs": True,
            "log_every": 500,
        },
        log_summary=True,
    )
    plt.close("all")

    py_draws = registry.to_econ_dict_subset(model._mcmc_draws_after_burn, names=registry.names)
    dynare_draws = pd.read_csv(
        ROOT / "dynare_comprobation/reference/dynare_posterior_draws.csv"
    )
    dynare_order = ["sig_d", "sig_s", "sig_m", "beta", "sigma", "kappa", "phi_pi", "phi_x"]
    dynare_raw = dynare_draws[dynare_order].to_numpy(dtype=float)

    validation_indices = np.linspace(0, len(dynare_raw) - 1, 25, dtype=int)
    stored_logposterior = dynare_draws["log_posterior"].to_numpy(dtype=float)
    validation_differences = []
    for idx in validation_indices:
        theta = registry.from_econ_dict(dict(zip(dynare_order, dynare_raw[idx])))
        py_value = log_posterior(
            theta,
            data,
            model._equations,
            model._y_t,
            model._y_tp1,
            model._eps_t,
            registry,
            y_tm1=model._y_tm1,
            eta_t=model._eta_t,
            div=float(cfg["solver"]["div"]),
            include_jacobian=False,
            likelihood_kwargs={"initial_covariance": "stationary"},
        )
        validation_differences.append(py_value - stored_logposterior[idx])

    output_dir = OUTPUT_ROOT / "dynare_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_posterior_comparison(py_draws, dynare_draws, output_dir / "posterior_comparison")

    _, _, impact, eu, _, measurement = _state_space(bundle, cfg, theta_mode)
    py_impact = measurement @ impact
    sigma = mode["sigma"]
    structural = np.array(
        [
            [1.0, 0.0, 1.0 / sigma],
            [-mode["kappa"], 1.0, 0.0],
            [-mode["phi_x"], -mode["phi_pi"], 1.0],
        ]
    )
    dynare_impact = np.linalg.inv(structural)
    max_irf_difference = _plot_dynare_irf_comparison(py_impact, dynare_impact, output_dir / "irf_comparison")

    metrics = {
        "bk": [int(eu[0]), int(eu[1])],
        "dynare_log_likelihood": float(dynare_loglike),
        "pydsgeforge_log_likelihood": float(py_loglike),
        "absolute_log_likelihood_difference": float(abs(py_loglike - dynare_loglike)),
        "dynare_log_posterior": float(dynare_logposterior),
        "pydsgeforge_log_posterior_at_dynare_mode": float(py_loglike + logprior),
        "absolute_log_posterior_difference": float(abs(py_loglike + logprior - dynare_logposterior)),
        "maximum_irf_difference": max_irf_difference,
        "posterior_validation_points": int(len(validation_indices)),
        "maximum_log_posterior_difference_on_dynare_draws": float(
            np.max(np.abs(validation_differences))
        ),
        "python_mcmc_acceptance_rate": float(result["mcmc"]["acceptance_rate"]),
        "python_posterior_draws": int(len(py_draws)),
        "dynare_posterior_draws": int(len(dynare_draws)),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    py_draws.to_csv(output_dir / "pydsgeforge_posterior_draws.csv", index=False)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=None, help="Override YAML MCMC draws for both showcase models.")
    parser.add_argument("--dynare-draws", type=int, default=4000, help="MCMC draws for the Python side of the Dynare comparison.")
    parser.add_argument("--only-dynare", action="store_true", help="Rebuild only the Dynare comparison outputs.")
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.only_dynare:
        metrics = build_dynare_comparison(draws=args.dynare_draws)
        report_path = OUTPUT_ROOT / "showcase_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {"models": []}
        report["dynare_comparison"] = metrics
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        return 0

    summaries = []
    for relative_path, seed in YAML_SHOWCASES:
        data_path = generate_synthetic_data(relative_path, seed)
        print(f"[showcase] generated {data_path.relative_to(ROOT)}")
        summaries.append(estimate_yaml_showcase(relative_path, seed, args.draws))
    metrics = build_dynare_comparison(draws=args.dynare_draws)
    report = {"models": summaries, "dynare_comparison": metrics}
    (OUTPUT_ROOT / "showcase_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
