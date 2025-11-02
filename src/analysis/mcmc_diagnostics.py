import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence, Union, Tuple

from src.inference.mcmc import unpack_draws
from src.specification.param_registry_class import ParamRegistry


def _hpdi(x: np.ndarray, cred_level: float = 0.9):
    x = np.sort(np.asarray(x).reshape(-1))
    n = x.size
    if n == 0:
        return np.nan, np.nan
    m = int(np.floor(cred_level * n))
    if m < 1 or m >= n:
        return np.nan, np.nan
    widths = x[m:] - x[: n - m]
    j = int(np.argmin(widths))
    return float(x[j]), float(x[j + m])


def _kde_gaussian(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    n = x.size
    if n == 0:
        return np.zeros_like(grid)
    std = np.std(x, ddof=1)
    if std == 0 or not np.isfinite(std):
        y = np.zeros_like(grid)
        y[np.argmin(np.abs(grid - x.mean()))] = 1.0
        return y / np.trapz(y, grid)
    h = 1.06 * std * n ** (-1 / 5)
    if h <= 0 or not np.isfinite(h):
        h = std or 1.0
    u = (grid[:, None] - x[None, :]) / h
    dens = np.exp(-0.5 * u**2).sum(axis=1) / (n * h * np.sqrt(2 * np.pi))
    return dens


def _autocorr_1d(x: np.ndarray, max_lag: int = 100) -> np.ndarray:
    x = np.asarray(x, float).reshape(-1)
    n = x.size
    x = x - x.mean()
    var = np.dot(x, x) / n
    if var == 0:
        return np.ones(max_lag + 1)
    ac = np.empty(max_lag + 1)
    ac[0] = 1.0
    for k in range(1, max_lag + 1):
        ac[k] = np.dot(x[:-k], x[k:]) / (n * var)
    return ac


def _ess_geyer(x: np.ndarray, max_lag: int = 100):
    ac = _autocorr_1d(x, max_lag)
    s = 0.0
    for k in range(1, max_lag, 2):
        pair = ac[k] + ac[k + 1]
        if pair <= 0:
            break
        s += pair
    tau = 1 + 2 * s
    ess = max(1.0, float(len(x) / tau))
    return ess, float(tau)


def _is_heavily_skewed_pos(x: np.ndarray) -> bool:
    x = np.asarray(x, float).reshape(-1)
    if np.any(x <= 0):
        return False
    med = np.median(x)
    mu = np.mean(x)
    cv = np.std(x, ddof=1) / (mu + 1e-16)
    return mu > 3 * (med + 1e-16) and cv > 1.0


C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_SKY = "#56B4E9"
C_GREY = "#999999"


def plot_mcmc_diagnostics(
    draws_work: np.ndarray,
    registry: ParamRegistry,
    subset: Union[int, Sequence[str]],
    *,
    bins: int = 25,
    burn_in: int = 0,
    figsize=(13, 4.8),
    cred_level: float = 0.90,
) -> None:
    draws = np.asarray(draws_work, dtype=float)
    if burn_in > 0:
        if burn_in >= draws.shape[0]:
            raise ValueError("burn_in no puede ser mayor o igual al número total de draws.")
        draws = draws[burn_in:]

    names = registry.names[:subset] if isinstance(subset, int) else list(subset)
    draws_econ = unpack_draws(draws, registry)

    for name in names:
        if name not in draws_econ.columns:
            print(f"Parámetro '{name}' no encontrado en el registro; se omite.")
            continue

        series = draws_econ[name].to_numpy().reshape(-1)
        n = series.size
        if n == 0:
            continue

        mean = float(np.mean(series))
        sd = float(np.std(series, ddof=1))
        median = float(np.median(series))
        lo, hi = _hpdi(series, cred_level)
        ess, tau = _ess_geyer(series, max_lag=min(100, max(10, n // 10)))
        mcse = sd * np.sqrt(tau / max(n, 1))

        use_logx = _is_heavily_skewed_pos(series)
        pad = sd * 3 if np.isfinite(sd) and sd > 0 else (series.max() - series.min() + 1e-6)
        x_min = max(0.0, min(series.min(), lo if np.isfinite(lo) else series.min()) - 0.1 * pad)
        x_max = max(series.max(), hi if np.isfinite(hi) else series.max()) + 0.1 * pad

        if use_logx:
            pos = series[series > 0]
            if pos.size == 0:
                use_logx = False

        if use_logx:
            log_series = np.log(pos)
            grid_log = np.linspace(log_series.min(), log_series.max(), 400)
            grid = np.exp(grid_log)
            dens_log = _kde_gaussian(log_series, grid_log)
            dens = dens_log / np.clip(grid, 1e-18, None)
            bins_edges = np.exp(np.linspace(log_series.min(), log_series.max(), bins + 1))
        else:
            grid = np.linspace(x_min, x_max, 400)
            dens = _kde_gaussian(series, grid)
            bins_edges = np.linspace(x_min, x_max, bins + 1)

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 3, width_ratios=[2.1, 2.1, 0.9])
        ax_post = fig.add_subplot(gs[0, 0])
        ax_trace = fig.add_subplot(gs[0, 1])
        ax_stats = fig.add_subplot(gs[0, 2])
        ax_stats.axis("off")
        fig.suptitle(name, fontweight="bold", fontsize=13)

        ax_post.hist(series, bins=bins_edges, density=True, alpha=0.25, color=C_SKY, edgecolor="white", label="Hist")
        ax_post.plot(grid, dens, color=C_BLUE, linewidth=2, label="KDE")

        mask = (grid >= lo) & (grid <= hi)
        if np.any(mask) and np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            ax_post.fill_between(
                grid[mask],
                0,
                dens[mask],
                color=C_SKY,
                alpha=0.4,
                edgecolor=C_BLUE,
                linewidth=0.8,
                label=f"{int(cred_level * 100)}% HPDI",
            )

        ax_post.axvline(mean, color=C_ORANGE, linestyle="--", linewidth=1.4, label=f"Media = {mean:.3g}")
        ax_post.axvline(median, color=C_GREY, linestyle=":", linewidth=1.4, label=f"Mediana = {median:.3g}")
        ax_post.set_title("Posterior (densidad)")
        ax_post.set_xlabel(name)
        ax_post.set_ylabel("Densidad")

        if use_logx:
            eps = max(1e-12, bins_edges[0])
            ax_post.set_xscale("log")
            ax_post.set_xlim(left=eps, right=bins_edges[-1])

        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            y0, y1 = ax_post.get_ylim()
            yb = y0 + 0.03 * (y1 - y0)
            ax_post.plot([lo, lo], [y0, yb], color=C_BLUE, linewidth=1.2)
            ax_post.plot([hi, hi], [y0, yb], color=C_BLUE, linewidth=1.2)
            ax_post.plot([lo, hi], [yb, yb], color=C_BLUE, linewidth=1.2)

        ax_post.legend(frameon=False, fontsize=8, ncol=2)

        ax_trace.plot(series, color=C_BLUE, linewidth=0.8, label="Cadena")
        run_mean = np.cumsum(series) / np.arange(1, n + 1)
        ax_trace.plot(run_mean, color=C_ORANGE, linestyle="--", linewidth=1.1, label="Media corrida")
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            ax_trace.axhspan(lo, hi, color=C_SKY, alpha=0.12, label=f"{int(cred_level * 100)}% HPDI")
        ax_trace.set_title("Trace")
        ax_trace.set_xlabel("Iteración")
        ax_trace.set_ylabel(name)
        ax_trace.legend(frameon=False, fontsize=8)

        stats_text = (
            f"n = {n:,}\n"
            f"ESS ≈ {ess:.0f}\n"
            f"MCSE ≈ {mcse:.3g}\n"
            f"Media = {mean:.3g}\n"
            f"Mediana = {median:.3g}\n"
            f"SD = {sd:.3g}\n"
            f"{int(cred_level * 100)}% HPDI:\n[{lo:.3g}, {hi:.3g}]")
        
        ax_stats.text(
            0.0,
            0.96,
            stats_text,
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
            color="#222",)

        plt.tight_layout()
        plt.show()
