import numpy as np 
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from src.inference.likelihoods import *
from functools import partial
from src.inference.posteriors import *
import pandas as pd

def metropolis_hastings(
    log_posterior: Callable[[np.ndarray, np.ndarray], float],
    R: int,
    theta0: np.ndarray,
    y: np.ndarray,
    V: np.ndarray,
    *,
    is_chol: bool = False,
    logs: bool = False,
    rng: Optional[np.random.Generator] = None,
    adapt: bool = False,
    warmup: int = 2000,
    adapt_block: int = 100,
    low: float = 0.15,
    high: float = 0.35,
    shrink: float = 0.5,
    expand: float = 2.0,
    stuck_shrink: float = 0.1,
    min_scale: float = 1e-8,
    max_scale: float = 1e8,
    log_every: int = 100):
    if rng is None:
        rng = np.random.default_rng()

    theta = np.asarray(theta0, dtype=float).reshape(-1)
    d = theta.size
    draws = np.zeros((R, d))

    if is_chol:
        L0 = np.asarray(V, dtype=float)
    else:
        S = 0.5 * (V + V.T)
        try:
            L0 = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            w, Q = np.linalg.eigh(S)
            w = np.maximum(w, 1e-12)
            L0 = np.linalg.cholesky(Q @ np.diag(w) @ Q.T)

    scale = 1.0
    L_eff = lambda: np.sqrt(scale) * L0

    logpost_curr = float(log_posterior(theta, y))
    if not np.isfinite(logpost_curr):
        raise ValueError("El estado inicial debe tener log-posterior finito.")

    accepts = 0
    accepts_blk = 0  
    inf_blk_iter = 0
    inf_blk_accepts = 0

    for r in range(R):
        theta_prop = theta + L_eff() @ rng.standard_normal(d)
        logpost_prop = float(log_posterior(theta_prop, y))

        accepted = False
        if np.isfinite(logpost_prop):
            log_alpha = logpost_prop - logpost_curr
            if log_alpha >= 0.0 or np.log(rng.random()) < log_alpha:
                theta = theta_prop
                logpost_curr = logpost_prop
                accepts += 1
                accepted = True

        draws[r] = theta

        if r < warmup:
            accepts_blk += int(accepted)

            if adapt and (r + 1) % adapt_block == 0:
                rate_blk = accepts_blk / adapt_block
                if accepts_blk == 0:
                    scale *= stuck_shrink
                elif rate_blk < low:
                    scale *= shrink
                elif rate_blk > high:
                    scale *= expand

                scale = float(np.clip(scale, min_scale, max_scale))

                if logs and (r + 1) % log_every == 0:
                    print(f"[adapt] it={r+1} rate={rate_blk:.3f} scale={scale:.3e}")

                accepts_blk = 0  
        else:
            inf_blk_iter += 1
            inf_blk_accepts += int(accepted)

            if logs and (inf_blk_iter % log_every == 0):
                rate_inf = inf_blk_accepts / max(1, inf_blk_iter)
                print(f"[Inference] it={r+1} rate={rate_inf:.3f} scale={scale:.3e}")
                inf_blk_iter = 0
                inf_blk_accepts = 0

    if logs:
        print(f"[Metropolis] acceptance rate: {accepts/R:.3f} ({accepts}/{R}) scale={scale:.3e}")

    return draws, accepts / R



def run_metropolis(
    theta_start: Sequence[float],
    y: np.ndarray,
    equations,
    y_t,
    y_tp1,
    eps_t,
    registry,
    *,
    steady = None,
    y_tm1=None,
    eta_t=None,
    measurement: Optional[Union[MeasurementSpec, Callable[[np.ndarray], MeasurementSpec]]] = None,
    cov_proposal: Optional[np.ndarray] = None,
    div: float = 0.0,
    R: int = 2000,
    rng: Optional[np.random.Generator] = None,
    adapt: bool = False,
    logs: bool = False,
    include_jacobian: bool = False,
    **adapt_kwargs,):

    if rng is None:
        rng = np.random.default_rng()

    state_dim = len(y_t)

    if measurement is None:
        def resolve_measurement(theta_work: np.ndarray):
            return measurement_from_registry(registry, theta_work, state_dim)
        
    elif callable(measurement):
        resolve_measurement = measurement
    else:
        def resolve_measurement(_theta_work: np.ndarray):
            return measurement

    theta_start = np.asarray(theta_start, dtype=float).reshape(-1)
    if cov_proposal is None:
        cov_proposal = np.eye(len(theta_start)) * 1e-3

    def log_post(theta_work: np.ndarray, data: np.ndarray) -> float:
        meas = resolve_measurement(theta_work)
        return log_posterior(
            theta_work,
            data,
            equations,
            y_t,
            y_tp1,
            eps_t,
            registry, steady=steady , 
            y_tm1=y_tm1,
            eta_t=eta_t,
            measurement=meas,
            div=div,
            include_jacobian=include_jacobian,)

    draws, acc_rate = metropolis_hastings(
        log_post,
        R,
        theta_start,
        y,
        cov_proposal,
        rng=rng,
        adapt=adapt,
        logs=logs,
        **adapt_kwargs,)

    return {
        "draws": draws,
        "acceptance_rate": acc_rate,
        "cov_proposal": cov_proposal,
        "theta_start": theta_start,
        "rng": rng,}



def unpack_theta(theta_work: np.ndarray, registry: ParamRegistry, names: Optional[Sequence[str]] = None):
    return registry.to_econ_dict_subset(theta_work, names=names)


def unpack_draws(draws: np.ndarray, registry: ParamRegistry, names: Optional[Sequence[str]] = None):
    records = [registry.to_econ_dict_subset(row, names=names) for row in draws]
    df = pd.DataFrame.from_records(records)
    if names is not None:
        df = df.loc[:, names]
    return df
