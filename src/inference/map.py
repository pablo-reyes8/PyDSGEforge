from scipy.optimize import minimize
import numpy as np 
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

from src.inference.posteriors import * 

def safe_eval_f(f, x):
    """Evalúa f(x). Si hay LinAlgError o NaN/Inf, devuelve +inf."""
    try:
        val = f(x)
    except Exception:
        return np.inf
    if not np.isfinite(val):
        return np.inf
    return float(val)


def make_spd_eig(A: np.ndarray, tau: float = 1e-6) -> np.ndarray:
    A_sym = 0.5 * (A + A.T)
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    eigvals_clipped = np.maximum(eigvals, tau)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T


def numerical_hessian_central(f, x, h=1e-3, ridge=1e-6):
    """
    Hessiano central robusto de f en x. Si f explota en algún punto,
    tratamos esa dirección como muy curva (gran penalización).
    Devolvemos H simétrica + ridge*I al final.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n), dtype=float)

    f0 = safe_eval_f(f, x)

    for i in range(n):
        ei = np.zeros(n); ei[i] = 1.0
        fp = safe_eval_f(f, x + h*ei)
        fm = safe_eval_f(f, x - h*ei)
        if not np.isfinite(fp) or not np.isfinite(fm) or not np.isfinite(f0):
            H[i, i] = 1e6
        else:
            H[i, i] = (fp - 2.0*f0 + fm)/(h**2)

    for i in range(n):
        ei = np.zeros(n); ei[i] = 1.0
        for j in range(i+1, n):
            ej = np.zeros(n); ej[j] = 1.0
            fpp = safe_eval_f(f, x + h*ei + h*ej)
            fpm = safe_eval_f(f, x + h*ei - h*ej)
            fmp = safe_eval_f(f, x - h*ei + h*ej)
            fmm = safe_eval_f(f, x - h*ei - h*ej)

            if (not np.isfinite(fpp) or not np.isfinite(fpm) or
                not np.isfinite(fmp) or not np.isfinite(fmm)):
                Hij = 0.0
            else:
                Hij = (fpp - fpm - fmp + fmm)/(4.0*h**2)

            H[i, j] = H[j, i] = Hij

    H = 0.5*(H + H.T) + ridge*np.eye(n)
    return H

def build_proposal_from_hessian(H, tau_scale=0.5, min_eig=1e-8):
    """
    Toma H ~ ∇²(-logpost)(MAP).
    Devuelve (cov_prop, chol_prop).
    Si todo falla, usa fallback diagonal.
    """

    H = 0.5*(H + H.T)

    try:
        H_inv = np.linalg.inv(H)
    
    except np.linalg.LinAlgError:
        H_inv = np.linalg.pinv(H)

    cov = tau_scale * H_inv

    try:
        cov_spd = make_spd_eig(cov, min_eig)
    except np.linalg.LinAlgError:
        diag_elements = np.maximum(np.diag(cov), min_eig)
        cov_spd = np.diag(diag_elements)

    # Cholesky robusto
    try:
        L = np.linalg.cholesky(cov_spd)
    except np.linalg.LinAlgError:
        jitter = 1e-6
        cov_spd_j = cov_spd + jitter*np.eye(cov_spd.shape[0])
        L = np.linalg.cholesky(cov_spd_j)

    return cov_spd, L


def build_proposal_from_covariance(cov, tau_scale=0.5, min_eig=1e-8):
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    cov_spd = make_spd_eig(tau_scale * cov, min_eig)
    try:
        L = np.linalg.cholesky(cov_spd)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(cov_spd + 1e-6 * np.eye(cov_spd.shape[0]))
    return cov_spd, L


def _optimizer_inverse_hessian(res, n_params: int):
    hess_inv = getattr(res, "hess_inv", None)
    if hess_inv is None:
        return None
    try:
        if hasattr(hess_inv, "todense"):
            H_inv = np.asarray(hess_inv.todense(), dtype=float)
        else:
            H_inv = np.asarray(hess_inv, dtype=float)
    except Exception:
        return None
    if H_inv.shape != (n_params, n_params) or not np.all(np.isfinite(H_inv)):
        return None
    return 0.5 * (H_inv + H_inv.T)


def run_map(
    theta0: Sequence[float],
    y: np.ndarray,
    equations,
    y_t,
    y_tp1,
    eps_t , 
    registry: ParamRegistry,
    *,
    steady = None,
    y_tm1=None,
    eta_t=None,
    measurement: Optional[Union[MeasurementSpec, Callable[[np.ndarray], MeasurementSpec]]] = None,
    bounds: Optional[Sequence[Tuple[float, float]]] = None,
    method: str = "L-BFGS-B",
    hess_step: float = 1e-4,
    tau_scale: float = 0.5,
    hessian_strategy: str = "auto",
    include_jacobian_prior: bool = False , div = 0,
    likelihood_kwargs: Optional[Dict] = None):

    theta0 = np.asarray(theta0, dtype=float).reshape(-1)
    state_dim = len(y_t)

    if measurement is None:
        def resolve_measurement(theta_work: np.ndarray):
            return measurement_from_registry(registry, theta_work, state_dim)
        
    elif callable(measurement):
        resolve_measurement = measurement
    else:
        def resolve_measurement(_theta_work: np.ndarray):
            return measurement

    def f_obj(theta_work: np.ndarray):
        meas = resolve_measurement(theta_work)
        return log_posterior2(
            theta_work,
            y,
            equations,
            y_t,
            y_tp1,
            eps_t,
            registry,
            y_tm1=y_tm1,
            eta_t=eta_t,
            measurement=meas,
            div=div, steady= steady , 
            include_jacobian=include_jacobian_prior,
            likelihood_kwargs=likelihood_kwargs)

    res = minimize(f_obj, theta0, method=method, bounds=bounds)

    theta_map = np.asarray(res.x, dtype=float)
    strategy = hessian_strategy.lower()
    if strategy not in {"auto", "optimizer", "numerical"}:
        raise ValueError("hessian_strategy debe ser 'auto', 'optimizer' o 'numerical'.")

    H_inv_opt = None
    if strategy in {"auto", "optimizer"}:
        H_inv_opt = _optimizer_inverse_hessian(res, theta_map.size)

    if H_inv_opt is not None:
        cov_prop, chol_prop = build_proposal_from_covariance(
            H_inv_opt,
            tau_scale=tau_scale,
            min_eig=1e-8,
        )
        try:
            H = np.linalg.pinv(H_inv_opt)
        except np.linalg.LinAlgError:
            H = np.eye(theta_map.size)
        hessian_source = "optimizer"
    else:
        H = numerical_hessian_central(f_obj, theta_map, h=hess_step)
        cov_prop, chol_prop = build_proposal_from_hessian(H, tau_scale=tau_scale, min_eig=1e-8)
        hessian_source = "numerical"


    return {
        "theta_map": theta_map,
        "neglogpost_map": float(res.fun),
        "success": bool(res.success),
        "message": res.message,
        "nit": res.nit,
        "hessian": H,
        "hessian_source": hessian_source,
        "cov_proposal": cov_prop,
        "chol_proposal": chol_prop}
