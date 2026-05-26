
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np 
from src.model_builders.linear_system import *
from src.solvers.gensys import *


from scipy.linalg import cho_factor, cho_solve
from scipy.linalg import solve_discrete_lyapunov
from numpy.linalg import slogdet

def st_sp(
    theta_work: Sequence[float],
    equations,
    y_t,
    y_tp1,
    eps_t,
    registry: ParamRegistry,
    *,
    y_tm1=None,
    eta_t=None,
    measurement=None, steady = None , 
    div: float = 0.0,):
    
    theta_work = np.asarray(theta_work, dtype=float).reshape(-1)
    state_dim = len(y_t)

    if measurement is None:
        meas_spec = measurement_from_registry(registry, theta_work, state_dim)
    elif callable(measurement):
        meas_spec = measurement(theta_work)
    else:
        meas_spec = measurement

    Gamma0, Gamma1, Psi, Pi, Psi0, Psi2, c = build_matrices(
        equations=equations,
        y_t=y_t,
        y_tp1=y_tp1,
        eps_t=eps_t,
        y_tm1=y_tm1,
        eta_t=eta_t,
        param_values=registry,
        measurement=meas_spec, steady_values=steady)

    G1, C, impact, eu = gensys(Gamma0, Gamma1, c, Psi, Pi, div=div)
    return G1, C, impact, eu, Psi0, Psi2


def _initial_filter_state(
    transition: np.ndarray,
    impact: np.ndarray,
    shock_cov: np.ndarray,
    *,
    initial_state=None,
    initial_covariance="stationary",
    diffuse_scale: float = 1e6):
    n_s = transition.shape[0]
    s_bar = (
        np.zeros(n_s, dtype=float)
        if initial_state is None
        else np.asarray(initial_state, dtype=float).reshape(n_s)
    )

    if isinstance(initial_covariance, str):
        mode = initial_covariance.lower()
        if mode == "stationary":
            process_cov = impact @ shock_cov @ impact.T
            if np.max(np.abs(np.linalg.eigvals(transition))) < 1.0:
                try:
                    P_bar = solve_discrete_lyapunov(transition, process_cov)
                except Exception:
                    P_bar = diffuse_scale * np.eye(n_s)
            else:
                P_bar = diffuse_scale * np.eye(n_s)
        elif mode == "diffuse":
            P_bar = diffuse_scale * np.eye(n_s)
        elif mode == "zero":
            P_bar = np.zeros((n_s, n_s), dtype=float)
        else:
            raise ValueError(
                "initial_covariance debe ser 'stationary', 'diffuse', 'zero' "
                "o una matriz n_state x n_state."
            )
    else:
        P_bar = np.asarray(initial_covariance, dtype=float)
        if P_bar.shape != (n_s, n_s):
            raise ValueError(f"initial_covariance debe ser {(n_s, n_s)}, recibido {P_bar.shape}")

    return s_bar, 0.5 * (P_bar + P_bar.T)


def log_like(
    theta_work: Sequence[float], y,
    equations,
    y_t, y_tp1, eps_t,
    registry: ParamRegistry,
    measurement = None,
    div: float = 0.0, y_tm1=None,
    eta_t=None, steady = None , 
    precomputed: Optional[Tuple[np.ndarray, ...]] = None,
    diffuse_scale: float = 1e6,
    initial_state=None,
    initial_covariance="stationary"):

    """
    Log-verosimilitud Gaussian Kalman filter:
      - BK check: si eu != [1,1] → -Inf
      - Predicción/actualización estándar:
           ŝ = Theta1 s̄
           P̂ = Theta1 P̄ Theta1' + Theta0 Q Theta0'
           e  = y_t - Psi0 - Psi2 ŝ
           S  = Psi2 P̂ Psi2' + H
      - Usa Cholesky si es posible; de lo contrario, cae a solve + slogdet.

    y: matriz T × n_y (en el MISMO orden que filas de Psi2)
    """

    if precomputed is None:
        Theta1, C, Theta0, eu, Psi0, Psi2 = st_sp(
            theta_work,
            equations,
            y_t,
            y_tp1,
            eps_t,
            registry,
            y_tm1=y_tm1,
            eta_t=eta_t,
            measurement=measurement, steady=steady ,
            div=div,)
    else:
        if len(precomputed) == 5:
            Theta1, Theta0, eu, Psi0, Psi2 = precomputed
            C = np.zeros(Theta1.shape[0], dtype=float)
        elif len(precomputed) == 6:
            Theta1, C, Theta0, eu, Psi0, Psi2 = precomputed
        else:
            raise ValueError("precomputed debe tener 5 elementos legacy o 6 elementos.")
        
    # BK: existencia y unicidad
    if eu[0] < 1 or eu[1] < 1:
        return float("-inf")

    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2:
        raise ValueError("y debe ser un vector o una matriz T x n_y.")

    T, n_y = y.shape
    n_s = Theta1.shape[0]
    C = np.asarray(C, dtype=float).reshape(n_s)

    # Q y H desde el registro (p.ej., Q diagonal con stds de shocks; H fijo o diag)
    Q = registry.build_Q(theta_work, n_eps=len(eps_t))
    H = registry.build_H(theta_work, n_y=n_y)

    s_bar, P_bar = _initial_filter_state(
        Theta1,
        Theta0,
        Q,
        initial_state=initial_state,
        initial_covariance=initial_covariance,
        diffuse_scale=diffuse_scale)

    const = n_y * np.log(2.0 * np.pi)
    ll_sum = 0.0

    for t in range(T):
        s_hat = C + Theta1 @ s_bar
        P_hat = Theta1 @ P_bar @ Theta1.T + Theta0 @ Q @ Theta0.T
        P_hat = 0.5 * (P_hat + P_hat.T)

        # Innovación
        e_t = y[t, :] - Psi0 - (Psi2 @ s_hat)
        S_t = Psi2 @ P_hat @ Psi2.T + H

        # Actualizacion
        try:
            cF, lower = cho_factor(S_t, lower=True, check_finite=False)
            logdet = 2.0 * np.sum(np.log(np.diag(cF)))
            Sinv_e = cho_solve((cF, lower), e_t, check_finite=False)
            K = P_hat @ Psi2.T @ cho_solve((cF, lower), np.eye(n_y), check_finite=False)
        except Exception:
            sign, logdet = slogdet(S_t)
            if sign <= 0 or not np.isfinite(logdet):
                return float("-inf")
            
            Sinv = np.linalg.inv(S_t)
            Sinv_e = Sinv @ e_t
            K = P_hat @ Psi2.T @ Sinv

        s_bar = s_hat + K @ e_t
        I_KZ = np.eye(n_s) - K @ Psi2
        P_bar = I_KZ @ P_hat @ I_KZ.T + K @ H @ K.T
        P_bar = 0.5 * (P_bar + P_bar.T)

        ll_sum += -0.5 * (const + logdet + e_t @ Sinv_e)

    return float(ll_sum)
