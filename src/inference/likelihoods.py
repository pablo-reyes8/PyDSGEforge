
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np 
from src.model_builders.linear_system import *
from src.solvers.gensys import *


from scipy.linalg import cho_factor, cho_solve
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
    return G1, impact, eu, Psi0, Psi2


def log_like(
    theta_work: Sequence[float], y,
    equations,
    y_t, y_tp1, eps_t,
    registry: ParamRegistry,
    measurement = None,
    div: float = 0.0, y_tm1=None,
    eta_t=None, steady = None , 
    precomputed: Optional[Tuple[np.ndarray, ...]] = None,
    diffuse_scale: float = 1e6):

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
        Theta1, Theta0, eu, Psi0, Psi2 = st_sp(
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
        Theta1, Theta0, eu, Psi0, Psi2 = precomputed
        
    # BK: existencia y unicidad
    if eu[0] < 1 or eu[1] < 1:
        return float("-inf")

    T, n_y = y.shape
    n_s = Theta1.shape[0]

    # Q y H desde el registro (p.ej., Q diagonal con stds de shocks; H fijo o diag)
    Q = registry.build_Q(theta_work, n_eps=len(eps_t))
    H = registry.build_H(theta_work, n_y=n_y)

    s_bar = np.zeros(n_s)
    P_bar = diffuse_scale * np.eye(n_s)

    const = n_y * np.log(2.0 * np.pi)
    ll_sum = 0.0

    for t in range(T):
        s_hat = Theta1 @ s_bar
        P_hat = Theta1 @ P_bar @ Theta1.T + Theta0 @ Q @ Theta0.T

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
        P_bar = P_hat - K @ Psi2 @ P_hat  

        ll_sum += -0.5 * (const + logdet + e_t @ Sinv_e)

    return float(ll_sum)
