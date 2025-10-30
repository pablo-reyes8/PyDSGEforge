import numpy as np 
from src.model_builders.linear_system import *
from src.inference.likelihoods import *

def log_posterior(
    theta_work: Sequence[float],
    y: np.ndarray,
    equations,
    y_t,
    y_tp1,
    eps_t,
    registry: ParamRegistry,
    *, y_tm1=None,
    eta_t=None,
    measurement=None, div: float = 0.0, include_jacobian: bool = False):

    lp = registry.log_prior(theta_work, include_jacobian=include_jacobian)

    if not np.isfinite(lp):
        return float("-inf")

    ll = log_like(
        theta_work,
        y,
        equations,
        y_t,
        y_tp1,
        eps_t,
        registry,
        y_tm1=y_tm1,
        eta_t=eta_t,
        measurement=measurement,
        div=div,)
    
    if not np.isfinite(ll):
        return float("-inf")

    return float(lp + ll)


def log_posterior2(
    theta_work,
    y,
    equations,
    y_t,
    y_tp1,
    eps_t,
    registry,
    *,  y_tm1=None,
    eta_t=None, measurement=None,
    div: float = 0.0, include_jacobian: bool = False,):

    return -log_posterior(
        theta_work,
        y,
        equations,
        y_t,
        y_tp1,
        eps_t,
        registry,
        y_tm1=y_tm1,
        eta_t=eta_t,
        measurement=measurement,
        div=div,
        include_jacobian=include_jacobian)