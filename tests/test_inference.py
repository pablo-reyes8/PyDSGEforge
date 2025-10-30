import numpy as np
import sympy as sp

from src.inference.likelihoods import log_like
from src.specification.param_registry_class import ParamRegistry


def _ar1_setup():
    x_t, x_tm1, eps_t = sp.symbols("x_t x_tm1 eps_t")

    equations = [sp.Eq(x_t, sp.Rational(4, 5) * x_tm1 + eps_t)]
    y_t = [x_t]
    y_tp1 = None
    eps = [eps_t]
    y_tm1 = [x_tm1]

    return equations, y_t, y_tp1, eps, y_tm1


def test_log_like_finite_for_stable_ar1():
    equations, y_t, y_tp1, eps_t, y_tm1 = _ar1_setup()

    registry = ParamRegistry(params=[])

    theta_work = np.array([], dtype=float)

    shocks = np.array([0.1, -0.05, 0.2, 0.0, -0.1], dtype=float)
    series = []
    state = 0.0
    for shock in shocks:
        state = 0.8 * state + shock
        series.append(state)
    y_obs = np.column_stack([series])

    ll_value = log_like(
        theta_work,
        y_obs,
        equations,
        y_t,
        y_tp1,
        eps_t,
        registry,
        y_tm1=y_tm1,
        eta_t=None,
    )

    assert np.isfinite(ll_value)
