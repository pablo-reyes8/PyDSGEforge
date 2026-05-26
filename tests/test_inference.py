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


def test_log_like_uses_solution_constant_and_known_initial_state():
    x_t, x_tm1, eps_t = sp.symbols("x_t x_tm1 eps_t")

    rho = 0.8
    const = 0.2
    equations = [sp.Eq(x_t, const + rho * x_tm1 + eps_t)]
    y_t = [x_t]
    eps = [eps_t]
    y_tm1 = [x_tm1]

    shocks = np.array([0.1, -0.05, 0.2, 0.0, -0.1], dtype=float)
    state = 0.0
    series = []
    for shock in shocks:
        state = const + rho * state + shock
        series.append(state)

    y_obs = np.asarray(series).reshape(-1, 1)
    expected = -0.5 * np.sum(np.log(2.0 * np.pi) + shocks**2)

    ll_value = log_like(
        np.array([], dtype=float),
        y_obs,
        equations,
        y_t,
        None,
        eps,
        ParamRegistry(params=[]),
        y_tm1=y_tm1,
        initial_covariance="zero",
    )

    np.testing.assert_allclose(ll_value, expected, atol=1e-9)
