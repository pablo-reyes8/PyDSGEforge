import numpy as np
import sympy as sp

from src.inference.likelihoods import st_sp
from src.model_builders.linear_system import build_matrices
from src.specification.param_registry_class import ParamRegistry


def _nk_colombia_setup():
    x_t, pi_t, i_t = sp.symbols("x_t pi_t i_t")
    x_tp1, pi_tp1 = sp.symbols("x_tp1 pi_tp1")
    eps_d, eps_s, eps_m = sp.symbols("eps_d eps_s eps_m")

    beta = 0.992493637415447
    sigma = 2.2615937232618766
    kappa = 0.30150700850328505
    phi_pi = 1.0007509678898938
    phi_x = 0.18385016833348625

    equations = [
        sp.Eq(x_t, x_tp1 - (i_t - pi_tp1) / sigma + eps_d),
        sp.Eq(pi_t, beta * pi_tp1 + kappa * x_t + eps_s),
        sp.Eq(i_t, phi_pi * pi_t + phi_x * x_t + eps_m),
    ]

    exact_impact = np.linalg.inv(
        np.array(
            [
                [1.0, 0.0, 1.0 / sigma],
                [-kappa, 1.0, 0.0],
                [-phi_x, -phi_pi, 1.0],
            ]
        )
    )

    return (
        equations,
        [x_t, pi_t, i_t],
        [x_tp1, pi_tp1],
        [eps_d, eps_s, eps_m],
        exact_impact,
    )


def test_forward_variables_create_forecast_error_identities():
    equations, y_t, y_tp1, eps_t, _ = _nk_colombia_setup()

    gamma0, gamma1, _, pi, _, psi2, _ = build_matrices(
        equations=equations,
        y_t=y_t,
        y_tp1=y_tp1,
        eps_t=eps_t,
    )

    # Augmented state is [x_t, pi_t, i_t, E_t x_{t+1}, E_t pi_{t+1}].
    np.testing.assert_allclose(gamma0[3:, :3], [[1, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(gamma0[3:, 3:], np.zeros((2, 2)))
    np.testing.assert_allclose(gamma1[3:, 3:], np.eye(2))
    np.testing.assert_allclose(pi[3:, -2:], np.eye(2))
    np.testing.assert_allclose(psi2, np.hstack([np.eye(3), np.zeros((3, 2))]))


def test_nk_colombia_forward_solution_matches_closed_form_impact():
    equations, y_t, y_tp1, eps_t, exact_impact = _nk_colombia_setup()

    transition, _, impact, eu, _, measurement = st_sp(
        np.array([], dtype=float),
        equations,
        y_t,
        y_tp1,
        eps_t,
        ParamRegistry(params=[]),
        div=1.0 + 1e-6,
    )

    assert tuple(eu) == (1, 1)
    np.testing.assert_allclose(transition, np.zeros((5, 5)), atol=1e-12)
    np.testing.assert_allclose(measurement @ impact, exact_impact, atol=1e-10)

