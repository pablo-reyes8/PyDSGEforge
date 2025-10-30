import numpy as np
import sympy as sp

from src.model_builders.linear_system import build_matrices


def test_build_matrices_ar1_structure():
    """Linearization should recover canonical AR(1) structure."""
    x_t, x_tm1, eps_t = sp.symbols("x_t x_tm1 eps_t")

    equation = sp.Eq(x_t, sp.Rational(4, 5) * x_tm1 + eps_t)

    gamma0, gamma1, psi, pi, psi0, psi2, c = build_matrices(
        equations=[equation],
        y_t=[x_t],
        y_tp1=None,
        eps_t=[eps_t],
        y_tm1=[x_tm1],
        eta_t=None,
        param_values=None,
        steady_values=None,
    )

    np.testing.assert_allclose(gamma0, [[1.0]])
    np.testing.assert_allclose(gamma1, [[0.8]])
    np.testing.assert_allclose(psi, [[1.0]])
    assert pi.shape == (1, 0)
    np.testing.assert_allclose(psi0, np.zeros(1))
    np.testing.assert_allclose(psi2, np.eye(1))
    np.testing.assert_allclose(c, np.zeros(1))
