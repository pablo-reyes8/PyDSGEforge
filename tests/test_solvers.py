import numpy as np

from src.solvers.gensys import gensys


def test_gensys_simple_ar1():
    """Gensys should reproduce the transition matrix for a stable AR(1)."""
    gamma0 = np.array([[1.0]])
    gamma1 = np.array([[0.8]])
    c = np.zeros(1)
    psi = np.array([[1.0]])
    pi = np.zeros((1, 0))

    g1, c_out, impact, eu = gensys(gamma0, gamma1, c, psi, pi)

    np.testing.assert_allclose(g1, [[0.8]])
    np.testing.assert_allclose(c_out, np.zeros(1))
    np.testing.assert_allclose(impact, [[1.0]])
    assert tuple(eu) == (1, 1)
