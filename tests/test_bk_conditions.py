import numpy as np

from src.solvers.gensys import gensys

DIVIDER = 1.0 + 1e-6


def _zeros(n, m=0):
    return np.zeros((n, m), dtype=float)


def test_gensys_all_stable_roots_unique_solution():
    """Stable eigenvalues should pass BK with no jump variables required."""
    gamma0 = np.eye(2)
    gamma1 = np.array([[0.8, 0.1], [0.0, 0.6]])
    c = np.zeros(2)
    psi = np.eye(2)
    pi = _zeros(2)

    g1, c_out, impact, eu = gensys(gamma0, gamma1, c, psi, pi, div=DIVIDER)

    np.testing.assert_allclose(g1, gamma1, atol=1e-8)
    np.testing.assert_allclose(c_out, np.zeros(2))
    np.testing.assert_allclose(impact, psi, atol=1e-8)
    assert tuple(eu) == (1, 1)


def test_gensys_requires_jump_for_single_unstable_root():
    """One unstable root with no jumps should fail the BK existence check."""
    gamma0 = np.eye(1)
    gamma1 = np.array([[1.2]])
    c = np.zeros(1)
    psi = np.zeros((1, 1))
    pi = _zeros(1)

    g1, c_out, impact, eu = gensys(gamma0, gamma1, c, psi, pi, div=DIVIDER)

    assert g1.size == 1
    assert tuple(eu) == (0, 0)
    np.testing.assert_allclose(c_out, np.zeros(1))
    np.testing.assert_allclose(impact, np.zeros((1, 1)))


def test_gensys_handles_unstable_root_when_jump_available():
    """Providing a jump variable for an unstable root should restore determinacy."""
    gamma0 = np.eye(1)
    gamma1 = np.array([[1.2]])
    c = np.zeros(1)
    psi = np.zeros((1, 1))
    pi = np.array([[1.0]])

    _, _, _, eu = gensys(gamma0, gamma1, c, psi, pi, div=DIVIDER)

    assert tuple(eu) == (1, 1)


def test_gensys_detects_all_unstable_without_enough_expectational_errors():
    """With zero stable roots and insufficient Pi rank, the model should be indeterminate."""
    gamma0 = np.eye(2)
    gamma1 = np.diag([1.1, 1.3])
    c = np.zeros(2)
    psi = np.eye(2)
    pi = np.array([[1.0], [0.0]])  # rank-1, cannot kill both explosive roots

    _, _, _, eu = gensys(gamma0, gamma1, c, psi, pi, div=DIVIDER)

    assert tuple(eu) == (0, 0)


def test_gensys_singular_pencil_reports_degeneracy():
    """Singular (S,T) pencil should trigger the early degeneracy flag."""
    gamma0 = np.zeros((1, 1))
    gamma1 = np.zeros((1, 1))
    c = np.zeros(1)
    psi = np.zeros((1, 1))
    pi = np.zeros((1, 1))

    g1, c_out, impact, eu = gensys(gamma0, gamma1, c, psi, pi, div=DIVIDER)

    assert g1.size == 0
    assert impact.size == 0
    assert tuple(eu) == (-2, -2)
    np.testing.assert_allclose(c_out, np.zeros(0))


def test_state_and_shock_permutation_is_consistent():
    """Permutation of states and shocks should map the solution via the same permutation."""
    gamma0 = np.eye(2)
    gamma1 = np.array([[0.7, 0.2], [0.1, 0.5]])
    c = np.zeros(2)
    psi = np.array([[1.0, 0.4], [0.3, 0.9]])
    pi = _zeros(2)

    g1_orig, c_orig, impact_orig, eu_orig = gensys(gamma0, gamma1, c, psi, pi, div=DIVIDER)
    assert tuple(eu_orig) == (1, 1)

    perm = np.array([[0, 1], [1, 0]])
    gamma0_perm = perm @ gamma0 @ perm.T
    gamma1_perm = perm @ gamma1 @ perm.T
    psi_perm = perm @ psi
    pi_perm = perm @ pi

    g1_perm, c_perm, impact_perm, eu_perm = gensys(
        gamma0_perm, gamma1_perm, c, psi_perm, pi_perm, div=DIVIDER
    )

    np.testing.assert_allclose(g1_perm, perm @ g1_orig @ perm.T, atol=1e-8)
    np.testing.assert_allclose(impact_perm, perm @ impact_orig, atol=1e-8)
    np.testing.assert_allclose(c_perm, perm @ c_orig, atol=1e-8)
    assert tuple(eu_perm) == tuple(eu_orig)
