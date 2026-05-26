import numpy as np
from scipy.special import gammaln
from scipy.stats import beta as scipy_beta

from src.dgse import DSGE
from src.specification.param_specifications import PriorSpec
from src.transformations import (
    beta_from_moments,
    dynare_inv_gamma1_from_moments,
    gamma_from_moments,
)


def test_beta_prior_matches_scipy_generalized_beta():
    prior = PriorSpec("beta", {"a": 391.05, "b": 3.95, "loc": 0.0, "scale": 1.0})

    actual = prior.logpdf(0.992493637415447)
    expected = scipy_beta.logpdf(0.992493637415447, a=391.05, b=3.95)

    np.testing.assert_allclose(actual, expected)


def test_dynare_inverse_gamma_type_1_prior_formula():
    s = 0.10875628191089724
    nu = 4.175125642356515
    x = 3.263933865830626
    prior = PriorSpec("invgamma1", {"s": s, "nu": nu})

    expected = (
        np.log(2.0)
        - gammaln(0.5 * nu)
        - 0.5 * nu * (np.log(2.0) - np.log(s))
        - (nu + 1.0) * np.log(x)
        - 0.5 * s / (x * x)
    )

    np.testing.assert_allclose(prior.logpdf(x), expected)


def test_public_prior_helpers_are_exported():
    assert gamma_from_moments(1.5, 0.5) == (9.0, 1.0 / 6.0)

    a, b = beta_from_moments(0.99, 0.005)
    assert a > 0.0
    assert b > 0.0

    s, nu = dynare_inv_gamma1_from_moments(0.20, 0.10)
    assert s > 0.0
    assert nu > 2.0


def test_prior_plot_distribution_builder_supports_beta_and_dynare_invgamma1():
    beta_dist, beta_support = DSGE._build_prior_distribution(
        PriorSpec("beta", {"a": 391.05, "b": 3.95, "loc": 0.0, "scale": 1.0})
    )
    np.testing.assert_allclose(beta_support, (0.0, 1.0))
    assert np.all(np.isfinite(beta_dist.ppf([0.01, 0.99])))

    ig1_dist, ig1_support = DSGE._build_prior_distribution(
        PriorSpec("invgamma1", {"s": 0.10875628191089724, "nu": 4.175125642356515})
    )
    np.testing.assert_allclose(ig1_support, (0.0, np.inf))
    assert np.all(np.isfinite(ig1_dist.ppf([0.01, 0.99])))
    assert np.isfinite(ig1_dist.mean())
    assert np.isfinite(ig1_dist.std())
