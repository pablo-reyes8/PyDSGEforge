import numpy as np
import sympy as sp

from src.analysis.impulse_responses import compute_irfs
from src.specification.param_registry_class import ParamRegistry


def test_irf_places_structural_shock_on_impact_period():
    x_t, x_tm1, eps_t = sp.symbols("x_t x_tm1 eps_t")
    rho = 0.8

    result = compute_irfs(
        draws_work=np.empty((1, 0)),
        registry=ParamRegistry(params=[]),
        equations=[sp.Eq(x_t, rho * x_tm1 + eps_t)],
        y_t=[x_t],
        y_tp1=None,
        eps_t=[eps_t],
        y_tm1=[x_tm1],
        horizon=4,
        quantiles=(0.5,),
        div=1.0 + 1e-6,
    )

    summary = result["eps_t"]["summary"]
    median_path = summary[0, 0, :]

    np.testing.assert_allclose(median_path, [1.0, 0.8, 0.64, 0.512], atol=1e-12)
