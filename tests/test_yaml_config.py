import numpy as np

from src.config import build_bundle_from_config, load_yaml_config
from src.dsge import DSGE


def test_build_bundle_from_self_contained_yaml_config():
    cfg = load_yaml_config("configs/tiny_ar1.yaml")
    bundle = build_bundle_from_config(cfg)

    model = bundle["model"]
    registry = bundle["registry"]

    assert isinstance(model, DSGE)
    assert registry.names == ["rho", "sigma_eps"]
    assert bundle["theta_econ"] == {"rho": 0.65, "sigma_eps": 0.35}


def test_dsge_from_yaml_returns_model_registry_and_start_values():
    model, registry, theta = DSGE.from_yaml("configs/tiny_ar1.yaml")

    assert model.summary().startswith("DSGE(equations=1")
    assert registry.names == ["rho", "sigma_eps"]
    np.testing.assert_allclose(
        registry.from_econ_dict(theta),
        [np.log(0.65 / 0.35), np.log(0.35)],
    )
