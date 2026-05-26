# YAML Configuration Mode

PyDSGEforge can build and run a model directly from a self-contained YAML file.
This is useful for experiments where equations, priors, transformations,
starting values, data, MAP settings, MCMC settings, and IRF settings should live
in one editable file.

## CLI

```bash
python scripts/run_pipeline.py --config configs/tiny_ar1.yaml
```

The CLI still supports the older factory-module mode. If `model.module` is
absent, the runner treats the YAML as a native model specification.

## Notebook

```python
from src.dsge import DSGE

model, registry, theta = DSGE.from_yaml("configs/tiny_ar1.yaml")

results = model.compute(
    registry=registry,
    theta_struct=theta,
    data=[[0.10], [0.18], [0.16], [0.23]],
    map=True,
    map_bounds="auto",
)
```

## Main Blocks

- `model.variables`: declares states, leads, lags, structural shocks, and
  expectation shocks.
- `model.equations`: SymPy-compatible equations. Use `=` for equalities.
- `parameters.specs`: declares symbols, transformations, priors, roles, and
  economic bounds.
- `parameters.theta_econ`: starting values in economic parameter space.
- `q.diag_params`: shock standard deviation parameters in shock order.
- `h.fixed` or `h.diag_params`: measurement-error covariance.
- `data.values`: inline data matrix, or `data.path` plus `data.columns`.
- `map`, `mcmc`, `likelihood`, `solver`, `irf`: runtime settings.

See `configs/tiny_ar1.yaml` for a minimal complete example.
