# API Notes

## `ParamRegistry`

`ParamRegistry` owns parameter order and maps the unrestricted work vector to
economic values. It also evaluates priors and builds structural covariance
matrices:

```python
registry.to_econ_dict(theta_work)
registry.from_econ_dict(theta_econ)
registry.log_prior(theta_work)
registry.build_Q(theta_work, n_eps)
registry.build_H(theta_work, n_y)
```

## `build_matrices`

`build_matrices` converts SymPy residuals or equalities into the canonical linear
system consumed by `gensys`.

## `log_like` And `log_posterior`

`log_like` evaluates the Gaussian state-space likelihood. `log_posterior` adds the
registry prior. Use `include_jacobian=True` only when the posterior density should
be expressed in the unrestricted work space.

## `DSGE`

The high-level facade is available from:

```python
from src.dsge import DSGE
```

`src.dgse` remains available for old notebooks.
