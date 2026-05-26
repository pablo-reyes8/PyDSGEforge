# Numerical Conventions

## Gensys Divider

If `div <= 0`, the solver now uses `1.0 + 1e-6`. This avoids the pathological
case where `div=0` classifies ordinary stable AR roots as unstable.

## Kalman Initialization

`log_like` defaults to `initial_covariance="stationary"` for stable systems. You
can choose:

```python
log_like(..., initial_covariance="stationary")
log_like(..., initial_covariance="diffuse", diffuse_scale=1e6)
log_like(..., initial_covariance="zero")
log_like(..., initial_covariance=np.eye(n_state))
```

Use the same initialization as the reference Octave/Dynare implementation when
comparing exact likelihood values.

## Constants

The solved transition constant `C` is included in the Kalman prediction:

```text
s_hat = C + G1 s_previous
```

Dropping this term is only harmless for models already expressed as deviations
from steady state with zero intercept.

## IRFs

Impulse responses are stored as deviations from baseline. A one-standard-deviation
structural shock is applied at horizon `0`, so an AR(1) with `rho=0.8` reports:

```text
[1.0, 0.8, 0.64, ...]
```

This matches the common Dynare-style impact-period convention.
