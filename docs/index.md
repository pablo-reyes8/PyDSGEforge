# PyDSGEforge

PyDSGEforge is a Python toolkit for small and medium-scale linear DSGE workflows:
symbolic model specification, linearization, Sims-style solution, Gaussian
state-space likelihood, MAP estimation, Metropolis-Hastings sampling, and impulse
responses.

The project is intentionally close to Dynare/Octave conventions. A model is
defined with SymPy equations and ordered lists of current variables, leads, lags,
structural shocks, and optional expectational errors. That ordering is carried
through the linear system, Kalman filter, posterior routines, and IRF utilities.

## What Is A DSGE?

A dynamic stochastic general equilibrium model describes how forward-looking
agents make choices over time under uncertainty. In practice, the code in this
repository works with a linearized system:

```text
Gamma0 y_t = Gamma1 y_{t-1} + C + Psi eps_t + Pi eta_t
```

`gensys` solves that system into:

```text
y_t = G1 y_{t-1} + C + impact eps_t
```

The Bayesian layer combines that transition equation with a measurement equation:

```text
obs_t = Psi0 + Psi2 y_t + measurement_error_t
```

and evaluates a Gaussian Kalman likelihood plus parameter priors.

## Repository Layout

```text
src/
  specification/      parameter transforms, priors, Q/H, measurement specs
  model_builders/     SymPy linearization and steady-state helpers
  solvers/            gensys and Blanchard-Kahn checks
  inference/          likelihood, posterior, MAP, MCMC
  analysis/           IRFs and MCMC diagnostics
configs/              reusable YAML experiment settings
scripts/              command-line pipeline entry points
tests/                numerical and API regression tests
```

Start with [Usage](usage.md), then read [Numerical Conventions](numerical-conventions.md)
before comparing output against Dynare or Octave.
