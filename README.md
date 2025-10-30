# DSGE Modeling Toolkit (Python)

This repository captures the core building blocks of a Dynare-style DSGE workflow implemented in Python. The code currently lives under `src/` and focuses on symbolic model definition, linearization, solution, and Bayesian inference primitives. Higher-level orchestration (for example, a `DSGE` facade, data loaders, and experiment scripts) is still under active development and intentionally omitted from this snapshot.

## Design Overview

The toolkit mirrors the familiar pipeline from model specification to empirical analysis:

1. **Specification (`src/specification/`)** - Parameter metadata lives in `ParamSpec` objects, while `ParamRegistry` enforces canonical ordering, handles work-to-economic transformations, applies priors, and builds structural covariance matrices `Q` and `H`. This gives estimators and solvers a single source of truth for the economic environment.
2. **Model Builders (`src/model_builders/`)** - Symbolic equilibrium conditions (SymPy) are converted into linear state-space matrices (`Gamma0`, `Gamma1`, `Psi`, `Pi`, `c`) via `build_matrices`. Optional measurement mappings (`Psi0`, `Psi2`) accommodate observation equations and auxiliary shocks.
3. **Solvers (`src/solvers/`)** - `gensys.py` wraps the generalized Schur decomposition to check Blanchard-Kahn conditions and recover transition matrices `(G1, impact)` from the linear system. The implementation mirrors the canonical Gensys algorithm used by Dynare and other DSGE toolkits.
4. **Inference (`src/inference/`)** - The Kalman-based Gaussian likelihood, posterior assembly, MAP optimizer, and Metropolis-Hastings sampler live here. All routines operate in the unrestricted parameter space and rely on the registry to evaluate priors, measurement systems, and stochastic blocks.
5. **Analysis (`src/analysis/`)** - Contains impulse-response utilities that propagate draws through the solved model and plotting helpers for MCMC diagnostics.
6. **Transformations (`src/transformations/`)** - Centralizes forward and inverse mappings (identity, exp, logistic, tanh01) that keep parameter constraints explicit and differentiable.

These modules are designed to be composable: a notebook or future `DSGE` class can register a model, call `build_matrices`, solve it with `gensys`, evaluate the likelihood or posterior, and then feed draws into IRF or diagnostic helpers without duplicating boilerplate.

## Current Status

- Core symbolic-to-linear builder and measurement helpers.
- Gensys solver with robustness features (eigenvalue sorting, fallback logic).
- Bayesian primitives: Gaussian state-space likelihood, MAP estimation, and Metropolis-Hastings with adaptive scaling.
- Analysis utilities: impulse-response envelopes and MCMC trace/summary plots.
- Pending integration layers: unified `DSGE` facade, dataset loaders, experiment scripts, and automated tests.

Run instructions are intentionally left out for now; the top-level orchestration layer will define the end-to-end workflow once those components stabilize.

## `src/` Layout (working set)

```
src/
  analysis/
    impulse_responses.py      # IRF simulation and plotting helpers
    mcmc_diagnostics.py       # Posterior histograms and trace plots
  inference/
    likelihoods.py            # State-space solution and Gaussian log-likelihood
    posteriors.py             # Prior combination and posterior helpers
    map.py                    # MAP optimizer and Hessian-based proposals
    mcmc.py                   # Metropolis-Hastings sampler utilities
  model_builders/
    linear_system.py          # SymPy residual handling and linearization
  solvers/
    gensys.py                 # Blanchard-Kahn compliant Gensys solver
  specification/
    param_registry_class.py   # Registry glue between parameters, Q/H, priors
    param_specifications.py   # ParamSpec, PriorSpec, measurement structures
  transformations/
    param_transformations.py  # Forward and inverse parameter mappings
```

## Modeling Example

The snippet below sketches a textbook New Keynesian system expressed with SymPy. Each block mirrors a familiar economic concept: state variables at time `t`, forward-looking variables, structural shocks, and deep parameters. Written in standard notation, the equilibrium conditions are

$$
\begin{aligned}
x_t &= x_{t+1} - \frac{1}{\sigma}\left(i_t - \pi_{t+1}\right) + \varepsilon_t^{d}, \\
\pi_t &= \beta\,\pi_{t+1} + \kappa\,x_t + \varepsilon_t^{s}, \\
i_t &= \phi_{\pi}\,\pi_t + \phi_{x}\,x_t + \varepsilon_t^{m}.
\end{aligned}
$$


```python
import sympy as sp

# State variables (IS gap, inflation, nominal rate)
x_t, pi_t, i_t = sp.symbols("x_t pi_t i_t")

# Leads entering expectations
x_tp1, pi_tp1 = sp.symbols("x_tp1 pi_tp1")

# Structural shocks: demand, supply, monetary policy
eps_d, eps_s, eps_m = sp.symbols("eps_d eps_s eps_m")

# Deep parameters: discounting, preferences, price stickiness, policy, shock stds
beta, sigma, kappa, phi_pi, phi_x, sig_d, sig_s, sig_m = sp.symbols(
    "beta sigma kappa phi_pi phi_x sig_d sig_s sig_m")

# Equilibrium conditions (residuals = 0)
eq1 = sp.Eq(x_t, x_tp1 - (1 / sigma) * (i_t - pi_tp1) + eps_d)   # Dynamic IS curve
eq2 = sp.Eq(pi_t, beta * pi_tp1 + kappa * x_t + eps_s)           # NK Phillips curve
eq3 = sp.Eq(i_t, phi_pi * pi_t + phi_x * x_t + eps_m)            # Taylor rule

equations = [eq1, eq2, eq3]
y_t       = [x_t, pi_t, i_t]         # endogenous state vector at t
y_tp1     = [x_tp1, pi_tp1]          # expectations block
eps_t     = [eps_d, eps_s, eps_m]    # structural shocks
y_tm1     = None                     # no lags here
eta_t     = None                     # no expectation shocks
```

Passing these objects to `build_matrices(...)` linearizes the system around its steady state, producing the familiar `Gamma0`, `Gamma1`, `Psi`, `Pi`, and `c` matrices consumed by `gensys`. The explicit lists (`y_t`, `y_tp1`, `eps_t`) fix the ordering inherited by the solver, Kalman filter, and IRF routines.

Parameter handling is centralized in `ParamRegistry`, which maps between free parameters in `R^k` and economically restricted values, evaluates priors, and constructs the stochastic blocks `Q` and `H`. A minimal registry for the same model looks like:

```python
import numpy as np
from src.specification.param_registry_class import ParamRegistry
from src.specification.param_specifications import ParamSpec, PriorSpec, QSpec, HSpec

REG_NK = ParamRegistry(
    params=[
        ParamSpec("beta",   beta,   transform="logistic",
                  prior=PriorSpec("uniform",   {"loc": 0.0, "scale": 1.0}), role="struct"),
        ParamSpec("sigma",  sigma,  transform="exp",
                  prior=PriorSpec("gamma",     {"a": 2.0, "scale": 1.0}),   role="struct"),
        ParamSpec("kappa",  kappa,  transform="exp",
                  prior=PriorSpec("gamma",     {"a": 2.0, "scale": 0.1}),   role="struct"),
        ParamSpec("phi_pi", phi_pi, transform="exp",
                  prior=PriorSpec("normal",    {"loc": np.log(1.5), "scale": 0.4}), role="struct"),
        ParamSpec("phi_x",  phi_x,  transform="exp",
                  prior=PriorSpec("normal",    {"loc": np.log(0.5), "scale": 0.4}), role="struct"),
        ParamSpec("sig_d",  sig_d,  transform="exp",
                  prior=PriorSpec("invgamma", {"a": 2.2, "scale": 0.5}), role="shock_std"),
        ParamSpec("sig_s",  sig_s,  transform="exp",
                  prior=PriorSpec("invgamma", {"a": 2.2, "scale": 0.5}), role="shock_std"),
        ParamSpec("sig_m",  sig_m,  transform="exp",
                  prior=PriorSpec("invgamma", {"a": 2.2, "scale": 0.5}), role="shock_std"),
    ],
    qspec=QSpec(diag_params=["sig_d", "sig_s", "sig_m"]),
    hspec=HSpec(fixed=np.zeros((3, 3))),
    measurement_builder=None
)
```

`ParamRegistry` is the glue between symbolic economics and numerical routines: it enforces parameter domains via smooth transforms (`id`, `exp`, `logistic`, `tanh01`), exposes priors to MAP and MCMC algorithms, and produces consistent covariance matrices that flow into the Kalman filter. Even before a high-level `DSGE` class exists, this pairing of symbolic equations and registry metadata delivers a fully functional pipeline for linearization, solution, and Bayesian estimation.

## Roadmap

- Wrap the existing primitives into a cohesive `DSGE` class with configuration-driven model assembly.
- Add regression tests and synthetic examples that validate the linearization and solver against known models.
- Publish structured experiment scripts (data ingestion, filtering, estimation, IRFs) once the integration layer is finalized.
- Document execution and configuration steps alongside reproducible notebooks.

Contributions and feedback on the current architecture are welcome while the higher-level API solidifies.

## License

Released under the MIT License. See `LICENSE` for the full text.

## References

- Sims, C. A. (2001). "Solving Linear Rational Expectations Models." *Computational Economics*, 20(1-2), 1-20. https://doi.org/10.1023/A:1013825826056  
  The core `gensys` routine implemented here is a direct adaptation of Sims' canonical algorithm; the linear solution layers in this repository would not exist without his code and documentation.

- Jacobo, J. (2025). *Una introduccion a los metodos de maxima entropia y de inferencia bayesiana en econometria*. Universidad Externado de Colombia, Bogota.  
  The Bayesian workflow (MAP, MCMC, diagnostics) follows the guidance and code patterns taught by Professor Jacobo in Advanced Quantitative Methods; his text and lecture materials are the primary reference for the inference stack implemented here.
