# Contributing to PyDSGEforge

Thanks for helping improve a general-purpose, pure-Python DSGE toolkit. Small,
well-tested pull requests are the easiest to review.

## Development setup

```bash
git clone https://github.com/pablo-reyes8/PyDSGEforge.git
cd PyDSGEforge
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
pre-commit install
make check
```

## Contribution workflow

1. Open an issue for substantial API, numerical, or model-format changes.
2. Create a focused branch and keep unrelated formatting out of the change.
3. Add a regression test for numerical fixes and a YAML example for new model
   features when appropriate.
4. Run `make check`. If the change affects likelihoods or solution matrices,
   also run `make dynare-check` and report the numerical tolerance.
5. Update documentation and `CHANGELOG.md` for user-visible behavior.
6. Open a pull request describing motivation, implementation, and verification.

## Numerical changes

Numerical equivalence matters more than visual similarity. State the parameter
point, data treatment, initialization convention, shock normalization, and
maximum absolute difference. Do not commit generated Dynare working trees;
refresh only the compact fixture described in `dynare_comprobation/README.md`.

By participating, you agree to follow the project `CODE_OF_CONDUCT.md`.
