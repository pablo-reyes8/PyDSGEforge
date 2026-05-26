# Testing

The test suite covers:

- canonical matrix construction for a known AR(1),
- Gensys Blanchard-Kahn edge cases,
- permutation invariance of states and shocks,
- Kalman likelihood constants and known initial states,
- IRF impact-period convention,
- public compatibility imports.

Run locally:

```bash
python -m pytest
```

Run with coverage:

```bash
python -m pytest --cov=src --cov-report=term-missing
```
