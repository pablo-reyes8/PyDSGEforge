# Usage

## Install

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest
```

## Minimal Model

```python
import numpy as np
import sympy as sp

from src.dsge import DSGE
from src.specification.param_registry_class import ParamRegistry

x_t, x_tm1, eps_t = sp.symbols("x_t x_tm1 eps_t")

model = DSGE(
    equations=[sp.Eq(x_t, 0.8 * x_tm1 + eps_t)],
    y_t=[x_t],
    y_tm1=[x_tm1],
    eps_t=[eps_t],
)

registry = ParamRegistry(params=[])
data = np.asarray([[0.1], [0.03], [0.224]])

result = model.compute(
    registry=registry,
    theta_struct=[],
    data=data,
    compute_steady=False,
    map=False,
    run_mcmc=False,
)
```

## Configured Runs

The CLI reads `configs/default.yaml` plus an experiment-specific YAML:

```bash
python scripts/run_pipeline.py --config configs/nk_example.yaml --dry-run
```

For a real run, provide `model.module` and `model.factory`. The factory receives
the merged config and returns a dictionary with at least:

```python
{
    "model": model,
    "registry": registry,
    "theta_econ": {...},  # or "theta_work": np.ndarray
    "data": data,        # optional if data.path/data.columns are configured
}
```
