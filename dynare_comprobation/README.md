# Dynare parity fixture

This directory is the small, auditable reference behind the Dynare comparison
shown in the project README. It keeps inputs and numerical evidence, not the
large working tree that Dynare generates on every run.

## What is committed

- `nk_colombia.mod`: the three-equation New Keynesian model, priors, and
  estimation command used in Dynare/Octave.
- `data/colombia_nk_quarterly.csv`: the 82 observations supplied to both
  implementations. Dynare applies `prefilter=1`; PyDSGEforge applies the same
  demeaning inside its likelihood.
- `reference/dynare_mode.json`: posterior mode and Dynare objective value.
- `reference/dynare_posterior_draws.csv`: 500 retained parameter draws and
  their Dynare log-posterior values.
- `reference/parity_metrics.json`: committed numerical tolerances and results.
- `validate.py`: independent Python-side likelihood, posterior, BK, and IRF
  check using only this fixture and the library.

Generated `+nk_colombia/`, `Output/`, `metropolis/`, `graphs/`, bytecode,
figures, logs, and matrices are intentionally ignored. They are products of a
run, not source material.

## Reproduce the checks

From the repository root, validate the committed Dynare result in Python:

```bash
python dynare_comprobation/validate.py
```

With Dynare available, reproduce the Octave/MATLAB side:

```bash
cd dynare_comprobation
dynare nk_colombia.mod
```

Dynare 7 accepts the checked-in CSV directly. Older versions can use an
equivalent MAT file with columns named `x`, `pi`, and `i`.

To refresh this fixture from the legacy local working directory after a new
Dynare run:

```bash
python scripts/export_dynare_reference.py --source nk_colombia
python scripts/build_showcase.py --only-dynare --dynare-draws 12000
```

The exporter is deliberately one-way: it extracts only the compact evidence
needed for review and never copies generated Dynare code or workspace files.
