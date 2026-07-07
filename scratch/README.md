# Scratch Experiments

This directory holds small, inspectable experiments that are useful for
transport performance work but are not part of the production API.

Generated outputs belong under `scratch/output/`, which is ignored by git.

Current layout experiments:

- `fyppm_layout_experiment.py` compares current `(tracer, lat, lon)` row PPM
  layout with tracer-last `(lat, lon, tracer)`.
- `fzppm_layout_experiment.py` compares current `(tracer, lev, lat, lon)`
  vertical PPM layout with tracer-last `(lev, lat, lon, tracer)`.

Run from the repo root with:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 scratch/fyppm_layout_experiment.py
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 scratch/fzppm_layout_experiment.py
```
