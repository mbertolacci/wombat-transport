# realistic_restart_noemis

Purpose: transport a spatially varying CO2 restart field with no emissions for
two days, then compare normal three-hourly SpeciesConc HISTORY output between
GEOS-Chem and Wombat.

This is the quick full-run parity case for realistic restart initialization
without emissions. It is not instrumented.

Expected generated layout:

```text
validation_runs/work/realistic_restart_noemis/main/geoschem/
validation_runs/work/realistic_restart_noemis/main/wombat/
```

Compare existing outputs:

```bash
.venv/bin/python tools/compare_validation_run.py \
  validation_runs/cases/realistic_restart_noemis \
  --mode quick
```

GEOS-Chem compile/run scripts live in `geoschem/main/`; see
`validation_runs/README.md` for materialization and `CodeDir` symlink setup.
