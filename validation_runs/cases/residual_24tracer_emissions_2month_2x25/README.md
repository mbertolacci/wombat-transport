# residual_24tracer_emissions_2month_2x25

Purpose: run two months of the 24-tracer residual emissions case and compare
daily averaged SpeciesConc HISTORY output plus monthly restart output.

This is the longer emissions-enabled full-run case. It uses the checked-in
Wombat emissions specification and the matching GEOS-Chem HEMCO config.

Expected generated layout:

```text
validation_runs/work/residual_24tracer_emissions_2month_2x25/main/geoschem/
validation_runs/work/residual_24tracer_emissions_2month_2x25/main/wombat/
```

Compare existing outputs:

```bash
.venv/bin/python tools/compare_validation_run.py \
  validation_runs/cases/residual_24tracer_emissions_2month_2x25 \
  --mode restart
```

GEOS-Chem compile/run scripts live in `geoschem/main/`; see
`validation_runs/README.md` for materialization and `CodeDir` symlink setup.
