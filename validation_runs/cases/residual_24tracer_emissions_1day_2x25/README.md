# residual_24tracer_emissions_1day_2x25

Purpose: run the first day of the 24-tracer residual emissions case and compare
normal three-hourly SpeciesConc HISTORY output.

This is the quick emissions-enabled full-run case. It uses the checked-in
Wombat emissions specification and the matching GEOS-Chem HEMCO config.

Expected generated layout:

```text
validation_runs/work/residual_24tracer_emissions_1day_2x25/main/geoschem/
validation_runs/work/residual_24tracer_emissions_1day_2x25/main/wombat/
```

Compare existing outputs:

```bash
.venv/bin/python tools/compare_validation_run.py \
  validation_runs/cases/residual_24tracer_emissions_1day_2x25 \
  --mode quick
```

GEOS-Chem compile/run scripts live in `geoschem/main/`; see
`validation_runs/README.md` for materialization and `CodeDir` symlink setup.
