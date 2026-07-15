# restart_chain_noemis_2month_2x25

Purpose: run a two-month no-emissions restart chain. The first one-month
window starts from the same realistic GEOS-Chem restart for both engines. The
second one-month window restarts GEOS-Chem from GEOS-Chem's first-window
restart and Wombat from Wombat's first-window restart.

This case is intended to catch restart-writing, restart-reading, and daily
averaged SpeciesConc drift over a longer window.

Expected generated layout:

```text
validation_runs/work/restart_chain_noemis_2month_2x25/window1/geoschem/
validation_runs/work/restart_chain_noemis_2month_2x25/window1/wombat/
validation_runs/work/restart_chain_noemis_2month_2x25/window2/geoschem/
validation_runs/work/restart_chain_noemis_2month_2x25/window2/wombat/
```

Compare existing outputs:

```bash
.venv/bin/python tools/compare_validation_run.py \
  validation_runs/cases/restart_chain_noemis_2month_2x25 \
  --mode restart
```

GEOS-Chem compile/run scripts live in each `geoschem/window*/` directory; see
`validation_runs/README.md` for materialization and `CodeDir` symlink setup.
