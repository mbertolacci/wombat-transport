# restart_chain_noemis_2x25

Purpose: run a no-emissions restart chain. The first window starts from the
same realistic GEOS-Chem restart for both engines. The second window restarts
GEOS-Chem from GEOS-Chem's first-window restart and Wombat from Wombat's
first-window restart.

This case is intended to catch restart-writing and restart-reading differences
that may not appear in a single continuous run.

Expected generated layout:

```text
validation_runs/work/restart_chain_noemis_2x25/window1/geoschem/
validation_runs/work/restart_chain_noemis_2x25/window1/wombat/
validation_runs/work/restart_chain_noemis_2x25/window2/geoschem/
validation_runs/work/restart_chain_noemis_2x25/window2/wombat/
```

Compare existing outputs:

```bash
.venv/bin/python tools/compare_validation_run.py \
  validation_runs/cases/restart_chain_noemis_2x25 \
  --mode restart
```

GEOS-Chem compile/run scripts live in each `geoschem/window*/` directory; see
`validation_runs/README.md` for materialization and `CodeDir` symlink setup.
