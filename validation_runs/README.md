# Validation Runs

This directory contains named full-run GEOS-Chem vs Wombat validation cases.
These are integration experiments, not unit tests.

Committed case directories contain only lightweight configs, manifests, and
purpose notes. Generated NetCDF, logs, build directories, and materialized run
directories belong under `validation_runs/work/`, which is ignored.

## Layout

```text
validation_runs/
  cases/<case>/
    README.md
    case.yml
    geoschem/<stage>/
    wombat/<stage>/
  work/<case>/
    <stage>/geoschem/
    <stage>/wombat/
    compare/<mode>/
```

Model behavior lives in the checked-in GEOS-Chem and Wombat run configs. The
case manifest describes orchestration and comparisons only: stages, restart
dependencies, output locations, and which collections should be compared.

## Compare Existing Outputs

The first tool is compare-only. It expects outputs to already exist in
`validation_runs/work/<case>/...`.

```bash
.venv/bin/python tools/compare_validation_run.py \
  validation_runs/cases/realistic_restart_noemis \
  --mode quick
```

The tool writes `metrics.csv` and `summary.json` under
`validation_runs/work/<case>/compare/<mode>/`.

Modes are case-defined. `quick` is intended for normal HISTORY diagnostics.
`restart` includes restart-file comparisons where a case defines them.
Instrumented tracing remains separate and opt-in via the GC harness tools.

## GEOS-Chem Stage Scripts

Each GEOS-Chem stage template includes:

- `compile_geoschem.sh`
- `run_geoschem.sh`

Materialize a stage by copying the template directory into the matching
`validation_runs/work/<case>/<stage>/geoschem/` directory, then create a
`CodeDir` symlink in that run directory:

```bash
ln -s ../../../../../GCClassic validation_runs/work/<case>/<stage>/geoschem/CodeDir
```

Compile in the `wombat-v3-forward` conda environment:

```bash
conda run -n wombat-v3-forward ./compile_geoschem.sh
```

Run from the stage run directory:

```bash
./run_geoschem.sh
```

The run script defaults to `OMP_NUM_THREADS=1`, writes normal outputs in the
run directory, and tees stdout/stderr to `gcclassic.log`.

For multi-stage cases that reuse the same GEOS-Chem build, later-stage
`compile_geoschem.sh` scripts may symlink `gcclassic` from an earlier stage
instead of rebuilding. The restart-chain second window does this by default
from `../../window1/geoschem`; set `GC_REUSE_RUN_DIR` to override that source.
