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
  validation_runs/cases/realistic_restart_noemis_2x25 \
  --mode quick
```

The tool writes `metrics.csv` and `summary.json` under
`validation_runs/work/<case>/compare/<mode>/`.

The 4x5 cases require the generated parity-test restart documented in
`external_data/README.md` and MERRA-2 data under
`external_data/geoschem/GEOS_4x5/MERRA2`. They reuse the established case
templates and apply their resolution-specific paths while materializing the
run directories.

Modes are case-defined. `quick` is intended for normal HISTORY diagnostics.
`restart` includes restart-file comparisons where a case defines them.
Instrumented tracing remains separate and opt-in via the GC harness tools.

## ObsOperator timing matrix

The checked-in validation configs enable ObsOperator for both engines. Daily
gzip inputs live outside Git under `external_data/obsoperator/`; the matrix
runner expands the same files to ordinary YAML for GEOS-Chem and keeps the gzip
inputs for Wombat. It reuses the compiled executables from
`validation_runs/work/` and creates fresh ignored work roots for each thread
count:

```bash
.venv/bin/python tools/run_validation_matrix.py \
  --threads 1 2 \
  --work-prefix validation_runs/work/obsoperator
```

For example, the command above writes one-thread results below
`validation_runs/work/obsoperator_t1/` and two-thread results below
`validation_runs/work/obsoperator_t2/`. Each engine stage records
`timing.txt`, `validation_timing.json`, and its normal log. Runs are sequential
so the timed processes do not compete with each other.

The runner refuses to overwrite an existing matrix directory. Use a new
`--work-prefix` below `validation_runs/work/` for a fresh timing experiment.
Pass one or more case directories positionally to run a subset, or
`--prepare-only` to validate materialization without starting either model.

Pass `--geoschem-executable /path/to/gcclassic` to use one freshly compiled
GEOS-Chem executable for every case and stage. The timing metadata records both
the Wombat repository commit and the nested GEOS-Chem source commit.

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
