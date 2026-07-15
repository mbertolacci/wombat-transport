# Wombat Transport

Wombat Transport is a Python/NumPy prototype for reproducing the relevant
GEOS-Chem Classic transport-only behavior for many CO2-like tracers. The goal
is GEOS-Chem numerical parity first, then efficient batched multi-tracer
transport.

The current target is deliberately narrow: global GEOS 2x2.5, 47 vertical
levels, 600 s transport timestep, TPCORE with negative-value filling, non-local
PBL mixing, and convection. Chemistry, dry deposition, and wet deposition are
not implemented for this prototype.

## Current Status

- Internal tracer fields use canonical `(lev, lat, lon, tracer)` order.
- The transport driver runs `TPCORE -> VDIFF -> convection`, matching the
  transport-only GEOS-Chem workflow for the current target.
- TPCORE, VDIFF, and convection have GEOS-Chem-backed parity harness coverage,
  including full-grid one-step fixtures and a cached
  `TPCORE -> VDIFF -> convection` handoff fixture set.
- Native emissions read configured source fields, scale factors, optional
  source dimensions such as `npft`, and GEOS 2x2.5 polar-row regridding
  behavior validated against GEOS-Chem/HEMCO.
- Output supports HISTORY-like `SpeciesConcVV_*` and `SpeciesRst_*` NetCDF
  collections, with configurable compression, chunking, and float dtype.
- The forcing loader follows the relevant GEOS-Chem MERRA2 cadence, record
  selection, and interpolation behavior for the current transport fields.
- Numba acceleration is available for the current performance path; see
  `performance.md`.

Short-run comparisons are currently consistent with GEOS-Chem for the tested
base no-emissions two-day window and residual 24-tracer one-day window. This is
not yet a long-horizon parity claim: monthly residual restart drift and
multi-week/month transport-only behavior still need explicit comparison.

## Repository Layout

- `src/wombat_transport/`: package source.
  - `run.py`, `runner.py`, `run_config.py`: CLI/config loading and simulation
    orchestration.
  - `transport/`: TPCORE, VDIFF/PBL, convection, forcing, pressure, metrics,
    and driver code.
  - `emissions.py`: native emissions operator.
  - `output.py`: HISTORY-like output writer.
  - `io.py`, `fields.py`, `grid.py`, `species.py`: restart, diagnostic, grid,
    and species helpers.
- `base/`: one-tracer GEOS-Chem reference run with restart and short-window
  diagnostics.
- `residual_20140901_part001_split01/`: 24-tracer residual GEOS-Chem reference
  run.
- `base_wombat/` and `residual_20140901_part001_split01_wombat/`: matching
  Wombat run configs.
- `fluxes/`: local emissions/flux inputs for the residual target.
- `GCClassic/`: vendored GEOS-Chem Classic reference source.
- `ExtData`: symlink to local GEOS-Chem meteorology and input data.
- `tools/gc_harness/`: GEOS-Chem-backed operator, output, met, oracle, and
  full-run trace harnesses.
- `tools/hemco_harness/`: standalone HEMCO emissions comparison scenarios.
- `validation_runs/`: named full-run GEOS-Chem vs Wombat validation case
  specs and compare-only tooling conventions.
- `oracle_data/`: ignored large-oracle payload cache with tracked fixture
  definitions under `oracle_data/manifests/`.

## Validation and Harnesses

GEOS-Chem is the numerical reference. Differences beyond expected
floating-point roundoff are treated as bugs or explicitly documented
deviations.

- `tools/gc_harness/README.md` documents the operator harnesses, large oracle
  cache, HISTORY/met harnesses, and instrumented full-run/main-loop tracing.
- `tools/hemco_harness/README.md` documents standalone HEMCO scenarios for
  source reads, regridding, scale factors, optional dimensions, and polar-row
  behavior.
- `validation_runs/README.md` documents full-run validation case specs and the
  compare-only workflow for existing GC/Wombat outputs.
- `oracle_data/README.md` documents the local large-fixture cache and the
  current full-chain fixture set.
- `performance.md` records benchmark and profiling results.

The strongest current validation chain is operator-by-operator GEOS-Chem
harness parity plus short-window comparison against archived GEOS-Chem outputs.
Use monthly restarts and longer transport windows before making any stronger
long-run parity claim.

## Running and Testing

Use the local virtual environment when available:

```bash
.venv/bin/python -m pytest
```

Selected transport parity tests run both pure NumPy and Numba transport paths
by default. The Numba half skips automatically when `numba` is not importable.

Optional Numba acceleration is controlled globally with `WOMBAT_NUMBA`; unset
or truthy values enable it when available, and `0`, `false`, `no`, `off`, or
`none` disable it. `WOMBAT_TPCORE_NUMBA`, `WOMBAT_VDIFF_NUMBA`, and
`WOMBAT_CONVECTION_NUMBA` override the global flag for individual operators.
`WOMBAT_HISTORY_NUMBA` similarly controls the parallel HISTORY time-average
accumulation, with `WOMBAT_HISTORY_NUMBA_THREADS` overriding the global thread
count for that operation.

The project can also be installed with standard Python packaging tools. Core
runtime dependencies are `numpy`, `netCDF4`, and `py-yaml12`; the
`dev` extra adds `pytest`.

Example run configs:

```text
base_wombat/run.yml
residual_20140901_part001_split01_wombat/run.yml
```

### ObsOperator output

Wombat can sample the post-transport species concentration field inline using
the GEOS-Chem ObsOperator schema. Configure it below `outputs`:

```yaml
outputs:
  obsoperator:
    activate: true
    verbose: false
    input_file: ./ObsOperator/obsoperator-YYYYMMDD.yml.gz
    output_file: ./OutputDir/GEOSChem.ObsOperator.YYYYMMDD_hhmmz.nc4
    restart_file: ./Restarts/Wombat.ObsOperator.Restart.YYYYMMDD_hhmmss.nc4
    restart_missing: warn
```

Relative paths are resolved from the run YAML directory. `YYYY`, `MM`, `DD`,
`hh`, `mm`, and `ss` are expanded from the current model time. Input files may
be ordinary YAML or gzip-compressed YAML ending in `.gz`; a missing expanded
daily input is logged and skipped.

Daily input and batched science output are synchronous. Restart snapshots are
made durable before shutdown returns. The former `input_mode` and `writer`
options are no longer supported.

ObsOperator numerical sampling uses the prepared serial Numba kernel when
Numba is available and `WOMBAT_NUMBA` is enabled. Set
`WOMBAT_OBSOPERATOR_NUMBA=0` to run the same prepared-array kernel in Python as
the reference implementation, or set it to `1` to override a disabled global
mode for focused comparisons.
The normal path validates daily inputs and restarts directly into flat arrays
for fields, selections, accumulators, and the time-sorted schedule. There are
no per-entry time, horizontal, vertical, completion, or restart objects in
either execution mode.

`restart_file` is required when ObsOperator is active. At startup it is
expanded using the simulation start time and, when present, restores unfinished
operators. At clean shutdown Wombat expands it using the actual stopping time
and atomically writes all unfinished state, including an empty restart when
nothing remains. `restart_missing` may be `warn` (the default), `error`, or
`ignore`; `error` is useful for continuation runs where losing the restart must
be fatal.

ObsOperator input is parsed as YAML 1.2. Each input contains an `entries`
sequence. Entries use `fields` with explicit
`SpeciesConcVV_<tracer>` names or the `SpeciesConcVV_?ALL?` and
`SpeciesConcVV_?ADV?` tokens. The older `species` key is not accepted. Time,
horizontal, and vertical operators follow the current GEOS-Chem ObsOperator
format. Pressure is in hPa, altitude is in metres, and `grid_index` and
`pressure_level` values are one-based. Pressure levels are counted bottom-up.
Entry IDs are the sole operator identity and must be unique among all active
entries. A daily input that duplicates an unfinished restart ID is rejected.

Time indices are zero-based and retain GEOS-Chem's end-of-timestep sampling
semantics: time index `0` samples the concentration after the first transport
step. Date-time ranges are half-open model periods, so `00:00`–`01:00` at a
10-minute timestep averages indices `0`–`5`; point timestamps and zero-duration
date-time ranges sample the period ending at that timestamp. Wombat
concentrations are already dry volume mixing ratios, so no
additional molecular-weight conversion is applied. Output uses the current
compressed ObsOperator NetCDF layout with `id`, `field`, `id_index`,
`field_index`, and `sample` variables. Completed entries are staged into bounded
batches before writing; the fixed v1 storage layout uses chunks of 256 IDs, 64
field names, and 16,384 samples or lookup indices.

Science output contains completed operators only. An operator spanning a run
boundary keeps its float64 accumulator, expanded fields, resolved horizontal
selection, vertical operator, and remaining absolute times and weights in the
dedicated NetCDF restart. The following run continues the same weighted sum and
writes the operator once it is complete; partial science files do not need to
be merged. Temporal normalization is applied at every sample (`1/N` for
`normalized`, `1` for `equal`) and there is no final division.

ObsOperator restarts require an exact boundary-time, transport-timestep, and
grid match. The grid check covers coordinates, cell areas, and hybrid vertical
coefficients. Fields are restored by name, so tracer reordering and unrelated
additional tracers are allowed, but a required field may not be removed and a
previous wildcard expansion is not expanded again. A sample whose model-step
start equals the restart boundary belongs to the new run and is evaluated after
that run's first transport step.

For a local GEOS-Chem parity check, set `WOMBAT_GC_OBSOPERATOR_OUTPUT` and
`WOMBAT_OBSOPERATOR_OUTPUT` to matching generated NetCDF files and run
`tests/test_obsoperator.py`. The check skips with a clear message when either
large local artifact is unavailable.
Set `WOMBAT_OBSOPERATOR_INPUT_DIR` to a directory containing the real daily
`obsoperator-YYYYMMDD.yml.gz` inputs to enable the optional cross-day restart
input scenario.

The main runner is available as a module:

```bash
.venv/bin/python -m wombat_transport.run --help
```

Benchmark entry points live under `tools/`, including standalone operator
scaling, transport-driver scaling, GEOS-Chem harness scaling, and Numba
profiling helpers. See `performance.md` for representative commands and
interpretation.

## Scope and Known Limits

- The supported numerical target is currently global GEOS 2x2.5 with 47
  vertical levels.
- Chemistry, dry deposition, wet deposition, nested grids, cubed-sphere grids,
  and alternate vertical grids are out of scope unless explicitly added later.
- Residual emissions currently read values verbatim with `unit_conversion:
  none`; generalized HEMCO unit conversion is not implemented.
- Short-window output parity is credible for the tested cases, but monthly
  restart and long-horizon drift are not yet validated.
- Large GEOS-Chem fixture payloads are local/ignored. Tracked manifests describe
  their contracts; payloads must be generated or fetched locally.

## Further Notes

- `AGENTS.md`: concise operating guide for coding agents.
- `performance.md`: benchmark history and profiling workflows.
- `tools/gc_harness/README.md`: GEOS-Chem oracle and trace tooling.
- `tools/hemco_harness/README.md`: HEMCO standalone emissions parity tooling.
- `validation_runs/README.md`: full-run validation case registry.
- `oracle_data/README.md`: large oracle cache layout.
