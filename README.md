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

The project can also be installed with standard Python packaging tools. Core
runtime dependencies are `numpy`, `netCDF4`, and `PyYAML`; the `dev` extra adds
`pytest`.

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
```

Relative paths are resolved from the run YAML directory. `YYYY`, `MM`, `DD`,
`hh`, `mm`, and `ss` are expanded from the current model time. Input files may
be ordinary YAML or gzip-compressed YAML ending in `.gz`; a missing expanded
daily input is logged and skipped.

Each input contains an `entries` sequence. Entries use `fields` with explicit
`SpeciesConcVV_<tracer>` names or the `SpeciesConcVV_?ALL?` and
`SpeciesConcVV_?ADV?` tokens. The older `species` key is not accepted. Time,
horizontal, and vertical operators follow the current GEOS-Chem ObsOperator
format. Pressure is in hPa, altitude is in metres, and `grid_index` and
`pressure_level` values are one-based. Pressure levels are counted bottom-up.

Time indices are zero-based and retain GEOS-Chem's end-of-timestep sampling
semantics: time index `0` samples the concentration after the first transport
step. Wombat concentrations are already dry volume mixing ratios, so no
additional molecular-weight conversion is applied. Output uses the current
compressed ObsOperator NetCDF layout with `id`, `field`, `id_index`,
`field_index`, and `sample` variables.

For a local GEOS-Chem parity check, set `WOMBAT_GC_OBSOPERATOR_OUTPUT` and
`WOMBAT_OBSOPERATOR_OUTPUT` to matching generated NetCDF files and run
`tests/test_obsoperator.py`. The check skips with a clear message when either
large local artifact is unavailable.

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
