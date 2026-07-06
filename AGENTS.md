# Wombat Transport Agent Guide

## Project Goal

Build a Python/NumPy transport prototype that reproduces the relevant
GEOS-Chem transport-only behavior closely enough for multi-tracer inversion
workflows, while making large tracer counts efficient.

The target use case is many CO2-like tracers with no chemistry. GEOS-Chem is
the numerical reference. Do not optimize by changing the modeled transport
semantics until parity tests show where differences are acceptable.

## Current Local Fixtures

- `base/` is a GEOS-Chem Classic transport run configured with one transported
  species, `CO2`. It contains the initial restart
  `base/Restarts/GEOSChem.Restart.20140901_0000z.nc4` and short-window
  diagnostics in `base/OutputDir/`. This run exercises the "restart file is
  present" path: initialize from the full, spatially varying restart field,
  not from the scalar `Background_VV` in `base/species_database.yml`.
- `residual_20140901_part001_split01/` is the first validation target. It is a
  24-tracer residual run with short-window diagnostics in
  `residual_20140901_part001_split01/OutputDir/` and monthly restart outputs
  for `2014-10-01`, `2014-11-01`, `2014-12-01`, and `2015-01-01`. This run
  also exercises the "restart file is absent for a species" path: GEOS-Chem
  initializes missing species from `Background_VV`, which is `0.0004`
  (`400 ppm`) for the residual tracers.
- `GCClassic/` is vendored GEOS-Chem Classic source. Treat it as reference
  source code unless the user explicitly asks to patch GEOS-Chem itself.
- `fluxes/` contains local flux input files used by these runs.
- `ExtData` is a symlink to `/home/mgnb/GEOS_Chem/ExtData`. Use `find -L`
  or equivalent when inspecting it; plain `find ExtData ...` will not descend
  into the directory symlink. It includes GEOS-Chem restarts, HEMCO data, and
  `GEOS_2x2.5/MERRA2` meteorology for the 2014 run window.
- `ExtData/GEOS_2x2.5/MERRA2/2014/{09,10,11,12}/` contains the daily A1,
  A3cld, A3dyn, A3mstC, A3mstE, and I3 files needed by the current
  `2014-09-01` to `2015-01-01` configuration.

The observed GEOS-Chem configuration to match first is:

- Grid: global 2.0 x 2.5, 47 vertical levels, 91 latitudes, 144 longitudes.
- Time: start `2014-09-01 00:00:00`, end `2015-01-01 00:00:00`,
  transport timestep `600 s`.
- Transport: GEOS-Chem Classic TPCORE enabled with `iord/jord/kord = 3/3/7`
  and negative-value filling enabled.
- Physics: convection enabled; PBL mixing enabled with non-local PBL mixing.
- Disabled: chemistry, dry deposition, and wet deposition.

Scope is intentionally limited to this grid for now. The Python prototype does
not need to support other horizontal resolutions, nested grids, cubed-sphere
grids, or alternate vertical level counts unless the user explicitly expands
scope later.

Current short-window outputs provide useful early validation targets:

- `base/OutputDir/GEOSChem.SpeciesConcThreeHourly.YYYYMMDD_0000z.nc4`:
  22 daily files from `2014-09-01` through `2014-09-22`, each with 8
  three-hourly time-averaged samples of `SpeciesConcVV_CO2`.
- `base/OutputDir/GEOSChem.LevelEdgeDiagsThreeHourly.YYYYMMDD_0000z.nc4`:
  matching daily files with `Met_PEDGE` and `Met_PEDGEDRY`.
- `base/OutputDir/GEOSChem.StateMetThreeHourly.YYYYMMDD_0000z.nc4`:
  matching daily files with `Met_BXHEIGHT` and `Met_AVGW`.
- `base/OutputDir/HEMCO_diagnostics.YYYYMMDDhh30.nc`: 527 hourly midpoint
  files from `2014-09-01 00:30` through `2014-09-22 22:30`.
- `residual_20140901_part001_split01/OutputDir/GEOSChem.SpeciesConcThreeHourly.YYYYMMDD_0000z.nc4`:
  5 daily files from `2014-09-01` through `2014-09-05`, each with 8
  three-hourly time-averaged samples for all 24 `SpeciesConcVV_r0002p001s*`
  tracers.
- `residual_20140901_part001_split01/OutputDir/HEMCO_diagnostics.YYYYMMDDhh30.nc`:
  119 hourly midpoint files from `2014-09-01 00:30` through
  `2014-09-05 22:30`, with all 24 `Emis_r0002p001s*` fields.

## Numerical Reference Points

Use these GEOS-Chem source files as the first reference for algorithmic
semantics:

- `GCClassic/src/GEOS-Chem/GeosCore/transport_mod.F90`
- `GCClassic/src/GEOS-Chem/GeosCore/tpcore_fvdas_mod.F90`
- `GCClassic/src/GEOS-Chem/GeosCore/mixing_mod.F90`
- `GCClassic/src/GEOS-Chem/GeosCore/vdiff_mod.F90`
- `GCClassic/src/GEOS-Chem/GeosCore/pbl_mix_mod.F90`
- `GCClassic/src/GEOS-Chem/GeosCore/convection_mod.F90`

Keep the first Python implementation correctness-oriented. Match GEOS-Chem
field ordering, units, pressure/mass bookkeeping, operator sequencing, and
restart conventions before adding optional acceleration.

`tools/gc_harness/` contains the first GEOS-Chem-backed operator harness. It
is intentionally full-state-shaped but narrow: it can populate the grid state
and array inputs needed by `DO_PJC_PFIX`, and can also populate minimal
`ChmState`/`DgnState` fields for one `TPCORE_FVDAS` tracer step when the input
fixture includes tracer concentrations. Extend this harness by filling more
state fields as operators need them, rather than creating unrelated one-off
interfaces.

## Initial Python Policy

Use `numpy` and `netCDF4` for the first prototype. These are available in the
local environment. Do not assume `xarray`, `scipy`, `numba`, `dask`, or
`pytest` are installed unless you check or add an explicit dependency workflow.

Represent tracer state internally as a single dense array. Tracer should be a
first-class dimension, with memory layout chosen deliberately and benchmarked.
The goal is to avoid GEOS-Chem's poor multi-tracer SIMD utilization by applying
the same transport operators across many tracers at once.

Recommended initial internal shapes:

- Restarts: stack `SpeciesRst_*` variables into one tracer array.
- Emissions: stack `Emis_*` variables from HEMCO diagnostics into one tracer
  array.
- Preserve a stable tracer-name list so unstacked NetCDF output can compare
  directly against GEOS-Chem variable names.

Initialization must support both GEOS-Chem behaviors used here:

- If a restart variable exists for a species, load that restart field exactly.
- If no restart variable exists for a species, initialize the full 3-D field
  from that species' `Background_VV` in `species_database.yml`. For the
  residual tracers, this means `0.0004 mol mol-1 dry` everywhere.

## Development Phases

### 1. I/O and Comparison Harness

Start by building tools that can:

- Parse `species_database.yml` enough to read tracer metadata, molecular
  weight, and `Background_VV`.
- Read GEOS-Chem restart files and stack all `SpeciesRst_*` variables.
- Read GEOS-Chem species concentration diagnostics and stack all
  `SpeciesConcVV_*` variables.
- Read HEMCO diagnostics and stack all `Emis_*` variables.
- Read the base-run level-edge and state-met diagnostics needed for pressure,
  water-vapor, and grid-box-height comparisons.
- Preserve coordinates, grid metadata, tracer names, and units needed for
  restart-like output.
- Write Python-generated restart-like NetCDF files for comparison.
- Report per-tracer and global metrics.

This phase should not attempt to parse all of HEMCO. For v1, use HEMCO
diagnostic emissions as precomputed source terms.

### 2. Emissions Tendency

Implement the CO2 emissions tendency using HEMCO diagnostic fields as input.
Be explicit about units and conversions:

- HEMCO diagnostics use `kg/m2/s` emissions.
- Restart tracers are dry mixing ratios in `mol mol-1 dry`.
- GEOS-Chem restart files include dry pressure thickness and grid-box area,
  which should be used for air-mass and mixing-ratio conversions.

Validate emissions independently before coupling them to transport.

### 3. Transport Operators

Port transport behavior in stages:

- Pressure and mass bookkeeping needed by TPCORE and restart conversion.
- Horizontal and vertical TPCORE advection for the configured global MERRA2
  grid.
- Negative-value filling behavior.
- PBL mixing, including non-local PBL mixing.
- Convection.

Prefer direct semantic ports from GEOS-Chem source before algebraic
simplification. Any simplification must be covered by comparison tests.

### 4. Validation

Compare Python output to GEOS-Chem fields in increasing time horizon. Use the
short-window three-hourly diagnostics first, then monthly residual restarts once
the short-window behavior is credible.

Recommended early sequence:

- Use the base run to validate one-tracer I/O, pressure/level-edge handling,
  water-vapor/grid-box-height handling, and three-hourly concentration output.
- Use the residual run to validate 24-tracer stacking, emission ordering, and
  three-hourly multi-tracer concentration output over the first five days.
- Use residual monthly restarts for longer-horizon drift and mass-budget checks.

Track at least:

- Maximum absolute error by tracer.
- Relative error by tracer, with sensible handling near zero.
- Column-integrated mass error.
- Global mass error.
- Tracer ordering and metadata mismatches.
- Negative values before and after the fill step.

Do not claim numerical parity based only on visual agreement or a single global
aggregate.

### 5. Performance

After correctness checks exist, benchmark operator time separately from I/O.
Include at least these tracer counts:

- 1 tracer, for parity with the base run shape.
- 24 tracers, for the residual fixture.
- Larger synthetic tracer counts, to expose scaling and memory-layout effects.

Report wall time, throughput per tracer, memory footprint, and the chosen array
layout. Optional acceleration such as numba or compiled kernels should only be
introduced after a baseline NumPy implementation has a parity harness.

## Testing Expectations

Add focused tests or scripts as implementation appears. Minimum scenarios:

- Metadata smoke test for `lev=47`, `lat=91`, `lon=144`, and expected residual
  tracer count.
- Restart stack/unstack roundtrip with exact value preservation.
- Initialization tests for both paths: existing restart variables take
  precedence, and missing residual restart variables initialize to
  `Background_VV = 0.0004`.
- Species concentration diagnostic stack/unstack test for one-tracer base files
  and 24-tracer residual files.
- HEMCO diagnostic stack ordering test for all `Emis_*` variables.
- Base diagnostic reader tests for `Met_PEDGE`, `Met_PEDGEDRY`,
  `Met_BXHEIGHT`, and `Met_AVGW`.
- Emissions-only tendency test with mass conservation and unit checks.
- Operator-stage parity tests against GEOS-Chem fields.
- Performance benchmark that can be run without modifying tracked fixtures.

When a test requires large local data that may not be present everywhere, make
that requirement explicit and fail with a clear message.

## Scope Boundaries

- Do not implement chemistry.
- Do not implement dry deposition or wet deposition for the initial prototype.
- Do not generalize beyond global 2.0 x 2.5 with 47 levels for the initial
  prototype.
- Do not refactor GEOS-Chem source unless the user asks for it.
- Do not replace the validation target with a faster-but-different scheme
  without making the numerical tradeoff explicit.
- Do not introduce broad dependencies before the NumPy/netCDF4 baseline is
  working and measured.

## Repository Hygiene

- Keep generated NetCDF outputs, benchmark artifacts, and scratch files out of
  tracked source unless the user asks to commit a specific fixture.
- Prefer small, inspectable Python modules and command-line scripts over hidden
  notebooks for core logic.
- Document every intentional deviation from GEOS-Chem behavior in code or
  comparison output.
- Keep temporary scaffold code clearly labeled, and remove it once the
  corresponding GEOS-Chem-parity operator or harness-backed replacement makes
  it irrelevant.
- If user changes are present in the worktree, preserve them and work around
  them rather than reverting.
