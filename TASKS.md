# Wombat Transport Status

Last completed checkpoint:
`f8764bb Wire VDIFF into transport driver`

Validation for that checkpoint:

- `tools/gc_harness/build_vdiff_harness.sh` rebuilt the local GEOS-Chem VDIFF
  oracle executable after tightening negative-count output.
- Compact VDIFF oracle comparisons match Python at roundoff for tracer,
  humidity, `kvh`, `kvm`, PBL height, perturbations, negative counts, and
  mass for:
  - zero constituent surface flux;
  - nonzero constituent surface flux;
  - negative tracer clipping and long-lived mass rescaling.
- Focused validation passed:
  - `tests/test_gc_harness.py`: `39 passed`
  - `tests/test_transport.py`: `11 passed`
  - `tests/test_runner.py`: `13 passed`
  - `tests/test_io.py`: `15 passed`
- Working tree was clean immediately after the commit.

## Where We Are

- Python package scaffolding exists under `src/wombat_transport`.
- Run configs exist for:
  - `base_wombat/run.yml`: one-tracer CO2 run with restart initialization.
  - `residual_20140901_part001_split01_wombat/run.yml`: 24 residual tracers
    with missing-restart initialization from `Background_VV = 0.0004`.
- HEMCO diagnostic replay is implemented as a cached-output source tendency.
  HEMCO diagnostics are treated as GEOS-Chem output, not raw input.
- Transport modes now include:
  - `transport-one-step`
  - `transport-window`
- Transport currently has:
  - pressure and dry-air-mass bookkeeping on the fixed 2x2.5, 47-level grid;
  - pressure-weighted MERRA2 72-to-47 level collapse;
  - a NumPy port of the GEOS-Chem PJC pressure-fixer mass-flux path for
    `XMASS`/`YMASS`;
  - a NumPy `TPCORE_FVDAS` one-step path matching the compact low-Courant
    oracle fixture, branch-isolating fixtures, the full-grid synthetic
    low-Courant fixture, the full-grid base initial-condition fixture, and the
    24-tracer residual initial-condition fixture for pressure, mass fluxes,
    and final tracer concentrations;
  - `TPCORE_FVDAS` followed by the configured non-local VDIFF/PBL stage in
    both `transport-one-step` and `transport-window`;
  - per-stage scalar mass reporting for `tpcore` and `vdiff`;
  - three-hour window averaging for equivalence checks against GEOS-Chem
    diagnostics;
  - pressure-thickness and pressure-edge comparison output against
    `Met_PEDGEDRY` when LevelEdge diagnostics are available.
- PBL work now includes a direct Python port of GEOS-Chem
  `Compute_Pbl_Height` bookkeeping, the compact mass-weighted full-PBL mixer
  core from `TurbDay`, and the configured non-local
  `VDIFFDR -> vdiff/pbldif/qvdiff` path. The production driver now runs that
  VDIFF path after TPCORE with zero constituent surface flux.
- A GEOS-Chem-backed operator harness exists under `tools/gc_harness/`. It
  writes NetCDF fixtures from a Wombat run config, calls `DO_PJC_PFIX` through
  a small Fortran executable linked against `base/build`, and can also run one
  `TPCORE_FVDAS` tracer step when the fixture includes `tracer_conc`. A
  separate VDIFF harness path generates a local trace-enabled copy of
  `vdiff_mod.F90`, exposes `VDIFFDR`, saves `kvh`, `kvm`, `tpert`, and
  `qpert`, and records negative tracer counts for oracle comparison.
- Fast tracked oracle snapshots now exist for:
  - PJC mass fluxes, under `tests/fixtures/pjc_snapshot_v1/`;
  - one-step PJC plus `TPCORE_FVDAS`, under
    `tests/fixtures/tpcore_snapshot_v1/`.
  - branch-isolating TPCORE snapshots for X full-PPM and large-Courant E-W
    behavior, under `tests/fixtures/tpcore_x_*_v1/`.
  - one-step non-local VDIFF zero-flux, under
    `tests/fixtures/vdiff_snapshot_v1/`;
  - one-step non-local VDIFF nonzero constituent surface flux, under
    `tests/fixtures/vdiff_nonzero_surface_flux_v1/`;
  - one-step non-local VDIFF negative clipping/rescaling, under
    `tests/fixtures/vdiff_negative_clipping_v1/`.
- Large real-run oracle fixtures now have a separate untracked cache policy
  under `oracle_data/`. Tracked manifests describe the fixture contract; NetCDF
  payloads are generated or fetched locally and verified by checksum before
  optional tests use them.

## Important Caveats

- `transport-one-step` and `transport-window` now route through the
  GEOS-Chem-oriented NumPy TPCORE port followed by the non-local VDIFF/PBL
  path. Convection is still not included in the production transport sequence.
- The supported transport driver path is the GEOS-Chem-oriented NumPy
  TPCORE+VDIFF path.
- The harness is now an isolated GEOS-Chem oracle for the pressure-fixer and
  one-step TPCORE stages. The Python `compare-python-tpcore-output` path and
  main transport driver modes route through the GEOS-Chem-oriented NumPy
  TPCORE port.
- The TPCORE fixtures verify the shared PJC mass-flux stage and GEOS-Chem's
  one-step final tracer field on compact, branch-isolating, and full-grid
  one-step fixtures. This is evidence of one-step `TPCORE_FVDAS` parity for
  the currently covered paths, not evidence that the full transport window
  matches GEOS-Chem.
- The tracked compact TPCORE snapshot is a low-Courant fixture. Current
  diagnostics report max `|cx|` about `0.0023` and max `|cy|` about `0.0008`,
  so the compact fixture covers the ordinary low-Courant branches. The
  full-grid base fixture exercises large-Courant E-W behavior and now matches
  the GEOS-Chem oracle at roundoff. Large-Courant N-S remains unsupported.
- The Python TPCORE path now preflights the active branch set and raises a
  clear `NotImplementedError` for currently unsupported large-Courant N-S
  behavior instead of continuing silently outside the validated path.
- Production VDIFF currently derives grid-box height hydrostatically from
  post-TPCORE dry pressure edges and virtual temperature because
  `TransportForcing` does not yet carry archived `Met_BXHEIGHT`. Base StateMet
  diagnostics remain the reference target for tightening this input path.
- VDIFF is wired with zero constituent surface flux in the production transport
  path. Nonzero tracer surface-flux behavior is covered by compact oracle
  fixtures but not yet sourced from emissions inside production transport.
- Convection, meaningful three-hourly production validation, and performance
  benchmarks are not implemented yet.
- Missing-operator gaps are not validation milestones. The project target is
  operator-by-operator numerical parity: isolate a GEOS-Chem operator, match it
  to roundoff or a documented floating-point tolerance, then move on.
- `oracle_data/manifests/base_initial_tpcore_v1.json` defines the first
  full-grid base initial-condition PJC+TPCORE fixture. It is useful for oracle
  coverage and branch reporting. Current Python TPCORE matches it at
  roundoff: `tracer_max_abs_error = 3.79470760e-18`,
  `surface_pressure_max_abs_error_hpa = 6.82121026e-13`, and trace-stage
  `q_after_cross_terms = 2.76471554e-18`.
- `oracle_data/manifests/residual_initial_tpcore_v1.json` defines the
  full-grid 24-tracer residual initial-condition PJC+TPCORE fixture. It guards
  residual tracer stacking/order and missing-restart `Background_VV`
  initialization through one TPCORE step.
- `oracle_data/manifests/fullgrid_synthetic_low_courant_tpcore_v1.json`
  defines a full-grid synthetic low-Courant control fixture. It matches Python
  TPCORE at tight tolerance, confirming that full-grid geometry is handled on
  the low-Courant path.
- Base run diagnostics are the best short-window target because base has
  matching SpeciesConc, LevelEdge, and StateMet files through 2014-09-22.
  Residual currently has SpeciesConc and HEMCO diagnostics but no StateMet or
  LevelEdge outputs in its `OutputDir`.

## Good Next Steps

1. Add a full-grid base TPCORE+VDIFF oracle fixture.
   - Use `mixing_mod.F90`, `vdiff_mod.F90`, and `pbl_mix_mod.F90` as references.
   - Compact edge cases are now stable for zero flux, nonzero flux, and
     negative clipping/rescaling.
   - The next VDIFF parity target is at least one full-grid base
     initial-condition `TPCORE + VDIFF` oracle comparing final tracer,
     humidity, diffusivities, negative counts, and scalar mass.

2. Port and verify transport in GEOS-Chem operator order.
   - Treat GEOS-Chem as the reference semantics, not as an approximate target.
   - For each operator, add the closest practical single-step or short-window
     check before moving on.
   - Keep each new operator behind a clear stage until it has its own parity
     check or direct GEOS-Chem justification.

3. Add negative-value filling where it is used outside current TPCORE coverage.
   - Match GEOS-Chem behavior before any high-order advection work depends on
     it.
   - Track negative counts/minima before and after filling in verification
     output.

4. Tighten VDIFF production inputs and diagnostics.
   - Use `transport_mod.F90`, `tpcore_fvdas_mod.F90`, and `pjc_pfix_mod.F90` as
     references.
   - Compare hydrostatically derived `bxheight` against archived
     `Met_BXHEIGHT` from the base StateMet diagnostics.
   - Preserve the current TPCORE and compact VDIFF parity checks while adding
     full-grid VDIFF oracle coverage.

5. Add convection.
   - Use `convection_mod.F90` as the reference.
   - Keep it behind a clear operator stage so it can be tested independently.

6. Add production verification once the relevant operators exist.
   - Walk base `SpeciesConcThreeHourly`, `LevelEdgeDiagsThreeHourly`, and
     `StateMetThreeHourly` files in chronological order.
   - Run Wombat in repeated three-hour windows only after the staged operator
     sequence is present enough for the comparison to be meaningful.
   - Emit CSV-style equivalence checks for concentration, scalar mass, dry
     pressure thickness, and dry pressure edges.

7. Benchmark vectorized multi-tracer scaling once the verification harness
   exists.
   - Measure 1 tracer, 24 tracers, and larger synthetic tracer counts.
   - Report operator time separately from NetCDF I/O.

## Useful Commands

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
tools/gc_harness/build_pjc_pfix_harness.sh
python -m wombat_transport.gc_harness pjc-pfix base_wombat/run.yml
python -m wombat_transport.gc_harness transport-step base_wombat/run.yml --max-tracers 1
python -m wombat_transport.gc_harness snapshot-tpcore tests/fixtures/tpcore_snapshot_v1
python -m wombat_transport.gc_harness snapshot-tpcore-branch x_fxppm_low_courant tests/fixtures/tpcore_x_fxppm_low_courant_v1
python -m wombat_transport.gc_harness snapshot-tpcore-branch x_large_courant_polar tests/fixtures/tpcore_x_large_courant_polar_v1
python -m wombat_transport.gc_harness compare-transport-step-output tests/fixtures/tpcore_snapshot_v1/tpcore_input.nc tests/fixtures/tpcore_snapshot_v1/tpcore_output.nc
python -m wombat_transport.gc_harness compare-python-tpcore-output tests/fixtures/tpcore_snapshot_v1/tpcore_input.nc tests/fixtures/tpcore_snapshot_v1/tpcore_output.nc
python -m wombat_transport.gc_harness oracle-fixture-generate base_initial_tpcore_v1
python -m wombat_transport.gc_harness oracle-fixture-generate fullgrid_synthetic_low_courant_tpcore_v1
python -m wombat_transport.gc_harness oracle-fixture-check base_initial_tpcore_v1
python -m wombat_transport.gc_harness oracle-fixture-compare base_initial_tpcore_v1
python -m wombat_transport.gc_harness oracle-fixture-trace-compare base_initial_tpcore_v1
tools/gc_harness/build_vdiff_harness.sh
python -m wombat_transport.gc_harness compare-vdiff-output tests/fixtures/vdiff_snapshot_v1/vdiff_input.nc tests/fixtures/vdiff_snapshot_v1/vdiff_output.nc
python -m wombat_transport.gc_harness compare-vdiff-output tests/fixtures/vdiff_nonzero_surface_flux_v1/vdiff_input.nc tests/fixtures/vdiff_nonzero_surface_flux_v1/vdiff_output.nc
python -m wombat_transport.gc_harness compare-vdiff-output tests/fixtures/vdiff_negative_clipping_v1/vdiff_input.nc tests/fixtures/vdiff_negative_clipping_v1/vdiff_output.nc
python -m wombat_transport.run base_wombat/run.yml --mode transport-window --max-steps 18
python -m wombat_transport.run residual_20140901_part001_split01_wombat/run.yml --mode transport-one-step
```
