# GEOS-Chem Operator Harness

This directory is for small GEOS-Chem-backed executables that isolate operator
behavior for the NumPy Wombat transport port.

The intended shape is one incremental full-state harness:

- start with just enough `GrdState` and arrays to call `DO_PJC_PFIX`;
- add minimal `ChmState`/`DgnState` population for `TPCORE_FVDAS`;
- fill more `MetState`, `ChmState`, and option fields for PBL mixing and
  convection later.

Generated NetCDF inputs and outputs belong in `tools/gc_harness/work/`,
`oracle_data/`, or another scratch directory and should not be committed.

The PJC executable is a GEOS-Chem oracle for this operator stage. The current
Python comparison command contrasts that output with Wombat's NumPy PJC port.
The tracked snapshot fixture keeps this parity check in the fast unit-test
suite without requiring the GEOS-Chem executable during normal test runs.

When the fixture includes `tracer_conc(tracer, lev, lat, lon)`, the same
executable runs one `DO_PJC_PFIX` plus `TPCORE_FVDAS` step and writes
`tracer_conc_after`, `xmass_hpa`, `ymass_hpa`, and `surface_pressure_hpa`.
The tracked TPCORE snapshot fixture records this one-step GEOS-Chem oracle
boundary for fast unit tests. The production transport driver now routes
one-step and window modes through the NumPy TPCORE port.

## Python Fixture Commands

```bash
python -m wombat_transport.gc_harness write-pjc-input \
  base_wombat/run.yml tools/gc_harness/work/pjc_input.nc

python -m wombat_transport.gc_harness compare-pjc-output \
  tools/gc_harness/work/pjc_input.nc tools/gc_harness/work/pjc_output.nc

python -m wombat_transport.gc_harness transport-step \
  base_wombat/run.yml --max-tracers 1

python -m wombat_transport.gc_harness snapshot-pjc \
  tests/fixtures/pjc_snapshot_v1

python -m wombat_transport.gc_harness snapshot-tpcore \
  tests/fixtures/tpcore_snapshot_v1

python -m wombat_transport.gc_harness snapshot-tpcore-branch \
  x_fxppm_low_courant tests/fixtures/tpcore_x_fxppm_low_courant_v1

python -m wombat_transport.gc_harness snapshot-tpcore-branch \
  x_large_courant_polar tests/fixtures/tpcore_x_large_courant_polar_v1

python -m wombat_transport.gc_harness compare-transport-step-output \
  tests/fixtures/tpcore_snapshot_v1/tpcore_input.nc \
  tests/fixtures/tpcore_snapshot_v1/tpcore_output.nc

python -m wombat_transport.gc_harness write-synthetic-vdiff-input \
  tests/fixtures/vdiff_snapshot_v1/vdiff_input.nc

python -m wombat_transport.gc_harness write-synthetic-vdiff-input \
  tests/fixtures/vdiff_nonzero_surface_flux_v1/vdiff_input.nc \
  --scenario nonzero_surface_flux

python -m wombat_transport.gc_harness write-synthetic-vdiff-input \
  tests/fixtures/vdiff_negative_clipping_v1/vdiff_input.nc \
  --scenario negative_clipping

python -m wombat_transport.gc_harness compare-vdiff-output \
  tests/fixtures/vdiff_snapshot_v1/vdiff_input.nc \
  tests/fixtures/vdiff_snapshot_v1/vdiff_output.nc

python -m wombat_transport.gc_harness write-synthetic-convection-input \
  tools/gc_harness/work/convection_input.nc \
  --scenario active_cloud

python -m wombat_transport.gc_harness write-real-convection-input \
  tools/gc_harness/work/convection_real_sampled_input.nc \
  --mode sampled-columns

python -m wombat_transport.gc_harness write-real-convection-input \
  tools/gc_harness/work/convection_real_fullgrid_input.nc \
  --mode full-grid --max-tracers 1

python -m wombat_transport.gc_harness python-convection-output \
  tools/gc_harness/work/convection_input.nc \
  tools/gc_harness/work/python_convection_output.nc

python -m wombat_transport.gc_harness compare-convection-output \
  tools/gc_harness/work/convection_input.nc \
  tools/gc_harness/work/convection_output.nc
```

The `snapshot-pjc` command regenerates the small tracked PJC oracle fixture
used by unit tests. Run it deliberately when the GEOS-Chem reference version or
the PJC fixture contract changes, then review the NetCDF/metadata diff.
The `snapshot-tpcore` command does the same for the compact one-step
`DO_PJC_PFIX` plus `TPCORE_FVDAS` oracle fixture.
The `snapshot-tpcore-branch` command creates small branch-isolating TPCORE
snapshots. `x_fxppm_low_courant` is a passing X full-PPM fixture;
`x_large_courant_polar` is a passing compact high-Courant E-W fixture.
The tracked VDIFF snapshot fixtures isolate the configured non-local
`VDIFFDR -> vdiff/pbldif/qvdiff` path for zero constituent surface flux,
nonzero constituent surface flux, and negative tracer clipping/rescaling.
The tracked convection real-met sampled fixture uses local 2014-09-01 MERRA2
fields from the residual 24-tracer configuration, maps native 72 center levels
and 73 edge levels onto the target 47/48 GEOS-Chem levels, and selects six
active convective columns plus one no-cloud column. It is intentionally small
enough for the fast unit-test suite while preserving the target 47-level
vertical contract and real field values.

## Large Oracle Fixture Cache

Full-grid, restart-derived, or multi-step fixtures are intentionally separate
from the tracked microscope snapshots. The local cache lives under
`oracle_data/`; NetCDF payloads there are ignored by git, while lightweight
fixture definitions under `oracle_data/manifests/` are tracked.

The first registered large fixture is `base_initial_tpcore_v1`: a full-grid
one-tracer base-run initial-condition `DO_PJC_PFIX` plus `TPCORE_FVDAS` oracle.
It is generated from `base_wombat/run.yml`, the base restart, and local MERRA2
met. It is a full-grid one-tracer TPCORE parity fixture.

`residual_initial_tpcore_v1` is the matching full-grid 24-tracer residual
initial-condition oracle generated from
`residual_20140901_part001_split01_wombat/run.yml`. It exercises the residual
species ordering and `Background_VV` initialization path used when restart
variables are absent.

`fullgrid_synthetic_low_courant_tpcore_v1` uses the same full GEOS-Chem grid
with smooth low-Courant synthetic pressure, winds, and tracers. It is the
control fixture for full-grid geometry and wide X full-PPM behavior without
real MERRA2/restart complexity.

```bash
python -m wombat_transport.gc_harness oracle-fixture-generate base_initial_tpcore_v1

python -m wombat_transport.gc_harness oracle-fixture-generate residual_initial_tpcore_v1

python -m wombat_transport.gc_harness oracle-fixture-generate fullgrid_synthetic_low_courant_tpcore_v1

python -m wombat_transport.gc_harness oracle-fixture-check base_initial_tpcore_v1

python -m wombat_transport.gc_harness oracle-fixture-compare base_initial_tpcore_v1

python -m wombat_transport.gc_harness oracle-fixture-compare residual_initial_tpcore_v1

python -m wombat_transport.gc_harness oracle-fixture-trace-generate base_initial_tpcore_v1

python -m wombat_transport.gc_harness oracle-fixture-trace-compare base_initial_tpcore_v1
```

When hosted artifacts are available, add URLs and SHA256 checksums to the
tracked manifest and populate the same cache layout with:

```bash
python -m wombat_transport.gc_harness oracle-fixture-fetch base_initial_tpcore_v1
```

`oracle-fixture-trace-generate` runs the instrumented GEOS-Chem harness and
writes an ignored `oracle_tpcore_trace.nc` beside the large fixture payload.
`oracle-fixture-trace-compare` writes the matching ignored Python trace named
`python_tpcore_trace.nc` and compares the TPCORE checkpoints stage by stage. If
the oracle trace is absent, it falls back to final-field attribution by level,
latitude, longitude, Courant bins, vertical mass-flux bins, and initial
tracer-gradient bins.

## Build Sketch

`build_pjc_pfix_harness.sh` links against the existing `base/build` tree. The
script is intentionally narrow and may need adjustment if the GEOS-Chem build
tree moves or was produced with different compiler wrappers.

```bash
tools/gc_harness/build_pjc_pfix_harness.sh

tools/gc_harness/build_pjc_pfix_harness.sh --with-tpcore-trace

tools/gc_harness/build_vdiff_harness.sh

tools/gc_harness/build_convection_harness.sh
```

The trace build generates an instrumented copy of
`GCClassic/src/GEOS-Chem/GeosCore/tpcore_fvdas_mod.F90` under
`tools/gc_harness/build/` and links it only into
`tools/gc_harness/build/pjc_pfix_harness_trace`. The vendored `GCClassic/`
source tree remains the reference input and is not patched.
The VDIFF build similarly generates a local copy of `vdiff_mod.F90` with
`VDIFFDR` exposed and a trace hook for `kvh`, `kvm`, `tpert`, and `qpert`.

Then run:

```bash
python -m wombat_transport.gc_harness pjc-pfix base_wombat/run.yml

tools/gc_harness/build/vdiff_harness \
  tests/fixtures/vdiff_snapshot_v1/vdiff_input.nc \
  tests/fixtures/vdiff_snapshot_v1/vdiff_output.nc

tools/gc_harness/build/convection_harness \
  tools/gc_harness/work/convection_input.nc \
  tools/gc_harness/work/convection_output.nc
```

Current smoke result on `base_wombat/run.yml` after building against
`base/build`:

```text
metric,value
xmass_max_abs_error_hpa,2.28936869e-11
xmass_mean_abs_error_hpa,7.84349743e-14
ymass_max_abs_error_hpa,1.77635684e-15
ymass_mean_abs_error_hpa,2.66716245e-17
```

Current one-tracer transport-step smoke result on `base_wombat/run.yml`:

```text
metric,value
tracer_min,3.81882596e-04
tracer_max,4.29874972e-04
xmass_min_hpa,-7.35175692e+01
xmass_max_hpa,3.33947648e+01
ymass_min_hpa,-5.25986229e+00
ymass_max_hpa,5.10715920e+00
surface_pressure_min_hpa,5.44398164e+02
surface_pressure_max_hpa,1.03315703e+03
```

Current real-met sampled convection snapshot result against the GEOS-Chem
convection harness:

```text
metric,value
tracer_max_abs_error,0.00000000e+00
tracer_mean_abs_error,0.00000000e+00
diag14_max_abs_error,0.00000000e+00
diag14_mean_abs_error,0.00000000e+00
negative_count_before_expected,0
negative_count_before_actual,0
negative_count_after_expected,0
negative_count_after_actual,0
initial_mass_max_abs_error,1.22070312e-03
final_mass_max_abs_error,1.22070312e-03
mass_change_max_abs,0.00000000e+00
expected_mass_change_max_abs,0.00000000e+00
top_error_index,0:0:0:0
internal_steps_expected,2
internal_steps_actual,2
```
