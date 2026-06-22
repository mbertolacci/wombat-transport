# GEOS-Chem Operator Harness

This directory is for small GEOS-Chem-backed executables that isolate operator
behavior for the NumPy Wombat transport port.

The intended shape is one incremental full-state harness:

- start with just enough `GrdState` and arrays to call `DO_PJC_PFIX`;
- add minimal `ChmState`/`DgnState` population for `TPCORE_FVDAS`;
- fill more `MetState`, `ChmState`, and option fields for PBL mixing and
  convection later.

Generated NetCDF inputs and outputs belong in `tools/gc_harness/work/` or
another scratch directory and should not be committed.

The PJC executable is a GEOS-Chem oracle for this operator stage. The current
Python comparison command contrasts that output with Wombat's existing
approximate mass-flux scaffold; it is a smoke check and diagnostic, not the
final PJC equivalence criterion.

When the fixture includes `tracer_conc(tracer, lev, lat, lon)`, the same
executable runs one `DO_PJC_PFIX` plus `TPCORE_FVDAS` step and writes
`tracer_conc_after`, `xmass_hpa`, `ymass_hpa`, and `surface_pressure_hpa`.

## Python Fixture Commands

```bash
python -m wombat_transport.gc_harness write-pjc-input \
  base_wombat/run.yml tools/gc_harness/work/pjc_input.nc

python -m wombat_transport.gc_harness compare-pjc-output \
  tools/gc_harness/work/pjc_input.nc tools/gc_harness/work/pjc_output.nc

python -m wombat_transport.gc_harness transport-step \
  base_wombat/run.yml --max-tracers 1
```

## Build Sketch

`build_pjc_pfix_harness.sh` links against the existing `base/build` tree. The
script is intentionally narrow and may need adjustment if the GEOS-Chem build
tree moves or was produced with different compiler wrappers.

```bash
tools/gc_harness/build_pjc_pfix_harness.sh
```

Then run:

```bash
python -m wombat_transport.gc_harness pjc-pfix base_wombat/run.yml
```

Current smoke result on `base_wombat/run.yml` after building against
`base/build`:

```text
metric,value
xmass_max_abs_error_hpa,1.83529079e+01
xmass_mean_abs_error_hpa,1.01810656e-01
ymass_max_abs_error_hpa,4.91098969e-03
ymass_mean_abs_error_hpa,5.55552666e-04
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
