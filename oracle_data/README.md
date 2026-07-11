# Oracle Data Cache

This directory is the local cache for large GEOS-Chem oracle fixtures.

Keep small microscope fixtures in `tests/fixtures/` so normal unit tests stay
fast and self-contained. Put full-grid, restart-derived, or multi-step NetCDF
oracle data here instead. NetCDF payloads and generated fixture directories are
ignored by git; only lightweight fixture definitions under `manifests/` should
be tracked.

## Layout

```text
oracle_data/
  manifests/
    base_initial_tpcore_v1.json
    residual_initial_tpcore_v1.json
    fullgrid_synthetic_low_courant_tpcore_v1.json
    base_initial_transport_chain_v3.json
    base_initial_vdiff_after_tpcore_v3.json
    base_initial_convection_fullgrid_v3.json
  base_initial_tpcore_v1/
    transport_step_input.nc
    transport_step_output.nc
    manifest.json
  residual_initial_tpcore_v1/
    transport_step_input.nc
    transport_step_output.nc
    manifest.json
  fullgrid_synthetic_low_courant_tpcore_v1/
    transport_step_input.nc
    transport_step_output.nc
    manifest.json
  base_initial_transport_chain_v3/
    transport_chain_input.nc
    transport_chain_output.nc
    manifest.json
  base_initial_vdiff_after_tpcore_v3/
    vdiff_input.nc
    vdiff_output.nc
    manifest.json
  base_initial_convection_fullgrid_v3/
    convection_input.nc
    convection_output.nc
    manifest.json
```

`manifest.json` inside a fixture directory is generated after the payload files
exist. It records file sizes, SHA256 checksums, provenance, and the TPCORE
branch report for that local oracle.

## Commands

Generate the first base-run fixture from local GEOS-Chem artifacts:

```bash
python -m wombat_transport.gc_harness oracle-fixture-generate base_initial_tpcore_v1
```

Generate the full-grid synthetic low-Courant diagnostic fixture:

```bash
python -m wombat_transport.gc_harness oracle-fixture-generate fullgrid_synthetic_low_courant_tpcore_v1
```

Generate the 24-tracer residual initial-condition fixture:

```bash
python -m wombat_transport.gc_harness oracle-fixture-generate residual_initial_tpcore_v1
```

Generate the current full transport-chain handoff fixtures:

```bash
python -m wombat_transport.gc_harness oracle-fixture-generate base_initial_transport_chain_v3

python -m wombat_transport.gc_harness oracle-fixture-generate base_initial_vdiff_after_tpcore_v3

python -m wombat_transport.gc_harness oracle-fixture-generate base_initial_convection_fullgrid_v3
```

Check a cached fixture:

```bash
python -m wombat_transport.gc_harness oracle-fixture-check base_initial_tpcore_v1
```

Compare a cached fixture against the current Python ports:

```bash
python -m wombat_transport.gc_harness oracle-fixture-compare base_initial_tpcore_v1
```

The base and residual initial-condition fixtures are TPCORE tracer-parity
fixtures. The residual fixture exercises the production 24-tracer stacking path;
its `oracle-fixture-compare` output should report 24 residual tracer names,
roundoff-scale tracer error, and zero Python negative values after fill.

The full-grid synthetic low-Courant fixture is a diagnostic control: it uses the
same horizontal and vertical grid but smooth synthetic pressure, wind, and
tracer fields. It should stay at tight TPCORE parity and helps separate
full-grid geometry issues from real-met/restart interactions.

The `base_initial_transport_chain_v3` fixture and its isolated VDIFF and
convection stage fixtures should be treated as one generation set. If the
handoff comparison reports pressure-derived or tracer mismatches between the
chain output and the stage inputs, regenerate all three local payloads before
changing thresholds.

Fetch support uses the same cache layout, but the tracked manifest must first be
updated with concrete URLs and checksums:

```bash
python -m wombat_transport.gc_harness oracle-fixture-fetch base_initial_tpcore_v1
```
