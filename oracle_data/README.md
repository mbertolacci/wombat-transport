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
  base_initial_tpcore_v1/
    transport_step_input.nc
    transport_step_output.nc
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

Check a cached fixture:

```bash
python -m wombat_transport.gc_harness oracle-fixture-check base_initial_tpcore_v1
```

Compare a cached fixture against the current Python ports:

```bash
python -m wombat_transport.gc_harness oracle-fixture-compare base_initial_tpcore_v1
```

Fetch support uses the same cache layout, but the tracked manifest must first be
updated with concrete URLs and checksums:

```bash
python -m wombat_transport.gc_harness oracle-fixture-fetch base_initial_tpcore_v1
```

