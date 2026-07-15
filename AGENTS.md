# Wombat Transport Agent Guide

This file is for agent operating instructions. Project documentation lives in
`README.md`; detailed notes live in `performance.md` and the harness READMEs.

## Core Rule

GEOS-Chem is the numerical reference. Treat any difference beyond expected
floating-point roundoff as meaningful until proven otherwise. Do not optimize by
changing transport semantics unless a parity test or explicit user decision
justifies the deviation.

Do not patch `GCClassic/` unless the user explicitly asks to modify GEOS-Chem
itself. It is vendored reference source for reading, harnessing, and tracing.

## Current Target

- Scope is global GEOS 2x2.5, 47 vertical levels, transport-only CO2-like
  tracers.
- Chemistry, dry deposition, and wet deposition are out of scope for the current
  prototype.
- The production tracer layout is canonical `(lev, lat, lon, tracer)`.
- The main transport path is `TPCORE -> VDIFF -> convection`, with non-local
  PBL surface emissions handled through VDIFF.
- Short-run parity is credible for the tested base-no-emissions two-day window
  and residual 24-tracer one-day window. Do not claim monthly or long-horizon
  parity until those comparisons are run explicitly.

## Repo Map

- `src/wombat_transport/run.py`, `runner.py`, `run_config.py`: run config,
  simulation loop, logging, and orchestration.
- `src/wombat_transport/transport/`: TPCORE, VDIFF/PBL, convection, forcing,
  pressure, and transport driver code.
- `src/wombat_transport/emissions.py`: native emissions source reading,
  regridding, scale factors, optional dimensions such as `npft`, and GEOS 2x2.5
  polar-row behavior.
- `src/wombat_transport/output.py`: HISTORY-like `SpeciesConcVV_*` and
  `SpeciesRst_*` output.
- `tools/gc_harness/`: GEOS-Chem operator, HISTORY, met, oracle, and
  full-run/main-loop trace tooling.
- `tools/hemco_harness/`: standalone HEMCO scenarios for emissions parity.
- `validation_runs/`: named full-run GEOS-Chem vs Wombat case specs and
  compare-only workflows for existing outputs.
- `oracle_data/`: ignored large oracle payload cache plus tracked lightweight
  manifests.
- `performance.md`: benchmark/profiling notes and Numba status.

## Local Fixtures and Data

- `external_data/` is the only supported local input root. Its tracked README
  documents the GEOS-Chem ExtData, flux, scaling-grid, ObsOperator, and restart
  contract; payloads below it remain ignored.
- The canonical initial restart is
  `external_data/restarts/2x25/GEOSChem.Restart.20140901_0000z.nc4`.
- `validation_runs/cases/realistic_restart_noemis_2x25/wombat/main/run.yml` and
  `validation_runs/cases/residual_24tracer_emissions_1day_2x25/wombat/main/run.yml`
  are the canonical one- and 24-tracer Wombat configs.
- Local generated run/debug directories are working artifacts unless the user
  explicitly asks to track them.
- When working from a separate Git worktree, check the parent
  `wombat-transport/` checkout for missing ignored `external_data/`, oracle
  payloads, and `validation_runs/work/` products before regenerating them.

## Validation Pointers

- Full test suite: `.venv/bin/python -m pytest`.
- HEMCO standalone: see `tools/hemco_harness/README.md`. Generated
  `HEMCO_sa_Grid.rc` must include explicit GEOS 2x2.5 latitude edges; polar-row
  failures usually mean the synthetic target grid is wrong.
- GEOS-Chem operator/oracle fixtures: see `tools/gc_harness/README.md`.
- Regenerate `base_initial_transport_chain_v3`,
  `base_initial_vdiff_after_tpcore_v3`, and
  `base_initial_convection_fullgrid_v3` together if handoff comparisons show
  stale pressure or tracer mismatches.
- Instrumented full-run/main-loop tracing compares selected-column CSV/NetCDF
  traces, optional full-grid tracer snapshots, and optional TPCORE internals;
  see `tools/gc_harness/README.md`.
- Large local-data tests should skip or fail with clear messages when required
  GEOS-Chem fixture files are absent.
- Transport parity tests that use the `transport_numba_mode` fixture run both
  pure NumPy and Numba paths by default; the Numba case skips when `numba` is
  unavailable.

## Git Workflow

- Prefer explicit merge commits when integrating finished feature branches into
  `main`, unless the user requests a fast-forward, rebase, or squash.

## Coding Constraints

- Preserve user changes and unrelated untracked outputs. Do not clean, reset, or
  delete without explicit instruction.
- Keep generated NetCDF outputs, benchmark artifacts, and temporary files out of
  tracked source unless the user asks to commit a specific fixture.
- Prefer existing module patterns and direct GEOS-Chem semantic ports before
  algebraic simplification.
- Keep comments sparse and useful; document intentional GEOS-Chem deviations
  where they occur.
- Numba is available in the local `.venv` and is part of the current
  performance path. `WOMBAT_NUMBA` is the global switch, with
  `WOMBAT_TPCORE_NUMBA`, `WOMBAT_VDIFF_NUMBA`, and `WOMBAT_CONVECTION_NUMBA`
  as per-operator overrides. Correctness is still judged by GEOS-Chem parity
  tests.
