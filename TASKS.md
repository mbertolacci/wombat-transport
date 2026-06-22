# Wombat Transport Status

Last checkpoint: `ac8aeb7 Document transport prototype status`

Validation at that checkpoint:

- Full test suite passed at that checkpoint: `44 passed`.
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
  - horizontal mass-flux advection scaffold;
  - closed-boundary vertical continuity/advection scaffold;
  - three-hour window averaging for equivalence checks against GEOS-Chem
    diagnostics;
  - pressure-thickness and pressure-edge comparison output against
    `Met_PEDGEDRY` when LevelEdge diagnostics are available.
- A GEOS-Chem-backed operator harness exists under `tools/gc_harness/`. It
  writes NetCDF fixtures from a Wombat run config, calls `DO_PJC_PFIX` through
  a small Fortran executable linked against `base/build`, and can also run one
  `TPCORE_FVDAS` tracer step when the fixture includes `tracer_conc`.

## Important Caveats

- The transport scaffold is still not a TPCORE parity implementation.
- Horizontal and vertical tracer reconstruction is first-order upwind, not the
  GEOS-Chem/PJC high-order limiter path.
- The vertical flux currently redistributes mass by column continuity rather
  than porting the full GEOS-Chem pressure fixer/TPCORE machinery.
- The harness is now an isolated GEOS-Chem oracle for the pressure-fixer and
  one-step TPCORE stages. The current NumPy transport path is still the older
  first-order scaffold and should be ported against this oracle next.
- PBL mixing, convection, negative-value filling, and performance benchmarks
  are not implemented yet.
- Base run diagnostics are the best short-window target because base has
  matching SpeciesConc, LevelEdge, and StateMet files through 2014-09-22.
  Residual currently has SpeciesConc and HEMCO diagnostics but no StateMet or
  LevelEdge outputs in its `OutputDir`.

## Good Next Steps

1. Port NumPy transport against the one-step GEOS-Chem oracle.
   - Use `python -m wombat_transport.gc_harness transport-step
     base_wombat/run.yml --max-tracers 1` to generate the reference fixture.
   - Start by matching PJC `XMASS`/`YMASS`, then TPCORE pressure/CFL setup,
     horizontal update, vertical update, pole handling, and negative fill.
   - Keep comparisons per substage; do not tune against only an aggregate
     concentration error.

2. Use the PJC harness to lock down pressure-fixer semantics.
   - Run `python -m wombat_transport.gc_harness pjc-pfix base_wombat/run.yml`
     as the first smoke check.
   - Compare GEOS-Chem `XMASS`/`YMASS` against an explicit NumPy port of
     `pjc_pfix_mod.F90`/`PJC_PFIX_WINDOW` internals, not just the current
     approximate flux scaffold.
   - Check dimension order, vertical orientation, surface-pressure timing
     (`P_TP1`/`P_TP2`), and pressure-coordinate constants before coupling to
     tracer advection.

3. Extend transport-step fixtures beyond one base tracer.
   - Run the same harness on residual tracers once the one-tracer orientation
     and units are verified.
   - Add larger synthetic tracer-count fixtures after the oracle comparison is
     stable enough to benchmark.

4. Add a transport verification command.
   - Walk base `SpeciesConcThreeHourly`, `LevelEdgeDiagsThreeHourly`, and
     `StateMetThreeHourly` files in chronological order.
   - Run Wombat in repeated three-hour windows.
   - Emit CSV-style equivalence checks for concentration, scalar mass, dry
     pressure thickness, and dry pressure edges.
   - Include `--max-windows` for quick smoke runs and `--output-csv` for longer
     verification runs.

5. Port and verify transport in GEOS-Chem operator order.
   - Treat GEOS-Chem as the reference semantics, not as an approximate target.
   - For each operator, add the closest practical single-step or short-window
     check before moving on.
   - Keep current scaffold behavior clearly labeled until the corresponding
     GEOS-Chem algorithm has been ported or directly justified.

6. Add negative-value filling.
   - Match GEOS-Chem behavior before any high-order advection work depends on
     it.
   - Track negative counts/minima before and after filling in verification
     output.

7. Replace the advection scaffold with a closer TPCORE/PJC port.
   - Use `transport_mod.F90`, `tpcore_fvdas_mod.F90`, and `pjc_pfix_mod.F90` as
     references.
   - Verify pressure fixer behavior, horizontal/vertical flux bookkeeping,
     tracer reconstruction, limiter behavior, and mass conservation as separate
     checks where possible.

8. Add PBL mixing.
   - Use `mixing_mod.F90`, `vdiff_mod.F90`, and `pbl_mix_mod.F90` as references.
   - Verify it as its own operator stage before coupling it into longer
     transport windows.

9. Add convection.
   - Use `convection_mod.F90` as the reference.
   - Keep it behind a clear operator stage so it can be tested independently.

10. Benchmark vectorized multi-tracer scaling once the verification harness
   exists.
   - Measure 1 tracer, 24 tracers, and larger synthetic tracer counts.
   - Report operator time separately from NetCDF I/O.

## Useful Commands

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
tools/gc_harness/build_pjc_pfix_harness.sh
python -m wombat_transport.gc_harness pjc-pfix base_wombat/run.yml
python -m wombat_transport.gc_harness transport-step base_wombat/run.yml --max-tracers 1
python -m wombat_transport.run base_wombat/run.yml --mode transport-window --max-steps 18
python -m wombat_transport.run residual_20140901_part001_split01_wombat/run.yml --mode transport-one-step
```
