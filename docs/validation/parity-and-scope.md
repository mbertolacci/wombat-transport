# Parity and scope

GEOS-Chem Classic is Wombat's numerical reference. A difference beyond
expected floating-point roundoff is treated as meaningful until it is traced
to an understood source or explicitly accepted.

## Current credible claim

Short-run comparisons are consistent with GEOS-Chem for:

- a realistic one-tracer restart with no emissions over two days;
- a 24-tracer residual-emissions case over one day;
- both global GEOS 2x2.5 and 4x5 grids with 47 levels;
- concentration, matched ObsOperator samples, and restart output where the
  case includes restart comparison.

Matched numerical outputs agree at floating-point roundoff in these cases.
The engines differ in whether an ObsOperator file is present at the terminal
boundary, because their restart/end-of-run behavior is not identical.

## What has not been established

Do not generalize the short-run result to:

- monthly or arbitrary long-horizon integrations;
- repeated long restart chains;
- chemistry or deposition;
- nested, cubed-sphere, or alternate vertical grids;
- emissions unit conversion beyond the currently explicit `none` policy.

Monthly residual restart drift and multi-week/month transport-only behavior
need explicit comparison before stronger parity claims are made.

## Validation layers

Wombat uses several complementary checks:

1. Small unit tests cover configuration, array conventions, I/O, and edge
   cases.
2. GEOS-Chem-backed operator fixtures compare TPCORE, VDIFF, convection, and
   their handoffs.
3. HEMCO harness scenarios compare source reading, scaling, optional
   dimensions, conservative regridding, and polar-row behavior.
4. Named full-run cases compare complete GEOS-Chem and Wombat outputs.
5. Instrumented tracing can follow selected columns, full tracer snapshots,
   and transport internals when a comparison fails.

The operational details remain close to their tools:

- [Validation runs](https://github.com/mbertolacci/wombat-transport/blob/main/validation_runs/README.md)
- [GEOS-Chem harness](https://github.com/mbertolacci/wombat-transport/blob/main/tools/gc_harness/README.md)
- [HEMCO harness](https://github.com/mbertolacci/wombat-transport/blob/main/tools/hemco_harness/README.md)
- [Oracle-data cache](https://github.com/mbertolacci/wombat-transport/blob/main/oracle_data/README.md)

Generated NetCDF payloads and local oracle caches are intentionally excluded
from Git. Their tracked manifests and READMEs define the expected contracts.
