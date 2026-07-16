# Wombat Transport

Wombat Transport provides parity-first GEOS-Chem Classic transport for
high-throughput ensembles of CO2-like tracers.

Many atmospheric inverse problems transport hundreds or thousands of passive
tracers through the same meteorology. Wombat isolates the relevant
GEOS-Chem-style transport path, batches tracers in one array, and uses Numba
for threaded execution. GEOS-Chem remains the numerical reference: performance
work is accepted only when parity tests or an explicit scientific decision
justify the resulting semantics.

## What Wombat runs

The current model target is intentionally narrow:

- global GEOS 2x2.5 and 4x5 horizontal grids;
- 47 GEOS-Chem vertical levels;
- CO2-like passive tracers in `(lev, lat, lon, tracer)` order;
- the `TPCORE -> VDIFF -> convection` transport chain;
- non-local PBL handling of surface emissions through VDIFF;
- GEOS-Chem-style HISTORY, restart, and ObsOperator output.

Wombat is not a replacement for the complete GEOS-Chem model. Chemistry, dry
deposition, wet deposition, nested grids, cubed-sphere grids, and alternate
vertical grids are outside the current scope.

## Why use it?

Wombat is aimed at workloads where tracer count is large enough for batched
transport to matter. Local end-to-end comparisons have shown roughly 3.4--3.5
times lower wall time than GEOS-Chem for measured 100-tracer, four-thread
cases. A separate 40-core CPU-socket experiment sustained about 700
tracer-steps/s with 400 tracers.

These figures are workload- and hardware-dependent. See
[performance and threading](user-guide/performance.md) for the complete
measurements and the definition of tracer-step throughput.

## Validation status

Short-run concentration, restart, and matched ObsOperator outputs agree with
GEOS-Chem at floating-point roundoff for the tested base no-emissions two-day
window and residual 24-tracer one-day window at both supported resolutions.
There is a known terminal-boundary difference in ObsOperator file presence.

The current evidence does not establish monthly or general long-horizon
parity. See [parity and scope](validation/parity-and-scope.md) for the precise
claim and validation pointers.

## Start here

1. [Install Wombat](getting-started/installation.md).
2. [Populate the external-data tree](getting-started/external-data.md).
3. [Run the included three-hour example](getting-started/first-run.md).
4. Use the [`run.yml` reference](reference/run-yml.md) to configure a study.
