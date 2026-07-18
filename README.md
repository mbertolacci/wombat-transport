# Wombat Transport

Wombat Transport provides parity-first GEOS-Chem Classic tracer transport for
high-throughput ensembles of CO2-like tracers. It reproduces the relevant
transport path in Python and NumPy, with Numba acceleration for production
runs, while treating GEOS-Chem as the numerical reference.

Wombat is deliberately narrower than GEOS-Chem. It runs global transport-only
simulations on the GEOS 2x2.5 and 4x5 grids with 47 vertical levels. Chemistry,
dry deposition, wet deposition, nested grids, and alternate vertical grids are
not currently supported.

[Read the documentation](https://mbertolacci.github.io/wombat-transport/)

## Why Wombat?

Atmospheric inversions and related ensemble studies may need to transport
hundreds or thousands of passive tracers through the same meteorology. Wombat
batches those tracers in a canonical `(lev, lat, lon, tracer)` layout and runs
the GEOS-Chem-style `TPCORE -> VDIFF -> convection` chain without the rest of
the GEOS-Chem chemistry model.

The priorities are:

1. reproduce GEOS-Chem Classic transport semantics;
2. make numerical differences visible through parity harnesses;
3. exploit batched tracers and threaded kernels for high throughput.

## Current status

Short-run comparisons are consistent with GEOS-Chem at floating-point
roundoff for the tested base no-emissions two-day window and residual
24-tracer one-day window at both supported resolutions. These comparisons
cover concentration, restart, and matched ObsOperator samples. There is a
known terminal-boundary difference in ObsOperator file presence.

This is not yet a monthly or long-horizon parity claim. Longer transport and
restart-chain comparisons must be run before relying on parity over those
timescales.

## Installation

Wombat requires Python 3.10 or newer.

```bash
git clone https://github.com/mbertolacci/wombat-transport.git
cd wombat-transport
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

Numba is strongly recommended for useful transport performance:

```bash
.venv/bin/python -m pip install numba
```

Real runs also require MERRA-2 meteorology and a compatible restart/grid
template. See the [external-data guide](https://mbertolacci.github.io/wombat-transport/getting-started/external-data/).

## First run

After populating `external_data/`, run the included three-hour, one-tracer
2x2.5 example:

```bash
.venv/bin/python -m wombat_transport.run examples/basic_2x25/run.yml
```

The [first-run guide](https://mbertolacci.github.io/wombat-transport/getting-started/first-run/)
explains the inputs, output, and configuration. The complete configuration
contract is documented in the [`run.yml` reference](https://mbertolacci.github.io/wombat-transport/reference/run-yml/).

Tracer state is always stored as contiguous tracer blocks. Transport uses
spatial parallelism within each block by default; parallel execution across
blocks is selected per process with:

```bash
WOMBAT_TRANSPORT_EXECUTOR=blocks WOMBAT_NUMBA_THREADS=8 \
  .venv/bin/python -m wombat_transport.run examples/basic_2x25/run.yml
```

`WOMBAT_NUMBA` enables or disables every optional Numba path. The
`WOMBAT_NUMBA_THREADS` value is applied once per process and shared by all
parallel Numba kernels; there are no subsystem-specific overrides.

Block width defaults to the full tracer count for `spatial` execution and to
8 for `blocks`. Either strategy accepts an explicit storage width:

```bash
WOMBAT_TRANSPORT_EXECUTOR=spatial WOMBAT_TRANSPORT_BLOCK_WIDTH=16 ...
```

Spatial execution processes configured blocks sequentially while retaining
within-operator threading. All tracer counts use the shared prepared one-block
transport step. Block execution uses one top-level Numba parallel region
across blocks. Storage layout is therefore independent of execution strategy.

## Performance snapshot

End-to-end local comparisons on an Intel Core i7-14700KF included MERRA-2
reading, residual emissions where configured, ObsOperator sampling, and
output:

- At 2x2.5 with 100 tracers and eight threads, width-16 block execution achieved
  228.8 tracer-steps/s and was 3.48 times faster than GEOS-Chem.
- At 4x5 with 100 tracers and eight threads, width-16 block execution achieved
  951.3 tracer-steps/s and was 3.59 times faster than GEOS-Chem.
- One-tracer runs were approximately 1.6--1.8 times faster than GEOS-Chem in
  the measured local cases.

A separate high-tracer experiment on one 40-core CPU socket of the Hercules
cluster at MSU sustained approximately 700 tracer-steps/s with 400 tracers.
See [performance and threading](https://mbertolacci.github.io/wombat-transport/user-guide/performance/)
for the full table, measurement conditions, and interpretation.

Deployment should be calibrated on the target machine rather than inferred
from core count alone. `tools/benchmark_transport_frontier.py` finds the
fastest process/thread topology, execution strategy, and block width at each
tracer load. For a fixed tracer count, the configuration with the shortest
step is also the one with the highest tracer throughput; the two aligned
panels show the resulting latency/throughput tradeoff.

![Example transport frontier on eight i7-14700KF P-cores](docs/assets/transport-frontier-i7-14700kf.svg)

In this example, 64 tracers already provide 328 tracer-steps/s at 0.195
seconds per step, while loading 512 tracers raises per-machine throughput to
423 tracer-steps/s but takes 1.211 seconds per step. A user can therefore
choose a knee for lower latency, fill one machine for aggregate throughput, or
use the curve to choose how many tracers to assign to each machine. See
[calibrating a new machine](https://mbertolacci.github.io/wombat-transport/user-guide/performance/#calibrating-a-new-machine)
for the workflow and caveats.

## Validation and development

GEOS-Chem is the numerical reference. Differences beyond expected
floating-point roundoff are treated as bugs or explicitly documented
deviations.

- [`validation_runs/README.md`](validation_runs/README.md) describes named
  full-run comparisons.
- [`tools/gc_harness/README.md`](tools/gc_harness/README.md) describes
  GEOS-Chem operator and tracing harnesses.
- [`tools/hemco_harness/README.md`](tools/hemco_harness/README.md) describes
  emissions parity scenarios.
- [`tools/benchmark_transport_frontier.py`](tools/benchmark_transport_frontier.py)
  calibrates process, thread, executor, and block-width choices on a selected
  CPU set using synthetic transport.
- [`performance.md`](performance.md) is the benchmark and profiling notebook.
- [`oracle_data/README.md`](oracle_data/README.md) describes the local
  large-fixture cache.

Run the test suite with:

```bash
.venv/bin/python -m pytest
```

Install the development tools and run the static-analysis baseline with:

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/ruff check
.venv/bin/vulture --config pyproject.toml
```

Ruff is configured for high-signal correctness and unused-code checks rather
than broad style enforcement. Vulture supplements it with conservative
dead-symbol detection.
