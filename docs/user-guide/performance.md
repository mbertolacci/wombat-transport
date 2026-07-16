# Performance and threading

Wombat's main performance opportunity is batching many passive tracers through
the same meteorology. Throughput is reported as:

```text
tracer-steps/s = transport steps * tracer count / wall time
```

This measures completed tracer work per second. For a fixed simulation length
and thousands of independent tracers, maximizing aggregate tracer-steps/s
minimizes wall time, subject to memory capacity and node allocation.

## Thread controls

Numba acceleration is enabled by default when installed. The main controls are:

| Variable | Purpose |
|---|---|
| `WOMBAT_NUMBA` | Global Numba enable/disable switch |
| `WOMBAT_NUMBA_THREADS` | Global Wombat Numba thread count; defaults to 1 |
| `WOMBAT_TPCORE_NUMBA` | Override Numba for TPCORE |
| `WOMBAT_VDIFF_NUMBA` | Override Numba for VDIFF |
| `WOMBAT_CONVECTION_NUMBA` | Override Numba for convection |
| `WOMBAT_HISTORY_NUMBA` | Override Numba for HISTORY accumulation |
| `WOMBAT_OBSOPERATOR_NUMBA` | Override Numba for serial ObsOperator sampling |

Each threaded operator also accepts an operator-specific `_THREADS` variable,
such as `WOMBAT_TPCORE_NUMBA_THREADS`. It overrides
`WOMBAT_NUMBA_THREADS` for that operator.

Numba's own `NUMBA_NUM_THREADS` is an upper bound established when Numba
starts. On a cluster, set it to at least the largest Wombat thread count:

```bash
export NUMBA_NUM_THREADS=8
export WOMBAT_NUMBA_THREADS=8
export OMP_NUM_THREADS=1
```

Falsy switch values are `0`, `false`, `no`, `off`, and `none`. If Numba is
unavailable or disabled for transport, Wombat emits a major performance
warning.

## Local end-to-end comparison

These timings were measured on 16 July 2026 on an Intel Core i7-14700KF. The
one-tracer cases cover two days; the 24- and 100-tracer cases cover one day.
The 100-tracer workload adds 76 synthetic background-only CO2 tracers to the
residual case.

Both engines read MERRA-2 meteorology, run ObsOperator, and write configured
output. Multi-tracer cases also read and apply residual emissions. Runs were
sequential and unpinned, so this is a practical snapshot rather than a
controlled microbenchmark.

| Grid | Tracers | Threads | GEOS-Chem wall s | Wombat wall s | GEOS-Chem tracer-steps/s | Wombat tracer-steps/s | Wombat speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2x2.5 | 1 | 1 | 66.10 | 36.20 | 4.4 | 8.0 | 1.83x |
| 2x2.5 | 1 | 2 | 49.25 | 26.29 | 5.8 | 11.0 | 1.87x |
| 2x2.5 | 1 | 4 | 39.06 | 21.03 | 7.4 | 13.7 | 1.86x |
| 2x2.5 | 24 | 1 | 200.46 | 49.16 | 17.2 | 70.3 | 4.08x |
| 2x2.5 | 24 | 2 | 124.01 | 35.30 | 27.9 | 97.9 | 3.51x |
| 2x2.5 | 24 | 4 | 83.28 | 25.79 | 41.5 | 134.0 | 3.23x |
| 2x2.5 | 100 | 1 | 710.41 | 158.86 | 20.3 | 90.6 | 4.47x |
| 2x2.5 | 100 | 2 | 436.12 | 108.35 | 33.0 | 132.9 | 4.03x |
| 2x2.5 | 100 | 4 | 286.84 | 81.70 | 50.2 | 176.2 | 3.51x |
| 4x5 | 1 | 1 | 16.51 | 9.45 | 17.4 | 30.5 | 1.75x |
| 4x5 | 1 | 2 | 13.61 | 7.67 | 21.2 | 37.5 | 1.77x |
| 4x5 | 1 | 4 | 11.49 | 6.70 | 25.1 | 43.0 | 1.71x |
| 4x5 | 24 | 1 | 49.38 | 13.30 | 70.0 | 259.8 | 3.71x |
| 4x5 | 24 | 2 | 30.46 | 10.02 | 113.4 | 345.0 | 3.04x |
| 4x5 | 24 | 4 | 20.30 | 7.75 | 170.3 | 445.7 | 2.62x |
| 4x5 | 100 | 1 | 176.16 | 37.81 | 81.7 | 380.9 | 4.66x |
| 4x5 | 100 | 2 | 105.78 | 26.58 | 136.1 | 541.7 | 3.98x |
| 4x5 | 100 | 4 | 69.75 | 20.46 | 206.5 | 703.9 | 3.41x |

Wombat speedup is GEOS-Chem wall time divided by Wombat wall time. Fixed
startup, meteorology, ObsOperator, and output costs make scaling with tracer
count deliberately nonlinear.

## High-tracer cluster result

A separate experiment on one 40-core CPU socket of the Hercules cluster at
MSU sustained approximately 700 tracer-steps/s with 400 tracers. That result
is not directly comparable to the local table, but it demonstrates that
single-node throughput can remain strong beyond the small validation cases.

For profiler commands, component timing history, and scaling experiments, see
the repository's
[performance notebook](https://github.com/mbertolacci/wombat-transport/blob/main/performance.md).
