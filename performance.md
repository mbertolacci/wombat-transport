# Performance Notes

## 2026-07-09 baseline

This branch started with a single-thread performance attribution pass for the
Numba transport path. At the time of the baseline run,
`WOMBAT_TPCORE_NUMBA=all` selected the fully accelerated TPCORE path. That mode
has since been replaced by a boolean switch; unset or truthy
`WOMBAT_TPCORE_NUMBA` now enables the fused Numba path, and false-like values
disable it.

The benchmark scripts now support `--warmup`, defaulting to one untimed run per
tracer count. This avoids timing first-call Numba compilation when collecting
steady-state kernel/runtime numbers.

Benchmarks were run from this worktree with the original checkout's local
fixtures:

```bash
PYTHONPATH=/home/mgnb/Projects/UWA/FluxInversion/wombat-transport-transport-performance/src \
NUMBA_CACHE_DIR=/tmp/wombat-numba-cache \
NUMBA_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
/home/mgnb/Projects/UWA/FluxInversion/wombat-transport/.venv/bin/python \
tools/benchmark_transport_driver_scaling.py \
  --run-config /home/mgnb/Projects/UWA/FluxInversion/wombat-transport/base_wombat/run.yml \
  --counts 1 24 96 \
  --repeat 2 \
  --warmup 1 \
  --output /tmp/wombat-driver-benchmark.csv
```

Equivalent commands were run for `tools/benchmark_tpcore_scaling.py`,
`tools/benchmark_vdiff_scaling.py`, and
`tools/benchmark_convection_scaling.py`.

## Driver attribution

Best timed run by tracer count:

| Tracers | Total s | TPCORE | VDIFF | Convection | Setup | Overhead |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.181 | 53.8% | 16.0% | 7.1% | 8.8% | 14.2% |
| 24 | 0.743 | 49.3% | 13.6% | 14.2% | 2.1% | 20.8% |
| 96 | 2.643 | 52.8% | 12.9% | 14.8% | 0.6% | 19.0% |

## Standalone operator timings

Best timed run by tracer count:

| Tracers | TPCORE s | VDIFF s | Convection s |
| ---: | ---: | ---: | ---: |
| 1 | 0.098 | 0.031 | 0.013 |
| 24 | 0.366 | 0.099 | 0.107 |
| 96 | 1.398 | 0.338 | 0.386 |

## Interpretation

TPCORE is the first optimization target. It accounts for roughly half of the
end-to-end transport step across the measured tracer counts, and its standalone
runtime is well above VDIFF and convection.

Driver overhead is also significant at multi-tracer counts, around 19-21% for
24 and 96 tracers in this run. After TPCORE kernel work, allocation/copy/object
wrapping around the staged driver should be measured directly.

VDIFF and convection are similar in cost by 24-96 tracers. They are not the
first target unless a narrower benchmark or real forcing fixture shows a
different profile.

## Verification notes

The benchmark-script tests passed under the project virtual environment with
this branch on `PYTHONPATH`, excluding only the two synthetic-builder tests that
require large fixture files to exist inside this sibling worktree:

```bash
PYTHONPATH=/home/mgnb/Projects/UWA/FluxInversion/wombat-transport-transport-performance/src \
NUMBA_CACHE_DIR=/tmp/wombat-numba-cache \
/home/mgnb/Projects/UWA/FluxInversion/wombat-transport/.venv/bin/python \
  -m pytest \
  tests/test_tpcore_scaling_benchmark.py \
  tests/test_vdiff_scaling_benchmark.py \
  tests/test_convection_scaling_benchmark.py \
  tests/test_transport_driver_scaling_benchmark.py \
  -k 'not synthetic_builder'
```

Result: 24 passed, 2 deselected.

The large GEOS-Chem fixture files are available in the original checkout, not
copied into this sibling worktree. Benchmark commands therefore used the
original checkout's `base_wombat/run.yml` so grid-template paths resolved to the
real files.

## 2026-07-09 fused TPCORE update

TPCORE was converted to use a fused Numba non-trace path controlled by a boolean
`WOMBAT_TPCORE_NUMBA` switch. The old per-axis `x`/`y`/`z`/`all` mode choices
were removed; unset or truthy values enable Numba, and false-like values disable
it.

The code is now split into a `tpcore` package:

- `src/wombat_transport/transport/tpcore/_core.py`: public API, setup, branch
  analysis, and the pure NumPy/reference trace path.
- `src/wombat_transport/transport/tpcore/_numba.py`: optional fused Numba
  non-trace path and compiled helper kernels.
- `src/wombat_transport/transport/tpcore/types.py`: TPCORE dataclasses.

Standalone TPCORE best timed run after fusion:

| Tracers | Previous optimized s | Fused s | Change |
| ---: | ---: | ---: | ---: |
| 1 | 0.098 | 0.072 | -26.8% |
| 24 | 0.343 | 0.284 | -17.2% |
| 96 | 1.308 | 1.164 | -11.0% |
| 256 | 3.638 | 3.376 | -7.2% |

Driver benchmark best timed run after fusion:

| Tracers | Previous total s | Fused total s | Total change | Previous TPCORE s | Fused TPCORE s | TPCORE change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.181 | 0.158 | -12.8% | 0.098 | 0.072 | -26.6% |
| 24 | 0.743 | 0.667 | -10.3% | 0.366 | 0.290 | -20.7% |
| 96 | 2.643 | 2.406 | -9.0% | 1.395 | 1.161 | -16.7% |

Focused TPCORE tests passed after the fused change:

```bash
PYTHONPATH=/home/mgnb/Projects/UWA/FluxInversion/wombat-transport-transport-performance/src \
NUMBA_CACHE_DIR=/tmp/wombat-numba-cache \
/home/mgnb/Projects/UWA/FluxInversion/wombat-transport/.venv/bin/python \
  -m pytest \
  tests/test_tpcore_scaling_benchmark.py \
  tests/test_gc_harness.py \
  -k 'tpcore and not fullgrid and not large_base and not residual_initial and not vdiff_after_tpcore'
```

Result: 33 passed, 41 deselected.

After the package split, the focused tests were rerun with
`PYTHONPYCACHEPREFIX=/tmp/wombat-pycache` because this linked worktree cannot
write local `__pycache__` directories under the sandbox:

```bash
PYTHONPATH=/home/mgnb/Projects/UWA/FluxInversion/wombat-transport-transport-performance/src \
PYTHONPYCACHEPREFIX=/tmp/wombat-pycache \
NUMBA_CACHE_DIR=/tmp/wombat-numba-cache \
/home/mgnb/Projects/UWA/FluxInversion/wombat-transport/.venv/bin/python \
  -m pytest \
  tests/test_tpcore_scaling_benchmark.py \
  tests/test_gc_harness.py \
  -k 'tpcore and not fullgrid and not large_base and not residual_initial and not vdiff_after_tpcore'
```

Result: 33 passed, 41 deselected.

The pure fallback smoke also passed:

```bash
WOMBAT_TPCORE_NUMBA=0 \
PYTHONPATH=/home/mgnb/Projects/UWA/FluxInversion/wombat-transport-transport-performance/src \
PYTHONPYCACHEPREFIX=/tmp/wombat-pycache \
NUMBA_CACHE_DIR=/tmp/wombat-numba-cache \
/home/mgnb/Projects/UWA/FluxInversion/wombat-transport/.venv/bin/python \
  -m pytest \
  tests/test_gc_harness.py \
  -k 'test_python_tpcore_matches_low_courant_oracle_tracer_step'
```

Result: 1 passed, 55 deselected.

Import and benchmark smoke after the split:

```bash
PYTHONPATH=/home/mgnb/Projects/UWA/FluxInversion/wombat-transport-transport-performance/src \
PYTHONPYCACHEPREFIX=/tmp/wombat-pycache \
NUMBA_CACHE_DIR=/tmp/wombat-numba-cache \
/home/mgnb/Projects/UWA/FluxInversion/wombat-transport/.venv/bin/python \
  -c 'from wombat_transport.transport import tpcore; print(tpcore._numba_tpcore_mode()); print(tpcore.run_tpcore_one_step.__module__)'
```

Output:

```text
1
wombat_transport.transport.tpcore._core
```

One-tracer TPCORE benchmark smoke after the split completed in `0.0759 s`,
consistent with the fused-path timing above.
