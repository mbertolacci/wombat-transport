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

## 2026-07-09 mass diagnostics hoist

The fused driver profile showed that most non-operator overhead at many tracer
counts came from always-on scalar mass diagnostics. These diagnostics have been
hoisted out of the normal transport path: `run_transport_one_step` and
`run_transport_window` now return transport state only, while validation callers
use trace output plus explicit mass-diagnostic helpers when they need budget
checks.

Driver benchmark best timed run after removing always-on mass diagnostics:

| Tracers | Previous fused total s | No mass-diag total s | Total change | Previous overhead s | No mass-diag overhead s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.158 | 0.166 | +4.9% | 0.0249 | 0.0235 |
| 24 | 0.667 | 0.538 | -19.4% | 0.1542 | 0.0229 |
| 96 | 2.406 | 1.930 | -19.8% | 0.4973 | 0.0268 |

The one-tracer total is effectively noise/regression at this scale, because mass
diagnostics were not a large cost there. The many-tracer case is the target:
overhead is now roughly flat with tracer count in this benchmark, and the step
is dominated by the three transport operators again.

## 2026-07-09 repeatable TPCORE Numba profiling

`tools/profile_tpcore_numba.py` captures the profiling workflow used for the
native TPCORE investigation. It runs a steady-state TPCORE benchmark, a staged
Numba timing breakdown, LLVM/ASM inspection summaries, and optional delayed
`perf stat`/`perf record` passes. With `--stage-perf`, it also attaches
`perf stat` to isolated worker processes for selected TPCORE suboperators.

Run it with a fresh Numba cache when codegen inspection matters; cached Numba
functions can otherwise report empty LLVM/ASM inspection output:

```bash
PYTHONPATH=/home/mgnb/Projects/UWA/FluxInversion/wombat-transport-transport-performance/src \
PYTHONPYCACHEPREFIX=/tmp/wombat-pycache \
NUMBA_CACHE_DIR=/tmp/wombat-numba-cache-profile \
NUMBA_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
/home/mgnb/Projects/UWA/FluxInversion/wombat-transport/.venv/bin/python \
tools/profile_tpcore_numba.py \
  --run-config /home/mgnb/Projects/UWA/FluxInversion/wombat-transport/base_wombat/run.yml \
  --tracers 96 \
  --repeat 5 \
  --warmup 2 \
  --stage-repeat 5 \
  --perf-repeat 8 \
  --output-dir /tmp/wombat-tpcore-profile
```

The `perf` pass requires host permissions such as
`kernel.perf_event_paranoid=1`. Use `--skip-perf` to collect the benchmark,
staged timing, and codegen summary without native sampling.

To split hardware counters by suboperator, add `--stage-perf`. This currently
profiles the top Numba stages (`ytp_horizontal_mass_flux`,
`xtp_horizontal_mass_flux`, and `fzppm_vertical`) by running each stage in an
isolated worker loop and attaching `perf stat -p` after Numba warmup:

```bash
PYTHONPATH=/home/mgnb/Projects/UWA/FluxInversion/wombat-transport-transport-performance/src \
PYTHONPYCACHEPREFIX=/tmp/wombat-pycache \
NUMBA_CACHE_DIR=/tmp/wombat-numba-cache-profile \
NUMBA_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
/home/mgnb/Projects/UWA/FluxInversion/wombat-transport/.venv/bin/python \
tools/profile_tpcore_numba.py \
  --run-config /home/mgnb/Projects/UWA/FluxInversion/wombat-transport/base_wombat/run.yml \
  --tracers 96 \
  --repeat 3 \
  --warmup 2 \
  --stage-repeat 3 \
  --skip-perf \
  --stage-perf \
  --stage-perf-iterations 10 \
  --output-dir /tmp/wombat-tpcore-stage-profile
```

The stage perf counters are not exact whole-TPCORE attribution: each worker
prepares representative arrays, repeats one stage, and measures that stage in
isolation. They are intended to compare hardware bottlenecks between stages.

## 2026-07-09 column-scratch FZPPM update

The Numba vertical PPM kernel now processes one latitude/longitude column at a
time with scratch arrays shaped `(nlev, ntracer)`, instead of row-wide scratch
arrays shaped `(nlev, nlon, ntracer)`. The tracer dimension remains the inner
contiguous loop. Synthetic TPCORE outputs for 1, 24, and 96 tracers matched the
previous Numba output exactly.

Standalone 96-tracer TPCORE improved from the native-profile baseline of about
`1.16-1.18 s` best wall time to about `1.02-1.04 s` best wall time.

Current repeatable profile output for 96 tracers:

```text
96 tracer TPCORE best wall: 1.022793 s
mean wall: 1.0257691 s
checksum: 0.0004048350081850619
```

Current staged Numba timing for 96 tracers:

| Stage | Mean s | Percent |
| --- | ---: | ---: |
| `ytp_horizontal_mass_flux` | 0.256 | 26.1% |
| `xtp_horizontal_mass_flux` | 0.171 | 17.5% |
| `fzppm_vertical` | 0.150 | 15.3% |
| `poles_plus_dq_init` | 0.068 | 6.9% |
| `yadv_dao2` | 0.068 | 6.9% |
| `copy/workspace/cross setup` | 0.066 | 6.8% |
| `xadv_dao2` | 0.054 | 5.6% |
| `calc_cross_terms` | 0.054 | 5.5% |
| `qckxyz_fill` | 0.036 | 3.6% |
| `q_prepass_update` | 0.032 | 3.3% |
| `finalize_output` | 0.025 | 2.6% |

Compared with the earlier staged profile, `fzppm_vertical` fell from about
`0.311 s` to about `0.150 s`. The top TPCORE cost is now horizontal Y transport,
then horizontal X transport.

Current delayed `perf stat -d` counters for the 96-tracer benchmark:

```text
IPC: 1.55
backend bound: 67.2%
branch misses: 0.28%
L1D load miss rate: 8.79%
LLC load miss rate: 82.37%
page faults: 30,468 over the delayed sampled region
```

Current delayed `perf record` DSO attribution on the core event:

```text
78.1%  Numba JIT code
 8.2%  kernel, mostly clear_page_erms
 6.7%  libc, mostly memset/memcpy
 4.2%  NumPy
 2.6%  Python
```

Current isolated stage `perf stat` counters for 96 tracers:

| Stage | Task ms | IPC | Backend bound | Branch miss | L1D miss | LLC miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ytp_horizontal_mass_flux` | 2575 | 1.65 | 67.5% | 0.26% | 7.94% | 83.95% |
| `xtp_horizontal_mass_flux` | 1695 | 2.42 | 51.2% | 0.25% | 3.27% | 92.00% |
| `fzppm_vertical` | 1532 | 2.24 | 52.2% | 0.52% | 10.15% | 48.25% |

This split makes `ytp_horizontal_mass_flux` the strongest next target: it is
the largest staged cost and has whole-kernel-like backend/cache pressure. `xtp`
has high IPC and low L1 miss rate despite being the second-largest stage, so it
may be less immediately memory-locality limited. `fzppm` is no longer the top
stage after the column-scratch rewrite, but its L1 miss rate remains visible.

Codegen checks still show clean native Numba compilation and vector bodies in
the main kernels. The FZPPM kernel still has NRT allocation references, but the
allocations are now for column-sized scratch. Further gains are more likely to
come from reducing horizontal transport memory traffic or hoisting/reusing
wrapper-level workspaces across repeated transport steps than from more FZPPM
scratch layout work.

Focused verification after the column-scratch change:

```bash
PYTHONPATH=/home/mgnb/Projects/UWA/FluxInversion/wombat-transport-transport-performance/src \
PYTHONPYCACHEPREFIX=/tmp/wombat-pycache \
NUMBA_CACHE_DIR=/tmp/wombat-numba-cache \
NUMBA_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
/home/mgnb/Projects/UWA/FluxInversion/wombat-transport/.venv/bin/python \
  -m pytest tests/test_tpcore_scaling_benchmark.py
```

Result: 18 passed.

The fixture-backed focused transport checks were run from the original checkout
with this branch on `PYTHONPATH`:

```bash
PYTHONPATH=/home/mgnb/Projects/UWA/FluxInversion/wombat-transport-transport-performance/src \
PYTHONPYCACHEPREFIX=/tmp/wombat-pycache \
NUMBA_CACHE_DIR=/tmp/wombat-numba-cache \
NUMBA_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
/home/mgnb/Projects/UWA/FluxInversion/wombat-transport/.venv/bin/python \
  -m pytest tests/test_transport.py tests/test_transport_driver_scaling_benchmark.py
```

Result: 22 passed.

```bash
PYTHONPATH=/home/mgnb/Projects/UWA/FluxInversion/wombat-transport-transport-performance/src \
PYTHONPYCACHEPREFIX=/tmp/wombat-pycache \
NUMBA_CACHE_DIR=/tmp/wombat-numba-cache \
NUMBA_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
/home/mgnb/Projects/UWA/FluxInversion/wombat-transport/.venv/bin/python \
  -m pytest tests/test_gc_harness.py -k 'tpcore'
```

Result: 15 passed, 4 skipped, 37 deselected.

Next candidate: focus first on `ytp_horizontal_mass_flux`, then `xtp` if the
Y-path changes help. The first thing to inspect is whether Y transport's full
`(nlat, nlon, ntracer)` work arrays can be made more local or reused without
changing the validated GEOS-Chem semantics.
