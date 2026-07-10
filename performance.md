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

## 2026-07-09 column/row scratch YTP and XTP update

YTP and XTP now avoid their remaining large horizontal work arrays in the fused
Numba path:

- YTP uses column scratch shaped `(nlat, ntracer)` for `dcy`, `al`, `ar`, and
  `a6`, instead of four full `(nlat, nlon, ntracer)` planes.
- XTP uses row scratch shaped `(nlon, ntracer)` for `dcx`, instead of a full
  `(nlat, nlon, ntracer)` plane.

Synthetic TPCORE outputs for 1, 24, and 96 tracers matched the previous Numba
output exactly after both changes.

Standalone TPCORE timing after the combined YTP/XTP scratch rewrite:

| Tracers | Best wall s | Mean wall s |
| ---: | ---: | ---: |
| 1 | 0.085 | 0.090 |
| 24 | 0.279 | 0.289 |
| 96 | 0.907 | 0.912 |

The 96-tracer best wall time was previously about `1.02-1.04 s` after the
FZPPM column-scratch rewrite, so this is another roughly 11-13% improvement for
the large-tracer case.

Current staged Numba timing for 96 tracers after the YTP/XTP rewrite:

| Stage | Mean s | Percent |
| --- | ---: | ---: |
| `ytp_horizontal_mass_flux` | 0.178 | 20.2% |
| `xtp_horizontal_mass_flux` | 0.151 | 17.1% |
| `fzppm_vertical` | 0.149 | 16.8% |
| `poles_plus_dq_init` | 0.069 | 7.8% |
| `copy/workspace/cross setup` | 0.067 | 7.6% |
| `yadv_dao2` | 0.066 | 7.5% |
| `xadv_dao2` | 0.060 | 6.8% |
| `calc_cross_terms` | 0.051 | 5.8% |
| `qckxyz_fill` | 0.036 | 4.0% |
| `q_prepass_update` | 0.032 | 3.6% |
| `finalize_output` | 0.025 | 2.9% |

Current isolated stage `perf stat` counters for 96 tracers:

| Stage | Task ms | IPC | Backend bound | Branch miss | L1D miss | LLC miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ytp_horizontal_mass_flux` | 1898 | 2.47 | 53.1% | 0.24% | 4.16% | 63.92% |
| `xtp_horizontal_mass_flux` | 1718 | 2.41 | 52.2% | 0.25% | 3.60% | 93.53% |
| `fzppm_vertical` | 1543 | 2.25 | 52.0% | 0.47% | 9.94% | 48.74% |

Compared with the pre-YTP/XTP split, YTP improved from `0.255 s` staged time,
`1.65 IPC`, and `67.5%` backend bound to about `0.178 s`, `2.47 IPC`, and
`53.1%` backend bound. XTP improved from about `0.171 s` staged time to about
`0.151 s`.

Focused verification:

```text
tests/test_tpcore_scaling_benchmark.py: 18 passed
tests/test_transport.py + tests/test_transport_driver_scaling_benchmark.py: 22 passed
tests/test_gc_harness.py -k tpcore: 15 passed, 4 skipped, 37 deselected
```

Next candidates are now less obvious. The leading stages are close together
(`ytp`, `xtp`, `fzppm`), and the fixed per-level passes
(`poles_plus_dq_init`, cross-term setup, DAO2 prepass, and q-update) together
are a comparable target. A plausible next pass is reducing/fusing those
prepass memory sweeps rather than continuing to optimize one PPM kernel in
isolation.

## 2026-07-09 VDIFF full-grid diagnostics-light path

VDIFF now has a production hot path for the normal zero-tracer-surface-flux
case used by the transport driver:

- `run_vdiffdr_one_step(..., diagnostics=True)` remains the default and keeps
  the existing full `kvh`, `kvm`, `tpert`, `qpert`, and mass diagnostics for
  oracle comparisons.
- `diagnostics=False` uses one full-grid Numba call for zero tracer surface
  flux, computes pressure/height setup inside compiled code, skips diagnostic
  output arrays, and avoids the Python latitude loop.
- The non-trace transport driver path requests `diagnostics=False`; trace and
  GEOS-Chem fixture paths keep `diagnostics=True`.
- `tools/benchmark_vdiff_scaling.py` benchmarks the diagnostics-light path.

Equivalence check against the existing diagnostic implementation on a 24-tracer
synthetic full grid:

```text
tracer_max_abs 0.0
sphu_max_abs 0.0
negative 0 0
diag_shape (0,)
```

Same-environment 96-tracer timing:

| Path | Best wall s | Mean wall s | Checksum |
| --- | ---: | ---: | ---: |
| Full diagnostics | 0.425 | 0.429 | 23936.383650816002 |
| Diagnostics-light full-grid Numba | 0.301 | 0.301 | 23936.383650816002 |

Updated VDIFF scaling benchmark:

| Tracers | Best wall s | Mean wall s | Seconds/tracer |
| ---: | ---: | ---: | ---: |
| 24 | 0.083 | 0.084 | 0.00345 |
| 96 | 0.299 | 0.304 | 0.00312 |

Updated VDIFF 96-tracer perf counters for the diagnostics-light path:

| Metric | Value |
| --- | ---: |
| IPC | 1.40 |
| Backend bound | 64.3% |
| Frontend bound | 7.4% |
| Retiring | 27.1% |
| Branch miss | 0.08% |
| L1D miss | 9.45% |
| LLC miss | 48.31% |
| Page faults | 1,484,287 |

The remaining profile was still memory/backend bound. This motivated the
reusable-workspace pass below, since the profile still showed page-fault and
NRT deallocation activity even after removing the full diagnostic array outputs.

## 2026-07-09 VDIFF reusable workspace pass

The diagnostics-light full-grid kernel now gets all scratch arrays from a
module-level reusable workspace keyed by `(nlev, nlon, ntracer)`. The retained
version keeps the previous full `(nlon, nlev, ntracer)` tracer scratch layout
because that loop shape is faster than a streamed one-column solve.

Tried and rejected: streaming the tracer solve through one `(nlev,)` column
buffer. It was numerically identical, but slower:

| Variant | 24 tracer best s | 96 tracer best s | 96 tracer mean s |
| --- | ---: | ---: | ---: |
| Pre-workspace diagnostics-light | 0.083 | 0.299 | 0.304 |
| Streamed column solve | 0.111 | 0.377 | 0.409 |
| Reusable workspace, tracer-cube solve | 0.078 | 0.260 | 0.265 |

The likely reason is that the tracer-cube solve exposes contiguous tracer lanes
more cleanly to LLVM, while the column solve trades memory footprint for much
less favorable vector/cache behavior.

Updated 24-tracer equivalence check against the full diagnostic implementation:

```text
tracer_max_abs 0.0
sphu_max_abs 0.0
negative 0 0
diag_shape (0,)
```

Updated VDIFF 96-tracer perf counters for the retained reusable-workspace path:

| Metric | Value |
| --- | ---: |
| Best wall | 0.258 s |
| Mean wall | 0.260 s |
| IPC | 1.43 |
| Backend bound | 66.5% |
| Frontend bound | 5.1% |
| Retiring | 28.3% |
| Branch miss | 0.10% |
| L1D miss | 8.36% |
| LLC miss | 11.98% |
| Page faults | 11,937 |

LLVM/ASM inspection of the retained kernel:

```text
NRT allocation mentions: 0
LLVM vector.body mentions: 195
LLVM <N x double> vector types: 308
ASM ymm mentions: 928
ASM xmm mentions: 1543
```

The page-fault collapse is the clearest proof that this change hit the intended
allocation churn. The kernel is still backend bound, but now much more of that
is actual compiled VDIFF work rather than repeated Numba scratch allocation.

Future note: splitting met/coefficient setup from the tracer solve remains a
good candidate, but it should wait until we know whether real driver forcing
cadence lets those coefficients be reused across multiple transport substeps.
If met changes every step, coefficient reuse is mostly an organization cleanup;
if met is reused, it could be a large win.

## Numba profiling workflow

Use three profiling layers for Numba transport kernels:

1. `cProfile` for Python wrapper overhead. It cannot see inside compiled Numba
   code, but it quickly answers whether time is still in Python, benchmark
   checksums, allocation wrappers, or the Numba dispatcher call.
2. `perf` for native and hardware-counter behavior. Use it to measure IPC,
   backend/frontend bound, cache misses, branch misses, page faults, and the
   split between JIT code, NumPy, libc, and kernel work.
3. `Profila` for approximate line-level attribution inside single-threaded
   Numba kernels.

Profila is useful when `perf` reports mostly `[JIT]` addresses and cannot map
them back to Python source lines. It is Linux-only, supports single-threaded
Numba, and samples every 10 ms, so run the target kernel in a loop for several
seconds.

Install/run Profila in a temporary environment rather than adding it as a
project dependency:

```bash
UV_CACHE_DIR=/tmp/wombat-uv-cache \
uv run --no-project --python 3.11 \
  --with numpy --with netCDF4 --with PyYAML --with numba --with profila \
  python -m profila setup
```

`profila setup` downloads an isolated `gdb` helper. It does not replace system
`gdb`. In this environment Profila needed ptrace permission, so the annotate
run had to be allowed outside the sandbox. A fresh Numba cache was also
important; otherwise Profila reused cached code without useful debug/source
mapping.

Template command:

```bash
PYTHONPATH=/home/mgnb/Projects/UWA/FluxInversion/wombat-transport-transport-performance/src \
UV_CACHE_DIR=/tmp/wombat-uv-cache \
NUMBA_CACHE_DIR=/tmp/wombat-numba-cache-profila-fresh \
NUMBA_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
uv run --no-project --python 3.11 \
  --with numpy --with netCDF4 --with PyYAML --with numba --with profila \
  python -m profila annotate -- /tmp/profile_vdiff_profila.py --tracers 96 --seconds 12
```

For repeatable use, create a small operator-specific script that:

- builds synthetic inputs,
- warms the target Numba function once,
- loops the warmed compiled function for at least 5-10 seconds,
- prints an iteration count and checksum.

Interpret Profila percentages directionally. Debug info and `gdb` sampling can
change runtime behavior; in the VDIFF run, Profila roughly doubled wall time.
Always validate candidate changes with the normal benchmark and keep only
wall-time-neutral-or-better changes.

## 2026-07-09 VDIFF Profila-guided zero-fill removal

Profila worked after running with ptrace permission and a fresh Numba cache so
the kernel was compiled with debug info. A successful 96-tracer diagnostics-light
run collected 1,788 samples and annotated the full-grid kernel. Treat these
percentages directionally, not as exact timings, because debug info and gdb
sampling roughly doubled wall time under Profila.

The useful signal was that the hot region is the tracer solve plus mass
rescale/output, not setup. The largest annotated lines were:

| Line/area | Profila share |
| --- | ---: |
| final `tracer_out[...] = tracer_diffused[...] * ratio` | 16.6% |
| load `value = tracer_diffused[...]` in mass scan | 9.7% |
| forward tracer diffusion body | ~8-9% |
| `before_mass += tracer_top * dry_mass` | 4.9% |
| backward substitution body | ~4-5% |
| negative check | 3.1% |
| zeroing `tracer_diffused[...] = 0.0` | 2.9% |

The full-grid fast path overwrote every `tracer_diffused` value before reading
it, so the explicit zero-fill loop was removed. The older diagnostic latitude
kernel was left unchanged.

Validation and timing:

```text
tracer_max_abs 0.0
sphu_max_abs 0.0
negative 0 0
```

| Variant | 24 tracer best s | 96 tracer best s | 96 tracer mean s | 192 tracer best s |
| --- | ---: | ---: | ---: | ---: |
| Reusable workspace before zero-fill removal | 0.078 | 0.258 | 0.260 | 0.498 |
| Zero-fill removed | 0.077 | 0.250 | 0.252 | 0.484 |

This is a small but real win, and it agrees with the Profila hint.

## 2026-07-09 VDIFF ratio/write-order rewrite

Profila's largest signal after zero-fill removal was still the mass-rescale and
output write phase. The original loop computed each `(lon, tracer)` ratio and
then wrote `tracer_out` with `lev` as the inner loop. That made the hottest
store stride through the output array rather than writing contiguous tracer
lanes.

The retained rewrite stores per-`(lon, tracer)` scale factors in a reusable
`tracer_ratio` workspace, then writes output with `lev` outer and `tracer`
inner:

```text
for lon:
    for tracer:
        compute tracer_ratio[lon, tracer]
    for lev:
        for tracer:
            tracer_out[lev, lat, lon, tracer] = tracer_diffused[lon, lev, tracer] * tracer_ratio[lon, tracer]
```

This keeps the two-pass dependency required by the column mass ratio, but makes
the final output write and `tracer_diffused` read run across contiguous tracer
lanes.

Validation:

```text
tracer_max_abs 0.0
sphu_max_abs 0.0
negative 0 0
checksum 5930.883189504  # 24-tracer synthetic check
```

Timing:

| Variant | 24 tracer best s | 96 tracer best s | 96 tracer mean s | 192 tracer best s |
| --- | ---: | ---: | ---: | ---: |
| Zero-fill removed | 0.077 | 0.250 | 0.252 | 0.484 |
| Ratio workspace + contiguous-tracer writeback | 0.064 | 0.198 | 0.211 | 0.359 |

Perf counters for a 96-tracer profile run after the rewrite:

| Metric | Value |
| --- | ---: |
| Best wall in profiler run | 0.215 s |
| IPC | 1.67 |
| Backend bound | 59.4% |
| Frontend bound | 6.7% |
| Retiring | 33.5% |
| Branch miss | 0.13% |
| L1D miss | 8.23% |
| LLC miss | 19.79% |
| Page faults | 10,655 |

The direct benchmark is faster than the profiler-run benchmark, but both show
the same direction. Compared with the pre-workspace diagnostics-light path, the
96-tracer VDIFF best wall time is now about `0.299 -> 0.198 s`. Compared with
the full-diagnostics path, it is about `0.425 -> 0.198 s`.

## 2026-07-09 VDIFF mass-scan loop-order rewrite

A fresh Profila pass after the ratio/write-order rewrite showed that the final
writeback remained the largest single line, but the conservation/negative scan
was now the clearest non-solve block:

| Region | Approx samples |
| --- | ---: |
| final `tracer_out[...] = tracer_diffused[...] * tracer_ratio[...]` | 13.2% |
| mass/negative scan loop, load, branch, and mass adds | ~17% |
| forward tracer solve | ~10% |
| backward tracer solve | ~7-8% |

The retained change rewrites the mass/negative scan to accumulate all tracers
for a longitude with `lev` outer and `tracer` inner. The existing
`tracer_ratio` workspace temporarily holds before-mass, and a new
`tracer_after_mass` workspace holds after-mass. This keeps each individual
tracer's vertical accumulation order unchanged, but makes the hot inner loop
walk contiguous tracer lanes.

A second retained pass then moved the before-mass accumulation into the forward
tracer solve, where `tracer_top` is already being read. The post-solve scan now
only clips negatives and accumulates after-mass before computing the ratio.

Old-vs-new validation against the previous commit for the 24-tracer synthetic
case was bitwise identical:

```text
tracer_max_abs 0.0
sphu_max_abs 0.0
negative 0 0
checksum 5930.883189504
```

Timing:

| Variant | 24 tracer best s | 96 tracer best s | 96 tracer mean s | 192 tracer best s |
| --- | ---: | ---: | ---: | ---: |
| Ratio workspace + contiguous-tracer writeback | 0.063 | 0.191 | 0.192 | 0.356 |
| Mass scan reordered across contiguous tracers | 0.059 | 0.171 | 0.172 | 0.325 |
| Before-mass fused into tracer solve | 0.059 | 0.160 | 0.162 | 0.295 |

Final Profila check after both retained changes:

| Region | Approx samples |
| --- | ---: |
| final writeback | 13.6% |
| fused before-mass accumulation inside forward solve | 9.5% |
| forward tracer solve arithmetic/store | ~8% |
| backward tracer solve | ~7% |
| post-solve after-mass/negative scan | ~4% |

The remaining likely VDIFF targets are the final writeback, the forward/backward
tridiagonal tracer solve, and call-level output allocation/page clearing.

## 2026-07-09 TPCORE follow-up targets after Profila

TPCORE was revisited after the VDIFF/Profila workflow was available. No code was
changed in that pass; the goal was to identify the next useful optimization
targets.

Current standalone fused-Numba TPCORE timings:

| Tracers | Best wall s | Mean wall s |
| ---: | ---: | ---: |
| 24 | 0.262 | 0.264 |
| 96 | 0.805 | 0.860 |
| 192 | 1.622 | 1.623 |

Current 96-tracer staged profile:

| Stage | Mean s | Percent |
| --- | ---: | ---: |
| `ytp_horizontal_mass_flux` | 0.170 | 21.5% |
| `xtp_horizontal_mass_flux` | 0.160 | 20.2% |
| `fzppm_vertical` | 0.150 | 19.0% |
| `poles_plus_dq_init` | 0.069 | 8.7% |
| `calc_cross_terms` | 0.058 | 7.3% |
| `python_copy_workspace_cross_terms` | 0.057 | 7.2% |
| `qckxyz_fill_finalize` | 0.053 | 6.7% |

Profila gave useful but noisy line-level attribution. The strongest signal was
not a single dominant stage, but limiter-heavy scalar inner loops across XTP,
YTP, and FZPPM. Numba's lowered `min(...)`/`max(...)` helpers also appeared in
the sample profile, matching the hot limiter regions.

Follow-up ideas to try later, in likely payoff/risk order:

1. Replace hot chained `min(...)`/`max(...)` limiter expressions in XTP, YTP,
   and FZPPM with explicit scalar comparisons. Validate tightly because NaN and
   equality behavior can differ from Python builtins.
2. Precompute horizontal remap indices, coefficients, and masks for X/Y paths
   to remove repeated `int`, `rint`, modulo, validity, and sign work from hot
   loops.
3. Add reusable or caller-managed TPCORE output/workspace to avoid per-call
   `q` copies, `dq1` allocation, and repeated cross-term setup where valid.
4. Revisit `qckxyz_fill_finalize`; the separate full-grid negative/finalize
   scan is now around 6-7% at 96 tracers.
5. Move FZPPM column scratch out of the compiled kernel into reusable workspace.
   This is less obvious than the first two, but FZPPM is again close to XTP/YTP
   in total cost.

## 2026-07-09 Convection diagnostics-light path

Convection was profiled after the VDIFF work. The old Python column chunking is
not used by the Numba path: Numba receives each cloud-base column group in one
call, while `_max_convection_group_columns` and `_iter_column_chunks` are only
used by the pure Python fallback.

Before this pass, standalone convection timings were:

| Tracers | Best wall s | Mean wall s |
| ---: | ---: | ---: |
| 24 | 0.108 | 0.110 |
| 96 | 0.397 | 0.399 |
| 192 | 0.808 | 0.835 |

Manual timing around the 96-tracer path showed that the Numba kernel was only
about half of wall time:

| Bucket | Mean s | Share |
| --- | ---: | ---: |
| Numba convection kernel | 0.200 | 49.7% |
| Mass reductions | 0.039 | 9.7% |
| Wrapper/copy/diag allocation | 0.164 | 40.6% |

The retained change adds `diagnostics=False` for the normal driver and
benchmark path, while preserving `diagnostics=True` as the default for direct
operator calls and harness/oracle comparisons. The light path skips
`diag14_mass_flux` allocation/writes, initial/final mass reductions, and
negative-count scans. It returns empty diagnostic arrays, matching the VDIFF
diagnostics-light convention.

Full-vs-light validation on the 24-tracer synthetic benchmark input:

```text
tracer_max_abs 0.0
light_diag_shape (0,)
light_mass_shapes (0,) (0,)
full_checksum 0.0004012190050963058
light_checksum 0.0004012190050963058
```

Updated diagnostics-light benchmark:

| Tracers | Best wall s | Mean wall s |
| ---: | ---: | ---: |
| 24 | 0.066 | 0.067 |
| 96 | 0.217 | 0.218 |
| 192 | 0.417 | 0.419 |

This is a large wrapper/diagnostic win: about `0.397 -> 0.217 s` at 96 tracers
and `0.808 -> 0.417 s` at 192 tracers. The remaining obvious convection target
is reusable output/workspace to reduce the still-present tracer copy and page
clearing, followed by profiling the now-more-dominant Numba kernel itself.

Current end-to-end driver benchmark after the convection change:

| Tracers | Best total s | TPCORE s | VDIFF s | Convection s | Overhead s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 0.398 | 0.236 | 0.059 | 0.063 | 0.023 |
| 96 | 1.214 | 0.797 | 0.163 | 0.212 | 0.026 |

## 2026-07-09 Convection workspace and fused light kernel

Two follow-up convection experiments were retained.

First, the diagnostics-light driver/benchmark path now opts into reusable
output workspace. Direct operator calls keep `reuse_output=False` by default so
the public-safe behavior still returns a fresh output array; the transport
driver uses `reuse_output=True` because it consumes the result immediately. The
workspace reuses the tracer output and dry-air-mass arrays, replacing repeated
large allocation with `copyto`/`multiply(..., out=...)`.

Second, the diagnostics-light Numba path now bypasses Python-side active-column
and cloud-base grouping. A fused full-grid Numba kernel scans active columns,
derives each column's cloud base, and runs the convection update in one compiled
pass. Full diagnostics and the pure Python fallback keep the existing grouped
implementation for oracle/debug behavior.

Full-vs-light validation for the synthetic 24- and 96-tracer inputs:

```text
24 tracer_max_abs 0.0 checksum 0.0004012190050963058
96 tracer_max_abs 0.0 checksum 0.00040481900509630625
```

Standalone benchmark progression:

| Variant | 24 tracer best s | 96 tracer best s | 96 tracer mean s | 192 tracer best s |
| --- | ---: | ---: | ---: | ---: |
| Diagnostics-light, fresh output | 0.065 | 0.216 | 0.218 | 0.419 |
| Reusable output workspace | 0.056 | 0.181 | 0.182 | 0.351 |
| Fused full-grid light Numba kernel | 0.048 | 0.173 | 0.173 | 0.334 |

Manual 96-tracer timing split after both retained changes:

| Region | Best s | Mean s |
| --- | ---: | ---: |
| output copy + dry-air-mass setup | 0.019 | 0.019 |
| fused Numba convection kernel | 0.152 | 0.153 |
| total measured hot path | 0.172 | 0.173 |

Current end-to-end driver benchmark:

| Tracers | Best total s | TPCORE s | VDIFF s | Convection s | Overhead s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 0.382 | 0.234 | 0.060 | 0.049 | 0.024 |
| 96 | 1.182 | 0.808 | 0.160 | 0.174 | 0.024 |

At this point convection is mostly the compiled kernel plus an unavoidable input
copy unless the broader driver becomes explicitly in-place/double-buffered. The
next useful convection work should profile the fused kernel itself, with likely
targets being active/cloud-base scans, repeated below-base plume setup, and
branch structure inside the per-column vertical loop.

## 2026-07-10 Convection delq split and entrainment hoist

Profila on the fused diagnostics-light kernel showed the dominant cost in the
`cmfmc_below > _TINYNUM` per-tracer update, especially the clamp/write sequence:

```text
if current + delq < 0.0        18.6%
q_all[...] = current + delq    22.8%
```

Three variants were tried:

1. Compute per-tracer `delq` into a small `(ntracer,)` workspace, then run a
   second tracer loop for clamp/write.
2. Hoist `entrains` outside the tracer loop, but keep the immediate clamp/write.
3. Keep the `delq` workspace and also hoist `entrains` outside the tracer loop,
   using separate entraining and non-entraining compute loops.

Variant 1 was neutral/slightly noisy. Variant 2 was clean and best at 24 tracers,
but slower than variant 3 at 96 and 192 tracers. Variant 3 was retained because
large tracer counts are the target workload.

Standalone benchmark progression:

| Variant | 24 tracer best s | 96 tracer best s | 96 tracer mean s | 192 tracer best s |
| --- | ---: | ---: | ---: | ---: |
| Fused full-grid light Numba kernel | 0.048 | 0.173 | 0.182 | 0.335 |
| Split `delq` workspace, no hoist | 0.049 | 0.172 | 0.173 | 0.333 |
| Hoisted `entrains`, no `delq` workspace | 0.047 | 0.170 | 0.171 | 0.331 |
| Split `delq` workspace, hoisted `entrains` | 0.048 | 0.167 | 0.168 | 0.326 |

Full-vs-light validation remained bitwise identical for 24 and 96 tracers.

Profila after the retained variant showed the clamp/write block was much less
dominant:

| Region | Approx samples |
| --- | ---: |
| entraining `q_all[level, col, tracer]` read in plume update | 4.8% |
| `upward = cmfmc * q_all[level - 1, ...]` | 5.0% |
| `delq -= cmfmc_below * q_all[level, ...]` | 7.9% |
| `delq_work[tracer] = delq * tendency_scale` | 6.4% |
| clamp check | 8.0% |
| final store | 8.5% |

The hot path is now more evenly distributed across the flux arithmetic,
temporary `delq` write, clamp, and final store.

## 2026-07-10 Convection current-value and scalar hoists

Two follow-up Profila-driven ideas were tested against the retained
`delq_work` + hoisted-`entrains` kernel:

1. Hoist repeated `cmfmc_all[level, col]` loads into a level-local scalar.
2. Cache each tracer's current `q_all[level, col, tracer]` value in a small
   `(ntracer,)` workspace during the `delq` compute loop, then reuse that value
   in the final clamp/write loop.

The full 2x2 experiment showed both changes help, and they are additive.

| Variant | 24 tracer best s | 96 tracer best s | 96 tracer mean s | 192 tracer best s |
| --- | ---: | ---: | ---: | ---: |
| Baseline `delq_work` + hoisted `entrains` | 0.048 | 0.166 | 0.167 | 0.326 |
| Scalar hoists only | 0.048 | 0.156 | 0.157 | 0.304 |
| Current-value workspace only | 0.039 | 0.130 | 0.131 | 0.258 |
| Scalar hoists + current-value workspace | 0.037 | 0.122 | 0.124 | 0.241 |

The retained variant adds one more small per-tracer workspace, but removes
repeated full-grid `q_all[level, col, tracer]` loads from the hottest branch.
This is a large convection-kernel win at the target large tracer counts.
A confirmation run of the retained combined variant gave best times of
`0.037`, `0.126`, and `0.239 s` for 24, 96, and 192 tracers respectively.

One final simplification was tested: keep the scalar hoists and local `current`
load, but remove `current_work`/`delq_work` and immediately clamp/write inside
the compute loop. This was rejected. It regressed to `0.048`, `0.169`, and
`0.329 s` best times for 24, 96, and 192 tracers, suggesting the split
compute-then-clamp loops are materially better for generated code even though
the temporary workspace stores appear in Profila.

## 2026-07-10 TPCORE follow-up experiments

TPCORE was profiled again after the VDIFF/convection work. Current 96-tracer
standalone timing was about `0.810-0.823 s`. The staged split was:

| Stage | Mean s | Share |
| --- | ---: | ---: |
| `ytp_horizontal_mass_flux` | 0.175 | 22.8% |
| `fzppm_vertical` | 0.147 | 19.2% |
| `xtp_horizontal_mass_flux` | 0.146 | 19.0% |
| `poles_plus_dq_init` | 0.068 | 8.8% |
| `python_copy_workspace_cross_terms` | 0.057 | 7.4% |
| `calc_cross_terms` | 0.053 | 6.9% |
| `qckxyz_fill_finalize` | 0.050 | 6.6% |

Whole-kernel `perf stat -d` at 96 tracers: IPC `1.89`, backend bound `59.4%`,
frontend bound `5.4%`, branch misses `0.25%`, L1D miss rate `5.15%`, and LLC
load miss rate `74.38%`. Isolated stage counters still show backend/cache
pressure, especially in fixed full-grid passes. Profila remained diffuse but
again highlighted limiter-heavy `min`/`max` code in XTP/YTP/FZPPM.

Five follow-up experiments were run:

| Experiment | 96-tracer result | Decision |
| --- | ---: | --- |
| Replace hot three-argument `min`/`max` limiters with explicit scalar comparisons | `0.863 s` best | Rejected; slower than baseline |
| Fuse pole averaging with `dq1 = q * mass` initialization | `0.824 s` best | Rejected; straightforward fusion did not remove enough traffic |
| Always run `qckxyz` fill instead of the guarded negative scan | `0.836 s` best | Rejected; guarded scan is cheaper for this workload |
| Use no-clip finalize after `fill=True` | `0.819 s` best | Rejected; extra kernel split did not pay |
| Hoist per-cell `ua[j, i]`/`va[j, i]` loads in cross-term setup | `0.809-0.815 s` best | Retained; neutral to slightly positive |

The retained cross-term hoist is deliberately small. Confirmation timings were
`0.236`, `0.815`, and `2.210 s` best for 24, 96, and 256 tracers respectively,
versus the prior current run of `0.243`, `0.810`, and `2.218 s`. The larger
precomputed-index workspace idea was not retained here because the cheap hoist
gave only noise-level movement, and a full precompute would add per-step setup
work unless those indices become reusable across multiple transport calls.

## 2026-07-10 TPCORE codegen-guided experiments

LLVM/ASM inspection was used after the Profila pass. Summary:

- XTP, YTP, FZPPM, cross-terms, and finalize all generate SIMD vector bodies.
- The generated vector width is AVX2-style `<4 x double>` / YMM, not AVX-512.
- `qckxyz_needs_fill` and `qckxyz` have no vector bodies.
- FZPPM still has 8 NRT allocation references from internal scratch arrays.
- The limiter-heavy loops are vectorized but branch/select heavy; this matches
  the earlier result that explicit scalar `min`/`max` rewrites were slower.

Four codegen-guided follow-up experiments were then tried:

| Experiment | 96-tracer result | Decision |
| --- | ---: | --- |
| Replace early-return `qckxyz_needs_fill` with a full reduction-style scan | `0.859 s` best | Rejected; early exit wins for this workload |
| Rework pole averaging with reusable per-tracer accumulators | `0.816 s` best, noisy mean | Rejected; not better than baseline |
| Move FZPPM scratch arrays into reusable workspace | `0.840 s` best | Rejected; removing NRT allocations worsened kernel shape |
| Hoist scalar coefficients/masses inside XTP/YTP | `0.827 s` best | Rejected; worse than baseline |

No additional TPCORE code changes were retained from this codegen pass. The
current retained TPCORE change remains only the earlier cross-term `ua`/`va`
scalar hoist.
