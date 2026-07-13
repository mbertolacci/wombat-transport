# Performance Notes

## 2026-07-09 baseline

This branch started with a single-thread performance attribution pass for the
Numba transport path. At the time of the baseline run,
`WOMBAT_TPCORE_NUMBA=all` selected the fully accelerated TPCORE path. That mode
has since been replaced by a boolean switch; unset or truthy
`WOMBAT_TPCORE_NUMBA` now enables the fused Numba path, and false-like values
disable it.

Current transport-wide control is `WOMBAT_NUMBA`: unset or truthy values enable
optional Numba paths when importable, while false-like values disable them.
`WOMBAT_TPCORE_NUMBA`, `WOMBAT_VDIFF_NUMBA`, and `WOMBAT_CONVECTION_NUMBA`
override that global switch for individual operators.

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

## 2026-07-12 Multi-step transport optimization notes

The next likely gains are from repeated work across many transport steps, not
from another isolated one-step kernel pass. The one-step operators have already
been pushed hard with direct timings, `cProfile`, `perf`, codegen inspection,
and Profila line-level Numba profiling. Extra stage timers are not a prerequisite
for obvious repeated-work cleanups; use `cProfile` for Python call attribution
and Profila/codegen inspection for Numba kernel internals.

A 96-tracer, 6-hour profile gave this broad split:

| Stage | Time s |
| --- | ---: |
| Total | 50.47 |
| TPCORE | 30.64 |
| VDIFF | 14.11 |
| Convection | 2.20 |
| TPCORE setup bucket | 0.61 |
| Meteorology forcing | 0.74 |

Two TPCORE setup cleanups were identified as direct, low-risk candidates. First,
the transport driver built a `TpcoreSetup`, then `run_tpcore_one_step()` built
the same setup again internally. This duplicated PJC mass-flux setup, pressure
terms, Courant and divergence setup, vertical flux setup, and branch validation
every normal driver step. Second, static geometry and hybrid-grid terms were
recomputed during setup even though area factors, `geofac`, `geofac_pc`,
`cose`/`cosp`, and hybrid coefficient deltas do not change step-to-step for the
current fixed 2 x 2.5, 47-level grid.

Both cleanups were tested on the same 96-tracer, 6-hour profile:

| Variant | Total s | TPCORE s | TPCORE setup s |
| --- | ---: | ---: | ---: |
| Before multi-step cleanup | 50.47 | 30.64 | 0.61 |
| Reuse driver-built setup | 48.89 | 29.68 | 0.59 |
| Reuse setup + static terms | 48.22 | 29.19 | 0.57 |

The static-term refactor is intentionally outside the Numba kernels, but it
still needs the same caution as any performance edit: if a future version
changes array layout, contiguity, or the exact arrays passed into the kernels,
recheck that it has not backed out earlier vectorization/codegen wins.

VDIFF has less obvious reusable work. Most of the expensive state depends on
evolving dry mass, pressure, humidity, temperature, PBL fields, and emissions.
Low-risk candidates are avoiding repeated zero surface-flux allocation/scans in
no-emissions runs and precomputing slow-cadence forcing views or simple scalars
such as water-flux conversions. These are probably secondary unless profiles of
the target run show VDIFF setup, rather than the solve itself, becoming large.

Convection has one reusable multi-step structure: A3 convection forcing fields
are fixed across a 3-hour block, so top-order forcing views, active-column
masks, and cloud-base indices can potentially be cached for the 18 ten-minute
transport steps in that block. This is lower priority at high tracer counts
because convection is only a small share of runtime, but it may matter more for
low-tracer full-run throughput.

Recommended order:

1. Keep the duplicate TPCORE setup removal if parity tests continue to pass.
2. Keep the TPCORE/PJC static geometry and hybrid-term cache if benchmark wins
   remain stable across normal run profiles.
3. Revisit VDIFF zero-emissions and convection A3-block caching after TPCORE
   setup cleanup is measured in a full output-enabled benchmark.

## 2026-07-12 VDIFF and convection multi-step follow-up

After the TPCORE setup cleanup, the next pass focused on the residual-run path
rather than the no-emissions case. The important VDIFF observation was that the
production full-grid Numba path was only used when surface tracer flux was
zero. Residual runs have nonzero surface emissions, so they fell back to the
older latitude-by-latitude Numba path.

The retained VDIFF change generalizes the full-grid diagnostics-light Numba
path to nonzero surface fluxes. The full-grid kernel now carries the same
surface-flux bookkeeping as the latitude path: bottom flux increments, nonlocal
`cgq`, adjusted `qmx`, and emitted tracer mass in the final mass ratio. A
regression test compares the nonzero-flux production path against the diagnostic
path on the tracked VDIFF fixture.

The 96-tracer, 6-hour residual-style profile moved as follows:

| Variant | Total s | VDIFF s | Convection s |
| --- | ---: | ---: | ---: |
| After TPCORE setup/static cleanup | 48.22 | 13.79 | 2.16 |
| VDIFF/convection pass, first run | 46.65 | 11.34 | 2.76 |
| VDIFF/convection pass, rerun | 43.15 | 8.44 | 2.17 |

The first post-edit run likely still had cache/compilation noise; the rerun is
the better indication of steady behavior. A 24-tracer, 6-hour reference after
this pass was `13.82 s` total, with `8.17 s` TPCORE, `2.68 s` VDIFF, and
`0.71 s` convection.

For convection, the retained change is deliberately conservative. Normal
transport no longer builds or passes wet-scavenging diagnostic fields that are
unused when `reconstruct_conv_precip_flux=False`: wet-pressure box heights,
`PFICU`, `PFLCU`, temperature, and convective precipitation. Trace/harness paths
still request those fields so handoff fixtures remain complete. The normal path
also avoids an extra `delp_hpa` copy.

The more aggressive convection idea was to cache A3 active-column/cloud-base
groups across the 18 ten-minute steps in each 3-hour A3 block. Real residual A3
records have roughly `9.5k-9.9k` active columns out of `13.1k`, with about
`18-21` cloud-base groups. That gives some potential fixed-cost upside, but the
current full-grid Numba kernel has been tuned as a single fused pass. Switching
to grouped cached kernels could accidentally back out the earlier vectorization
and loop-shape wins. Leave this for a dedicated convection-kernel profiling pass
rather than mixing it into the VDIFF improvement.

## 2026-07-12 TPCORE buffer/workspace experiment

The TPCORE explorer produced the following broader idea list, ranked by payoff
versus risk before the buffer/workspace pass:

1. Reusable `q`/`dq1` buffers: avoid repeated full tracer cube
   allocation/page clearing inside TPCORE. Medium impact, manageable aliasing
   risk.
2. Explicit TPCORE output ownership / double buffering: let driver-owned
   next-state buffers flow into VDIFF instead of fresh TPCORE output
   allocation. Medium impact, broader lifecycle change.
3. Store `ua`/`va`/`jn`/`js` in `TpcoreSetup`: avoid recomputing cross Courant
   averages and branch bounds each call. Low risk, likely low-medium impact.
4. TPCORE setup workspace: reuse setup arrays like `delp1`, `delpm`, `pu`,
   `cx`, `cy`, `dpi`, `wz`, `xmass`, and `ymass`. Medium implementation risk,
   low-medium impact.
5. Low-Courant specialized kernels: dispatch to branch-light XTP/YTP kernels
   for validated common path. Medium payoff, medium parity/codegen risk.
6. YTP/FZPPM longitude tiling: improve cache locality while keeping tracer
   inner. Medium possible payoff, high vectorization risk.
7. Precompute horizontal remap indices/masks: cache X/Y integer indices, signs,
   and branch masks per setup. Possible medium payoff, but extra memory traffic
   may hurt.
8. Precompute A3 wind-block PJC helpers: cache wind-only pieces across the 18
   steps sharing A3 winds. Low-medium payoff, mostly setup-side.
9. Carry `p1_hpa` directly between steps: avoid reconstructing pressure from
   dry mass every step. Low risk, low payoff.
10. Avoid normal-path TPCORE flux wrapping: do not allocate/carry
    `xmass`/`ymass`/`zmass` unless diagnostics need them. Low risk, low payoff.
11. Amortize branch validation: validate on first step or met-record changes
    instead of every step. Low numerical risk, small payoff.
12. Pole/init/cross-term fusion: reduce full-grid sweeps. Prior simple fusion
    did not help, so this needs a more careful memory-traffic experiment.
13. `qck` scan/finalize policy by negative frequency: dispatch no-fill or
    fused-fill variants when safe. Small-medium payoff, higher semantic risk.
14. FZPPM scratch allocation surgery: remove remaining Numba NRT scratch
    allocations. Tempting, but prior workspace attempt worsened code shape.
15. Overlap next-step setup in a helper thread: prepare next setup while
    VDIFF/convection run. Fits the allowed threading exception, but setup is
    probably too small for huge wins.
16. Native C++/Fortran TPCORE rewrite: highest ceiling, highest cost/risk.
    Only worth it if Numba hits a hard wall and we can stage it
    operator-by-operator against oracle traces.

Four low-risk TPCORE ideas were tested sequentially on the 96-tracer path. Two
were retained and two were rejected.

Retained:

- The fused Numba TPCORE path now reuses a caller-owned contiguous input buffer
  inside the existing per-shape Numba workspace. This replaces a fresh
  `ascontiguousarray(...).copy()` allocation with `np.copyto` into the reusable
  buffer. The direct TPCORE benchmark improved from `0.804 s` best / `0.807 s`
  mean to `0.781 s` best / `0.790 s` mean.
- The normal transport driver now opts into reusable TPCORE output ownership.
  The traced/diagnostic and public direct-call defaults still allocate a fresh
  output array, but the production driver can safely let the next operator
  consume the workspace-owned result before the next TPCORE call. The 96-tracer
  one-step driver benchmark improved from `1.111 s` best with `0.782 s` in
  TPCORE to `1.045 s` best with `0.721 s` in TPCORE after the retained changes.

Rejected:

- Storing `ua`, `va`, `jn`, and `js` in `TpcoreSetup` regressed the direct
  TPCORE benchmark to `0.791 s` best / `0.803 s` mean. Keeping this work inside
  the Numba workspace/kernel setup appears better for codegen/cache behavior.
- A NumPy-side `TpcoreSetupWorkspace` that filled setup arrays in place regressed
  the one-step driver benchmark to `1.115 s` best. Reducing allocation did not
  offset the cost of the changed NumPy expression shape.

The main lesson is that TPCORE remains sensitive to loop and vectorization
shape. Moving small setup arrays out of the Numba-owned flow or converting
broadcast expressions to manual `out=` forms can make the code slower even when
it allocates less. Future TPCORE edits should measure both the direct TPCORE
benchmark and the driver benchmark, because reusable output only helps the
production driver path.

An isolated C++ YTP prototype under `validation_runs/work/tpcore_cpp_ytp/`
showed that a native port is not automatically faster. With strict
`-O3 -march=native`, the C++ YTP kernel was slower than the current Numba YTP
kernel (`0.207 s` vs `0.170 s` for 96 tracers). With
`-O3 -march=native -ffast-math`, the best simple indexed C++ kernel improved to
`0.144 s` for 96 tracers and `0.041 s` for 24 tracers, about `15-23%` faster
than Numba on the isolated YTP stage. The local parity check was roundoff-clean
for that stage, but `-ffast-math` is a GEOS-Chem parity risk over full
transport windows. The next native-code experiment, if any, should be a full
TPCORE-step prototype comparing strict and fast-math builds against full-step
fixtures before considering production integration.

## 2026-07-12 TPCORE follow-up ideas 5/10/11/7

Four more ideas from the ranked list were tried after the buffer reuse commit.

Rejected:

- Low-Courant specialized XTP dispatch. The experiment routed validated
  non-large-`cx` setups to a branch-light XTP kernel. It passed the TPCORE
  oracle tests, including the large-Courant fixture through the generic path,
  but direct 96-tracer TPCORE regressed from `0.781 s` to `0.787 s`, while the
  driver movement was only noise-level (`1.040 s` vs `1.045 s`). The duplicate
  kernel shape was not worth retaining.
- Precomputed X remap indices. The experiment cached the
  `int((i + 1) - cx) - 1` upstream index array in the TPCORE Numba workspace.
  It passed oracle tests, but direct TPCORE regressed (`0.791 s` on the first
  run and `0.844 s` on rerun, versus `0.781 s` baseline). Driver timings were
  noisy and did not justify the extra memory traffic.

Retained:

- Normal transport no longer wraps TPCORE `xmass`/`ymass`/`zmass` flux
  diagnostics unless explicitly requested with `include_flux_diagnostics=True`.
  The production runner does not use these arrays; trace/oracle paths still opt
  in where needed. The one-step 96-tracer driver benchmark moved from
  `1.045 s` best / `1.050 s` mean to `1.040 s` best / `1.043 s` mean, with the
  overhead bucket dropping from about `0.021 s` to `0.0185 s`.
- Multi-step loops now amortize TPCORE branch validation. Public one-step calls
  still validate by default, but `run_transport_window` validates only the first
  step of its fixed-forcing window, and the full runner validates at the start
  of each 3-hour met window. A synthetic 18-step, 96-tracer comparison gave
  identical checksums and improved from `19.20-19.26 s` when validating every
  step to `19.04-19.14 s` when validating once for the window.

The main lesson from ideas 5 and 7 is that pulling scalar remap work out of XTP
does not automatically help. That work is not tracer-inner, and the additional
kernel/code shape or memory traffic can cost more than the saved integer
operations. Future X/Y work needs to target a larger structural change than
precomputing one index array or duplicating the low-Courant branch.

## 2026-07-12 TPCORE Numba threading stage 1-2

The first Numba threading pass parallelized the largest independent TPCORE
kernels over horizontal rows/columns. That helped, but the 8-thread 96-tracer
profile then became dominated by formerly serial support stages:
`poles_plus_dq_init`, `calc_cross_terms`, `xadv_dao2`, and `yadv_dao2`.

Two follow-up groups were implemented and retained:

1. Parallelize pole averaging over tracers and replace the profiler's
   standalone `dq1 = q * delp1` helper with the production
   `_init_dq_mass_numba_kernel`.
2. Parallelize cross-term setup plus X/Y DAO2 application. Y DAO2 needs
   deterministic pole treatment, so it writes per-longitude south/north pole
   increments to workspace buffers and reduces those in fixed longitude order.

The focused transport/TPCORE test subset passed with
`WOMBAT_TPCORE_NUMBA_THREADS=8`: `55 passed`.

Direct TPCORE scaling after the change:

| Threads | 1 tracer best s | 8 tracer best s | 24 tracer best s | 96 tracer best s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0700 | 0.1205 | 0.2355 | 0.7996 |
| 2 | 0.0537 | 0.0914 | 0.1717 | 0.5155 |
| 4 | 0.0449 | 0.0700 | 0.1267 | 0.3617 |
| 8 | 0.0393 | 0.0607 | 0.0968 | 0.3015 |

For the target 96-tracer case, this improves the 8-thread path from the
previous post-YTP-threading `~0.438 s` profile to `0.303 s` best, while the
1-thread path stays effectively unchanged at `~0.80 s`.

The new 8-thread staged profile for 96 tracers is:

| Stage | Mean s | Share |
| --- | ---: | ---: |
| `python_copy_workspace_cross_terms` | 0.0567 | 20.0% |
| `qckxyz_fill_finalize` | 0.0489 | 17.3% |
| `poles_plus_dq_init` | 0.0436 | 15.4% |
| `ytp_horizontal_mass_flux` | 0.0333 | 11.7% |
| `xtp_horizontal_mass_flux` | 0.0332 | 11.7% |
| `fzppm_vertical` | 0.0297 | 10.5% |
| `calc_cross_terms` | 0.0215 | 7.6% |
| `yadv_dao2` | 0.0105 | 3.7% |
| `xadv_dao2` | 0.0062 | 2.2% |

The remaining obvious bottlenecks are no longer the main X/Y/Z transport
kernels. `python_copy_workspace_cross_terms` is mostly setup/copy overhead,
`qckxyz_fill_finalize` is mostly global scan/finalize traffic, and
`poles_plus_dq_init` still contains pole reductions plus full-grid mass
initialization. Further improvement probably needs larger structural changes
than simply adding more `prange`: reduce full-grid passes, avoid normal
negative-fill scans when validated safe, or move more of the whole TPCORE step
into a single compiled/native flow.

## 2026-07-12 VDIFF full-grid threading

The diagnostics-light VDIFF production path now uses the full-grid Numba kernel
for all configured VDIFF thread counts. The previous multi-thread path used the
older latitude helper shape; the new path parallelizes the tuned full-grid
kernel directly over latitude and expands its scratch workspace with a leading
`nthreads` dimension. Each worker uses `get_thread_id()` to select private
`(lon, lev, tracer)` work arrays, so the per-latitude algorithm and tracer-inner
loop order stay unchanged.

The old latitude-parallel dispatch helper was removed from the production code.
Diagnostics/debug fallback paths still use the scalar latitude implementation
when full diagnostics are requested or Numba is unavailable.

Validation:

- `WOMBAT_VDIFF_NUMBA_THREADS=2 pytest tests/test_transport.py -q`:
  `30 passed`.
- `WOMBAT_VDIFF_NUMBA_THREADS=8 pytest -q`: `225 passed, 2 skipped`.

Direct VDIFF scaling after the change:

| Threads | 1 tracer best s | 8 tracer best s | 24 tracer best s | 96 tracer best s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0264 | 0.0364 | 0.0582 | 0.1582 |
| 2 | 0.0155 | 0.0227 | 0.0367 | 0.0934 |
| 4 | 0.0085 | 0.0131 | 0.0223 | 0.0713 |
| 8 | 0.0050 | 0.0088 | 0.0168 | 0.0674 |

The 96-tracer case improves from `0.158 s` at one thread to `0.067 s` at eight
threads, about `2.35x`. Scaling tapers after four threads, which is consistent
with the kernel becoming memory/cache-bandwidth limited rather than scheduler
limited. Checksums were stable across all thread counts in the benchmark.

## 2026-07-12 Convection scratch padding

Convection initially showed pathological 2-thread behavior at low and moderate
tracer counts. A quick `numba.set_parallel_chunksize()` experiment did not fix
the pattern consistently: chunk size `1` helped 8 tracers but regressed 24 and
96 tracers, so scheduler imbalance was not the dominant issue.

The retained fix pads the per-thread convection scratch rows to at least 32
tracers. The kernel still loops over the logical tracer count; the padding only
separates each thread's `qc`, `qb_num`, `delq_work`, and `current_work` rows in
memory. This strongly suggests false sharing was a major part of the bad
2-thread behavior.

Convection best times before and after 32-wide scratch padding:

| Threads | Tracers | Before s | Padded s | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 0.0345 | 0.0060 | 0.17 |
| 2 | 8 | 0.0764 | 0.0113 | 0.15 |
| 2 | 24 | 0.0363 | 0.0237 | 0.65 |
| 2 | 96 | 0.1270 | 0.0843 | 0.66 |
| 4 | 24 | 0.0545 | 0.0152 | 0.28 |
| 8 | 24 | 0.0292 | 0.0120 | 0.41 |
| 8 | 96 | 0.0525 | 0.0554 | 1.06 |

A smaller 16-wide pad was rejected because it regressed the 24-tracer cases
badly (`2` threads: `0.0568 s`, `8` threads: `0.0344 s`) and did not recover
the 96-tracer 8-thread result. With 32-wide padding, low/moderate tracer
threading is much healthier. The 96-tracer 8-thread case is approximately
flat to slightly slower in the standalone benchmark, so the best production
thread mix may still differ by tracer count and workload.

Validation after retaining the padding:

- `WOMBAT_CONVECTION_NUMBA_THREADS=8 pytest tests/test_transport.py -q`:
  `30 passed`.

## 2026-07-12 Modest-thread follow-up experiments

Three follow-up experiments targeted the poor or sublinear modest-thread
behavior, especially for a future strategy where tracers may be split across
multiple processes with a small thread count per process.

Rejected code experiments:

- TPCORE shared pole/reduction buffer padding. The YTP and Y-DAO2 shared
  south/north pole buffers were padded from `(nlon, ntracer)` to
  `(nlon, max(ntracer, 32))` to test for false sharing between longitude
  workers. It did not help the important 2-thread cases and was backed out:

| Threads | Tracers | Before s | Padded s | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 0.0537 | 0.0545 | 1.02 |
| 2 | 8 | 0.0914 | 0.0920 | 1.01 |
| 2 | 24 | 0.1717 | 0.1777 | 1.03 |
| 2 | 96 | 0.5155 | 0.5023 | 0.97 |

- Manual-block convection scheduling. Replacing Numba's `prange(col)` scheduler
  with `prange(worker)` and contiguous manual column blocks helped some small
  tracer cases and the 8-thread/96-tracer case, but it regressed the 2- and
  4-thread 96-tracer cases badly. It was backed out because the goal was one
  implementation that scales acceptably up to 8 threads:

| Threads | Tracers | Padded scheduler s | Manual block s | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 8 | 0.0113 | 0.0108 | 0.95 |
| 2 | 24 | 0.0237 | 0.0233 | 0.99 |
| 2 | 96 | 0.0843 | 0.1281 | 1.52 |
| 4 | 96 | 0.0645 | 0.0741 | 1.15 |
| 8 | 96 | 0.0554 | 0.0464 | 0.84 |

Process-level concurrency experiment:

A fixed total of 96 tracers was split across 1, 2, or 4 concurrent benchmark
processes, with either 1 or 2 Numba threads per process. Each process ran the
full synthetic TPCORE + VDIFF + convection driver. The effective step time is
the slowest process's reported best timed step, excluding warmup/compilation.

| Processes | Tracers/process | Threads/process | Effective step s | Aggregate tracers/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 96 | 1 | 1.0674 | 89.9 |
| 2 | 48 | 1 | 0.6101 | 157.3 |
| 4 | 24 | 1 | 0.4191 | 229.1 |
| 1 | 96 | 2 | 0.8092 | 118.6 |
| 2 | 48 | 2 | 0.4708 | 203.9 |
| 4 | 24 | 2 | 0.3343 | 287.1 |
| 8 | 12 | 1 | 0.3357 | 285.9 |

This is encouraging for the split-tracer strategy: at least up to four
concurrent processes and eight total worker threads, aggregate throughput
continued improving. Adding the core-count-equivalent `8 processes x 12 tracers
x 1 thread` point gave `286 tracer-steps/s`, essentially the same as
`4 processes x 24 tracers x 2 threads` at `287 tracer-steps/s`. That suggests
the tested workload is close to saturating useful eight-worker throughput, but
that either process-only or modest per-process threading can reach that point.
The next concurrency sweep should include CPU affinity/SMT placement controls
before drawing a deployment policy from it.

Validation after backing out the rejected code experiments:

- `WOMBAT_TPCORE_NUMBA_THREADS=2 WOMBAT_VDIFF_NUMBA_THREADS=2
  WOMBAT_CONVECTION_NUMBA_THREADS=2 pytest tests/test_transport.py -q`:
  `30 passed`.
