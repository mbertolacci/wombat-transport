# Performance Notes

## 2026-07-09 baseline

This branch started with a single-thread performance attribution pass for the
Numba transport path. At the time of the baseline run,
`WOMBAT_TPCORE_NUMBA=all` selected the fully accelerated TPCORE path. That mode
has since been replaced by a boolean switch; unset or truthy
`WOMBAT_TPCORE_NUMBA` now enables the fused Numba path, and false-like values
disable it.

Current repository-wide control is `WOMBAT_NUMBA`: unset or truthy values
enable all optional Numba paths when importable, while false-like values disable
them. `WOMBAT_NUMBA_THREADS` is the single process-wide worker count and is
applied once. Operator-specific switches and thread controls mentioned in older
entries below are retained only as historical records and no longer exist.

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
  --run-config /home/mgnb/Projects/UWA/FluxInversion/wombat-transport/validation_runs/cases/realistic_restart_noemis_2x25/wombat/main/run.yml \
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
original checkout's `validation_runs/cases/realistic_restart_noemis_2x25/wombat/main/run.yml` so grid-template paths resolved to the
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
  --run-config /home/mgnb/Projects/UWA/FluxInversion/wombat-transport/validation_runs/cases/realistic_restart_noemis_2x25/wombat/main/run.yml \
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
  --run-config /home/mgnb/Projects/UWA/FluxInversion/wombat-transport/validation_runs/cases/realistic_restart_noemis_2x25/wombat/main/run.yml \
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
  --with numpy --with netCDF4 --with py-yaml12 --with numba --with profila \
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
  --with numpy --with netCDF4 --with py-yaml12 --with numba --with profila \
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

## 2026-07-13 initial VTune characterization

Intel VTune Profiler 2026.3 was run on the local Intel Core i7-14700KF. The
single-thread runs were pinned to logical CPU 8, a performance core. Four-thread
runs were pinned to logical CPUs `0,2,4,6`, which are four distinct performance
cores rather than SMT siblings. This matters on the hybrid 8-P-core/12-E-core
host; unpinned results mix different core types and are not directly comparable.

For Numba source/JIT registration, the profile environment used
`NUMBA_ENABLE_PROFILING=1` and `NUMBA_DEBUGINFO=1`. Compile once into a dedicated
cache before collecting steady-state results. A first TPCORE trace that included
fresh debug compilation was dominated by compiler front-end behavior and was not
representative. The retained traces used the warmed cache and repeated each
operator long enough to dominate Python startup and fixture loading.

Representative single-thread setup:

```bash
source /opt/intel/oneapi/vtune/latest/env/vars.sh
PYTHONPATH=src \
PYTHONPYCACHEPREFIX=/tmp/wombat-pycache-vtune \
NUMBA_CACHE_DIR=/tmp/wombat-numba-cache-vtune \
NUMBA_ENABLE_PROFILING=1 \
NUMBA_DEBUGINFO=1 \
NUMBA_NUM_THREADS=1 \
WOMBAT_NUMBA_THREADS=1 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
vtune -collect uarch-exploration \
  -knob collect-memory-bandwidth=true \
  -cpu-mask=8 \
  -target-duration-type=short \
  -finalization-mode=full \
  -result-dir=/tmp/wombat-vtune-single-tpcore-uarch \
  -- taskset -c 8 .venv/bin/python tools/benchmark_tpcore_scaling.py \
    --run-config validation_runs/cases/realistic_restart_noemis_2x25/wombat/main/run.yml \
    --counts 96 --repeat 20 --warmup 2 \
    --output /tmp/wombat-vtune-single-tpcore.csv
```

The warmed 96-tracer microarchitecture summaries were:

| Operator | Retiring | Frontend bound | Bad speculation | Backend bound | Memory bound | Core bound |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TPCORE | 52.9% | 16.0% | 1.1% | 29.9% | 22.4% | 7.6% |
| VDIFF | 30.3% | 4.8% | 0.4% | 64.5% | 50.7% | 13.8% |
| Convection | 44.0% | 7.4% | 1.2% | 47.4% | 27.4% | 20.0% |

The VTune pass refines the earlier broad backend-bound interpretation:

- TPCORE is mixed rather than overwhelmingly backend bound on this CPU. Within
  its backend share, the strongest signals are L1 dependency, split loads,
  store latency, and moderate DRAM pressure. Large speculative algorithmic
  changes justified only by a generic memory-bound label are unlikely to pay.
- VDIFF is clearly memory/store bound. Store Bound was 32.3% of clockticks and
  Store Latency 48.5%, with Memory Bandwidth active for 45.1% of clockticks.
  This strengthens the case for reducing or combining the final rescale/output
  writeback and other full-workspace stores. False sharing was 0% in the
  single-thread trace and is not relevant to this result.
- Convection has a previously hidden divider bottleneck: Divider was 44.2% of
  clockticks, nearly all floating-point division. DRAM Bound was 15.0%, Split
  Loads 14.7%, and Memory Bandwidth active for 45.7% of clockticks. The first
  source experiment should hoist reciprocal values that are invariant across
  the tracer loop, especially `1 / cmout`, while preserving operation ordering
  in a separately parity-checked candidate. The per-level
  `internal_dt_s / bmass` values are already outside tracer loops but may still
  be candidates for cached reciprocal multiplication if parity permits.

Four-thread `threading` analysis used the same warmed cache and four pinned
performance cores. The profile benchmarks and utilization were:

| Operator | 1-thread best s | 4-thread best s | Speedup | Effective use of four pinned cores | Spin/overhead share |
| --- | ---: | ---: | ---: | ---: | ---: |
| TPCORE | 1.503 | 0.560 | 2.69x | 74% | 0.9% |
| VDIFF | 0.161 | 0.066 | 2.45x | 71% | 0.0% |
| Convection | 0.126 | 0.080 | 1.56x | 55% | 0.1% |

These wall times include the debug/profiling build and should not replace the
normal benchmark numbers elsewhere in this file. They are useful for comparing
the matched VTune runs. TBB spin and scheduler overhead are small, so the
sublinear scaling is primarily useful-work imbalance, serial regions, and
hardware bottlenecks rather than task-runtime overhead. TPCORE worker CPU times
were closely balanced (`13.31-13.62 s` effective across the three worker
threads), while the main thread also performs serial/setup work. Convection is
the weakest threading target and remains a good candidate for process-level
tracer parallelism unless the divider/memory bottlenecks are reduced first.

VTune result directories from this pass are temporary under `/tmp` and are not
tracked. Use `uarch-exploration` for single-thread hardware diagnosis,
`memory-access` only after a source region is narrowed, and `threading` for
multi-thread waits/utilization. Profila remains useful to map a VTune-identified
Numba JIT bottleneck to approximate Python source lines, but it should not be
used for final wall-time decisions.

## 2026-07-13 TPCORE stage-level VTune and unified opportunity report

`tools/profile_tpcore_numba.py` now supports direct, duration-based isolated
stage workers. `--stage-worker-direct` bypasses the stdin handshake used by the
older attach-based `perf` workflow, while `--stage-worker-seconds` keeps the
selected warmed stage active long enough for sampling. The worker performs one
selected-stage warmup before timing, reports the actual iteration count, and
retains the checksum output. The original stdin/attach behavior is unchanged
when `--stage-worker-direct` is absent.

Nine 96-tracer stages were collected for at least six seconds each using
single-thread `uarch-exploration` pinned to P-core logical CPU 8. The table uses
the current staged timing shares recorded above to distinguish an expensive
stage from a dramatic bottleneck in a small stage:

| TPCORE stage | Approx TPCORE share | Retiring | Backend | Memory | Core | Primary VTune signal |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| YTP horizontal mass flux | 22.8% | 54.9% | 27.7% | 21.7% | 6.0% | L1 dependency; mixed/healthy overall |
| FZPPM vertical | 19.2% | 57.9% | 21.8% | 16.8% | 4.9% | front-end 19.1%; highest retiring |
| XTP horizontal mass flux | 19.0% | 53.4% | 28.7% | 23.1% | 5.6% | L1 dependency; mixed/healthy overall |
| poles plus `dq` initialization | 8.8% | 25.6% | 65.9% | 54.9% | 11.0% | split loads, DRAM and store latency |
| copy/workspace/cross setup | 7.4% | 23.2% | 67.8% | 64.1% | 3.6% | 41.7% Store Bound; 37.2% FB Full |
| cross terms | 6.9% | 34.7% | 56.8% | 47.1% | 9.7% | L1 dependency and store traffic |
| final fill/finalize | 6.6% | 33.8% | 57.5% | 36.3% | 21.2% | DRAM/split-load signal; see caveat below |
| X DAO2 | ~4-5% | 34.1% | 56.8% | 38.5% | 18.3% | DRAM and split accesses |
| Y DAO2 | ~4-5% | 38.5% | 53.1% | 35.9% | 17.2% | L3/DRAM and split accesses |

This changes the TPCORE interpretation. The three largest PPM/remap kernels are
not individually dominated by back-end stalls on this CPU. They retire more
than half of available slots and have low bad speculation. Their size still
makes small improvements valuable, but VTune does not point to one obvious
memory-layout rewrite after the earlier scratch work. The strongest hardware
pathology is instead in the collection of full-grid setup, initialization,
cross-term, DAO2, and finalize passes. Together these smaller passes are a
large target even though none dominates alone.

The isolated worker changes its input repeatedly to keep the same machine code
active. That is representative for XTP, YTP, FZPPM and the other stable kernels,
but finalization is not idempotent and its checksum grows over many iterations.
Treat the isolated finalize counters as directional only. A candidate finalize
change must be judged in whole TPCORE with a fresh input each step.

Matched non-debug isolated timings for 20 stage repetitions gave:

| TPCORE stage | 1 thread s | 4 threads s | Speedup |
| --- | ---: | ---: | ---: |
| copy/workspace/cross setup | 1.102 | 1.127 | 0.98x |
| poles plus `dq` initialization | 0.800 | 0.411 | 1.95x |
| cross terms | 0.994 | 0.326 | 3.05x |
| X DAO2 | 0.729 | 0.515 | 1.41x |
| Y DAO2 | 0.905 | 0.627 | 1.44x |
| YTP horizontal mass flux | 3.504 | 1.293 | 2.71x |
| XTP horizontal mass flux | 3.341 | 1.282 | 2.61x |
| FZPPM vertical | 3.224 | 1.046 | 3.08x |
| final fill/finalize | 1.013 | 0.976 | 1.04x |

Four-thread VTune `threading` traces on X/Y DAO2, XTP, YTP, and FZPPM showed
only `0.2-1.4%` spin/overhead. During the whole trace they averaged about
`2.93-3.25` effective CPUs out of the four pinned P-cores. X/Y DAO2 still scale
poorly because the stage wrapper launches and synchronizes level-sized parallel
kernels 47 times; the trace shows waits rather than a TBB spin problem. The
normal TPCORE path has the same per-level call structure. A fused all-level DAO2
kernel is therefore a more plausible threading experiment than scheduler or
chunk-size tuning. It could also improve single-thread instruction/data flow by
removing dispatcher boundaries, but parity and direct wall time remain decisive.

### Unified operator opportunity matrix

| Area | Runtime importance | VTune diagnosis | Most defensible next experiment | Confidence |
| --- | --- | --- | --- | --- |
| TPCORE fixed passes | Highest operator plus roughly one-third of TPCORE across setup/init/cross/DAO2/finalize | memory/store pressure; serial setup/finalize; weak DAO2 scaling | fuse compatible full-grid passes or all-level DAO2 calls without changing arithmetic order within each cell | Medium-high |
| TPCORE XTP/YTP/FZPPM | About 61% of TPCORE | relatively healthy retiring; diffuse L1/front-end pressure | do not start another broad layout rewrite; use source/assembly inspection only for a narrowly supported traffic or code-size hypothesis | High |
| VDIFF writeback/solve | Second-tier operator cost but clearest memory pathology | 64.5% backend, 50.7% memory, 32.3% Store Bound | reduce/combine final mass-rescale and output stores; test caller-managed output only if ownership remains safe | High |
| Convection tracer update | Smallest of the three at high tracer count, but a clear arithmetic issue | 44.2% divider-active clockticks plus DRAM/split loads | hoist `1 / cmout` outside tracer loops and test reciprocal multiply as an explicitly parity-sensitive candidate | High diagnosis, medium acceptance |
| Multi-thread runtime | Useful but secondary to process-level tracer splitting | negligible TBB spin/overhead; sublinear useful-core occupancy | improve kernel granularity/serial fractions, not TBB scheduler knobs | High |

Suggested experiment order when effort is assigned:

1. Run the small convection reciprocal-hoist experiment first because it is
   cheap and VTune gives a precise hypothesis. Reject it immediately if strict
   parity or normal wall time is worse.
2. Prototype VDIFF writeback/store reduction. It has the strongest confirmed
   single-thread memory bottleneck and a known hot source region from Profila.
3. Prototype TPCORE all-level DAO2 fusion, then consider compatible fixed-pass
   fusion. TPCORE offers the largest end-to-end leverage, but each change is
   broader and requires the strongest parity coverage.
4. Leave XTP/YTP/FZPPM unchanged until a more specific source/assembly finding
   justifies another experiment; the stage-level VTune results do not currently
   identify a clean dominant defect in those kernels.

This ordering ranks experiment clarity and cost, not just operator runtime.
If only one larger workstream can be funded, TPCORE still has the greatest
end-to-end ceiling. If short probes can be run first, convection and VDIFF have
more sharply diagnosed hypotheses.

## 2026-07-13 VTune-guided experiments 1-3

The three experiments from the unified opportunity report were implemented and
measured with matched, P-core-pinned, non-debug runs. Two were retained and the
TPCORE experiment was rejected.

### Retained: convection reciprocal hoists

The diagnostics-light full-grid kernel now computes the reciprocals of
`denominator`, `denom_qc`, and `cmout` outside their tracer loops, replacing
per-tracer divisions with multiplication. This directly targets VTune's 44.2%
divider-active signal.

Matched single-thread timings:

| Tracers | Before s | Reciprocal s | Change |
| ---: | ---: | ---: | ---: |
| 24 | 0.03736 | 0.03556 | -4.8% |
| 96 | 0.12472 | 0.11352 | -9.0% |
| 192 | 0.23841 | 0.22218 | -6.8% |

The benchmark checksums were unchanged. The diagnostic and light kernels differ
at one of 564 values in the small active-cloud equivalence fixture, with maximum
absolute difference `5.42101086e-20`, maximum relative difference
`1.35452979e-16`, and maximum ULP difference one. The user explicitly accepted
this bounded one-ULP deviation for the performance win. The equivalence test now
enforces `maxulp=1`; the tracked real sampled GEOS-Chem convection snapshot still
uses the diagnostic/reference path and remains exact.

### Retained: VDIFF reusable production output

The diagnostics-light VDIFF path now has an opt-in `reuse_output` mode. The
transport driver and scaling benchmark enable it; direct API calls default to
fresh output arrays and preserve safe ownership. Reused tracer and humidity
outputs live in the existing shape/thread-keyed VDIFF workspace and are
overwritten by the next compatible reuse-enabled call.

Matched single-thread fresh-output versus reused-output timings:

| Tracers | Fresh s | Reused s | Change |
| ---: | ---: | ---: | ---: |
| 24 | 0.05921 | 0.04713 | -20.4% |
| 96 | 0.16848 | 0.12158 | -27.8% |
| 192 | 0.30807 | 0.21768 | -29.3% |

Outputs and checksums were unchanged. A regression test confirms that two
reuse-enabled calls return the same workspace-owned buffers while a default
call returns fresh buffers. This does not eliminate VDIFF's required final
rescale store, but it removes repeated allocation/page ownership churn around
that store and is a much larger win than expected from the source-level pass
alone.

### Rejected: fused TPCORE X/Y DAO2

A low-memory DAO2 prototype fused the X and Y interior traversal within each
level, preserving the original X-then-Y additions and avoiding the roughly
one-gigabyte intermediate cost of a literal all-level `qqu/qqv` design. Focused
TPCORE/transport tests were bitwise compatible.

Matched 96-tracer results:

| Mode | Separate DAO2 s | Fused DAO2 s | Change |
| --- | ---: | ---: | ---: |
| 1 thread | 0.79114 | 0.78227 | -1.1% |
| 4 P-cores | 0.35779 | 0.38173 | +6.7% |

The single-thread movement is too close to noise and the important four-core
case regressed, so the fused kernel was removed and the original separate DAO2
kernels remain in production. A literal all-level design was not pursued after
the stage analysis showed it would require two additional full tracer-sized
intermediate arrays plus private pole state to preserve dependencies.

### Updated prioritization

The two clearest VTune findings have now paid off. At 96 tracers the retained
operator-level improvements are approximately 9% for convection and 28% for
VDIFF. TPCORE remains the dominant end-to-end cost, but the first DAO2 fusion
hypothesis did not survive matched multi-thread measurement. Future TPCORE work
should require a new specific hypothesis rather than extending this fusion.

The next profiling decision should therefore be based on fresh end-to-end runs:

1. quantify how much VDIFF output reuse and convection reciprocals reduce the
   target multi-step residual workload, including real nonzero emissions;
2. re-run whole-TPCORE VTune after the retained surrounding-operator changes
   only if TPCORE's end-to-end share justifies another source investigation;
3. if TPCORE remains the focus, investigate the serial setup/finalize passes or
   code-generation details in XTP/YTP/FZPPM rather than more DAO2 fusion.

## 2026-07-13 residual reprofile and incremental TPCORE follow-up

A new P-core-pinned, single-thread residual-style profile was run for six hours
with real nonzero surface emissions and outputs disabled. It used 36 ten-minute
transport steps and 18 emissions evaluations. The retained VDIFF output reuse
and convection reciprocal changes were active.

| Tracers | Total s | TPCORE s | VDIFF s | Convection s | TPCORE setup s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 12.76 | 7.77 | 2.31 | 0.72 | 0.59 |
| 96 | 39.80 | 28.28 | 6.61 | 2.19 | 0.60 |

Compared with the prior steady 96-tracer residual reference (`43.15 s` total,
`8.44 s` VDIFF), total time improved by about 7.8% and VDIFF by about 21.7%.
The standalone VDIFF reuse gain therefore survives real nonzero-emissions
multi-step execution, though it is smaller than the isolated 27.8% result.
TPCORE now accounts for 60.9% of the 24-tracer run and 71.1% of the 96-tracer
run, confirming that further small TPCORE improvements have high leverage.

### Retained: remove duplicate TPCORE negative scan

`run_tpcore_one_step_with_setup` and its trace wrapper scanned the complete
final tracer cube with `tracer[tracer < 0] = 1e-26` after the lower-level
TPCORE finalize path had already performed the same clipping. Both the Numba
finalize kernel and NumPy finalize path cover the complete output, including
the copied polar rows, so the wrapper scan was redundant.

Matched direct timings:

| Tracers | Before s | Without duplicate scan s | Change |
| ---: | ---: | ---: | ---: |
| 24 | 0.23062 | 0.23147 | neutral |
| 96 | 0.79893 | 0.75925 | -5.0% |

Checksums were identical and focused TPCORE, transport-chain, and oracle tests
passed. In the six-hour 96-tracer residual profile, total time moved from
`39.80` to `38.14 s` (-4.2%) and attributed TPCORE time from `28.28` to
`26.61 s` (-5.9%). The cProfile wrapper self-time collapsed from about one
second across 36 calls to effectively zero, which confirms that the intended
full-cube Python mask was removed.

### Retained: opt-in TPCORE input consumption

A 96-tracer cube is about 451 MiB. Copying it into TPCORE's reusable `q`
workspace measured `18.7 ms` on the pinned P-core, or roughly 0.67 seconds over
36 steps. The long-run runner now calls `run_transport_one_step` with
`consume_input=True`; this is threaded to TPCORE as `reuse_input=True`, allowing
the writable contiguous incoming tracer cube to serve directly as TPCORE's
mutable `q` buffer.

The ownership relaxation is opt-in. Public one-step calls, trace paths, and
direct TPCORE calls retain copy-safe defaults. The runner immediately replaces
the consumed state with the returned state, and output recording consumes the
new state synchronously before the following step. A regression test verifies
that the default call leaves its input bitwise unchanged, the consume-enabled
call mutates it, and both produce bitwise-identical transport output.

The follow-up six-hour 96-tracer profile was:

| Variant | Total s | TPCORE s | VDIFF s | Convection s |
| --- | ---: | ---: | ---: | ---: |
| After duplicate-scan removal | 38.14 | 26.61 | 6.59 | 2.22 |
| Consume TPCORE input | 37.79 | 26.24 | 6.65 | 2.18 |

This is a smaller but repeatable target-scale win: about 0.9% total and 1.4%
within TPCORE. At very large ensemble/run counts it is worth retaining.

### Current next-target assessment

After these changes, the obvious full-cube wrapper traffic has been removed.
The remaining 96-tracer TPCORE time is almost entirely inside the compiled
kernel. Stage VTune still says XTP/YTP/FZPPM are relatively healthy, while the
smaller fixed passes have memory pressure but no single clean source rewrite.
Already rejected experiments include pole/init fusion, reusable FZPPM scratch,
limiter rewrites, precomputed-index variants, and DAO2 fusion.

Do not start another broad TPCORE rewrite from the existing profile alone. The
next productive options are narrower:

1. collect `memory-access` data for the pole/init and cross-term stage workers
   if exact load/store source attribution is needed;
2. inspect whether the `qckxyz_needs_fill` scan ever triggers in representative
   long residual windows before considering any fill-policy change;
3. revisit the strict-versus-fast-math native TPCORE prototype only as a
   separately parity-gated longer-term path;
4. retain sub-percent non-TPCORE improvements, such as caching repeated
   emissions path resolution, when they are independently measurable.

## 2026-07-13 fresh convection VTune profile after reciprocal hoists

Convection was collected again from the current tree rather than relying on the
older pre-hoist result. The matched synthetic case has 96 tracers, uses the
diagnostics-light full-grid Numba path, and is pinned to one P-core. Numba
profiling/debug information was enabled in a dedicated warm cache. The VTune
run used 120 timed calls after two warmups.

| Metric | Before reciprocal hoists | Current |
| --- | ---: | ---: |
| Retiring | 44.0% | 47.2% |
| Front-end bound | 7.4% | 7.9% |
| Bad speculation | 1.2% | 1.1% |
| Back-end bound | 47.4% | 43.8% |
| Memory bound | 27.4% | 30.2% |
| Core bound | 20.0% | 13.5% |
| Divider active | 44.2% of clockticks | 1.7% |
| DRAM bound | 15.0% of clockticks | 16.6% |
| Store bound | not previously decisive | 3.2% |

The retained reciprocal rewrite therefore removed the intended bottleneck:
divider activity fell by about 96%. Convection is now principally a
memory-traffic problem, with nearly half of the sampled clockticks seeing
memory-bandwidth activity. The 16% FPU vector-capacity figure is not evidence
that a C++ rewrite alone would help; the generated Numba loop is already
retiring efficiently and is increasingly constrained by data movement.

The timed benchmark was `0.11486 s` best and `0.11646 s` mean. VTune attributed
`11.58 s` across the 120 calls to the compiled parallel gufunc. The hottest
resolved source locations are the entraining tracer update at lines 683-697 and
its clamp/writeback pass at lines 712-717. In particular, the calculation of
`qc_next`, the several flux terms, filling `current_work`/`delq_work`, and then
rereading those scratch arrays dominate the resolved kernel samples.

The collection also attributed `2.30 s` to `memmove`, about `19.2 ms` per call.
This agrees with copying the 451 MiB 96-tracer input into convection's reusable
output buffer. Unlike pairwise checksum work in the benchmark harness, that
copy is inside the timed production-style call. An opt-in consume/in-place
handoff from VDIFF to convection could therefore have an approximate 16%
standalone convection ceiling and roughly a 1% six-hour end-to-end ceiling,
before implementation overhead and measurement noise.

The current four-P-core run was `0.05539 s` best and `0.06167 s` mean, versus
`0.11486 s` and `0.11646 s` on one P-core: 2.07x best-time and 1.89x mean-time
scaling. VTune reported only 0.9% spin/overhead. The poor scaling is therefore
not primarily a Numba scheduler problem; it is consistent with shared memory
bandwidth pressure. Do not spend another round on manual convection scheduling
without a new contrary measurement.

### Revised cross-operator opportunity list

Rank these by a mixture of evidence, plausible end-to-end value, and experiment
cost rather than by speculative maximum alone:

1. **Consume the VDIFF result in-place in convection.** This directly removes
   the measured 451 MiB copy and is the clearest new convection experiment.
   Preserve safe public defaults, as with TPCORE input consumption.
2. **Remove convection's `current_work`/`delq_work` round trip.** Test a fused
   arithmetic, clamp, and writeback tracer loop. The exact hot lines and the
   new memory-bound diagnosis both support this, but parity and vectorization
   must be checked because the split loop may help LLVM.
3. **Instrument TPCORE's `qckxyz_needs_fill` decision over representative
   residual windows.** If the slow fill is never needed, replace repeated
   checking with a justified policy or cheaper invariant. Measure first.
4. **Use VTune memory-access on TPCORE's fixed passes.** Pole/init, cross-term,
   DAO, and finalize are more memory-bound than XTP/YTP/FZPPM. Exact load/store
   attribution is needed before another fusion attempt.
5. **Attack VDIFF's final rescale/writeback traffic.** Output allocation is
   already reused, but the final full-cube mass-to-mixing-ratio pass remains.
   Test whether it can be combined with its producer or the downstream
   convection handoff without changing GEOS-Chem ordering.
6. **Tile VDIFF by columns and tracer blocks.** The aim is to retain its
   vertical coefficients and working state in cache while reducing full-array
   rereads; test single-thread first and retain only if four-core scaling is not
   harmed.
7. **Formalize whole-chain buffer ownership.** TPCORE already consumes its
   input and VDIFF reuses output; a two-buffer/ping-pong contract could eliminate
   remaining operator-boundary copies and allocations across all three stages.
8. **Cache convection forcing classification for an A3 block.** Active-column
   and cloud-base discovery uses forcing that is often repeated for three
   transport steps. This targets scalar setup rather than the dominant tracer
   loop and ranks below the two newly measured convection ideas.
9. **Investigate TPCORE code generation, specialization, and instruction
   footprint in XTP/YTP/FZPPM.** These stages retire reasonably well, so use
   assembly/code-size evidence and narrow variants rather than broad algebraic
   rewrites.
10. **Maintain a parity-gated native TPCORE prototype.** C++ may provide more
    control over alignment, restrict/alias contracts, vector reports, and
    streaming stores, but Numba is already producing bare native loops. Native
    code is most justified for TPCORE's large remaining share, not as a blanket
    rewrite of all three operators.

The recommended parallel experiment tracks are: (A) convection in-place
handoff followed by scratch-loop fusion; (B) TPCORE fill instrumentation plus a
targeted memory-access collection; (C) VDIFF final-writeback fusion and tiling;
and (D) a separately maintained native TPCORE feasibility prototype. A fifth
design-only track can specify whole-chain buffer ownership, then feed concrete
changes into A-C. Tracks A-C are production Numba work; D is a benchmark and
parity probe until it demonstrates a compelling advantage.

## 2026-07-13 parallel A-E experiment follow-up

Five isolated worktrees investigated the opportunity list. Production retained
only the convection handoff and two-buffer ownership work from Tracks A and E.
The TPCORE fill instrumentation, local VDIFF variants, and native C++ backend
remain out of the integrated production tree.

### Retained: in-place convection and two-buffer transport ownership

Track A added an explicitly destructive diagnostics-light convection mode.
The transport driver transfers ownership of VDIFF's output to convection, so
convection mutates it directly instead of copying a complete tracer cube. Safe
public calls remain non-destructive. The standalone 96-tracer result was:

| Mode | 1 P-core best s | 1 P-core mean s | 4 P-core best s |
| --- | ---: | ---: | ---: |
| reusable output with copy | 0.11960 | 0.12164 | 0.08076 |
| consume input | 0.10099 | 0.10136 | 0.06361 |
| change | -15.6% | -16.7% | -21.2% |

Checksums were exact and the existing diagnostics/light comparison remained
within the explicitly accepted one ULP. A separate attempt to fuse convection's
arithmetic, clamp, and writeback loops regressed `0.11774 -> 0.16342 s` (39%)
and was removed. The split pass evidently gives LLVM a substantially better
loop/vectorization shape despite its scratch traffic.

Track E then removed the extra persistent VDIFF cube introduced by ownership
alternation. Production already allows TPCORE to consume the incoming state.
After TPCORE writes its output, that old input allocation is dead; it is now
passed as VDIFF's explicit output and subsequently consumed by convection. The
steady chain therefore uses two tracer-sized cubes rather than three:

1. incoming TPCORE state, later recycled as VDIFF output and convection state;
2. reusable TPCORE output.

At 96 tracers one cube is 451.09 MiB. A matched three-hour RSS measurement fell
from `2,100,052` to `1,594,940 KiB`, a reduction of `505,112 KiB` or about
493.3 MiB including allocator/page effects. Isolated virtual allocation was
only about four microseconds; touched-memory capacity and the removed copy are
the material benefits.

Two matched, serialized, CPU8-pinned six-hour nonzero-emissions comparisons
were run around the integrated commits. Each includes 36 transport steps:

| Tracers | Pair | Before total s | After total s | Change | Before convection s | After convection s | Convection change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 1 | 12.310 | 12.096 | -1.74% | 0.715 | 0.520 | -27.3% |
| 24 | 2, reverse order | 12.253 | 12.025 | -1.86% | 0.727 | 0.512 | -29.6% |
| 96 | 1 | 36.999 | 36.769 | -0.62% | 2.289 | 1.484 | -35.2% |
| 96 | 2, reverse order | 37.553 | 37.335 | -0.58% | 2.175 | 1.475 | -32.2% |

The convection saving is stable and directly attributable. At 96 tracers the
roughly `0.7-0.8 s` stage saving is partly masked in total time by run-to-run
TPCORE/VDIFF and shared-memory-system variation. Report the measured whole-run
gain as about 0.6% for this six-hour 96-tracer workload, not the larger isolated
ceiling. The deterministic 493 MiB RSS reduction is independently valuable for
running many processes. The full integrated suite passed: 229 tests, two skips.

### Track B: TPCORE fill remains enabled

The `QCKXYZ` correction was needed zero times in 216 representative smooth
residual steps (24 tracers for six hours and one day; 96 tracers for six hours).
Skipping its full-cube negative scan has a roughly 3.1% standalone TPCORE
ceiling. Deliberately discontinuous step and point tracers did trigger the
correction, however, so disabling it globally would change GEOS-Chem semantics
for valid sharp tracer fields. The user rejected that risk. Neither an opt-out
nor the profiling instrumentation was integrated.

Fresh memory-access collections found pole/init 51.9% memory-bound, cross-term
45.6%, X DAO2 35.8%, Y DAO2 32.4%, and directional finalize 35.0%. An all-level
dq initialization experiment sometimes improved 96-tracer and four-core runs
by 2-5%, but its thresholded confirmation was neutral/noisy and regressed the
important 24-tracer single-thread case. It was removed.

### Track C: no retained local VDIFF rewrite

All tested VDIFF variants preserved exact checksums but failed performance
confirmation:

- Writing the vertical solve directly into canonical output doubled the
  96-tracer single-thread time (`~0.121 -> ~0.247 s`) because the recurrence
  became badly strided.
- Tracer blocks of 32 were neutral; blocks of 64 moved 96-tracer time only
  0.2-0.7% and were not reproducible enough to retain.
- Avoiding untouched upper-level `qmx` copies improved an initial 24-tracer
  probe but regressed the 96-tracer nonzero-emissions case by 4-8%.

The existing longitude-major scratch layout and tracer-contiguous whole-stage
passes are already strong. The retained cross-operator buffer ownership work is
more productive than another local VDIFF loop-order rewrite.

### Track D: full native C++ TPCORE is correct but slower

A complete optional single-thread C++17 TPCORE backend was implemented and
tuned in its isolated branch, including pole handling, cross/DAO2, X/Y PPM and
large-Courant branches, vertical PPM, fill, and finalize. It used GCC `-O3
-march=native -fno-fast-math -ffp-contract=off`, explicit 64-byte scratch
alignment, vectorization reports, and VTune. Full synthetic 24/96 results were
bitwise identical to one-thread Numba; a compact fixture differed at 16 of
5,264 values by at most one ULP.

| Tracers | Numba s | Tuned strict C++ s | C++ change |
| ---: | ---: | ---: | ---: |
| 24 | 0.2067 | 0.2428 | +17.5% |
| 96 | 0.7423 | 0.9682 | +30.4% |

Alignment and dead-store work improved the strict C++ baseline by about 4.7%.
PGO was neutral. Relaxed reassociation/reciprocal transformations were slower
and changed roughly 24 million cells, so they were rejected. C++ VTune reported
63.6% retiring, 16.1% front-end-bound, 17.6% backend-bound, 11.8% memory-bound,
and 23.5% FPU vector-capacity use. GCC reported widespread AVX2, but sampled
uops were still dominated by scalar FP (20.5% scalar versus 6.9% vector).

This result does not mean Python interpretation beat C++. Numba compiles the
TPCORE kernel ahead of execution into specialized native machine code, so both
paths are native-code compiler outputs. Numba/LLVM had the better code shape
for this dependency-heavy PPM implementation. C++ syntax alone cannot remove
the vertical recurrences, limiters, branches, alias questions, or instruction
footprint, and the strict parity flags deliberately prohibit many unsafe
reassociations. The C++ backend remains useful experimental work on its branch
but was not integrated because it adds maintenance without improving current
production performance.

## 2026-07-13 post-A/E VTune and native-compiler follow-up

### Matched VTune confirms the removed transport copy

A clean matched one-hour 96-tracer comparison was collected from the preserved
pre-A/E worktree and integrated post-A/E main tree. Both used the same dedicated
profiling cache, Numba debug/profiling build, CPU8 affinity, one thread, six
transport steps, and VTune `uarch-exploration` with bandwidth collection. An
earlier post-build capture warned of PMU contention and was discarded; neither
matched result reported that warning.

| Metric | Pre A/E | Post A/E | Change |
| --- | ---: | ---: | ---: |
| profiler total | 11.831 s | 11.522 s | -2.61% |
| TPCORE | 8.707 s | 8.655 s | -0.6% |
| VDIFF | 1.231 s | 1.137 s | -7.6% |
| convection | 0.410 s | 0.260 s | -36.6% |
| `memmove` self CPU | 0.218 s | 0.090 s | -58.9% |
| CPI | 0.368 | 0.359 | -2.4% |
| retiring | 46.9% | 48.0% | +1.1 points |
| back-end bound | 35.3% | 33.8% | -1.5 points |
| memory bound | 26.8% | 25.4% | -1.4 points |
| store bound | 10.6% | 9.9% | -0.7 points |

The lower `memmove` time directly confirms that the full tracer-cube convection
copy disappeared. The residual `memmove` belongs to other application/runtime
work. DRAM-bound percentage moved `7.8 -> 8.2%` and bandwidth-active percentage
`28.0 -> 28.5%`; these are shares of a shorter execution and not evidence of
more absolute work. The broader memory-bound and store-bound signals both fell.

The profiling/debug build and short window amplify the total movement. Retain
the non-debug matched six-hour result (about 0.6% total at 96 tracers) as the
production wall-time claim. After A/E, TPCORE is about 75% of this instrumented
run, VDIFF about 10%, and convection only 2.3%, further concentrating future
performance leverage in TPCORE.

### Corrected GCC, Clang, and Intel C++ compiler matrix

The native branch was revisited after Clang 18 and Intel oneAPI 2026.1 became
available. A serious benchmark-harness problem was found during the study: an
explicit C++ request could silently fall back to Numba when the extension's
runtime libraries were unavailable. This produced initially attractive but
false Intel results. All figures below come from corrected subprocesses that
asserted the extension-reported compiler and FP mode. The native branch now
makes explicit `cpp` selection fail with loader detail; only `auto` may fall
back, and the extension reports compiler, FP model, architecture, and LTO state.

The corrected pre-alias CPU8, one-thread minimum timings were:

| Backend | FP mode | 24 tracers s | 96 tracers s | Numerical result |
| --- | --- | ---: | ---: | --- |
| Numba/LLVM | strict | ~0.204 | ~0.750 | reference |
| GCC 13.3 C++ | strict | 0.2714 | 1.0665 | bitwise identical |
| Clang 18.1 C++ | strict | 0.2396 | 0.8281 | bitwise identical |
| Intel 2026.1 C++ | strict | 0.2829 | 1.1129 | bitwise identical |
| GCC C++ | fast | 0.2125 | 0.8449 | max 131/137 ULP |
| Clang C++ | fast | 0.2271 | 0.8058 | max 114/116 ULP |
| Intel C++ | fast | 0.2648 | 1.0416 | max 126/132 ULP |

Granular relaxations did not provide a useful frontier. GCC contraction reached
`1.0212 s` at 96 tracers with two ULP error; GCC reciprocal remained bitwise
but gave no speedup. Clang contraction reached `0.8021 s` with 29 ULP error;
Clang reciprocal reached `0.8259 s` with two ULP error. No mode within one ULP
beat strict Clang, and strict Clang remained about 10% slower than Numba.

The table above predates the final alias-contract pass. Adding correct
`restrict` contracts to the native TPCORE arguments enabled vectorization that
the compilers had previously declined. It introduced no new numerical
difference. The final locked comparison was:

| Backend | 24 tracers s | 96 tracers s | Change from Numba |
| --- | ---: | ---: | ---: |
| Numba/LLVM | 0.20383 | 0.74405 | reference |
| strict Clang C++ with alias contracts | 0.19233 | 0.69167 | -5.6% / -7.0% |
| strict Intel C++ with alias contracts | 0.19401 | 0.78341 | -4.8% / +5.3% |

Thus `restrict` was material; alignment, `ivdep`, LTO, and broad fast-math were
not useful on top of the final strict frontier. LLVM explains much of the
original GCC gap, while Numba's specialized parfor lowering explains why the
initial strict C++ ports were not automatically faster. The optimized Clang
prototype is now modestly faster, but it remains isolated on the Track D branch
because the supported Numba path is close, exact, and much cheaper to maintain.

## 2026-07-13 supported-Numba follow-up experiments

This pass deliberately excluded low-Courant specialization and fixed-grid or
fixed-tracer shapes. It also stayed within supported Numba/NumPy: no custom
compiler alias metadata, C++ production backend, fast-math, or changed transport
semantics.

### Retained: defer TPCORE finalization into VDIFF

TPCORE normally finishes by converting its full `(lev, lat, lon, tracer)` mass
array back to mixing ratio. VDIFF immediately traverses the same values. When
both production Numba operators are enabled, TPCORE now leaves the interior in
mass form and VDIFF performs the division and negative clamp on its existing
first reads. This removes one complete tracer-grid read/write pass. The two
TPCORE pole-copy rows are finalized by a small pole-only kernel before VDIFF so
their ordering and values remain unchanged.

Standalone TPCORE, standalone VDIFF, pure-Python, and mixed-backend calls retain
their previous public semantics. Synthetic 24- and 96-tracer comparisons were
bitwise exact. A real nonzero-emissions residual step was also bitwise exact
(`array_equal`, maximum absolute difference zero). The initial integrated suite
passed with 231 tests and two skips.

The destructive two-buffer ownership chain is enabled only when TPCORE, VDIFF,
and convection are all using Numba. An explicit `WOMBAT_NUMBA=0` production run
uses ordinary allocating reference calls even when the runner offers ownership
with `consume_input=True`; the caller's input remains unchanged and the result
does not alias it. This guard was added after the original buffer-recycling
commit exposed a pure-runner coverage gap. The added regression test brings the
suite to 232 passes and two skips.

Two serialized, CPU8-pinned, one-thread six-hour real comparisons were run with
opposite candidate/baseline and tracer-count order:

| Tracers | Pair | Before s | After s | Change |
| ---: | ---: | ---: | ---: | ---: |
| 24 | 1 | 12.092 | 11.890 | -1.67% |
| 24 | 2, reverse order | 12.127 | 11.944 | -1.51% |
| 96 | 1 | 37.166 | 36.500 | -1.79% |
| 96 | 2, reverse order | 37.134 | 35.553 | -4.26% |

The 24-tracer result is stable at about 1.5-1.7%. The 96-tracer TPCORE time is
more sensitive to memory-system state. Use 1.8% as the conservative repeatable
96-tracer claim; several reverse-order runs reached about 4.3%, but larger
movements are not a safe production estimate.

Repeated synthetic full-driver timings provide a shorter controlled check:

| Threads | Tracers | Before best/mean s | After best/mean s | Best / mean change |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 24 | 0.31283 / 0.31418 | 0.30452 / 0.30560 | -2.66% / -2.73% |
| 1 | 96 | 0.96291 / 0.96600 | 0.94270 / 0.95783 | -2.10% / -0.85% |
| 4 | 24 | 0.18901 / 0.19357 | 0.18177 / 0.18593 | -3.83% / -3.95% |
| 4 | 96 | 0.53300 / 0.54745 | 0.51947 / 0.53136 | -2.54% / -2.94% |

The optimization therefore helps both the preferred one-thread/process mode
and modest within-process threading.

### Post-fusion VTune

Two warmed one-hour 96-tracer `uarch-exploration` captures used the same
profiling/debug settings as the prior post-A/E result. Both were clean (MUX
reliability 0.994 and 0.998) and agreed on the hardware movement:

| Metric | Post A/E | Deferred-finalization range |
| --- | ---: | ---: |
| CPI | 0.359 | 0.360-0.361 |
| retiring | 48.0% | 47.9-48.1% |
| back-end bound | 33.8% | 32.1-32.4% |
| memory bound | 25.4% | 23.6-23.9% |
| DRAM bound | 8.2% | 6.4-6.5% |
| bandwidth active | 28.5% | 23.7-25.5% |
| store bound | 9.9% | 10.0-10.1% |

This confirms that removing the full-grid pass reduces memory pressure. The
debug/profiling build itself ran about 1% slower (`11.522 -> 11.620-11.665 s`)
because conversion branches and instructions move into VDIFF; it does not
reproduce the normal-build wall-time win. Use normal pinned timings for the
speed claim and VTune only for the microarchitectural explanation.

### Rejected scheduling and code-shape variants

- Naively parallelizing the outer level loop regressed 24 tracers by 64.7% due
  to nested parallel scheduling.
- A correct outer scheduler with serial inner stages and per-worker scratch was
  exact but about 2.9% slower.
- Tracer blocking through non-contiguous last-axis slices caused a very large
  Numba specialization whose compilation was repeatedly killed. Threading
  explicit bounds through roughly 70 loops was not justified without evidence
  of a runtime win.
- Manual groups of eight in FZPPM regressed full TPCORE by 1.9-5.1%.
- A divisible-by-eight tracer-loop specialization was neutral within 0.5%.
- Outlining the rare X fallback helped the isolated X stage but was neutral or
  worse in whole TPCORE. Removing a logically dead edge branch regressed the
  96-tracer four-core best by 6.3%.

The TPCORE profiler's `--path-census` mode explains these results. In
the measured workload, XTP uses common PPM for 86.2% of rows, the near-pole path
for 11.5%, and fallback for 2.3%. No fallback cell actually has `|cx| > 1`.
X flux is 97.7% positive, but Y and vertical flux signs are essentially 50/50,
and limiter work overwhelmingly belongs to common paths. There is no large,
rare branch with enough cost to justify duplicating the supported Numba kernel.

### Rejected cross-operator chunking and forcing caches

Latitude-chunked VDIFF-to-convection execution was exact but slower: best full
driver time moved `0.31064 -> 0.31158 s` at 24 tracers and
`0.92949 -> 0.93339 s` at 96, with the 96-tracer mean 1.4% slower. It was fully
removed.

An offline census checked all 35 adjacent transitions in a six-hour
nonzero-emissions run. A3 winds shared storage and equal values for 34/35
transitions, and hourly PBL fields did so for 30/35. However, pressure,
interpolated temperature, and humidity changed on 35/35. Every complete TPCORE
setup product and the complete VDIFF coefficient input set therefore had zero
legal reuse hits.

A narrower west-neighbour U cache was bitwise exact and improved its raw PJC
stage by 11.5-14.5%, but that projects to only 17-27 ms over 36 steps
(0.05-0.07% of the whole run) and retains another 18.8 MiB per A3 block. It was
rejected and reverted. Do not retry broad setup/coefficient caching unless the
forcing interpolation cadence or semantics change.

## ObsOperator profiling (2026-07-13)

Two-day real-input runs used the global 2x2.5 grid, 47 levels, 10-minute
transport, CPU8 pinning, one Numba thread, and synchronous compressed HISTORY
(three-hour species concentration plus daily restart). The enabled cases read
the parent checkout's daily gzip YAML inputs. Fresh matched results were:

| Tracers | Off s | Object sampler s | Prepared Numba s | Batched output s | ID-only restart s | YAML 1.2 s | Array state s | Array added s | Array added wall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 97.92 | 117.60 | 103.29 | 102.69 | 100.68 | 98.71 | 99.20 | 1.28 | 1.3% |
| 1 | 35.40 | 55.05 | 40.63 | 38.77 | 37.84 | 36.56 | 35.69 | 0.29 | 0.8% |

Before this pass, enabled wall time was 458.64 s for 24 tracers and 398.12 s
for one tracer. The main correction computes the full-grid wet-pressure and
box-height diagnostics once per timestep instead of once per observation.
The Rust-based `py-yaml12` loader and bulk restart arrays remove most of the
remaining startup and shutdown cost.

The production path now validates daily YAML directly into a struct-of-arrays
state: flattened fields, selections, accumulators, and a stable time-sorted
schedule. Restart loading rebuilds the same state directly from NetCDF, and
completion passes array slices to the batched science writer. It no longer
creates one Python entry/time/horizontal/vertical object per observation. One
serial nopython kernel per timestep evaluates all scheduled entries directly
from the tracer and forcing arrays. With `WOMBAT_OBSOPERATOR_NUMBA=0`, the same
kernel function runs directly in Python over those arrays; there is no separate
object parser, adapter, or sampler.

Preparing 5,144 entries took about 8 ms; rebuilding tables for 12,770 active
entries took about 22 ms. Across the first six hours, 36 kernel calls covering
7,653 scheduled observations took 15 ms, including selective dynamic pressure
and altitude diagnostics. The equivalent object sampler plus full-grid
diagnostics took about 1.9 s. Cython is not justified for preparation or
sampling at these costs.

Experiments with background input prefetch and threaded science writing found
the synchronous path faster or effectively neutral on the pinned CPU. Those
options were subsequently removed; daily input, batched science output, and
restart snapshots now use one straightforward synchronous path.

The science writer now stages at most 256 completed entries or 16,384 samples,
flattens each batch into contiguous arrays, and performs one slice assignment
per NetCDF variable. The fixed v1 chunks are 256 IDs, 64 field names, and
16,384 samples or lookup indices (roughly 64 KiB for each main chunk). An
instrumented six-hour 24-tracer run completed five compressed flushes in 5.9 ms
and spent 1.9 ms closing the writer. The daily science files shrank from
794 KiB and 1.2 MiB with library-selected chunks to 139 KiB and 172 KiB. Their
dimensions, attributes, registry/index order, and every stored value were
identical to the pre-batching products; only physical chunk layout changed.

For comparison, before batching the final two-day Numba run spent 1.77 s on
12,584 individual science-entry writes. The matched batched wall times above
show a 0.60 s improvement at 24 tracers and a 1.86 s improvement at one tracer;
the difference between internal write time and wall improvement includes
compression behavior and ordinary run-to-run variation. In the instrumented
six-hour batched run, gzip YAML load and resolution was 1.14 s of the manager's
1.22 s. Restart/daily identity now uses the required unique entry ID directly;
removing canonical YAML serialization for definition hashes reduced the final
PyYAML measurements to 100.68 s and 37.84 s. The 24-tracer movement is larger
than the isolated 0.72 s two-day hash cost, so it includes run variation.

The real daily inputs were also parsed with the Rust-based `py-yaml12` 0.1.0.
It produced equal Python structures and numerically identical resolved entries.
Parse time fell from 0.575 s and 1.038 s to about 0.058 s and 0.085 s. Direct
validation and array/schedule construction takes about 0.055 s for 5,144
entries and 0.117 s for 7,626 entries, down from about 0.221 s and 0.325 s for
object construction, registration, and flattening. Disabling repeated-operator
resolution caching was tested and was slower (0.208 s and 0.306 s for array
construction), so the small per-file caches remain.

A 36-step cProfile run attributed 0.216 s to YAML plus array construction,
0.007 s to all 36 prepared sampling calls, 0.007 s to five batched science
flushes, and 0.156 s to shutdown. Of shutdown, 0.127 s prepared the compressed
restart snapshot and 0.014 s wrote it. The direct two-day 24-tracer science
files matched the saved pre-array files exactly for IDs, field registry,
one-based lookup indices, and float32 samples. The end-to-end figures are
single matched runs and retain ordinary wall-time noise, especially in the
transport-dominated 24-tracer case.

After removing the compatibility classes entirely, a matched 36-step,
24-tracer run took 12.97 s with the Numba array kernel and 13.45 s with that
same function executed directly in Python. Their logical science-output
SHA-256 digests were identical. The Python reference therefore adds about
0.48 s to this full run while retaining exactly one sampling implementation
and one in-memory representation.

## 2026-07-16 parity-gated local optimization follow-up

This pass tested the low-risk local and compiler ideas left after the broader
transport study. Block-major storage, prepare/apply plans, noalias compiler
internals, QCK restructuring, fixed-grid kernels, level tiling, and aggregate
scheduling were explicitly deferred.

### Retained changes

- Convection now computes the subcloud dry-pressure sum, mass sum, base mass
  flux, and their reciprocals once per column rather than once per 300-second
  internal step. The 24/96-tracer one-thread synthetic best times moved from
  `0.03451/0.11377 s` to `0.03407/0.11366 s` before later compiler work.
- FZPPM no longer clears its complete `dc` workspace or writes the unused last
  `dpi` row. A NaN-poisoned workspace test proves that no removed value is
  consumed. In the reverse-order comparison, 24/96-tracer TPCORE moved from
  `0.23214/0.77375 s` to `0.22499/0.76797 s`.
- VDIFF no longer clears `kvh`/`kvm` before overwriting them from `kvf`, clears
  only the required `potbar` boundary, and removes redundant coefficient and
  humidity-solve initialization. NaN-poison tests cover the affected arrays.
  Zero-flux 24/96-tracer best times moved from `0.04762/0.12094 s` to
  `0.04742/0.11934 s`.
- The nonzero-emission VDIFF path computes `cgq` and the bottom source directly
  with the original parenthesization. This removes the per-thread `cgq` and
  `dqbot` workspaces. A new benchmark option supplies a uniform nonzero surface
  flux; its 24/96-tracer best times moved from `0.06295/0.18870 s` to
  `0.06038/0.16561 s`, improvements of 4.1% and 12.2%, with identical
  checksums.
- Granular `fastmath={"contract"}` is retained only for FZPPM and the
  diagnostics-light convection kernel. FZPPM remained bitwise equal on the
  compact oracle fixture and improved 96-tracer TPCORE by about 2.4% in the
  isolated probe. Convection remained within its existing one-ULP contract and
  moved from `0.03407/0.11366 s` to `0.03321/0.10394 s` at 24/96 tracers.

The cumulative one-P-core synthetic driver reached `0.29750 s` at 24 tracers
and `0.92281 s` at 96 tracers. Four P-cores reached `0.13925 s` and
`0.40448 s`. A real 2x2.5 nonzero-emissions 24-tracer six-hour run completed
all 36 transport steps in `11.69 s`, including `10.51 s` in transport. The
contract and strict variants produced bitwise-identical values for every one
of the 24 species in the six-hour HISTORY output. The full suite passed with
233 tests and 52 local-data skips.

### Rejected probes

- Activating convection's existing inactive-column `continue` was exact, but
  the all-active 24-tracer benchmark regressed by about 7%; the branch was
  removed. The new inactive-column precipitation-field regression test remains.
- `error_model="numpy"` was neutral or regressive on TPCORE, VDIFF, and
  convection and was removed.
- `fastmath={"contract"}` on XTP traded a 96-tracer improvement for a
  24-tracer regression; YTP regressed both primary counts. Both were removed.

All retained workspace and arithmetic changes passed the tracked low- and
large-Courant TPCORE, zero/nonzero/negative VDIFF, active/inactive convection,
NumPy fallback, and transport ownership tests. Generated benchmark, LLVM,
profile, and multistep artifacts remained untracked under `/tmp`.

The four canonical full-run windows were then rerun from copies of the parent
checkout's materialized directories while loading this worktree on
`PYTHONPATH`: two-day no-emissions and one-day 24-tracer residual emissions at
both 2x2.5 and 4x5. All 864 transport steps completed. Case-defined HISTORY,
ObsOperator, and available restart comparisons were bitwise identical to the
saved pre-optimization Wombat outputs at both resolutions. Consequently the
GEOS-Chem comparison metrics were also unchanged: species concentration and
CO2 restart maximum absolute differences remained at the established
float32-quantization scale (`2.91e-11` at 2x2.5 and up to `5.82e-11` in 4x5
HISTORY), with zero ObsOperator tolerance failures. Comparison artifacts live
under `/tmp/wombat-opt-validation-compare` and remain untracked.

## Experimental persistent TPCORE tracer blocks (2026-07-16)

An opt-in prototype tests contiguous padded tracer blocks without changing the
production dispatch or canonical public layout. It prepares `ua`, `va`, `jn`,
and `js` once, recompiles the existing TPCORE leaf implementations as serial
Numba kernels, and assigns independent blocks to a Python thread pool. Block
storage accepts every positive tracer count; unused lanes in the final block
are zero-filled and discarded on unpack. Exact tests cover widths 8 and 16,
one- and multi-block inputs, and partial final blocks.

Packing canonical state into blocks and unpacking it after every TPCORE call is
not viable. On global 2x2.5 with eight workers, the complete 96-tracer blocked
call took `0.500-0.554 s` for widths 8-24 versus `0.242 s` for the fused path.
The extra full-state memory copies erase the scheduling benefit.

The already-packed apply measurement is promising at larger tracer counts. It
includes block scheduling and the per-step shared plan cost, but excludes the
canonical pack and unpack that persistent state would avoid:

| Grid | Tracers | Fused s | Best block width | Block apply s | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2x2.5 | 24 | 0.0706 | 8 | 0.1115 | 0.63x |
| 2x2.5 | 48 | 0.1264 | 8 | 0.1275 | 0.99x |
| 2x2.5 | 64 | 0.1569 | 8 | 0.1317 | 1.19x |
| 2x2.5 | 96 | 0.2421 | 16 | 0.2043 | 1.18x |
| 2x2.5 | 128 | 0.3411 | 16 | 0.2435 | 1.40x |
| 2x2.5 | 192 | 0.5631 | 8 | 0.3543 | 1.59x |
| 4x5 | 24 | 0.0198 | 8 | 0.0270 | 0.73x |
| 4x5 | 96 | 0.0669 | 16 | 0.0487 | 1.37x |
| 4x5 | 192 | 0.1283 | 16 | 0.0791 | 1.62x |

Every measured result was bitwise equal to the fused Numba output, including
full-grid padded tails. Width 24 was slower than widths 8 and 16 at the primary
2x2.5 counts, so 24 has no special status. Width 8 reaches the eight-worker
frontier sooner and is the safer default for arbitrary counts; width 16 can be
better once enough blocks exist. The eventual policy should remain measured
dispatch rather than a user-visible tracer-count restriction.

The prototype was retained for the next architecture decision, but was not
used by production. The next useful test was
to let VDIFF and convection consume the same persistent block storage, avoiding
conversion across a complete timestep. Until that succeeds, small ensembles
and all canonical-state calls should continue using the existing fused path.

### Persistent zero-flux VDIFF handoff

The experiment now captures the exact `cch`, `zeh`, and `termh` coefficients
produced by the full-grid VDIFF kernel during a one-tracer preparation pass.
Independent serial tracer-block solves consume TPCORE's packed output directly
and write into its alternate buffer. The preparation also computes humidity
once. Padded-tail tracer output, humidity, and negative-count diagnostics are
bitwise equal to the production full-grid path.

The table charges both TPCORE and VDIFF per-step plan costs but excludes initial
canonical packing, representing state retained in block form across operators:

| Grid | Tracers | Fused chain s | Best width | Block chain s | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2x2.5 | 24 | 0.0720 | 8 | 0.1296 | 0.56x |
| 2x2.5 | 64 | 0.1675 | 8 | 0.1662 | 1.01x |
| 2x2.5 | 96 | 0.2642 | 16 | 0.2544 | 1.04x |
| 2x2.5 | 128 | 0.3892 | 8 | 0.2960 | 1.31x |
| 2x2.5 | 192 | 0.6446 | 8 | 0.4310 | 1.50x |
| 4x5 | 24 | 0.0263 | 8 | 0.0347 | 0.76x |
| 4x5 | 96 | 0.0627 | 8 | 0.0646 | 0.97x |
| 4x5 | 192 | 0.1329 | 16 | 0.1009 | 1.32x |

This retains the architecture on net even though the serial VDIFF block solve
uses some of TPCORE's isolated gain. It does not justify production dispatch
yet: nonzero surface flux and convection must share the layout, and the fused
canonical path remains materially faster for small ensembles. Reproduction is
available in `tools/benchmark_transport_blocks.py`.

The same block solve now supports nonzero surface flux without precombining
source coefficients: `cgs`, `kvh`, `potbar`, `rpdel`, `rrho`, and the bottom
source scale are captured from the full-grid preparation, and the existing
parenthesization is retained per tracer. The tracked emitting fixture is
bitwise equal through tracer output, humidity, clipping count, and a padded
tail block. With a uniform `1e-9 kg m-2 s-1` synthetic source on 2x2.5, the
plan-charged chain improved from `0.3238` to `0.2620 s` at 96 tracers (1.24x)
and from `0.7676` to `0.4604 s` at 192 tracers (1.67x). Surface-emission work
therefore strengthens rather than erases the persistent-layout case.

### Unified block state and convection

The two persistent tracer buffers now use single C-contiguous arrays with
shape `(block, lev, lat, lon, lane)`. Each slowest-dimension block slice remains
C-contiguous for the serial TPCORE and convection kernels, while allocation,
swapping, and future cross-block scheduling become simpler. TPCORE stayed
bitwise exact and showed no layout regression.

An explicit Numba `prange(block * lat * lon)` VDIFF variant was also tested.
Despite its larger task pool and smaller per-worker vertical scratch, it was
2-8% slower than the existing coarse block executor at 96 and 192 tracers. The
flattened kernel was removed; the unified 5-D state remains because it is
neutral-to-positive independently of that scheduling experiment.

Convection now consumes the VDIFF result in the same persistent buffer. It
recompiles the production full-grid arithmetic as a serial per-block kernel;
the tracked 24-tracer sampled fixture and padded synthetic cases are bitwise
equal. Complete plan-charged TPCORE -> VDIFF -> convection results were:

| Grid/source | Tracers | Fused s | Best width | Block s | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2x2.5 zero flux | 24 | 0.0826 | 8 | 0.1667 | 0.50x |
| 2x2.5 zero flux | 96 | 0.3230 | 16 | 0.2987 | 1.08x |
| 2x2.5 zero flux | 192 | 0.7590 | 8 | 0.4977 | 1.52x |
| 2x2.5 emitting | 96 | 0.3743 | 16 | 0.3110 | 1.20x |
| 2x2.5 emitting | 192 | 0.8391 | 8 | 0.5515 | 1.52x |
| 4x5 emitting | 96 | 0.0885 | 16 | 0.0841 | 1.05x |
| 4x5 emitting | 192 | 0.1739 | 16 | 0.1274 | 1.37x |

All complete-chain benchmark outputs were bitwise equal. The evidence supports
a hybrid policy: retain the fused canonical path for small ensembles and use
persistent blocks only above a measured grid- and worker-dependent threshold.

### Top-level Numba block pipeline

The persistent executor now also has an opt-in single-region Numba variant.
One outer `prange(block)` assigns each block to a Numba worker and runs the
serial TPCORE, VDIFF, and convection kernels consecutively. Scratch that is
only live within an operator is indexed by Numba worker rather than duplicated
for every block. Neither path was in production dispatch at this stage.

Direct executor comparisons below exclude the common plan cost. All outputs,
including a padded tail and nonzero surface flux, were bitwise equal:

| Grid/source | Tracers | Width | Python threads s | Numba pipeline s | Pipeline change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4x5, zero flux | 24 | 8 | 0.0446 | 0.0310 | 30% faster |
| 4x5, zero flux | 96 | 16 | 0.0774 | 0.0575 | 26% faster |
| 4x5, zero flux | 192 | 8 | 0.1316 | 0.1081 | 18% faster |
| 2x2.5, zero flux | 96 | 16 | 0.2815 | 0.2531 | 10% faster |
| 2x2.5, zero flux | 192 | 8 | 0.5279 | 0.4691 | 11% faster |
| 2x2.5, emitting | 96 | 16 | 0.3040 | 0.2869 | 6% faster |
| 2x2.5, emitting | 192 | 8 | 0.5477 | 0.5090 | 7% faster |

The improvement is not universal: the 2x2.5 emitting 96-tracer width-8 case
regressed by about 1%, and width 16 was nearly neutral at 192 tracers. The best
width also changes with the number of available blocks. Retain the Numba
pipeline as the lower-overhead executor candidate. The Python block scheduler
was subsequently removed, leaving the Numba pipeline as the sole concurrent
block executor. It gives every block an uninterrupted TPCORE-to-convection
path and avoids maintaining two scheduling implementations.

The benchmark must set `WOMBAT_NUMBA_THREADS` as well as Numba's runtime thread
count. Production operator wrappers reapply the environment-controlled count
on entry; without the environment setting, the nominally fused eight-worker
baseline silently runs with one thread. The benchmark now sets both controls.

With that correction, the complete zero-flux chain, all per-step block plan
costs charged, and initial canonical packing excluded, has the following
crossover on eight workers. Each block result remained bitwise equal:

| Grid | Tracers | Best width | Fused s | Numba blocks s | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4x5 | 1 | 1 | 0.0098 | 0.0155 | 0.63x |
| 4x5 | 2 | 1 | 0.0105 | 0.0166 | 0.63x |
| 4x5 | 4 | 1 | 0.0137 | 0.0175 | 0.78x |
| 4x5 | 8 | 1 | 0.0142 | 0.0192 | 0.74x |
| 4x5 | 16 | 2 | 0.0233 | 0.0227 | 1.02x |
| 4x5 | 32 | 4 | 0.0299 | 0.0271 | 1.10x |
| 4x5 | 64 | 8 | 0.0552 | 0.0401 | 1.38x |
| 4x5 | 128 | 16 | 0.1022 | 0.0664 | 1.54x |
| 4x5 | 192 | 8 | 0.1520 | 0.1093 | 1.39x |
| 2x2.5 | 1 | 1 | 0.0219 | 0.0640 | 0.34x |
| 2x2.5 | 2 | 1 | 0.0244 | 0.0654 | 0.37x |
| 2x2.5 | 4 | 1 | 0.0293 | 0.0694 | 0.42x |
| 2x2.5 | 8 | 1 | 0.0397 | 0.0756 | 0.53x |
| 2x2.5 | 16 | 2 | 0.0606 | 0.0905 | 0.67x |
| 2x2.5 | 32 | 4 | 0.1167 | 0.1156 | 1.01x |
| 2x2.5 | 48 | 8 | 0.1566 | 0.1601 | 0.98x |
| 2x2.5 | 64 | 8 | 0.2112 | 0.1712 | 1.23x |
| 2x2.5 | 96 | 16 | 0.3234 | 0.2695 | 1.20x |
| 2x2.5 | 128 | 16 | 0.4447 | 0.3170 | 1.40x |
| 2x2.5 | 192 | 8 | 0.7231 | 0.4903 | 1.47x |

The 2x2.5 results around 32-48 tracers are effectively the noisy crossover,
not a useful dispatch win. A conservative local policy is spatial below 64
tracers, then parallel across blocks, while 4x5 can cross around 16-32 tracers. Expressed
in terms of work per worker, the useful frontier is roughly eight tracers per
worker on 2x2.5 and four per worker on 4x5. The selected width should leave at
least one block per worker; extra blocks can help dynamic scheduling. This also
explains why a 400-tracer, 40-core socket is promising for width 8: it supplies
50 independent blocks, whereas a one-tracer run supplies only one and cannot
use block-level concurrency.

### Block-native simulation state

`TracerField` always owns physical storage with shape
`(time, block, lev, lat, lon, lane)`. Logical tracer names exclude padded tail
lanes. Individual blocks and individual tracers are zero-copy views; joining
multiple blocks into the old canonical array requires explicit
`to_canonical()` conversion. The former `BlockedTracerField` type and the
blocked-only transport driver were removed, so layout no longer changes with
execution policy.

`WOMBAT_TRANSPORT_EXECUTOR=spatial` is the default. Its default width is the
complete tracer count, and every all-Numba tracer count uses the shared
prepared one-block transport step. An explicit `WOMBAT_TRANSPORT_BLOCK_WIDTH`
makes the spatial executor process several block views sequentially with
within-operator parallelism. Both Numba forms and the native NumPy path were
within one ULP at widths 8 and 24 on the real 24-tracer fixture; the spatial
form remains bitwise equal to the standalone one-block adapter.

`WOMBAT_TRANSPORT_EXECUTOR=blocks` defaults to width 8. One transport workspace
owns persistent TPCORE state, block-shared intermediates, and worker-local
scratch. Its top-level `prange(block)` calls a serial one-block TPCORE -> VDIFF
-> convection step. The Python thread scheduler and its per-operator wrappers
were removed. Two consecutive production-driver steps on the real 24-tracer
fixture agree within one ULP through tracer state, with bitwise-equal humidity
and dry mass. Zero- and nonzero-emission fixtures retain the same bound.

HISTORY accumulation operates directly on contiguous block storage. Species
and restart writers map each logical tracer index to `(block, lane)`, and the
ObsOperator sampling kernel performs the same mapping for its prepared global
field indices. Consequently the runner does not repack the complete tracer
cube at output or observation boundaries.

### Unified one-block transport policies

The prepared Numba transport step now supports three policies over the same
state, plans, and workspace:

- `serial`: a serial block loop calling the serial one-block step;
- `spatial`: a serial block loop calling spatially parallel TPCORE, VDIFF, and
  convection variants;
- `blocks`: one outer `prange(block)` calling the serial one-block step.

VDIFF's tracer solve is one source function containing `prange(latitude)`,
compiled with and without `parallel=True`. Convection already follows the same
pattern through its Python kernel source. TPCORE shares all leaf arithmetic
while retaining thin serial and spatial orchestration variants.

Before tracer-free persistent VDIFF preparation, the local 2x2.5 results with
eight workers, including cold plan construction, showed that the
full-width spatial policy was about neutral at 8, 16, and 24 tracers and ranged
from roughly neutral to 14% faster at 32--96 tracers across repeated runs. It
was 8--11% slower at 1--4 tracers because fixed plan cost dominates, which
initially motivated a below-eight fallback. At 24 and 96 tracers it was
neutral-to-positive with one, two, four, and eight workers except for noisy
comparisons within roughly 2%.

Width-8 outer-block execution retained its high-thread crossover. With eight
workers it was 27% faster than the direct chain at 64 tracers in one repeat and
3% faster at 96 in a noisier repeat. At two workers it was 28% slower for 96
tracers and at four workers roughly neutral. Block execution therefore remains
explicit rather than automatically selected; spatial execution remains the
default.

### Tracer-free VDIFF preparation and steady-state plan cost

The first unified low-tracer comparison charged a single cold VDIFF plan after
allocation while warming the cached workspaces used by the direct chain. It
also prepared coefficients by running the full VDIFF kernel with a dummy
one-tracer field. Stage timings showed that the unified spatial apply itself
was faster at 1--8 tracers; the apparent regression came from this preparation
pass.

The retained preparation path now exits after coefficient and humidity work,
before all tracer loops. Coefficient, humidity, and dummy-input arrays live in
the persistent transport workspace. The benchmark warms and repeats plan
construction on those buffers, matching its treatment of the direct chain.
The original combined VDIFF path retains the same tracer arithmetic, and a
shared inlined humidity solve keeps the two paths in source-order parity.

On global 2x2.5 with eight workers, zero flux, full-width one-block spatial
execution, and all plan costs charged, the steady-state frontier was:

| Tracers | Direct chain s | Unified s | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.02117 | 0.02036 | 1.04x |
| 2 | 0.02436 | 0.02380 | 1.02x |
| 4 | 0.03034 | 0.02699 | 1.12x |
| 8 | 0.04041 | 0.03691 | 1.09x |
| 16 | 0.06435 | 0.05540 | 1.16x |
| 32 | 0.11143 | 0.10828 | 1.03x |
| 64 | 0.21379 | 0.19398 | 1.10x |
| 96 | 0.32226 | 0.27399 | 1.18x |

Uniform nonzero surface flux was within 1% at one tracer and 1.07x/1.15x
faster at four/eight tracers. Width-8 outer-block execution also retained its
gain: 1.23x at 64 tracers and 1.08x at 96 in this run. Every benchmark result
was bitwise equal. The persistent VDIFF plan adds roughly 38 MiB at 2x2.5,
independent of tracer count; this is the main cost to weigh before making the
unified executor unconditional at low tracer counts.

The executor is now unconditional whenever all three Numba operators are
enabled. The default spatial width is exactly the tracer count, so it performs
no padded lanes. A subsequent full-chain run including convection measured
0.996x at one tracer and 1.045x at four tracers, which is close parity at the
smallest case and a gain thereafter. The same run retained outer-block gains
of 1.26x at 64 tracers with width 8 and 1.15x at 96 tracers with width 16;
width 32 did not improve the block-parallel frontier.

The standalone Numba VDIFF and convection APIs now adapt to these same
one-block production kernels. Their diagnostic outputs are accumulated by the
production arithmetic rather than separate diagnostic kernels. The obsolete
latitude-by-latitude Numba VDIFF and grouped-column Numba convection
implementations were removed; the NumPy implementations remain as independent
semantic references.

After making block storage universal and removing the transitional field and
driver APIs, a warmed global 2x2.5 full-chain check with eight workers measured
1.05x at one tracer and 1.13x at four tracers for exact-width spatial
execution. Parallel block execution measured 1.29x at 64 tracers with width 8
and 1.19x at 96 tracers with width 16. Spatial results were bitwise equal to
the direct operator chain; block-parallel results remained within one ULP.
