# CUDA implementation working plan

Status: temporary design and task document. This is a best-current-guess plan,
not a stable public interface. Revise it when experiments contradict it.

## Objective

Add an optional CUDA execution path using CuPy for device ownership, transfers,
streams, NumPy-style device operations, and raw CUDA C++ kernels where they are
actually needed. Preserve the existing NumPy/Numba path as the CPU
implementation and numerical reference within Wombat.

The CUDA path must support:

- canonical fixed-width tracer blocks;
- `float64` and `float32` computation;
- explicit measurement of speed, transfer cost, and numerical drift;
- HISTORY accumulation and ObsOperator sampling without downloading the full
  tracer state every step;
- both supported GEOS grids and the existing one- and multi-tracer workflows.

GEOS-Chem remains the ultimate numerical reference.

## Design constraints

### Keep the integration asymmetric

Do not first turn the entire project into a general multi-backend framework.
The CPU path already works and should not be rewritten merely to make it look
like the CUDA path.

Add the CUDA path at a few coarse ownership seams:

1. tracer-state storage;
2. transport execution;
3. HISTORY accumulation;
4. ObsOperator sampling;
5. explicit host/device input and output boundaries.

The existing CPU functions should continue to be directly callable and
testable.

### Share semantics, not necessarily loops

Share:

- runner scheduling and configuration;
- field layouts and tracer metadata;
- grid and forcing definitions;
- operator plans, constants, and indexing metadata where practical;
- observation plans and output scheduling;
- test fixtures and parity rules.

Backend-specific code should be limited mainly to:

- CPU Numba loops;
- CUDA C++ kernels and their launch adapters;
- workspace allocation;
- reductions whose execution order is backend-specific;
- host/device staging.

Some arithmetic loops will therefore exist in CPU and CUDA forms. That is
preferable to hiding numerical behavior behind a large abstraction or code
generator. A semantic change should normally update one shared plan or
orchestration function plus the affected CPU and CUDA kernel, not two copies of
the whole operator.

### Avoid a global `xp` conversion

Do not mechanically replace every `numpy` import with a NumPy/CuPy namespace.
That tends to hide transfers, synchronizations, reduction-order differences,
and accidental CPU fallback.

Small, identical array expressions may be shared through one narrowly scoped
array helper. Complex and parity-sensitive operators should use an explicit
CPU or CUDA implementation.

Prefer ordinary CuPy operations for:

- allocation and explicit copying;
- elementwise arithmetic with `out=`;
- broadcasting and shape manipulation;
- slicing and indexing;
- simple transformations whose operation order is already clear.

Do not write a raw kernel merely because an operation runs on the GPU. Add a
custom kernel when an operation has irregular loops, needs a controlled
reduction, benefits materially from fusion, or cannot be expressed clearly and
efficiently with supported CuPy operations. Profiling should justify custom
kernels for otherwise simple array expressions.

Do not:

- wrap every NumPy function;
- introduce an inheritance hierarchy for every operator;
- pass `backend` or `xp` through every low-level function;
- silently convert a CuPy array to NumPy;
- silently run a CPU implementation in the middle of a CUDA timestep.

### Keep host/device boundaries explicit

The intended steady state is:

- tracer state remains on the GPU across timesteps;
- grid and static transport terms upload once;
- meteorology uploads by chunk or timestep;
- emissions upload only when refreshed;
- HISTORY sums stay on the GPU until their write boundary;
- ObsOperator plans upload when changed and completed samples download in
  compact batches;
- complete tracer state downloads only for requested output, restart, tracing,
  or validation.

Configuration, calendars, NetCDF access, input parsing, and file writing remain
host-side.

## Proposed source shape

Start with a small CUDA support package rather than a repository-wide backend
package:

```text
src/wombat_transport/
    cuda/
        __init__.py
        runtime.py
        modules.py
        transport_step.py
        sources/
            common.cuh
            tpcore.cu
            vdiff.cu
            convection.cu
    transport/
        tpcore/_cuda.py
        pbl/_cuda.py
        convection/_cuda.py
    obsoperator/
        sampling_cuda.py
```

Responsibilities:

- `cuda/runtime.py`: lazy CuPy import, device selection, explicit upload and
  download, streams, synchronization, and optional transfer accounting.
- `cuda/modules.py`: when the first custom kernel is justified, construct and
  cache `RawModule` instances by dtype and numerical mode.
- `cuda/transport_step.py`: compose the three resident operator executors
  without introducing a second simulation loop.
- CUDA source files: arithmetic kernels templated for `float` and `double`.
- subsystem `_cuda.py` files: launch geometry, argument validation, and mapping
  between shared plans and named CUDA kernels.

Do not import CuPy from the normal CPU operator modules. CPU-only installation
and import must continue to work without the CUDA extra.

This layout is provisional. Do not add more infrastructure until at least two
subsystems demonstrate that it is needed.

## Minimal execution seam

Prefer extending the existing coarse seams over inventing a universal backend
API:

- `TransportExecutor` is the natural transport ownership seam.
- `select_sampling_kernel()` is already a useful ObsOperator seam.
- HISTORY already has a small accumulation API.
- the runner already constructs these long-lived objects once.

The runner should select execution once, near initialization. Hot operator code
should not repeatedly inspect configuration or array types.

One small execution selection object or enum is acceptable. A family of
abstract base classes is not justified initially.

`TracerField` now accepts NumPy or CuPy storage without coercing device arrays
to the host. Coordinate metadata remains on the host, and storage-module
selection is confined to its construction and allocation helpers in
`fields.py`. Block views, tracer views, reblocking, and canonical reconstruction
preserve the original storage device. Keep this boundary small; if backend
checks start spreading beyond field storage and explicit I/O, stop and
reconsider the design.

## Numerical modes

Treat compute dtype and arithmetic policy as explicit CUDA choices.

Initial modes:

- `float64-strict`;
- `float32-strict`.

A later fast mode is allowed only after strict parity and performance are
understood.

For strict kernels:

- do not enable global fast math;
- make FMA contraction policy explicit;
- use precise division and square root settings where relevant;
- type constants deliberately for `float` and `double`;
- specify reduction order where it affects parity;
- record any intentional difference from the CPU or GEOS-Chem operation order.

Do not assume that disabling FMA is always the most GEOS-Chem-like choice.
Measure both contraction policies against the relevant reference build before
choosing the production default.

HISTORY elementwise accumulation can preserve per-cell timestep addition order.
Pole averaging, transport mass calculations, and ObsOperator horizontal or
vertical weighting are reductions and need explicit parity analysis rather
than an unexamined `cupy.sum`.

## Work plan

### Phase 0: freeze measurement cases

- [ ] Define the development GPU and record driver, CUDA, CuPy, and compute
  capability.
- [x] Select small operator fixtures for TPCORE, VDIFF/PBL, convection,
  HISTORY, and ObsOperator.
- [ ] Select end-to-end cases:
  - [ ] one tracer at 2x2.5 and 4x5 (2x2.5 selected; 4x5 remains);
  - [ ] 24 tracers at 2x2.5 and 4x5 (2x2.5 selected; 4x5 remains);
  - [x] at least one larger tracer-block throughput case.
- [x] Record initial CPU Numba timings for the selected 2x2.5 and throughput
  cases.
- [ ] Define reported numerical metrics: maximum absolute error, relative
  error, ULP distance, mass error, and error growth by timestep.
- [ ] Estimate device memory for state, forcing, static terms, HISTORY,
  ObsOperator plans, and worst-case workspaces.

Exit criterion: the CUDA work has fixed correctness and performance comparisons
rather than a moving collection of ad hoc smoke tests.

### Phase 1: build a disposable CUDA residency skeleton

- [x] Add a lazy CuPy availability check with a clear error when CUDA is
  selected but unavailable.
- [x] Add explicit `to_device` and `to_host` operations.
- [x] Add optional byte and synchronization counters to those operations.
- [x] Exercise resident float32 and float64 arrays with ordinary CuPy
  operations.
- [x] Accumulate several HISTORY timesteps on-device with `cupy.add` or
  equivalent broadcasting.
- [x] Verify that importing the CPU path and CUDA launch adapters does not
  import CuPy.
- [x] Add GPU-marked tests that skip clearly when no device is available.

Exit criterion: a test can upload one field, accumulate several HISTORY steps
in both dtypes, download the result, and report all transfers. No raw kernel or
general backend framework should exist at this point.

### Phase 2: establish array ownership

- [x] Decide whether a minimally storage-neutral `TracerField` is sufficient.
- [x] Reblock on the host before initial upload.
- [x] Keep tracer names, units, and coordinates as shared host metadata.
- [x] Allocate device storage and VDIFF workspaces through the CUDA runtime.
- [x] Make conversion to host explicit for output and comparison.
- [x] Add assertions preventing accidental NumPy coercion of resident device
  arrays.
- [x] Test padded final blocks and both one-block and multi-block layouts.

Decision: retain one storage-neutral `TracerField`. The required changes remain
local to `fields.py`; a separate CUDA state holder is not currently justified.
Reopen this decision if supporting device storage requires backend checks
throughout the existing field and I/O code.

Exit criterion: state can remain resident across repeated test-kernel launches
with zero hidden transfers.

### Phase 3: port simple shared services

- [x] Port HISTORY accumulation using ordinary CuPy elementwise operations.
- [ ] Port pressure and mass elementwise preparation that is demonstrably
  identical across array libraries, or write raw kernels when reduction order
  matters.
- [x] Port emissions application to resident state.
- [x] Add strict float32 and float64 parity tests for HISTORY.
- [x] Measure launch overhead before deciding whether any adjacent operations
  should be fused into a custom kernel.

Exit criterion: routine per-step bookkeeping no longer forces a state download.

### Phase 4: port column-oriented transport

- [x] Add the initial `RawModule` loader and templated CUDA source only when the
  first column kernel requires it.
- [x] Port VDIFF/PBL tracer application against a shared, CPU-generated
  meteorology plan.
- [ ] Decide whether VDIFF meteorology-plan preparation remains on the CPU or
  moves to CUDA after transfer and whole-chain profiling.
- [x] Port convection plans and kernels.
- [x] Reuse host-generated VDIFF plan metadata for the first operator
  checkpoint.
- [x] Reuse other host-generated plan metadata where it does not dominate transfer
  cost.
- [x] Allocate persistent VDIFF workspaces once per executor.
- [x] Test VDIFF columns containing zero flux, nonzero flux, countergradient
  rejection, negative clipping, and mass rescaling.
- [x] Test VDIFF tracer counts below and above block width, including padded
  final lanes and reusable output storage.
- [x] Compare strict float64 first, then characterize float32 drift for VDIFF.
- [x] Benchmark initial VDIFF block-width and tracer-launch mappings.

Exit criterion: VDIFF/PBL and convection run entirely on resident arrays and
pass their operator and handoff parity suites.

## Current checkpoint: resident CUDA VDIFF tracer application

The first complete operator slice is the tracer-dependent part of VDIFF/PBL.
The existing CPU/Numba code prepares one tracer-independent `VdiffPlan`; the
plan is uploaded explicitly, then `CudaVdiffExecutor` applies vertical
diffusion, countergradient surface-flux adjustment, the tridiagonal solve,
negative clipping, and mass rescaling to resident tracer blocks.

The CUDA kernel consumes the production physical layout directly:
`(block, lev, lat, lon, lane)`. One launch covers every active tracer across
all blocks. The active tracer count prevents padded final lanes from doing
column work, and padded output lanes are kept zero. The same executor also
accepts a contiguous one-block `(lev, lat, lon, lane)` operator fixture.

Hardware validation on 2026-07-26 used an NVIDIA GeForce RTX 4070 Ti SUPER
(compute capability 8.9), CuPy 13.6.0, CUDA runtime 12.9, and strict compilation
with FMA contraction disabled and precise division and square root enabled.
The full-grid case was `base_initial_vdiff_after_tpcore_v3` at
47x91x144. Timings are warm CUDA-event timings of the resident tracer
application only; they exclude plan preparation, compilation, and transfers.

| active tracers | block width | dtype | kernel ms | max abs vs CPU | max relative vs CPU | max ULP vs CPU | mass relative error |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 1 | float64 | 0.262 | 0 | 0 | 0 | 0 |
| 1 | 1 | float32 | 0.226 | 1.997e-10 | 5.081e-7 | 7 | 1.088e-8 |
| 24 | 8 | float64 | 6.211 | 0 | 0 | 0 | 0 |
| 24 | 8 | float32 | 6.015 | 3.703e-10 | 6.704e-7 | 9 | 1.088e-8 |

Strict float64 was bitwise identical to Wombat's CPU operator for the measured
tracer output. Against the GEOS-Chem output, maximum absolute error was
1.084e-19 for the source one-tracer case. Float32 retained the expected
negative-clipping count and finite output in all tracked branch fixtures.
There were zero host-to-device and zero device-to-host transfers during every
timed application.

This VDIFF kernel is dominated by its vertical state traffic on the development
GPU: float32 was about 14% faster for one tracer and only about 3% faster for
24 tracers. Poor double-precision arithmetic throughput is therefore not the
limiter for this operator. Block width still matters for sparse blocks: forcing
one tracer into width-eight physical storage measured about 1.86 ms because
the useful lane is strided through padded storage. CUDA initialization should
therefore select width one for a one-tracer run and use wider blocks for
multi-tracer throughput; this decision needs to be tested again once TPCORE
and convection share the storage.

The benchmark is reproducible with:

```bash
python tools/benchmark_cuda_vdiff.py VDIFF_INPUT.nc \
  --oracle-output VDIFF_OUTPUT.nc \
  --counts 1 24 --dtypes float64 float32 --block-width 8
```

Run the one-tracer case separately with `--counts 1 --block-width 1`.
`cpu_full_operator_wall_s` in the JSON output includes CPU meteorology-plan
preparation and is intentionally not presented as a like-for-like kernel
speedup. The next comparable end-to-end number should wait until plan staging,
convection, and TPCORE are resident.

### Phase 5: port TPCORE

- [x] Separate shared setup/plan semantics from CPU loop implementation where
  this can be done without rewriting the reference.
- [x] Port prepass and Courant calculations through the shared CPU-generated
  plan.
- [x] Port X advection.
- [x] Port Y advection and polar handling.
- [x] Port vertical PPM advection.
- [x] Port fill/fix and finalization logic.
- [x] Preserve the GEOS-Chem-directed operation sequence in strict mode.
- [x] Add targeted tests for every supported TPCORE branch.
- [x] Compare intermediate handoffs, not only final concentration.
- [ ] Profile memory traffic, occupancy, register pressure, and useful fusion.

Exit criterion: the full transport chain can execute one and many steps without
moving tracer state to the host.

## Current checkpoint: one resident transport step

`CudaTransportStepExecutor` now composes:

```text
TPCORE -> VDIFF/PBL -> convection
```

The production block layout remains
`(block, lev, lat, lon, lane)` throughout. TPCORE writes into executor-owned
storage, VDIFF writes into a second reusable buffer, and convection updates
that VDIFF result in place. No tracer array is staged through the host between
operators. Optional TPCORE and VDIFF handoff copies exist only for validation.

`tools/cuda_transport_step_harness.py` validates each handoff against a
full-grid transport-chain oracle and reports transfer counts, kernel-chain
timing, pointwise drift, and final dry-mass-weighted drift. It deliberately
prepares the shared CPU TPCORE and VDIFF plans before constructing any CuPy
`RawModule`. On the current environment, constructing a RawModule before the
first Numba/TBB VDIFF plan invocation can corrupt that first plan preparation;
subsequent preparations are unaffected. Until that library initialization
interaction is isolated upstream, runner integration must preserve this
ordering and validate prepared plans before upload.

Hardware validation on 2026-07-26 used the same RTX 4070 Ti SUPER and software
stack as the VDIFF checkpoint above. The case was
`base_initial_transport_chain_v3`, 47x91x144, with the source tracer expanded
linearly for the 24-tracer measurement. Timings are warm CUDA-event timings
for the complete three-operator step and include one device-to-device reset of
the input state per repetition. They exclude compilation, CPU plan
preparation, plan upload, and host transfers.

| active tracers | block width | dtype | step ms | max abs final error | max relative final error | mass relative error |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 1 | float64 | 64.976 | 6.234e-18 | 3.150e-14 | 0 |
| 1 | 1 | float32 | 36.971 | 1.891e-9 | 9.764e-6 | 1.228e-8 |
| 24 | 8 | float64 | 90.569 | 1.843e-17 | 3.544e-14 | 4.136e-16 |
| 24 | 8 | float32 | 63.157 | 6.752e-9 | 1.422e-5 | 1.228e-8 |
| 128 | 8 | float64 | 193.149 | 2.071e-17 | 3.853e-14 | 5.323e-16 |
| 128 | 8 | float32 | 145.695 | 8.404e-9 | 1.448e-5 | 1.228e-8 |

There were zero host-to-device and zero device-to-host transfers during every
timed step. On this consumer GPU, float32 was 1.76 times faster for one tracer
and 1.43 times faster for 24 tracers. Moving from one to 24 tracers increased
float64 step time by only 39% and float32 time by 71%, showing useful
tracer-level occupancy despite the correctness-first serial horizontal mapping
inside each TPCORE `(level, tracer)` worker.

At 128 tracers, throughput reached 408 million grid-cell-tracers/s in float64
and 541 million in float32. One full tracer-state buffer is about 601 MiB in
float64 or 301 MiB in float32 at this grid and tracer count. The current
correctness-first executor retains several full-state workspaces, so buffer
reuse and memory accounting remain important before scaling substantially
beyond this point.

The harness can also time Wombat's prepared-plan Numba chain with the exact
same inputs, width-eight block layout, and per-repetition state reset by adding
`--cpu-baseline`. The development CPU is an Intel Core i7-14700KF: logical CPUs
`0/1, 2/3, ..., 14/15` are the eight P-core SMT sibling pairs and CPUs `16-27`
are E-cores. The clean CPU comparison pins eight workers to
`0,2,4,6,8,10,12,14`, using one hardware thread from each P-core and no E-cores.

At 128 tracers, this pinned eight-P-core path measured 322 ms best and 328 ms
mean. One CPU worker measured 1826 ms. The corresponding CUDA float64 time was
193-194 ms: 9.43 times faster than one CPU worker and 1.66 times faster than the
eight pinned P-cores. CUDA float32 at 146 ms was 12.49 and 2.21 times faster
respectively, but the latter comparisons also change precision. These are
prepared operator-chain comparisons; they exclude plan preparation, forcing
I/O, diagnostics, and output on both sides.

The full-chain harness is reproducible with:

```bash
python tools/cuda_transport_step_harness.py ORACLE_DIRECTORY \
  --counts 1 24 128 --dtypes float64 float32 --warmup 2 --repeat 10

taskset --cpu-list 0,2,4,6,8,10,12,14 \
python tools/cuda_transport_step_harness.py ORACLE_DIRECTORY \
  --counts 128 --dtypes float64 float32 --cpu-baseline \
  --cpu-threads 8 --warmup 2 --repeat 10
```

The current composition hands finalized TPCORE concentration to VDIFF. The CPU
production path can defer that pressure division into VDIFF; adopting the same
handoff on CUDA is a later optimization and must be judged at the intermediate
handoff, not assumed equivalent. The current result validates one step. It
does not yet establish multi-step or long-horizon float32 drift.

### Phase 6: make diagnostics resident

- [x] Add a CUDA ObsOperator sampling implementation behind the existing
  sampling-kernel selection seam.
- [x] Keep plan parsing and daily plan merging on the host.
- [x] Upload compact observation-plan arrays only when the plan changes.
- [x] Retain observation accumulators on the device.
- [x] Download only completed observation batches for NetCDF writing.
- [x] Preserve verbose logging without forcing tracer-state downloads.
- [x] Keep HISTORY accumulators resident and download a collection only when it
  is written.
- [x] Exercise restart and partial-window close boundaries for HISTORY and
  ObsOperator state in the real-run smoke harness.

Exit criterion: enabling normal diagnostics adds no full-state per-step
downloads.

### Phase 7: integrate the runner

- [x] Add an experimental CPU/CUDA execution selection at runner
  initialization.
- [x] Construct the CUDA executor, state, diagnostics, and workspaces once.
- [x] Retain uploaded static transport terms and reuse device plan buffers.
- [x] Upload A1, A3, and I3 forcing by source chunk and select timestep views
  without assembling interpolated host forcing.
- [x] Interpolate pressure, humidity, and temperature on the GPU.
- [x] Prepare TPCORE, VDIFF, and convection plans into persistent device
  buffers.
- [x] Upload refreshed emissions only when their scheduled field changes.
- [x] Reject unsupported mixed CPU/CUDA execution rather than silently falling
  back.
- [x] Ensure snapshots do not implicitly copy device state.
- [x] Download only fields required at an actual writer or tracing boundary.
- [x] Log backend, dtype, numerical mode, device, and transfer totals in run
  metadata or the run log.

Keep the selection experimental until the full-chain parity cases pass. An
environment variable is sufficient for the early prototype; add a durable
`run.yml` contract only after the behavior stabilizes.

Exit criterion: a configured CUDA run completes through the ordinary runner
without duplicating the simulation loop.

## Current checkpoint: resident forcing and device preparation

The normal simulation loop now selects the experimental CUDA path once through
`WOMBAT_BACKEND=cuda`. NetCDF reads remain on the host, but the forcing
provider now exposes its natural A1, A3, and I3 source chunks and timestep
offsets. Each new chunk is copied once into reusable float64 device buffers.
Pressure and field interpolation, TPCORE preparation, VDIFF/PBL preparation,
and convection orientation then run on the GPU. Static grid terms, plan
storage, tracer blocks, and the current dry surface pressure remain resident.
The CUDA runner no longer depends on Numba.

TPCORE preparation is split into parallel mass-flux, divergence, pressure,
vertical-flux, and cross-term stages. Reductions that define strict numerical
semantics retain the CPU loop order. VDIFF preparation assigns one CUDA thread
per horizontal column and retains the original vertical operation order.
Float32 transport still prepares meteorology in float64 and casts finalized
plans, matching the previous CPU-plan-then-cast policy.

The same completed-step snapshot is passed to both diagnostics:

- HISTORY uses float64 device accumulators for both float32 and float64 tracer
  state, downloading one completed collection only at its write boundary.
- ObsOperator keeps its compact float64 accumulator and parsed-plan arrays on
  the device. Plan parsing and daily merge remain on the host, and only
  completed compact observations cross back for NetCDF writing.

The first realistic 2x2.5 CPU-prepared baseline used 24 tracers, active residual emissions,
18 ten-minute transport steps, one three-hour HISTORY window, one ObsOperator
entry expanded across all tracers, and final restart materialization. CPU plan
preparation and the CPU baseline were pinned to CPUs
`0,2,4,6,8,10,12,14`, one hardware thread from each P-core.

| execution | wall s | peak RSS MiB | final max abs vs CPU float64 | final max relative |
|---|---:|---:|---:|---:|
| CPU float64, 8 P-cores | 3.39 | 1085 | reference | reference |
| CUDA float64 | 5.32 | 1319 | 4.570e-17 | 1.142e-13 |
| CUDA float32 | 4.89 | 1259 | 2.972e-8 | 7.429e-5 |

The float32 endpoint RMSE was `1.318e-9` in mole fraction after 18 steps.
Float64 therefore provides an excellent implementation-parity baseline, while
float32 already exposes measurable but bounded short-run drift that must be
tracked over longer windows.

The resident-forcing implementation was then measured on the same realistic
18-step case. Timings include process startup, forcing reads, and final restart
materialization:

| execution | wall s | H2D bytes | final max abs vs CPU float64 |
|---|---:|---:|---:|
| resident CUDA float64 | 6.00 | 319,848,352 | 2.992e-17 |
| resident CUDA float32 | 5.45 | 249,348,104 | 2.972e-8 |

The float32 endpoint RMSE remains `1.318e-9`. The H2D totals include the initial
tracer state, static terms, the first forcing chunks, and nine scheduled
emissions refreshes. They no longer grow by roughly 140 MiB for every transport
step. A float64 run with a three-hour HISTORY collection, instantaneous restart
meteorology, and ObsOperator sampling took 6.56 seconds. Against the pinned
eight-P-core CPU reference, maximum differences were `4.743e-17` for HISTORY,
`2.954e-17` for restart tracers, and `1.158e-12 hPa` for `Met_DELPDRY`; the
other written meteorology fields matched exactly.

At 24 tracers this is still not an end-to-end GPU speedup. The immediate
performance targets are VDIFF preparation's column-local storage pressure,
overlap/double-buffering when forcing chunks change, and profiling repeated
A3-dependent work that may be reusable across the 18 steps in a meteorology
window.

### 128-tracer resident-run checkpoint

A larger ordinary-run fixture extends the 24 real residual tracers to 128
CO2-like tracers. The original 24 retain their configured emissions; the
additional 104 use the same `4e-4` background and zero emissions. This keeps
the meteorology and active flux workload realistic while measuring the complete
128-tracer transport and emissions-packing path.

The process was pinned to CPUs `0,2,4,6,8,10,12,14`. Each backend received a
one-step warm-up before the timed 18-step run. Timings include tracer
initialization, real forcing reads, nine emissions evaluations, transport, and
final host materialization. HISTORY, ObsOperator, and NetCDF output were
disabled.

| execution | wall s | speedup vs 8-P-core CPU | final state bytes | max abs vs CPU f64 | RMSE |
|---|---:|---:|---:|---:|---:|
| CPU float64, 8 P-cores | 8.701 | reference | 630,669,312 | reference | reference |
| resident CUDA float64 | 6.853 | 1.27x | 630,669,312 | 2.992e-17 | 1.735e-18 |
| resident CUDA float32 | 5.845 | 1.49x | 315,334,656 | 2.972e-8 | 1.318e-9 |

This establishes an end-to-end speedup at 128 tracers even on the development
GPU's relatively weak float64 hardware. The gain is smaller than the
prepared-operator result because initialization, forcing I/O, emissions
evaluation, device setup, and final materialization remain fixed or host-side.
The sequential comparison process peaked at about 3.71 GiB RSS while retaining
the CPU reference for both drift comparisons; that is not a per-backend memory
measurement.

#### 128-tracer event profile

The following subsection records the profile before the first focused
optimization pass. It is retained as the measurement baseline and is
superseded by the optimization notes below.

Nested CUDA events and host timers were added around the ordinary 18-step run.
This is the same synchronized timing mechanism used by
`cupyx.profiler.benchmark`, applied at the runner, operator, and raw-kernel
boundaries. Instrumentation changed wall time by less than one percent.

The following operator-level regions are non-overlapping except that the final
row includes everything:

| region | float64 ms | float32 ms | float64 / float32 |
|---|---:|---:|---:|
| TPCORE preparation | 181.6 | 184.6 | 0.98x |
| VDIFF/convection preparation | 473.6 | 476.0 | 0.99x |
| TPCORE application | 2,767.0 | 1,924.3 | 1.44x |
| VDIFF application | 684.7 | 643.4 | 1.06x |
| convection application | 92.8 | 33.1 | 2.81x |
| final device-to-host state copy | 87.6 | 44.1 | 1.99x |
| other host, setup, and transfer work | 2,580.5 | 2,578.8 | 1.00x |
| complete run | 6,867.8 | 5,884.3 | 1.17x |

Run-to-run variation accounts for the small difference from the benchmark
above. The profile makes the limited overall float32 gain unsurprising: about
2.58 seconds is dtype-independent, and another 0.66 seconds is preparation
that deliberately computes float64 plans for strict CPU parity before casting
the state-dependent values. Of the roughly 0.98 second saving, 0.84 seconds
comes from TPCORE alone.

Raw-kernel event timings explain why TPCORE itself is only 1.44x faster:

| kernel or preparation step | float64 ms | float32 ms | float64 / float32 |
|---|---:|---:|---:|
| TPCORE horizontal | 1,503.3 | 1,082.8 | 1.39x |
| TPCORE vertical | 622.4 | 244.0 | 2.55x |
| TPCORE finalize | 65.0 | 29.3 | 2.21x |
| VDIFF raw kernel | 168.0 | 126.6 | 1.33x |
| convection raw kernel | 92.8 | 33.0 | 2.81x |
| VDIFF start-pressure preparation | 453.5 | 454.1 | 1.00x |
| remaining VDIFF preparation kernel | 18.4 | 18.4 | 1.00x |

Operator times include fills, launches, allocations, and synchronization gaps
that are not inside the named raw kernels. These contribute about 0.57 seconds
to TPCORE and 0.52 seconds to VDIFF in both dtypes, so the VDIFF raw kernel's
1.33x gain becomes only 1.06x at the operator boundary.

The horizontal TPCORE mapping assigns one CUDA thread to a level/tracer pair
and performs long serial loops over the horizontal grid. Each thread also has
seven 144-element scratch arrays: about 8 KiB in float64 and 4 KiB in float32.
That structure is dominated by serial dependencies, indexing, control flow,
and likely local-memory traffic rather than floating-point peak throughput.
VDIFF similarly assigns a column/tracer to one thread and serializes its
47-level tridiagonal recurrence. Performance-counter confirmation is not
currently available because the installed driver denies non-admin access to
the NVIDIA hardware counters.

The clearest first profile-driven improvements are:

1. replace the single-thread, roughly 25 ms-per-step VDIFF start-pressure
   calculation with a parallel reduction;
2. attribute and remove the fixed fills or synchronization gaps around TPCORE
   and VDIFF;
3. remap or tile horizontal TPCORE so that more of each grid calculation is
   parallel and per-thread scratch storage is smaller;
4. retain `cupyx.profiler.time_range` annotations in a disposable profiling
   harness for an Nsight Systems timeline when report import is available.

The actual-GPU validation comprises 29 CUDA tests. It covers both state
dtypes, all three transport operators and their handoffs, padded tracer blocks,
mixed float32-state/float64-HISTORY accumulation, and ObsOperator point, box,
pressure, altitude, area, and normalized weighting modes. Resident preparation
is compared directly with the CPU strict plans in both transport dtypes,
including the unstable near-surface PBL branch.

### Phase 8: end-to-end parity and drift

- [ ] Run the canonical short one-tracer no-emissions cases at both grids.
- [ ] Run the canonical 24-tracer emissions cases at both grids.
- [ ] Compare concentration, restart, HISTORY, ObsOperator samples, mass, and
  intermediate transport handoffs.
- [ ] Record float64 strict differences separately from float32 strict
  differences.
- [ ] Measure drift over increasing step counts.
- [ ] Test restart chaining.
- [ ] Run longer comparisons before making monthly or long-horizon claims.
- [ ] Repeat float64 performance tests on hardware with strong double-precision
  throughput.

Exit criterion: supported numerical claims are written down with exact cases,
hardware, tolerances, and observed drift.

### Phase 9: performance work

- [x] Separate initialization/JIT time, input time, transfer time, and kernel
  time for the 128-tracer transport case.
- [ ] Profile diagnostics and output in a representative enabled run.
- [ ] Benchmark warm and cold kernel caches.
- [x] Tune block width and launch geometry for the 128-tracer development
  case; retain separate CPU and CUDA policy.
- [x] Test pinned asynchronous forcing transfers and reject them because they
  did not improve end-to-end wall time.
- [ ] Consider CUDA graphs only after launch overhead is shown to matter.
- [x] Fuse vertical TPCORE fill/finalization after handoff tests isolated the
  change.
- [x] Compare float32 and float64 throughput and memory use at 128 tracers.
- [x] Compare 128-tracer GPU throughput with an eight-P-core CPU baseline.

Do not use microkernel speed alone as the success criterion. The relevant
result is end-to-end tracer-steps per second with normal forcing and requested
diagnostics.

### Profiling support

`tools/profile_cuda_run.py` is the maintained ordinary-run profiler. It accepts
any run configuration plus dtype, step count, block width, device, warm-up,
and optional simulation-end controls. The end override permits longer profiles
without maintaining a separate benchmark configuration. Its JSON report
contains:

- synchronized complete-run wall and CUDA-event spans;
- nested runner, operator, and named raw-kernel CUDA-event regions;
- host orchestration timings;
- H2D and D2H call counts and bytes;
- CuPy device-pool total bytes as a peak-allocation proxy after clearing the
  warm-up pool;
- compiled raw-kernel attributes such as registers and local-memory bytes;
- per-transfer byte counts, host time, and the enclosing profiled operation;
- redirected output artifact names and sizes;
- device, compute capability, CUDA, CuPy, and CPU-affinity metadata.

Nested device regions overlap their parent regions and must not be summed.
Use `--nvtx` to add `cupyx.profiler.time_range` annotations from the profiling
process for an Nsight Systems capture without putting profiling branches into
the production runner. Use `--summary-only` for authoritative end-to-end wall
and device-span measurements without per-region event overhead, and use the
fully instrumented run for the stage breakdown.

```bash
taskset -c 0,2,4,6,8,10,12,14 \
  python tools/profile_cuda_run.py RUN.yml \
  --dtype float64 --steps 18 --block-width 32 \
  --output /tmp/wombat-cuda-profile.json

taskset -c 0,2,4,6,8,10,12,14 \
  python tools/profile_cuda_run.py RUN.yml \
  --dtype float64 --steps 18 --block-width 32 \
  --run-dir /tmp/wombat-cuda-run \
  --output /tmp/wombat-cuda-profile.json

nsys profile --trace=cuda,nvtx,osrt --output=/tmp/wombat-cuda \
  python tools/profile_cuda_run.py RUN.yml \
  --dtype float64 --steps 18 --block-width 32 --nvtx
```

`tools/cuda_transport_step_harness.py` remains the repeatable prepared-chain
microbenchmark and handoff-parity tool. `tools/benchmark_cuda_vdiff.py` provides
the corresponding isolated VDIFF measurement.

Both `nsys` and `ncu` launchers are installed in the development environment.
An NVTX capture succeeds and produces a `.qdstrm` file, but the local Nsight
Systems importer binary and its dependencies are missing, so it cannot produce
or inspect an `.nsys-rep` locally. The raw capture can be imported with a
complete compatible Nsight Systems installation elsewhere. The installed
NVIDIA driver also denies non-admin hardware-performance-counter access, so
Nsight Compute cannot yet provide dynamic occupancy, cache,
memory-throughput, divergence, or stall counters. Static compiled-kernel
attributes remain available through CuPy and are included in the JSON report.

Profiling writes never use the source run directory. Input paths retain their
source-relative meaning, while HISTORY, ObsOperator output/restart, run
metadata, and `output_dir` are redirected to an isolated root. Without
`--run-dir`, that root is temporary and its artifact inventory is captured in
the JSON report before cleanup.

The first maintained 128-tracer profiles reproduced the disposable profiler's
event timings:

| dtype | wall s | device-pool total bytes | horizontal ms | vertical/finalize ms |
|---|---:|---:|---:|---:|
| float64 | 4.363 | 2,869,593,088 | 1,295.8 | 664.5 |
| float32 | 3.256 | 1,636,246,528 | 855.4 | 151.0 |

The pool is cleared after warm-up; `total_bytes` is therefore a useful
allocation high-water proxy for the timed run. `used_bytes` is zero after the
runner releases its CUDA executor.

Compiled TPCORE attributes are:

| kernel | dtype | registers/thread | local bytes/thread | static shared bytes |
|---|---|---:|---:|---:|
| horizontal | float64 | 48 | 4,608 | 0 |
| horizontal | float32 | 40 | 2,304 | 0 |
| vertical/finalize | float64 | 72 | 1,504 | 0 |
| vertical/finalize | float32 | 48 | 752 | 0 |

The initial horizontal local allocation was exactly four longitude scratch
arrays. The scratch-lifetime pass below reduced that to three without changing
the transport result. This still defines the first alternate decomposition
cleanly: assign one warp to a level/tracer task, allocate its three 144-element
arrays once in shared memory, and distribute longitude-independent loops
across lanes. Keep longitude recurrences and pole reductions on one lane
initially so their strict operation order is unchanged. Four or eight warps
per block require 13.5 or 27 KiB of shared memory in float64, within the
reported 48 KiB dynamic-shared limit. Develop this as a side-by-side kernel
selected only by the profiling harness until every intermediate handoff
passes.

## Performance optimization notes

### First focused pass, 2026-07-26

The first pass targeted work already visible in the event and Python profiles.
It did not change transport scheduling, introduce a fast-math mode, move file
I/O to the GPU, or alter the strict float64-plan policy.

The benchmark is the same 128-tracer, 18-step, 2x2.5 resident-run fixture used
above. The process is pinned to one hardware thread on each of the eight
P-cores. The CPU uses the locally established 16-lane balanced block strategy;
CUDA uses 32-lane blocks. Each backend receives a one-step warm-up. Timings
include initialization, forcing reads, nine real emissions evaluations,
transport, and final host materialization, but exclude HISTORY, ObsOperator,
and NetCDF output.

| execution | before s | after s | speedup vs tuned CPU after | max abs vs CPU float64 | RMSE |
|---|---:|---:|---:|---:|---:|
| CPU float64, 8 P-cores, width 16 | 8.701* | 8.050 | reference | reference | reference |
| CUDA float64, width 32 | 6.853 | 4.377 | 1.84x | 2.992e-17 | 1.735e-18 |
| CUDA float32, width 32 | 5.845 | 3.239 | 2.49x | 2.972e-8 | 1.318e-9 |

`*` The earlier CPU measurement used width eight, so the before/after CPU
figures are not an implementation speedup. The width-16 result is the correct
local CPU comparison.

CUDA float64 wall time fell by 36% and CUDA float32 by 45%. Float64 remains an
implementation-parity baseline at this horizon. Float32 has the same measured
endpoint drift as before the optimization pass.

The final event breakdown is:

| region | float64 ms | float32 ms | float64 / float32 |
|---|---:|---:|---:|
| TPCORE preparation | 25.8 | 28.8 | 0.90x |
| VDIFF/convection preparation | 27.7 | 29.5 | 0.94x |
| TPCORE application | 1,965.5 | 1,039.6 | 1.89x |
| VDIFF application | 63.3 | 27.5 | 2.30x |
| convection application | 92.1 | 23.7 | 3.88x |
| final device-to-host state copy | 88.4 | 48.1 | 1.84x |
| complete run | 4,374.7 | 3,252.7 | 1.35x |

The end-to-end float32 ratio is lower than the transport ratio because tracer
initialization, forcing and emissions I/O, source-field selection, strict
float64 plan arithmetic, and other host work do not become twice as fast when
the resident tracer dtype changes.

#### Retained changes

| change | evidence and result |
|---|---|
| Replace `cupy.shares_memory` in the hot output check with a constant-time contiguous device-range check | Python profiling attributed 3.34 seconds across 38 warm and timed calls to CuPy's general overlap implementation, which sorts array addresses. The operator contract already requires contiguous arrays. |
| Reduce wet surface pressure once for the VDIFF start level | The old serial kernel repeatedly summed the same 13,104 surface cells for all 47 levels. Algebraically deriving the layer means from one ordered surface-pressure sum reduced 18-step time from about 454 ms to 7.6 ms. Strict start-level comparisons pass. |
| Stage TPCORE pressure-plan work into parallel term, copy, row, and correction kernels | The two numerically sensitive global sums retain their original left-to-right addition order. Independent cells and latitude rows run in parallel. TPCORE preparation fell from 181.6 ms to 25.8 ms. |
| Compute `jn` and `js` while producing horizontal Courant values | Initialized per-level bounds plus atomic min/max remove a second full scan without changing the selected indices. |
| Use 32-lane CUDA tracer blocks by default | Width 32 gave the best 128-tracer device time and maps one tracer block to one warp. Widths 16, 64, and 128 were close but slower. `WOMBAT_TRANSPORT_BLOCK_WIDTH` remains an override. |
| Pack and reblock by contiguous tracer slices | Replacing one assignment per tracer with one per block removed roughly half a second of 128-tracer initialization work. |
| Skip the TPCORE output clear for full blocks | Exactly 128 tracers in width-32 storage have no padded lanes, so every output value is overwritten by the kernels. Padded final blocks still clear before launch. |
| Reduce horizontal TPCORE per-thread longitude scratch from seven arrays to four | Reusing expired scratch values and accumulating pole sums directly reduced local-memory traffic. The 18-step horizontal kernel region fell to 1,278 ms in float64 and 858 ms in float32. |
| Reuse expired vertical TPCORE scratch | Removing one 47-element local array gave a small timing change but lowers per-thread local storage without changing the arithmetic sequence. |
| Elide VDIFF source preparation for zero-flux tracer-columns | Only 24 of the 128 tracers have active emissions. Exact zero-flux columns now bypass the `qmx` preparation and source-adjustment path. Float64 VDIFF application fell from about 115 ms after the earlier fixes to 63 ms; float32 is 28 ms. |

These optimizations are independent of the development GPU's weak float64
throughput. They remove unnecessary work or memory traffic and should remain
useful on a bandwidth-limited device and on a device with strong float64
capacity. The exact best block width and launch geometry must still be tuned
per architecture and tracer count.

CPU execution policy is deliberately separate. The development CPU performs
well with balanced 16-lane tracer blocks, and its spatial-to-block crossover
was measured near 128 tracers. Those are machine-specific CPU scheduling facts,
not shared storage semantics and not CUDA defaults.

#### Rejected or deferred experiments

| experiment | decision |
|---|---|
| Persistent pinned host staging plus a nonblocking copy stream | Rejected for now. Recorded refresh events fell from apparent multi-second host waits to about 7 ms, but complete wall time did not improve: the original host timer was largely waiting for already queued GPU work. The extra pinned allocation and stream/event coordination therefore bought no measured throughput. |
| Explicit coalesced horizontal scratch workspace | Rejected. It removed all reported compiler-local storage, but raised registers from 46 to 56 and horizontal time from 1,267.9 to 1,313.3 ms in float64. Float32 rose from 845.8 to 901.2 ms. CUDA local memory was already coalesced effectively; explicit indexing added overhead and persistent storage. |
| Horizontal launch sizes 16, 64, or 128 | Rejected for the current kernel and device. Thirty-two threads was fastest. |
| Vertical launch sizes 64 or 256 | Rejected for the current kernel and device. The differences were small; 128 remained marginally best. |
| Native-float32 plan preparation | Deferred as a distinct numerical experiment. Strict float32 currently computes plans in float64 and casts the finalized arrays. All TPCORE plus VDIFF/convection preparation is only 58.3 ms over the complete 3.239-second run, and casting itself is 4.7 ms. Even making plan preparation free would save at most 1.8%; actual savings would be smaller. A native-float32 mode is therefore primarily a drift and memory-policy question, not an obvious performance fix. It should be opt-in and compared over increasing step counts before adoption. |

### Scratch-lifetime pass, 2026-07-26

The horizontal kernel originally kept four 144-element private arrays live for
each level/tracer task. Two lifetime changes reduce that to three:

- horizontal flux differences are applied as each successive flux is computed,
  retaining only the first and previous scalar flux rather than a full flux
  array;
- the meridional slope array is reused for the PPM curvature after all slopes
  have been consumed.

The expressions used to construct each flux and the subtraction applied to
each destination cell retain their original ordering. The 18-step 128-tracer
run produced exactly the same measured endpoint drift as before:

| dtype | max abs vs CPU float64 | RMSE |
|---|---:|---:|
| float64 | 2.9923979961e-17 | 1.7349636207e-18 |
| float32 | 2.9715031337e-8 | 1.3176232434e-9 |

Compiled storage and repeated maintained-profile timings changed as follows:

| dtype | local bytes/thread before | after | horizontal before ms | after ms | change |
|---|---:|---:|---:|---:|---:|
| float64 | 4,608 | 3,456 | 1,294.8 | 1,267.2–1,267.9 | -2.1% |
| float32 | 2,304 | 1,728 | 855.4 | 845.0–845.8 | -1.1% |

The corresponding complete-run observations were 4.33–4.35 seconds for
float64 and 3.20–3.22 seconds for float32, compared with the maintained
4.36- and 3.26-second pre-change samples. Complete-run improvement is small
enough to remain somewhat sensitive to host and I/O noise; the repeated raw
kernel events are the more useful signal.

This result indicates that spilled private-array traffic contributes to the
horizontal cost but is not its dominant limit. In particular, halving scalar
width from float64 to float32 still yields only about 1.5x horizontal speed,
so serialized arithmetic, control flow, indexing, and common memory traffic
remain important.

Launch retuning after the storage reduction kept 32 threads as the best local
choice. At 128 tracers, 16 threads increased horizontal time to 1,538 ms
(float64) and 883 ms (float32); 64 threads measured 1,316 ms and 853 ms.
No launch-selection abstraction was added.

An attempted vertical scratch reuse was rejected by the operator parity test.
The pressure-difference array appears dead after reconstruction but is read
again by the later monotonic limiter, so it cannot alias the right-edge array.
The experiment was removed before benchmarking or commit.

### Full-run HISTORY and ObsOperator profile, 2026-07-26

The full-run profile uses the same 2x2.5, eighteen-step window at 24 and 128
tracers. Each case has one resident float64 HISTORY accumulator sampled every
step, one uncompressed float32 three-hour SpeciesConc output, and one
ObsOperator entry containing every tracer and point-sampled every step. The
entry and HISTORY interval both complete at the three-hour boundary. One
warm-up step is outside the measurement.

The core-transport column sums only the four raw application kernels.
Preparation, transfers, initialization, I/O, and nested parent regions are
excluded.

| tracers | dtype | wall s | core transport ms | HISTORY accumulation ms | Obs kernel ms | device pool bytes |
|---:|---|---:|---:|---:|---:|---:|
| 24 | float64 | 2.690 | 1,374.3 | 13.0 | 0.142 | 1,125,305,856 |
| 24 | float32 | 2.147 | 853.2 | 10.8 | 0.146 | 842,995,200 |
| 128 | float64 | 5.080 | 2,084.5 | 53.9 | 0.154 | 3,500,381,184 |
| 128 | float32 | 3.935 | 1,050.5 | 45.1 | 0.153 | 2,267,034,624 |

HISTORY accumulation is not a substantial execution cost. It is intentionally
float64 for both state precisions, however, so its memory cost is one complete
float64 tracer state. Relative to the transport-only 128-tracer profiles, the
device pool grows by approximately 631 MB in both modes.

At the 128-tracer boundary:

| operation | float64 state | float32 state |
|---|---:|---:|
| HISTORY D2H bytes | 630,669,312 | 630,669,312 |
| HISTORY D2H host time | 87.1 ms | 87.0 ms |
| complete HISTORY append | 574.9 ms | 572.1 ms |
| NetCDF field loop within append | 469.1 ms | 466.6 ms |
| float32 HISTORY file bytes | 316,025,480 | 316,025,480 |
| final-state/met D2H bytes | 635,596,416 | 320,261,760 |

The complete append contains the nested transfer and NetCDF regions and must
not be added to them. The output path is the largest non-transport boundary
cost. It currently transfers float64 sums and divides/casts on the CPU even
when the configured output is float32.

ObsOperator tracer scaling is negligible for this point workload. All eighteen
kernel launches total about 0.15 ms at both tracer counts because the selected
fields fit in one 128-thread block. Across completion and close, the
128-tracer run copies only 2,056 bytes back to the host. Its much larger host
`sync_to_host` duration is time waiting for previously queued transport, not
ObsOperator copy work. Completion immediately precedes HISTORY here, so it
moves the synchronization boundary rather than adding equivalent work. A
workload with entries completing more frequently than HISTORY output may
behave differently and should be profiled separately.

Eight resident emissions refreshes transfer 107,347,968 bytes and take only
about 5.3 ms (float64) or 2.9 ms (float32) as device events, while their
synchronous host calls wait 1.85 seconds or 0.92 seconds behind queued
transport. This is not a new optimization target: the earlier pinned,
nonblocking copy-stream experiment removed the apparent wait but did not
improve complete wall time. The full profile confirms that the blocked host
timer must not be treated as removable work.

The resulting HISTORY boundary experiment is recorded below. ObsOperator
kernel optimization is not justified by these profiles.

### GPU-finalized HISTORY averages, 2026-07-26

HISTORY continues to accumulate in float64. At a completed interval, CuPy now
divides by the sample count in float64, casts into one reusable buffer in the
configured NetCDF dtype, and transfers that buffer. The synchronous NetCDF
writer accepts the pre-averaged values directly, avoiding its former CPU
float64 divide buffer.

The CUDA operation is element-for-element identical to
`(host_float64_sum / count).astype(output_dtype)` for both float32 and float64
output. An end-to-end test also reads the resulting float32 NetCDF file and
checks the completed average exactly.

With the full-profile float32 output configuration:

| tracers | dtype | wall before s | wall after s | HISTORY append before ms | after ms |
|---:|---|---:|---:|---:|---:|
| 24 | float64 | 2.690 | 2.655 | 105.8 | 77.0 |
| 24 | float32 | 2.147 | 2.101 | 115.8 | 76.5 |
| 128 | float64 | 5.080 | 4.912 | 574.9 | 395.5 |
| 128 | float32 | 3.935 | 3.754 | 572.1 | 399.0 |

At 128 tracers, HISTORY D2H falls from 630,669,312 to 315,334,656
bytes. Complete wall time improves by 3.3% in float64 and 4.6% in float32;
the output boundary itself improves by about 31%.

The reusable float32 staging buffer adds 315,334,656 bytes to peak device-pool
allocation at 128 tracers. This is the direct space-for-time tradeoff; float64
accumulation and transport storage are unchanged. Chunked staging could reduce
that peak but would complicate the writer boundary, so it is deferred unless a
future device-memory target requires it.

The remaining NetCDF field loop is about 331–334 ms. The simple staging change
captures enough benefit to keep, but not enough to justify a background writer
yet. The next performance work returns to the side-by-side horizontal TPCORE
decomposition.

### Parallel horizontal initialization, 2026-07-26

The first TPCORE decomposition separates three phases that were formerly
serialized by one thread for each level/tracer task:

1. ordered north/south pole averaging remains one task per level/tracer;
2. initial mass plus `qqu`/`qqv` cross-term construction runs independently
   across grid-cell/tracer work;
3. the existing horizontal adjustment and flux recurrences run only after
   those two kernels complete.

The pole reductions retain their original loop and operation order. Every
cellwise expression is unchanged; only independent destinations execute in
parallel. Obsolete serial initialization code was removed rather than retained
as a second implementation.

At 128 tracers over eighteen steps:

| dtype | horizontal before ms | pole ms | parallel initialize ms | remaining recurrence ms | total after ms | improvement |
|---|---:|---:|---:|---:|---:|---:|
| float64 | 1,267.9 | 2.6 | 77.3 | 1,002.8 | 1,082.6 | 14.6% |
| float32 | 845.8 | 1.1 | 38.7 | 648.7 | 688.5 | 18.6% |

Transport-only wall time measured 4.157 seconds in float64 and approximately
3.04–3.08 seconds in float32, compared with 4.345 and 3.224 seconds before the
split. The established CPU comparison is unchanged:

| dtype | max abs vs CPU float64 | RMSE |
|---|---:|---:|
| float64 | 2.9923979961e-17 | 1.7349636207e-18 |
| float32 | 2.9715031337e-8 | 1.3176232434e-9 |

Combined with GPU-finalized HISTORY, the 128-tracer complete-output profile
falls from 5.080 to 4.732 seconds in float64 and from 3.935 to 3.602 seconds in
float32. These are 6.8% and 8.5% full-run improvements respectively.

The remaining horizontal recurrence kernel is now isolated more cleanly. It
still owns the three longitude scratch arrays and about 1,003 ms float64 or
649 ms float32 over the profile. The next substantial experiment is a
side-by-side cooperative-warp implementation of that recurrence region; there
is no longer value in moving more initialization work around it.

### Cooperative-warp horizontal transport, 2026-07-26

The horizontal recurrence now follows the natural independence boundaries
inside TPCORE rather than assigning the complete horizontal grid to one CUDA
thread:

1. one warp owns a level/tracer pair for the horizontal adjustments and zonal
   transport;
2. four longitude-sized arrays are shared by the warp, eliminating the
   recurrence kernel's compiler-local scratch;
3. zonal slopes, interface values, fluxes, and flux differences are distributed
   across the warp while each individual flux expression retains its original
   operation order;
4. meridional transport is a separate kernel with one worker per
   level/tracer/longitude column; and
5. a small final kernel performs the two polar flux sums in their original
   left-to-right longitude order and broadcasts the resulting polar mass.

The first experiment kept the complete meridional longitude loop on lane zero
of each warp. It was numerically exact but slow: the 24-tracer float64 zonal
kernel took 1,554 ms over eighteen steps. The idle lanes remained allocated
while one lane performed all 144 meridional columns. That version was removed.
Making columns independent reduced the same complete horizontal sequence to
about 217 ms, including pole averaging and initialization.

The established resident full-chain harness gives the cleanest before/after
comparison. Timings include a device-to-device state reset but no compilation,
plan preparation, or host transfers:

| tracers | dtype | serial recurrence step ms | cooperative/column step ms | speedup |
|---:|---|---:|---:|---:|
| 24 | float64 | 90.569 | 22.073 | 4.10x |
| 24 | float32 | 63.157 | 9.784 | 6.46x |
| 128 | float64 | 193.149 | 98.856 | 1.95x |
| 128 | float32 | 145.695 | 51.782 | 2.81x |
| 256 | float64 | n/a | 195.824 | n/a |
| 256 | float32 | n/a | 110.148 | n/a |
| 512 | float32 | n/a | 231.570 | n/a |

Throughput remains nearly flat from 128 through 256 tracers at about 0.8
billion grid-cell-tracers/s in float64. Float32 sustains 1.36--1.52 billion
through 512 tracers. This indicates that the warp mapping does not depend on
the CPU's 16-tracer blocking crossover and continues to scale when outer
tracer parallelism is abundant.

The strict tests pass for every packaged TPCORE case and the complete CUDA
transport-step handoffs. At 128 tracers, float64 maximum final drift remains
`2.071e-17` with mass-relative drift `5.323e-16`; float32 remains
`8.404e-9` with mass-relative drift `1.228e-8`. The 256- and 512-tracer
measurements remain finite and nonnegative with the same drift class. The
validation-heavy 512-tracer float64 harness exceeded the 16 GiB device after
allocating captured handoff copies; this is a harness-memory limit, not a
kernel failure, and no 1,024-tracer claim is made.

An ordinary 24-tracer run including forcing, HISTORY, ObsOperator, and NetCDF
output measured 1.718 seconds in float64 and 1.571 seconds in float32. Its
eighteen-step horizontal breakdown was 217 ms and 122 ms respectively. The
profiler also now resolves relative meteorology input paths before relocating
its output root; previously a temporary run could incorrectly resolve them
under `/external_data`.

### Horizontal and vertical follow-up experiments, 2026-07-26

Three parity-preserving follow-ups tested the remaining TPCORE kernels. Only
the zonal launch policy was retained.

#### Cooperative meridional columns

A warp-per-level/tracer/longitude kernel moved the three 91-element latitude
arrays from compiler-local storage into shared memory. Warp lanes computed
independent latitude values for the slope, interface, limiter, flux, mass, and
flux-difference passes, with warp synchronization between passes. Both four-
and eight-warps-per-block launches passed the strict TPCORE and complete-chain
handoff tests.

The shape was nevertheless slower. With eight warps/block, the 128-tracer
complete step rose from 98.9 to 114.7 ms in float64 and from 51.8 to 62.7 ms
in float32. Four warps/block was no better and reached 72.2 ms in float32.
The latitude-strided global accesses and synchronization cost more than the
private-array traffic saved. The cooperative implementation was removed.
Retuning the retained one-thread-per-column kernel at 64, 128, and 256
threads/block produced differences below one percent, so 128 remains the
simple default.

#### Staged vertical finalization

The vertical recurrence and negative-fill walk remained column-local while
the final pressure division and clamp moved to one worker per
level/column/tracer, followed by a small polar-row copy kernel. Results were
bitwise-equivalent to the fused path. The complete-step change was inconsistent
and below one percent: 128-tracer float64 moved from 98.9 to 98.8 ms and
float32 from 51.8 to 51.3 ms, while 256-tracer float64 regressed from 195.3
to 196.0 ms. Two extra launches and a synchronization boundary were not
justified, so the existing fused vertical kernel was restored.

#### Zonal launch geometry

The shared-memory zonal warp kernel has a real precision- and count-dependent
launch optimum:

- float64 retains four warps/block at every measured tracer count;
- float32 retains four warps/block below 32 tracers; and
- float32 uses eight warps/block from 32 tracers upward.

This is implemented as one small launch-policy function; both cases execute
the same kernel. Two warps/block was slower except for a smaller float32 gain
at high counts, and eight warps/block was slower for float64 and 24-tracer
float32.

| float32 tracers | four-warps step ms | eight-warps step ms | improvement |
|---:|---:|---:|---:|
| 32 | 11.792 | 11.133 | 5.6% |
| 64 | 24.629 | 23.131 | 6.1% |
| 96 | 38.411 | 33.981 | 11.5% |
| 128 | 51.788 | 45.913 | 11.3% |
| 256 | 110.201 | 93.240 | 15.4% |
| 512 | 231.570 | 188.294 | 18.7% |

The final retained complete-step profile, using seven repetitions after two
warm-ups, is:

| tracers | float64 ms | float32 ms | float32 grid-cell-tracers/s |
|---:|---:|---:|---:|
| 24 | 22.319 | 9.743 | 1.52 billion |
| 128 | 99.281 | 45.913 | 1.72 billion |
| 256 | 196.499 | 93.240 | 1.69 billion |

Maximum and mass-relative drift remain exactly in the previously reported
classes. At 128 tracers they are `2.071e-17` and `5.323e-16` for float64,
and `8.404e-9` and `1.228e-8` for float32.

The final ordinary 24-tracer, 18-step profile includes real forcing,
emissions, HISTORY, ObsOperator, and NetCDF output:

| region | float64 ms | float32 ms |
|---|---:|---:|
| complete wall | 1,703.1 | 1,579.9 |
| TPCORE application | 352.0 | 160.4 |
| VDIFF application | 30.4 | 11.7 |
| convection application | 19.8 | 5.8 |
| TPCORE plus VDIFF/convection preparation | 54.1 | 59.0 |
| HISTORY device work | 19.7 | 17.6 |
| initial forcing host boundary | 607.3 | 623.4 |
| emissions host boundary | 320.8 | 315.6 |

The host boundaries include synchronization inherited from earlier queued GPU
work and are not additive with device events. At 24 tracers, forcing, emissions,
and output boundaries dominate the complete run. For larger tracer counts,
the retained float32 launch policy materially reduces the scaling transport
term while those input costs remain mostly fixed.

### TPCORE/VDIFF lifetime and vertical pass, 2026-07-26

The next pass tested the CPU executor's deferred pressure-mass handoff directly.
It was numerically sound but not faster on this GPU. Removing TPCORE's
concentration division saved about 40 ms over 18 steps, while repeating that
division inside the more complex VDIFF kernel added about 48 ms. The complete
float64 run rose to 4.46 seconds. Fusing the remaining fill work into VDIFF was
slightly slower again at 4.49 seconds. The mass-aware VDIFF experiment was
therefore removed rather than retained as an unused alternate path.

The production path instead keeps the concentration handoff and makes two
orthogonal changes:

1. The vertical TPCORE kernel now performs QCK fill, pressure division,
   negative-floor handling, and polar-row copies immediately after its vertical
   flux work. This removes the separate full-state finalization launch without
   changing the TPCORE/VDIFF data contract.
2. VDIFF writes its result into the expired input-state buffer and borrows an
   expired TPCORE horizontal workspace for `qmx`. The dedicated VDIFF output
   and `qmx` allocations are no longer needed in the composed executor.

At 128 tracers, this removes two complete tracer-state allocations:
`1,261,338,624` bytes (1.17 GiB) in float64 or `630,669,312` bytes
(601 MiB) in float32. Standalone VDIFF still owns a normal private workspace;
borrowing is explicit only at the composed executor seam.

Vertical local storage was also reduced by replacing the 47-element `a6` array
with four endpoint scalars and exact on-demand recomputation for interior
levels. Recomputing the separate `dpi` array from tracer values was tested but
rejected: extra state reads increased the float64 vertical region by about
18 ms.

Final 18-step event timings are:

| region | float64 ms | float32 ms |
|---|---:|---:|
| TPCORE horizontal | 1,296.0 | 861.1 |
| TPCORE vertical, fill, and finalization | 663.7 | 150.9 |
| complete TPCORE application | 1,960.2 | 1,012.5 |
| VDIFF application | 64.1 | 27.8 |
| complete run | 4,363.6 | 3,237.9 |

The old separate vertical plus finalization regions totalled about 687.6 ms in
float64 and 181.6 ms in float32. The fused region is therefore 3.5% faster in
float64 and 17% faster in float32. End-to-end improvement is deliberately
modest because the pass primarily removes peak device memory rather than the
dominant horizontal computation.

The final pinned endpoint benchmark measured 4.358 seconds for CUDA float64 and
3.214 seconds for CUDA float32. Maximum drift against the tuned CPU float64
reference remains exactly `2.992e-17` and `2.972e-8`, respectively; RMSE remains
`1.735e-18` and `1.318e-9`.

The hot region that remains is horizontal TPCORE. Further substantial gains
will probably require a different parallel decomposition or tiling of its
serial longitude/latitude recurrences. That changes the kernel's shape and
reduction behavior, so it belongs in a separate parity-instrumented experiment
rather than this no-compromise cleanup pass.

### Two-day ordinary-run profile

On 2026-07-26 the canonical 24-tracer 2x2.5 emissions case was extended in
place to 48 model hours with the profiler's `--simulation-end` override. The
workload comprised 288 ten-minute transport steps, 144 emissions evaluations,
16 three-hour HISTORY averages, two daily restarts, and two days of
ObsOperator inputs. Processes were pinned to CPU cores
`0,2,4,6,8,10,12,14` on the RTX 4070 Ti SUPER.

Three summary-only repetitions, after a one-step JIT warm-up, measured:

| dtype | wall repetitions s | median s | tracer-steps/s |
|---|---|---:|---:|
| float64 | 14.718, 14.524, 14.542 | 14.542 | 475.3 |
| float32 | 12.900, 12.780, 12.832 | 12.832 | 538.6 |

The matching fully instrumented profiles give this non-additive breakdown.
CUDA parent regions in the first group are useful device-queue totals; host
regions in the second group can overlap CUDA execution and must not be summed
with them as a wall-time partition.

| CUDA parent region | float64 ms | float32 ms |
|---|---:|---:|
| TPCORE | 5,579 | 2,546 |
| VDIFF | 485 | 187 |
| convection | 314 | 93 |
| forcing selection and plan preparation | 944 | 999 |
| HISTORY accumulation and materialization | 392 | 323 |
| ObsOperator sampling and D2H completion | 128 | 128 |

| host boundary region | float64 ms | float32 ms |
|---|---:|---:|
| initial forcing load | 615 | 610 |
| forcing chunk selection/load, 289 calls | 2,434 | 2,424 |
| emissions evaluation, 144 calls | 888 | 878 |
| ObsOperator plan update/check, 288 calls | 987 | 995 |
| ObsOperator NetCDF flush, 281 calls | 296 | 290 |
| HISTORY NetCDF field writes, 16 calls | 988 | 992 |

Within float64 TPCORE, the zonal, meridional, and vertical kernels took 1,864,
1,299, and 2,063 ms. Their float32 times were 1,074, 712, and 589 ms. Float32
therefore accelerates the transport arithmetic substantially, especially the
vertical kernel, but total wall time is only 12% lower because CPU input,
emissions, diagnostic planning, and NetCDF costs are almost dtype-independent.

Float32 output was finite but much less compressible: the instrumented run
wrote 235 MB versus 51 MB for float64 transport, even though the configured
output dtype was float32 in both cases. Comparing the float32-transport outputs
with the float64-transport outputs after two days gave a maximum restart
difference of `4.57e-8` mole fraction (`0.0457 ppm`) and RMS `3.04e-9`.
This is a dtype comparison, not a new GEOS-Chem parity claim.

The first production-sized float32 ObsOperator attempt also exposed an unsafe
assumption: post-VDIFF humidity is a reversed-level resident CUDA view with a
negative stride. The kernel previously treated it as contiguous. It now
accepts the actual humidity and temperature level strides, retaining the
zero-copy handoff, and a reversed-device-view regression covers both dtypes.

### Same-thread host-step lookahead

The CUDA runner now exploits asynchronous kernel launch without introducing a
CPU worker pool. After enqueueing transport, ObsOperator sampling, and HISTORY
accumulation for the current step, the main thread loads the next forcing
selection and evaluates any scheduled emissions. Only then does it complete
current ObsOperator/HISTORY boundaries. A single `_CudaHostStep` retains the
future host selection and emissions result until the next iteration.

This scheduling remains confined to the CUDA branch. The CPU runner retains
its synchronous forcing, emissions, transport, and diagnostic order.
ObsOperator exposes CUDA-specific launch and completion methods while its
ordinary synchronous `sample()` contract remains available.

The first version deliberately does not add a transfer stream or duplicate
device forcing buffers. Current H2D work is below 1% of wall time. Uploading
the prefetched host selection at the next iteration provides a natural join
boundary when a chunk changes; otherwise default-stream ordering safely queues
the next preparation behind current diagnostics. This captures the material
CPU-I/O overlap without pinned-buffer lifetime or cross-stream dependencies.

Three uninstrumented two-day repetitions measured:

| dtype | before median s | lookahead repetitions s | lookahead median s | tracer-steps/s |
|---|---:|---|---:|---:|
| float64 | 14.542 | 13.915, 13.904, 13.973 | 13.915 | 496.7 |
| float32 | 12.832 | 12.252, 12.398, 12.349 | 12.349 | 559.7 |

This is a 4.5% float64 and 3.9% float32 throughput improvement. Matching fully
instrumented profiles show unchanged TPCORE device time and about 0.55-0.59
seconds less host time inherited by ObsOperator completion. Every variable in
the two HISTORY files, two restarts, and 280 ObsOperator files was bit-for-bit
identical to the pre-lookahead run for both dtypes.

### Event-bounded single-thread CUDA batches

The CUDA runner now extends same-thread lookahead across multiple transport
steps instead of completing ObsOperator work after every step. Transport,
resident preparation, ObsOperator sampling, and HISTORY accumulation remain
ordered on the default CUDA stream. The main thread prepares the following
host step while that queued work executes, and synchronizes only at the next
host-visible event.

The conservative first implementation ends a batch before:

- replacing a resident A1, A3, or I3 forcing chunk;
- uploading a newly evaluated emissions field;
- materializing a HISTORY average or instantaneous restart;
- changing the daily ObsOperator input plan;
- the final or `max_steps` boundary.

ObsOperator output filenames are not CUDA boundaries. Pending samples retain
their intended output paths, the accumulator is copied to the host once per
batch, and completed entry ranges are then written to their original files in
order. HISTORY exposes only a schedule query; its existing synchronous CPU
path and writer behavior are unchanged.

The canonical residual configuration evaluates emissions every two transport
steps, so that cadence is the limiting event in this profile. The maintained
profiler reports 144 `cuda.batch_synchronize` calls for 288 steps, down from
288 per-step ObsOperator completions. An hourly emissions configuration would
naturally permit six-step batches until a different event intervened.

One summary-only two-day measurement after a warm-up step gave:

| dtype | pre-batch median s | batched s | tracer-steps/s | throughput change |
|---|---:|---:|---:|---:|
| float64 | 13.915 | 13.178 | 524.5 | +5.6% |
| float32 | 12.349 | 11.104 | 622.5 | +11.2% |

These are single post-change measurements against the previous three-run
medians, so they describe the observed effect rather than a final statistical
benchmark. A fully instrumented float32 run measured 144 batch joins, 288
transport/ObsOperator/HISTORY launches, and the same 281 ObsOperator NetCDF
flushes. The joins occupied 1.995 seconds of host time; forcing selection,
emissions evaluation, and HISTORY field writes took 2.426, 0.942, and 0.661
seconds respectively and overlap queued device work where the event schedule
allows.

All 1,570 variables in 284 retained NetCDF files were bit-for-bit identical to
the pre-batch two-day output for both float32 and float64. No transfer stream,
pinned staging allocation, worker thread, queue, lock, or CPU-runner
abstraction was added.

### Effective-hour emissions batches

The 1,200-second emissions timestep is a logical evaluation and accounting
interval, not the source read frequency. In the residual case, hourly GPP
fields select the same source slice for three consecutive evaluations while
monthly fields and masks remain unchanged. Component arrays were already
cached, but each evaluation still rebuilt an identical `SurfaceEmissions`
array and made CUDA treat it as a new upload boundary.

`EmissionsOperator` now caches the assembled surface field by the effective
selection times of every configured field and scale. The 00:10, 00:30, and
00:50 evaluations therefore return the same field object; 01:10 creates a new
one. Mass accounting still records three 20-minute intervals per hour, and
VDIFF still applies the active flux on every ten-minute transport step. The
CUDA runner ends a batch only when the next scheduled evaluation actually
replaces that object.

This reduces the canonical two-day run from 144 two-step joins to 48 six-step
joins without read-ahead or an emissions-slot array. Three summary-only
repetitions measured:

| dtype | repetitions s | median s | tracer-steps/s | change from two-step batch |
|---|---|---:|---:|---:|
| float64 | 12.854, 12.880, 12.881 | 12.880 | 536.7 | +2.3% |
| float32 | 11.065, 11.010, 11.065 | 11.065 | 624.7 | +0.4% |

The float32 result explains the limited incremental benefit. Relative to the
two-step profile, assembled-emissions host time fell from 0.942 to 0.696
seconds, but time waiting at the fewer, longer batch joins rose from 1.995 to
2.186 seconds. Most of the saved host work moved behind the GPU critical path.
The larger float64 gain was stable across the three repetitions.

All 1,570 variables in the 284 retained two-day NetCDF files remained
bit-for-bit identical to the two-step implementation for both transport
dtypes.

### Deferred output pipeline

The first batch executor still wrote the current batch's ObsOperator,
HISTORY, and restart files before enqueueing the next batch. It overlapped
input preparation, but left the GPU idle during most NetCDF work.

CUDA diagnostic completion is now split into two phases. At boundary A,
device results are materialized into an owned host payload, HISTORY sums are
reset in stream order, and ObsOperator state advances without opening output
files. The main thread then enqueues all six steps of batch B. Immediately
before waiting for B, it writes A's detached payload:

```text
sync A -> detach A -> enqueue B -> write A -> sync B -> detach B
```

Only one detached host-output slot is retained. This is the current
single-thread pipeline; no writer thread, queue, lock, transfer stream, or CPU
runner change is involved. Manager `close()` methods drain a retained payload
so ordinary cleanup and partial-run behavior remain safe.

Three summary-only two-day repetitions measured:

| dtype | repetitions s | median s | tracer-steps/s | throughput change |
|---|---|---:|---:|---:|
| float64 | 11.912, 11.899, 12.006 | 11.912 | 580.2 | +8.1% |
| float32 | 10.181, 10.142, 10.218 | 10.181 | 678.9 | +8.7% |

The comparison is against the effective-hour six-step medians of 12.880 and
11.065 seconds. Instrumented float64 wall time fell from 12.898 to 11.943
seconds, while explicit batch-wait time fell from 5.739 to 4.714 seconds.
Its detached HISTORY/restart writes still occupied 2.206 seconds and
ObsOperator writes 0.386 seconds, but those regions now run while the
following batch is queued.

Instrumented float32 wall fell from 11.079 to 10.224 seconds and batch waiting
from 2.186 to 1.292 seconds. Float32 HISTORY/restart writing remains expensive
at 4.102 seconds because its output is poorly compressible; a six-step GPU
batch cannot hide all of it, but the available overlap is now captured.

All 1,570 variables in 284 retained NetCDF files were again bit-for-bit
identical to the non-pipelined six-step outputs for both dtypes.

### Phase 10: consolidate or retreat

- [ ] Remove spike-only APIs and unused abstractions.
- [ ] Confirm that ordinary CPU changes still touch only the normal CPU code in
  most cases.
- [ ] Document the small set of semantic changes that genuinely require both a
  CPU and CUDA kernel update.
- [ ] Keep CUDA source and launch adapters close to the corresponding operator.
- [ ] Add developer documentation for profiling and parity regeneration.
- [ ] Decide whether this file should become maintained documentation or be
  deleted in favor of smaller permanent documents.

If the implementation requires pervasive backend conditionals, frequent
full-state transfers, or duplicated runners, stop and redesign before merging.

## Testing structure

Use four levels:

1. kernel tests against small explicit arrays;
2. operator tests against the NumPy reference and CPU Numba path;
3. full transport-chain handoff tests;
4. named end-to-end GEOS-Chem parity runs.

Every CUDA test should state:

- compute dtype;
- numerical mode and compiler flags;
- device compute capability;
- whether comparison is against NumPy, CPU Numba, or GEOS-Chem;
- exact tolerance or ULP rule;
- whether the operation contains a reduction.

Do not weaken a shared tolerance merely to accommodate float32. Report float32
as a separate numerical mode with its own measured behavior.

## Questions to resolve experimentally

- Which tracer/block/thread mapping is best for each transport direction?
- Should block width remain common between CPU and CUDA execution?
- Which TPCORE operations benefit from fusion without obscuring parity?
- What reduction order best matches the GEOS-Chem reference?
- Is strict float64 useful on the development GPU despite its low throughput?
- Where does float32 first produce scientifically meaningful drift?
- Does forcing transfer overlap materially improve end-to-end performance?
- How much device memory is needed for realistic high-tracer runs?
- Is a storage-neutral `TracerField` genuinely small, or is a CUDA state holder
  cleaner?

These questions should be answered by fixtures and profiles, not by adding
abstraction in anticipation of every possible answer.
