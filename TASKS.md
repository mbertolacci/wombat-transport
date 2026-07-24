# Wombat Transport Tasks

This is the proposed order for the next transport performance experiments.
Keep each experiment on its own branch and retain it only after a matched,
warmed benchmark. Record completed experiments and detailed results in
`performance.md`.

GEOS-Chem parity remains the correctness gate. Experiments described as exact
should pass bitwise comparisons; explicitly low-ULP experiments require
multi-step ULP, mass-drift, pole, sharp-tracer, and emissions checks.

## Discrete experiments

### 1. Restore deferred TPCORE finalization

- Restore the consuming executor path in which TPCORE leaves interior values
  as pressure-weighted mass and VDIFF converts them on first read.
- Preserve the special pole-row finalization and fully finalized standalone
  TPCORE API.
- Confirm both spatial and block execution paths.

This is a likely regression rather than a new optimization. The earlier
implementation was bitwise exact and had an established whole-chain gain.

### 2. Make TPCORE cross-term scratch worker-local

- Replace per-block `qqu`, `qqv`, and Y-boundary scratch with worker-indexed
  storage where the outer block executor guarantees one active block per
  worker.
- Avoid constructing unused full standalone TPCORE workspaces for production
  blocks if they can be allocated lazily.
- Measure both wall time and peak/resident memory at several block counts.

Expected result: exact, with a modest speed or working-set improvement.

### 3. Extend `noalias` to spatial fixed passes

- Test executor-private pole, mass-initialization, cross-term, DAO2, and
  finalization kernels one at a time.
- Inspect LLVM to confirm the intended attributes and vector bodies.
- Retain only passes that improve whole-TPCORE or whole-chain timing.

Expected result: exact. Keep the experiment narrow because the compiler hook
uses private Numba APIs.

### 4. Add production-only control variants

- Test variants with negative counting disabled, diagnostics disabled,
  precipitation reconstruction disabled, `fill=True`, and common fixed
  `internal_steps`.
- Inspect generated code first: Numba may already have removed some branches.
- Keep diagnostic/public variants unchanged.

Expected result: exact and probably small, but cheap to evaluate.

### 5. Specialize the vertical level count

- Prototype a kernel specialized for the supported `nlev=47` recurrence bounds.
- Do not retry fixed tracer-width specialization as part of this work.
- Compare code size, compilation cost, and both grid resolutions.

Expected result: exact, with uncertain payoff.

### 6. Remove VDIFF's separate post-solve mass scan

- Accumulate positive `after_mass` values as levels become final during
  backward substitution.
- Leave negative scratch values untouched until the existing final ratio/output
  pass so the recurrence is unchanged.
- Test negative-count removal separately from scan fusion.

This changes the order of the `after_mass` reduction, so treat it as a low-ULP
experiment rather than an exact one.

### 7. Try DAO2-only contraction

- Apply `fastmath={"contract"}` only to DAO2 polynomial helpers.
- Do not broaden contraction to XTP/YTP or complete kernels.
- Retain only after multi-step parity and GEOS-Chem discrepancy checks.

Expected result: possibly low-ULP and low confidence, but bounded in scope.

### 8. Precompute only cheap block-amortized coefficients

Try independently, in this order:

1. VDIFF bottom solve coefficient.
2. Deferred-finalization conversion factor.
3. Normalized vertical Courant and compact upwind selection.

Stop if the additional plan traffic outweighs repeated divisions. Do not add
the larger FZPPM coefficient set until a smaller case demonstrates a clear
block-path win. Keep stored-division and reciprocal-multiply variants separate,
as only the latter intentionally changes rounding.

## Broader execution experiments

### 9. Allow execution policy to differ by operator

**Skipped after design review (2026-07-24).**

- Benchmark complete-operator combinations of spatial and block TPCORE,
  VDIFF, and convection.
- Charge the global barrier and loss of immediate operator cache locality.
- Calibrate by grid, workers, tracer count, and block width.

Do this after the discrete kernel experiments so the policy comparison uses the
best retained kernels.

The project is prioritizing very large tracer ensembles. In that limit,
independent tracer blocks provide enough work to occupy all workers across the
whole chain, while operator-specific policies would introduce global barriers
and lose immediate cross-operator cache locality. Mixed policies were therefore
judged mainly useful near the medium-tracer crossover and were not benchmarked
or retained.

### 10. Fuse driver-side forcing preparation

- Fill persistent top-order and flattened VDIFF/convection inputs directly.
- Remove repeated temporary reversals and contiguous repacking without caching
  forcing values between timesteps.
- Prioritize low-tracer and multi-process cases, where fixed costs matter most.

Expected result: exact, but broader than a single-kernel experiment.

### 11. Exploit zero and sparse surface flux

Treat sparsity as a staged programme after the more discrete experiments:

1. Extend the benchmark with zero, realistic sparse, and dense-uniform sources;
   report active block and block-column fractions.
2. Add persistent block-native flux storage and avoid the per-step allocation,
   zeroing, and global cube scan.
3. Replace the global `has_flux` with per-block flags and explicit zero/nonzero
   VDIFF entry points.
4. Add per-block, per-column flags so inactive columns skip `qmx` and `adjust`
   source preparation.
5. Only if emitted tracers are too scattered for block flags to help, cluster
   them into physical blocks using the existing logical-to-physical mapping.
6. Consider cost-aware block/process scheduling only after measured block
   imbalance justifies it.

Each stage should be useful independently. Preserve exactly the established
zero-flux arithmetic rather than relying on the nonzero path to add floating
zero.

### 12. Prototype matrix-free DAO2/FZPPM

- Prototype only in the serial block leaf.
- Form DAO2-corrected column values in worker scratch immediately before
  FZPPM, avoiding the corrected full-grid `q` write/read.
- Compare width 8 and 16 before considering any spatial implementation.

This is the last experiment: it has the largest semantic and performance risk.

## Not planned

- Dormant/dead tracer block skipping.
- Shared plans per NUMA domain.
- NUMA placement, first-touch, or huge-page tuning.
