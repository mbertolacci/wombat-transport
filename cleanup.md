# Cleanup Audit

This is the working backlog from the July 2026 repository-wide engineering
review. It deliberately excludes performance opportunities. GEOS-Chem parity
remains the governing constraint for any transport refactor.

## Current tranche

- **Remove stale oracle dashboard — resolved.** `plot_oracle.py` was both
  layout-incompatible and compared different operator chains. The utility was
  deleted rather than repaired.
- **ObsOperator output rotation — resolved.** Resolve and rotate the output
  independently of whether the templated input path changed.
- **Manager cleanup after failed runs — resolved.** Output and ObsOperator
  managers are now registered with the simulation resource stack as soon as
  they are constructed.
- **Unused `max_courant` API — resolved.** Removed from the public transport,
  trace, and window entry points and the internal block path.
- **Instantaneous restart alignment — resolved.** Reject configurations whose
  restart boundaries do not land on the transport-step grid.
- **Forcing interpolation boundary — resolved.** Reject a transport step that
  crosses the three-hour interpolation bracket.
- **Trace comparisons — accepted as-is.** These are targeted debugging tools,
  not comprehensive parity gates; they are intentionally adjusted when a new
  divergence is investigated.
- **Harness directory cleanup — deferred.** Some utilities recursively replace
  caller-selected output directories without a marker or explicit force flag.

## Production correctness and lifecycle backlog

- ObsOperator sampling is scheduled by `step_start` but consumes a post-step
  snapshot. Make that relationship explicit and assert it before changing any
  timing semantics.
- The ObsOperator writer mutates its staged-field registry before NetCDF writes
  complete, making retry behavior inconsistent after an I/O failure.
- Fresh ObsOperator YAML permits some non-finite vertical values and weights
  that its restart reader later rejects.
- ObsOperator character limits differ between ingestion (characters) and
  serialization (UTF-8 bytes).
- Output mode, field, frequency, and interval validation is spread between
  parsing, writer construction, and first use. Consolidate semantic validation
  before a run begins.
- Supported grids are recognized mainly by latitude/longitude counts. Validate
  canonical coordinate centers and ordering with a documented tolerance.
- Dry-pressure comparison infers vertical orientation from array rank. Require
  an explicit ordering or pressure representation.
- Conservative emissions remapping is duplicated between the cached operator
  and leading-dimension implementation, including polar-row behavior.

## Transport API and abstraction backlog

- TPCORE deferred finalization returns pressure-weighted tracer mass through a
  field named `tracer_conc_after`. Use a distinct result type or field name for
  the deferred handoff.
- TPCORE `reuse_input` mutates caller storage and `reuse_output` may return
  thread-local scratch that is overwritten by a later call. Make ownership and
  borrowed lifetimes explicit.
- TPCORE workspace rebinding currently relies on distributed alias invariants
  between the operator and executor. Provide one owned rebinding operation.
- Supplied TPCORE static terms are shape-checked but not tied to the grid and
  hybrid coordinates from which they were prepared.
- Pole averaging assumes zonally invariant cell area although the public API
  accepts arbitrary two-dimensional areas. Validate the supported invariant.
- Scalar reference helpers and several packing/predicate helpers appear to be
  stale compatibility surface. Confirm external use, then quarantine as oracle
  code or remove.
- The unified executor independently reconstructs pressure, vertical reversal,
  VDIFF inputs, and convection handoffs instead of sharing driver preparation.
  Consolidate only in small, parity-tested changes.
- VDIFF plan preparation uses dummy tracer and flux arrays to reach met-only
  planning logic. Give the kernel a dedicated met-plan interface.
- Executor ownership accepts any `shares_memory` relationship; validate exact
  shape, strides, offset, and expected layout.
- Several convection fields are built and validated but not consumed:
  `bxheight`, `pficu`, `pflcu`, `temp`, and `precccon`. Remove them or document
  them as intentionally reserved inputs.
- `TracerField` is frozen but owns writable NumPy arrays and mutable coordinate
  metadata. Model mutable buffer ownership directly; copying storage would be a
  parity-sensitive behavioral change.

## Proposed design-debt solutions

1. Give deferred TPCORE output a distinct result type and a field such as
   `tracer_mass_after`; reserve `tracer_conc_after` for finalized concentration.
2. Replace the `reuse_input` and `reuse_output` booleans with explicit in-place
   and workspace entry points; keep borrowed scratch results internal.
3. Add one TPCORE `bind_state_storage()` operation that validates and updates
   all workspace aliases atomically.
4. Store grid identity metadata in `TpcoreStaticTerms` and validate it when an
   operator is prepared, computing any fingerprint only once.
5. Validate the supported zonally uniform cell-area invariant before TPCORE
   pole averaging. **Selected for the current low-risk batch.**
6. Confirm use of the scalar TPCORE/reference compatibility helpers, then
   remove unreachable code or quarantine deliberately retained oracle code.
   **Selected for the current low-risk batch.**
7. Extract backend-neutral pressure, TPCORE, VDIFF, and convection preparation
   functions, proving their arrays exactly equal before switching the unified
   executor. This is a later, parity-sensitive change.
8. Add a met-only `prepare_vdiff_met_plan()` and retain the existing
   tracer-shaped entry point temporarily as a compatibility wrapper.
9. Replace the executor's broad `shares_memory` ownership test with exact
   constant-time storage validation: pointer, shape, strides, dtype, and
   layout. **Selected for the current low-risk batch.**
10. Remove inert convection arguments from the operator API while retaining
    any fields actually needed as trace/oracle diagnostic payload.
    **Selected for the current low-risk batch.**
11. Make `TracerField` buffer mutability explicit while keeping names, units,
    and coordinate metadata immutable; do not introduce implicit array copies.
12. Parse and semantically validate every output collection before constructing
    writers or loading simulation data. **Selected for the current low-risk
    batch.**
13. Validate grid coordinates and orientation against the canonical GEOS 2x2.5
    and 4x5 centers with a documented tight tolerance. **Selected for the
    current low-risk batch.**
14. Introduce one immutable conservative-remapping weights object and one
    leading-dimension application routine, preserving existing polar arithmetic
    exactly. This is a later, parity-sensitive change.
15. Normalize ObsOperator YAML and restart data into one intermediate form and
    pass both through a shared validator and array-state builder.

## Low-risk design batch outcome

- **Zonal-area invariant — resolved.** TPCORE static-term construction rejects
  cell areas that vary with longitude within a latitude row. The check uses an
  allocation-light exact comparison and production runs perform it only during
  static setup.
- **Reference compatibility surface — resolved.** Scalar oracle helpers remain
  in the private `_reference` module for branch tests, but are no longer
  re-exported through the TPCORE package. Packing helpers are explicitly marked
  as private harness/benchmark utilities.
- **Executor ownership — resolved.** Persistent executor state now requires an
  exact pointer, shape, strides, dtype, contiguity, and writability match rather
  than any overlapping memory region.
- **Inert convection arguments — resolved.** The operator no longer accepts or
  validates fields it does not consume. Those fields remain in trace input
  state where harnesses use them as GEOS-Chem handoff diagnostics.
- **Output semantics — resolved.** Collection path, interval, mode, and field
  combinations are validated before any writer or thread is constructed.
- **Canonical grid identity — resolved.** Supported grids must match canonical
  GEOS 2x2.5 or 4x5 coordinates and orientation within `1e-10` degrees. The
  tracked synthetic I/O fixture was corrected to use standard half-polar rows.

Benchmark: the 2x2.5, 24-tracer, single-thread compiled driver retained the
same checksum. The initial seven-repeat baseline was 0.298517 seconds/step;
after replacing an allocation-heavy validation implementation, the 15-repeat
post-change best was 0.299569 seconds/step (0.35% slower than that baseline).
A contemporaneous old-source control varied by about one percent across
unchanged numerical kernels, so this difference is below the observed host
noise. Production runs also prebuild the newly validated static terms once.

## Harness and tooling backlog

- `--skip-oracle-run` accepts cached fixed-name files without checking config,
  version, checksum, or executable provenance.
- Several comparison helpers write Python results beside oracle fixtures by
  default, coupling comparison to cache mutation.
- Large fixture generation accepts a TPCORE executable but hard-codes VDIFF and
  convection binaries, allowing mixed toolchain provenance.
- Harness README commands reference older fixture versions than the code and
  tracked manifests.
- Fixture downloads write directly to final paths. Download to a unique sibling
  temporary file, verify it, then replace atomically.
- Benchmark readiness loops can block indefinitely on `stdout.readline`, and
  profiler cleanup does not consistently reap all child processes.
- Benchmark `--resume` trusts an existing directory without confirming that its
  manifest, configuration, CPU metadata, and experiment dimensions match.
- Harness source generators use inconsistent anchor replacement rules: some
  replace every match, some the first match, and some require exactly one.
- Restart regridding uses a fixed `<output>.tmp` path, so concurrent invocations
  can collide.
- Process-memory and cgroup-limit handling is duplicated and differs among
  benchmark tools.

## Test backlog

- Species concentration, restart, and HEMCO reader tests often assert names and
  shapes without asserting representative values; transposition or wrong-field
  bugs could pass.
- The 4x5 polar-row test checks uniformity but not the expected weighted mean or
  global integral; an all-zero result can pass.
- Expand runner failure-path coverage to assert both concrete output and
  ObsOperator managers close when a transport operation raises.
- Optional ObsOperator parity comparison stores rows in a dictionary keyed by
  `(id, field)`, losing duplicate-row multiplicity.
- External-data markers check broad directories instead of the exact dated
  files required by a test, leading to lower-level failures rather than skips.
- The broad HEMCO scenario test checks shape and finiteness but allows all-zero
  output; standalone parity covers only a subset of scenarios.
