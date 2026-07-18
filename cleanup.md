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
