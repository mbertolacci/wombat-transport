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

No active items from the original audit remain in this section.

## Transport API and abstraction backlog

- The unified executor still independently prepares vertical reversals, VDIFF
  inputs, and convection handoffs. Consolidate only where an existing operator
  boundary can be shared without new per-step objects or array copies.
- `TracerField` is frozen but owns writable NumPy arrays and mutable coordinate
  metadata. Model mutable buffer ownership directly; copying storage would be a
  parity-sensitive behavioral change.

## Proposed design-debt solutions

1. Give deferred TPCORE output a distinct result type and a field such as
   `tracer_mass_after`; reserve `tracer_conc_after` for finalized concentration.
   **Resolved in the ownership-API batch.**
2. Replace the `reuse_input` and `reuse_output` booleans with explicit in-place
   and workspace entry points; keep borrowed scratch results internal.
   **Resolved in the ownership-API batch.**
3. Add one TPCORE `bind_state_storage()` operation that validates and updates
   all workspace aliases atomically. **Resolved in the ownership-API batch.**
4. Store grid identity metadata in `TpcoreStaticTerms` and validate it when an
   operator is prepared, capturing the immutable identity once.
   **Resolved in the met-planning batch.**
5. Validate the supported zonally uniform cell-area invariant before TPCORE
   pole averaging. **Resolved in the low-risk design batch.**
6. Confirm use of the scalar TPCORE/reference compatibility helpers, then
   remove unreachable code or quarantine deliberately retained oracle code.
   **Resolved in the low-risk design batch.**
7. Extract backend-neutral pressure, TPCORE, VDIFF, and convection preparation
    functions, proving their arrays exactly equal before switching the unified
    executor. **Pressure boundary resolved; later handoffs remain
    parity-sensitive.**
8. Add a met-only `prepare_vdiff_met_plan()` and remove the misleading
   tracer-shaped planning surface.
   **Resolved in the met-planning batch.**
9. Replace the executor's broad `shares_memory` ownership test with exact
   constant-time storage validation: pointer, shape, strides, dtype, and
   layout. **Resolved in the low-risk design batch.**
10. Remove inert convection arguments from the operator API while retaining
    any fields actually needed as trace/oracle diagnostic payload.
    **Resolved in the low-risk design batch.**
11. Make `TracerField` buffer mutability explicit while keeping names, units,
    and coordinate metadata immutable; do not introduce implicit array copies.
12. Parse and semantically validate every output collection before constructing
    writers or loading simulation data. **Resolved in the low-risk design batch.**
13. Validate grid coordinates and orientation against the canonical GEOS 2x2.5
    and 4x5 centers with a documented tight tolerance.
    **Resolved in the low-risk design batch.**
14. Introduce one immutable conservative-remapping weights object and one
    leading-dimension application routine, preserving existing polar arithmetic
    exactly. **Resolved in the emissions-remapping batch.**
15. Normalize ObsOperator YAML and restart data into one intermediate form and
    pass both through a shared validator and array-state builder.
    **Rejected.** That intermediate representation previously made large input
    setup substantially slower. The current separate ingestion paths converge
    directly on the flat sampling arrays and share focused validators without
    adding per-entry objects or another full-state construction pass.

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

## TPCORE ownership-API batch outcome

- **Deferred result semantics — resolved.** Compiled deferred finalization now
  returns `TpcoreDeferredState.tracer_mass_after_hpa`; finalized public results
  continue to use `TpcoreState.tracer_conc_after`.
- **Input and result ownership — resolved.** The public prepared-step API is
  copy-safe and caller-owned. Private production adapters separately express a
  borrowed finalized result, borrowed deferred mass, and consuming deferred
  mass. Borrowed lifetime and synchronous VDIFF consumption are documented.
- **No hidden consuming copy — resolved.** Consuming calls require writable,
  C-contiguous float64 arrays and reject inputs that would require conversion.
- **Workspace rebinding — resolved.** `bind_state_storage()` validates the
  complete storage contract and updates every block alias without `copy` or
  `copyto`.
- **Benchmark attribution — resolved.** Driver timing wraps all explicit
  TPCORE entry points, so introducing the adapters does not misclassify TPCORE
  time as driver overhead.

There were no numerical or performance-semantic changes to the kernels. The
2x2.5, 24-tracer, one-thread copy-safe benchmark retained its checksum and
moved from 0.300658 to 0.299742 seconds best-of-15. A matched consuming-chain
control built from the parent commit measured 0.295158 seconds best-of-15;
the new source measured 0.295155 seconds with the identical chained checksum.
Their means differed by 0.16%, within host noise. Ownership tests also verify
input mutation, owned-result stability, borrowed-result invalidation, and
zero-copy workspace binding directly.

## TPCORE identity and VDIFF met-planning batch outcome

- **TPCORE static identity — resolved.** Cached terms carry immutable snapshots
  of cell area, hybrid coefficients, and latitude. Reuse validates those source
  values before applying cached geometry or pressure terms and rejects a
  mismatch explicitly.
- **VDIFF met planning — resolved.** `prepare_vdiff_met_plan()` and its kernel
  adapter accept meteorology only. Persistent plan workspaces no longer contain
  dummy tracer or surface-flux arrays, and the misleading old plan name was
  removed rather than retained as compatibility surface.
- **Numerical identity — verified.** Independent met-plan workspaces produce
  exactly equal plan arrays, while the full VDIFF and transport oracle tests
  continue to pass.

The immutable TPCORE snapshots add about one grid's area plus three coordinate
vectors once per run. Their full reuse check measured about 6 microseconds on
2x2.5. The 24-tracer one-thread driver retained its checksum and moved from
0.299870 to 0.298970 seconds best-of-15; alternating cached-static old/new
controls varied in both directions by more than the observed difference, so
there is no measurable regression.

## ObsOperator correctness batch outcome

- **Sampling semantics — resolved.** Production scheduling remains keyed by
  transport `step_start`, while the runner now asserts that the sampled
  snapshot is the corresponding completed-step state at `step_end`. No timing
  or time-index semantics changed.
- **Transactional writer registry — resolved.** Pending field names and indices
  are built locally and committed only after every NetCDF assignment succeeds.
  Failed flushes retain their pending batches, counters, and prior registry for
  a consistent retry.
- **Vertical validation — resolved.** Fresh YAML and restart reconstruction use
  the same finite-value, finite-weight, range, and level checks before building
  sampling state.
- **String widths — resolved.** IDs and field names are validated in UTF-8
  bytes during ingestion and serialization, including exact multibyte boundary
  tests and rejection of embedded NUL characters.

The 5,000-entry setup benchmark moved from 63.244/66.305 ms best/mean to
63.229/65.384 ms after adding an ASCII validation fast path. Compiled sampling
moved from 77.992/79.827 microseconds to 77.847/78.929 microseconds with the
same checksum. The validation and writer changes do not enter the sampling
kernel or per-observation loop.

## Safety-net batch outcome

- **Dry-pressure ordering — resolved.** Harness comparisons require callers to
  declare whether oracle levels are bottom-to-top or top-to-bottom, and a
  reversed-level regression test covers both representations.
- **Reader values — resolved.** Restart, species-concentration, and HEMCO tests
  compare representative interior and boundary values with named NetCDF source
  variables, including the expected vertical reversal and tracer selection.
- **4x5 polar conservation — resolved.** The polar-row test now checks the
  independently calculated overlap-weighted polar means and the global area
  integral in addition to row uniformity.
- **Failure cleanup — resolved.** A concrete runner test forces transport to
  raise after both managers are constructed and verifies ObsOperator and output
  closure, including the restart boundary passed to ObsOperator.
- **ObsOperator duplicate parity — resolved.** Optional parity preserves row
  multiplicity and has direct coverage for duplicate `(id, field)` samples.

These changes affect harness and test code only; no production transport or
sampling path changed.

## Emissions-remapping batch outcome

- **Shared geometry — resolved.** `ConservativeRemappingWeights` owns immutable
  snapshots of source and target coordinates, overlap matrices, and
  denominators. Its single application path handles arbitrary leading
  dimensions while preserving the existing polar-row operation order.
- **Production cache — resolved.** `EmissionsOperator` caches one complete
  geometry object for each source grid instead of maintaining a separate 2-D
  remapping implementation.
- **Restart reuse — resolved.** Restart conversion builds geometry once and
  applies it to every species instead of rebuilding latitude and longitude
  weights per variable.
- **Contracts — verified.** Tests cover exact agreement with the former 2-D
  arithmetic, leading dimensions, polar behavior, global conservation,
  coordinate snapshots, read-only geometry, source preservation, shape
  rejection, and operator cache reuse.

For a deterministic 1-degree-to-2x2.5 field, the shared application was bitwise
identical to the former specialized arithmetic. Best/mean application times
were 0.651/0.668 ms shared versus 0.648/0.671 ms specialized. Geometry
construction was 17.95 ms versus 17.94 ms for the former raw-weight setup.
There is no measurable regression and no new per-application input copy.

## Shared pressure-preparation batch outcome

- **Dry-pressure boundary — resolved.** Regular, trace, and unified transport
  paths now use the same allocation-neutral functions to reconstruct current
  dry surface pressure and to construct paired next-boundary pressure thickness
  and dry-air mass.
- **Arithmetic and ownership — preserved.** The helpers contain the former
  expressions unchanged, return the same newly required pressure/mass arrays,
  and introduce no intermediate state object or additional array copy.
- **Vertical preparation — deliberately unchanged.** The regular driver enters
  the complete VDIFF operator while the unified executor prepares a persistent
  VDIFF plan. Sharing their reversal bundle would require a new per-step object
  or a wider operator-boundary change, so it remains deferred.

The exact pressure tests and complete transport parity suite retained identical
pressure and mass arrays. In alternating 2x2.5, 24-tracer, one-thread controls,
the second parent/current pair measured 0.299489/0.299400 seconds best-of-15
with the same checksum. The 0.03% difference is below host noise.

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

- External-data markers check broad directories instead of the exact dated
  files required by a test, leading to lower-level failures rather than skips.
- The broad HEMCO scenario test checks shape and finiteness but allows all-zero
  output; standalone parity covers only a subset of scenarios.
