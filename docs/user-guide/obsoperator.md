# ObsOperator

Wombat can sample post-transport tracer concentrations inline using the
GEOS-Chem ObsOperator schema.

## Run configuration

```yaml
outputs:
  obsoperator:
    activate: true
    verbose: false
    input_file: ./ObsOperator/obsoperator-YYYYMMDD.yml.gz
    output_file: ./OutputDir/GEOSChem.ObsOperator.YYYYMMDD_hhmmz.nc4
    restart_file: ./Restarts/Wombat.ObsOperator.Restart.YYYYMMDD_hhmmss.nc4
    restart_missing: warn
```

When active, `input_file`, `output_file`, and `restart_file` are required.
`restart_missing` may be `warn`, `error`, or `ignore`. The former asynchronous
`input_mode` and `writer` options are not supported.

Relative paths resolve from the run YAML directory. `YYYY`, `MM`, `DD`, `hh`,
`mm`, and `ss` expand from model time. Missing daily input is logged and
skipped.

## Input schema

Input is YAML 1.2, optionally gzip-compressed, with an `entries` sequence.
Entries identify concentration fields explicitly as `SpeciesConcVV_<tracer>`
or use `SpeciesConcVV_?ALL?` and `SpeciesConcVV_?ADV?`. The older `species`
key is not accepted.

Pressure is in hPa, altitude in metres, and `grid_index` and `pressure_level`
values are one-based. Pressure levels count from the surface upward. Entry IDs
must be unique among all active entries.

An entry normally has one `time_operator`, `horizontal_operator`, and
`vertical_operator` mapping. Each key may instead contain a nonempty sequence
of mappings. Components in the same dimension are additive, so disjoint time
windows can accumulate into one observation without repeating its accumulator
or ID. Overlapping components deliberately add their weights.

## Time semantics

Time indices are zero-based and use GEOS-Chem end-of-timestep sampling. Index
zero samples the concentration after the first transport step.

Date-time ranges describe half-open model periods. At a ten-minute timestep,
`00:00`--`01:00` averages six periods, indices zero through five. Point times
and zero-duration ranges sample the period ending at that timestamp.

Internally, ranges remain compact half-open bounds; Wombat does not create one
schedule record per sampling timestep. Entries are kept in completion order
and expired portions are removed as the model advances.

Wombat concentrations are already dry volume mixing ratios, so ObsOperator
does not apply another molecular-weight conversion.

## Output and continuation

Science output contains completed operators only. It uses `id`, `field`,
`id_index`, `field_index`, and `sample` variables.

At clean shutdown Wombat atomically writes unfinished state to the dedicated
NetCDF restart, including an empty restart when nothing remains. A continuation
requires an exact boundary time, transport timestep, and grid match. Required
fields are restored by name, so tracer order may change and unrelated tracers
may be added, but a required field may not be removed.

The current restart format is version 3. It stores the packed operator arrays
and float64 accumulator directly. Older ObsOperator restart versions are
rejected rather than translated because their expanded schedules have a
different state contract.

An operator spanning a boundary retains its float64 accumulator, tracer
mapping, spatial and vertical components, and compact remaining time windows.
It appears in science output only after completion.

## Performance switch

The repository-wide `WOMBAT_NUMBA` switch controls the block sampling and
packed-plan maintenance kernels. The manager currently visits tracer blocks
serially, so `WOMBAT_NUMBA_THREADS` does not parallelize ObsOperator work.
