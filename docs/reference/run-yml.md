# `run.yml` reference

`run.yml` is the public configuration for a Wombat simulation. It is parsed as
YAML 1.2. Paths may be absolute; relative paths resolve from the directory
containing the YAML file.

This page documents the current format exactly. There is no schema-version
field yet, and unknown keys are not a supported extension mechanism.

## Complete shape

```yaml
name: example
source_run_dir: .
species_database: ./species_database.yml
initial_restart: /data/restarts/GEOSChem.Restart.20140901_0000z.nc4
grid_template: /data/restarts/GEOSChem.Restart.20140901_0000z.nc4
output_dir: ./OutputDir

simulation:
  start: "2014-09-01 00:00"
  end: "2014-09-01 03:00"
  transport_timestep_s: 600
  emissions_timestep_s: 1200

meteorology:
  root: /data/GEOS_2x2.5/MERRA2
  initial_time_index: 0
  chunk_multiple: 1

emissions:
  unit_conversion: none
  missing_species: zero
  scales: {}
  fields: []

logging:
  level: info

outputs:
  writer: sync
  expid: OutputDir/Wombat
  dtype: float32
  compression:
    enabled: true
    level: 1
    shuffle: true
  chunking:
    rank1: [512]
    rank2:
    rank3: [1, 91, 144]
    rank4: [1, 1, 91, 144]
  collections:
    SpeciesConcThreeHourly:
      template: "%y4%m2%d2_%h2%n2z.nc4"
      frequency: 00000000 030000
      duration: 00000000 030000
      mode: time-averaged
      fields: [SpeciesConcVV_?ADV?]
  obsoperator:
    activate: false

diagnostics: {}
comparison: {}
validation: {}
```

## Top-level fields

| Field | Required | Type | Meaning |
|---|---|---|---|
| `name` | yes | string | Run identifier written to logs and metadata |
| `source_run_dir` | yes | path | Compatibility/inspection metadata; does not change the working directory |
| `species_database` | yes | path | GEOS-Chem-style species mapping |
| `initial_restart` | no | path or null | Restart concentrations; missing species use `Background_VV` |
| `grid_template` | yes | path | Coordinates, area, and hybrid grid; always required |
| `output_dir` | yes | path | Compatibility/inspection field; collection templates control actual output paths |
| `simulation` | yes | mapping | Dates and timesteps |
| `meteorology` | yes | mapping | Meteorology root and loading controls |
| `emissions` | no | mapping or path | Inline emissions configuration or separate YAML; default is empty |
| `logging` | no | mapping | Logging controls |
| `outputs` | no | mapping | HISTORY, restart, and ObsOperator output; empty disables output |
| `diagnostics` | no | mapping | Developer inspection inputs |
| `comparison` | no | mapping | Legacy comparison settings for focused CLI modes |
| `validation` | no | mapping | Comparison settings preferred over `comparison` when nonempty |

Although `initial_restart` is optional, `grid_template` is not. If no restart
is supplied, every tracer starts from its `Background_VV`.

## `simulation`

| Field | Required | Default | Rules |
|---|---|---|---|
| `start` | yes | none | `YYYY-MM-DD hh:mm` |
| `end` | yes for a full run | none | Same format; must be after `start` |
| `transport_timestep_s` | no | `600` | Positive whole seconds |
| `emissions_timestep_s` | no | transport timestep | Positive whole seconds and an integer multiple of transport timestep |

The simulation interval is half-open: transport advances from `start` until
the `end` boundary.

## `meteorology`

| Field | Required | Default | Rules |
|---|---|---|---|
| `root` | yes | none | Directory containing target-resolution MERRA-2 collections |
| `initial_time_index` | no | `0` | Integer initial record offset |
| `chunk_multiple` | no | `1` | Integer at least one |

## `emissions`

The value may be a mapping or a path string. A path is resolved from the run
YAML directory; paths inside that emissions file are then resolved from the
emissions file's directory.

Inline and external mappings accept:

| Field | Default | Current support |
|---|---|---|
| `unit_conversion` | `none` | Only `none` |
| `missing_species` | `zero` | Only `zero` |
| `scales` | `{}` | Named constant or NetCDF scaling fields |
| `fields` | `[]` | NetCDF surface-flux fields assigned to configured species |

See [inputs](../user-guide/inputs.md) for source fields and regridding.

## `logging`

`logging.level` defaults to `warning` and may be `warning`, `info`, or `debug`.

## `outputs`

| Field | Default | Accepted values |
|---|---|---|
| `writer` | `sync` | `sync`, `threaded` |
| `expid` | `OutputDir/GEOSChem` | Filename prefix used with collection `template` |
| `dtype` | `float32` | `float32`, `float64` |
| `compression.enabled` | `true` | boolean |
| `compression.level` | `1` | integer 0--9 |
| `compression.shuffle` | `true` | boolean |
| `chunking.rank1`--`rank4` | null | null or matching-length positive integer array |
| `collections` | `{}` | Named collection mappings |
| `obsoperator` | `{}` | Inline ObsOperator configuration |

Each collection requires `frequency`, `mode`, and `fields`, plus either
`filename` or `template`. `duration` defaults to `frequency`.

Currently supported collection combinations are:

- `time-averaged` with exactly `SpeciesConcVV_?ADV?`;
- `instantaneous` including `SpeciesRst_?ALL?`, optionally with supported
  restart meteorology fields.

See [outputs](../user-guide/outputs.md) for intervals, field tokens, and file
naming.

### `outputs.obsoperator`

| Field | Default | Rules |
|---|---|---|
| `activate` | `false` | Enables sampling |
| `verbose` | `false` | Logs each sampled entry |
| `input_file` | null | Required when active |
| `output_file` | null | Required when active |
| `restart_file` | null | Required when active |
| `restart_missing` | `warn` | `warn`, `error`, or `ignore` |

See [ObsOperator](../user-guide/obsoperator.md) for compact time-window and
continuation semantics.

## Diagnostic and comparison settings

These settings support inspection and focused transport CLI modes; they are
not needed for a production `--mode run` simulation.

- `diagnostics.species_conc_sample` and `diagnostics.hemco_sample` are read by
  the inspection command.
- `diagnostics.level_edge_sample` enables pressure-edge comparison for focused
  transport modes.
- `validation.species_conc_sample` or the legacy equivalent under `comparison`
  enables a post-run concentration comparison. `species_conc_time_index`
  selects the reference record.

Prefer named cases under `validation_runs/` for complete GEOS-Chem versus
Wombat validation experiments.
