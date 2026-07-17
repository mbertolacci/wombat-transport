# First run

The repository includes a three-hour, one-tracer 2x2.5 example under
`examples/basic_2x25/`. It has no emissions or ObsOperator input and writes one
time-averaged concentration collection.

## Prerequisites

Populate these paths first:

```text
external_data/restarts/2x25/GEOSChem.Restart.20140901_0000z.nc4
external_data/geoschem/GEOS_2x2.5/MERRA2/
```

The meteorology tree must contain the required collections for 1 September
2014. See [external data](external-data.md) for details.

## Run the example

From the repository root:

```bash
.venv/bin/python -m wombat_transport.run examples/basic_2x25/run.yml
```

For a faster smoke test, stop after one ten-minute transport step:

```bash
.venv/bin/python -m wombat_transport.run \
  examples/basic_2x25/run.yml \
  --max-steps 1
```

The process reports the state shape, transport operators, step count, and
timestep. It also writes `wombat_run_metadata.json` beside the run file.

Numba acceleration is enabled by default. For larger tracer ensembles, the
[performance and threading guide](../user-guide/performance.md) explains the
block-native tracer layout, spatial and block execution strategies, and their
environment controls.

The complete three-hour run writes:

```text
examples/basic_2x25/OutputDir/
  Wombat.SpeciesConcThreeHourly.20140901_0000z.nc4
```

The collection filename uses the beginning of its three-hour averaging window.
The output contains `SpeciesConcVV_CO2` in GEOS-Chem HISTORY-style dimensions.

## Adapt the example

Copy the example directory before changing it. Paths in `run.yml` are resolved
relative to the YAML file, so update the restart, grid-template, and
meteorology paths if the copy moves to a different depth.

Use the [`run.yml` reference](../reference/run-yml.md) for every supported
field. The [inputs](../user-guide/inputs.md) and
[outputs](../user-guide/outputs.md) pages explain the linked files and output
collections.
