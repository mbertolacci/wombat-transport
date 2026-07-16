# Inputs

A normal Wombat simulation combines a run configuration, a species database,
a grid template, optional restart concentrations, MERRA-2 meteorology, and
optional emissions and ObsOperator inputs.

All relative paths in `run.yml` resolve from the directory containing that
file. When `emissions` names a separate YAML file, paths inside it resolve from
the emissions file's directory.

## Species database

The species database uses the GEOS-Chem species mapping style. Wombat reads
entries with `Is_Tracer: true` and requires `MW_g` and `Background_VV`.
`FullName` is optional.

```yaml
CO2:
  Formula: CO2
  Is_Gas: true
  MW_g: 44.01
  Is_Tracer: true
  Background_VV: 0.0004
  FullName: carbon dioxide
```

The mapping order defines tracer order. Internally, tracer state uses
`(lev, lat, lon, tracer)` order.

## Grid template and restart

`grid_template` is always required. It supplies latitude, longitude, area, and
hybrid vertical-coordinate fields on either the global 2x2.5 or 4x5,
47-level grid.

`initial_restart` may be a NetCDF restart path or null. For every configured
tracer:

- a matching `SpeciesRst_<name>` variable initializes the tracer when present;
- otherwise Wombat fills the field from the species `Background_VV` value.

The restart and grid template must use the same supported grid.

## Meteorology

`meteorology.root` points at the resolution-specific MERRA-2 directory. Wombat
selects and interpolates the `A1`, `A3dyn`, `A3mstC`, `A3mstE`, and `I3`
fields required by TPCORE, VDIFF, and convection.

`initial_time_index` offsets initial record selection and defaults to zero.
`chunk_multiple` controls how many forcing windows are loaded together and
defaults to one; it must be an integer of at least one.

## Emissions

`emissions` may contain an inline mapping or the path to a separate YAML file.
An empty configuration produces zero emissions.

```yaml
emissions:
  unit_conversion: none
  missing_species: zero
  scales: {}
  fields: []
```

The currently supported policies are exactly `unit_conversion: none` and
`missing_species: zero`. Configured fields name a target `species`, source
`path_template`, NetCDF `variable`, selection `frequency`, units, and optional
scale names. Sources must reduce to a two-dimensional latitude/longitude
field after time and optional-dimension selection.

Wombat accepts `lat`/`lon` and `latitude`/`longitude` coordinate names. Fields
on another regular latitude/longitude grid are conservatively regridded to the
configured GEOS grid before scaling and accumulation.

For full examples, see the checked-in
[residual emissions configuration](https://github.com/mbertolacci/wombat-transport/blob/main/validation_runs/cases/residual_24tracer_emissions_1day_2x25/wombat/main/emissions.yml)
and the [HEMCO harness guide](https://github.com/mbertolacci/wombat-transport/blob/main/tools/hemco_harness/README.md).

## ObsOperator inputs

ObsOperator is optional. When activated, it reads daily YAML or `.yml.gz`
files expanded from a timestamp template. Its input schema and restart behavior
are documented on the [ObsOperator page](obsoperator.md).
