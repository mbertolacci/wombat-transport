# External data

Real Wombat runs read GEOS-Chem-compatible meteorology, a restart or grid
template, and any configured emissions and ObsOperator inputs. These large
payloads are not distributed in Git and Wombat does not currently download
them automatically.

The supported local entry point is `external_data/` at the repository root.
Directories may contain data directly or be symbolic links to a shared store.

```text
external_data/
  geoschem/
    GEOS_2x2.5/MERRA2/
    GEOS_4x5/MERRA2/
  fluxes/
  obsoperator/
  scaling-grids/
  restarts/
    2x25/GEOSChem.Restart.20140901_0000z.nc4
    4x5/GEOSChem.Restart.20140901_0000z.nc4
```

## Meteorology

For each simulated date, Wombat needs the MERRA-2 `A1`, `A3dyn`, `A3mstC`,
`A3mstE`, and `I3` collections at the target resolution. The forcing loader
uses the GEOS-Chem record-selection and interpolation cadence for the current
transport fields.

The 2x2.5 files belong below `GEOS_2x2.5/MERRA2`. The 4x5 files belong below
`GEOS_4x5/MERRA2` and use filenames ending in `.4x5.nc4`.

## Restart and grid template

The grid template supplies coordinates, cell areas, and the 47-level hybrid
vertical grid. A restart may also initialize any matching `SpeciesRst_*`
variables. Species absent from the restart are initialized from
`Background_VV` in the species database.

The repository's 4x5 parity-test restart is generated from the canonical
2x2.5 restart. The exact command and expected checksum are maintained in the
[external-data contract](https://github.com/mbertolacci/wombat-transport/blob/main/external_data/README.md).

## Optional study inputs

- Emissions configurations can read flux fields and scaling grids from any
  paths accessible to the run.
- ObsOperator daily inputs may be ordinary YAML or gzip-compressed YAML.
- The checked-in residual validation cases expect their fluxes, masks, and
  observations beneath the corresponding `external_data/` directories.

See the repository's
[external-data contract](https://github.com/mbertolacci/wombat-transport/blob/main/external_data/README.md)
for the exact validation-data layout.
