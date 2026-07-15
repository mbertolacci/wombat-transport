# External data

This directory is the single local entry point for large inputs that are
required by real-data transport and validation runs but are not stored in Git.
Only this README is tracked. Each child may be a directory or a symlink to a
shared data store.

## Layout

```text
external_data/
  geoschem/
  fluxes/
  obsoperator/
  scaling-grids/
  restarts/
    GEOSChem.Restart.20140901_0000z.nc4
    4x5/
      GEOSChem.Restart.20140901_0000z.nc4
```

### `geoschem/`

GEOS-Chem ExtData. The current validation cases require MERRA-2 GEOS 2x2.5
meteorology under `GEOS_2x2.5/MERRA2`, including the A1, A3dyn, A3mstC,
A3mstE, and I3 collections for the simulated dates. GEOS-Chem also expects
the standard `HEMCO`, `CHEM_INPUTS`, and related ExtData trees beneath this
directory. On the original development machine this is a symlink to
`/home/mgnb/GEOS_Chem/ExtData`.

The 4x5 validation cases use the corresponding daily collections under
`GEOS_4x5/MERRA2`; filenames end in `.4x5.nc4`.

### `fluxes/`

Local emissions inputs for the residual-tracer cases. The September 2014
one-day case requires:

- `SOM_FFN_vBAMS2024v2_residual.nc`, variable `residual`;
- `sib4-residual-gpp/2014/sib4-residual-gpp-2014-09-01.nc`, variable
  `residual` with an `npft` dimension.

Longer validation cases require the matching daily/monthly files through the
end of their configured periods. Other directories here are optional inputs
for additional HEMCO experiments.

### `scaling-grids/`

NetCDF regional masks named `regionRegionNN_monthYYYY-MM.nc`, with variable
`value`, used by residual emissions configurations. The one-day case uses the
September 2014 masks referenced by its checked-in `emissions.yml`.

### `obsoperator/`

Daily gzip-compressed ObsOperator inputs named
`obsoperator-YYYYMMDD.yml.gz`. Validation matrix runs expand these for
GEOS-Chem and pass the compressed files directly to Wombat.

### `restarts/`

`GEOSChem.Restart.20140901_0000z.nc4` is the canonical one-tracer initial
condition and grid template. It must contain `SpeciesRst_CO2` and the standard
GEOS-Chem grid/pressure coordinates on the global 2x2.5, 47-level grid.

The ignored `4x5/GEOSChem.Restart.20140901_0000z.nc4` parity-test template is
generated from that canonical restart:

```bash
PYTHONPATH=src .venv/bin/python tools/regrid_restart.py \
  external_data/restarts/GEOSChem.Restart.20140901_0000z.nc4 \
  external_data/restarts/4x5/GEOSChem.Restart.20140901_0000z.nc4
```

The migrated reference file has:

```text
size:   63670600 bytes
sha256: baa953b0aa1a08a77e2a48dc90604a1724f68f9de2420bc7b1086e4a221813b5
```

## Setup

Populate the tree directly or link individual datasets, for example:

```bash
mkdir -p external_data
ln -s /path/to/GEOS_Chem/ExtData external_data/geoschem
ln -s /path/to/fluxes external_data/fluxes
ln -s /path/to/scaling-grids external_data/scaling-grids
ln -s /path/to/obsoperator external_data/obsoperator
mkdir -p external_data/restarts
cp /path/to/GEOSChem.Restart.20140901_0000z.nc4 external_data/restarts/
```

The compact unit-test fixtures under `tests/fixtures/` do not require this
tree. Full-grid transport tests, oracle regeneration, GEOS-Chem harnesses, and
`validation_runs/` cases skip or fail with a specific missing-input message
when their required external inputs are unavailable.
