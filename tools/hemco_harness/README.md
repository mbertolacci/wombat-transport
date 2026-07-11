# HEMCO Emissions Harness

This harness generates small HEMCO standalone run directories for checking
Wombat's `EmissionsOperator` against HEMCO's source read, regridding, scale
factor, and species-emission assembly behavior.

Generated files belong under `tools/hemco_harness/work/` and are ignored by
git.

The generated `HEMCO_sa_Grid.rc` deliberately writes explicit GEOS 2x2.5
latitude edges, including the polar half-cells `[-90, -89]` and `[89, 90]`.
Do not remove those edges: HEMCO standalone reconstructs different polar edges
from `YMID` alone, which exercises a different regridding branch than the
GEOS-Chem Classic target run.

## Generate a Scenario

```bash
python -m wombat_transport.hemco_harness list-scenarios

python -m wombat_transport.hemco_harness generate \
  source_regrid_then_scale \
  tools/hemco_harness/work/source_regrid_then_scale
```

Each generated run directory contains:

- `HEMCO_sa_Config.rc`, `HEMCO_Config.rc`, `HEMCO_Diagn.rc`
- `HEMCO_sa_Grid.rc`, `HEMCO_sa_Spec.rc`, `HEMCO_sa_Time.rc`
- synthetic NetCDF files under `inputs/`
- matching Wombat config in `wombat_emissions.yml`
- scenario metadata in `scenario.yml`

## Build HEMCO Standalone

From a generated run directory, configure HEMCO standalone with that directory
as `RUNDIR`:

```bash
cmake -S GCClassic/src/HEMCO -B tools/hemco_harness/build \
  -DRUNDIR=$PWD/tools/hemco_harness/work/source_regrid_then_scale

cmake --build tools/hemco_harness/build --target hemco_standalone -j
```

Set the executable path if it is not found automatically:

```bash
export HEMCO_STANDALONE=$PWD/tools/hemco_harness/build/bin/hemco_standalone
```

## Run and Compare

```bash
python -m wombat_transport.hemco_harness run \
  tools/hemco_harness/work/source_regrid_then_scale

python -m wombat_transport.hemco_harness compare \
  tools/hemco_harness/work/source_regrid_then_scale \
  --csv tools/hemco_harness/work/hemco_emissions_compare.csv
```

The comparison reports per-scenario, per-species max/mean error, global mass
error, max grid-cell mass error, nonzero-mask mismatches, and whether both
outputs are bottom-level-only.

The optional standalone comparison should stay tight for same-grid fields,
source regridding, scale factors, and source-plus-scale regridding. If only the
first or last latitude row fails, inspect the generated target grid edges
before relaxing tolerances.
