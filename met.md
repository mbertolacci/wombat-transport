# Meteorology Timing Notes

These notes record the GEOS-Chem behavior we need to match for the current
transport-only Wombat target. They are based on the local GEOS-Chem Classic
source and the `residual_20140901_part001_split01/HEMCO_Config.rc` met entries.

## Source Trail

- `GCClassic/src/GEOS-Chem/GeosCore/hco_interface_gc_mod.F90`
  - `Get_Met_Fields` reads CN/A1/A3/I3 fields through HEMCO.
- `GCClassic/src/GEOS-Chem/GeosUtil/time_mod.F90`
  - `GET_A1_TIME`, `GET_A3_TIME`, `GET_I3_TIME`, and `ITS_TIME_FOR_*` define
    met read times.
- `GCClassic/src/GEOS-Chem/GeosCore/flexgrid_read_mod.F90`
  - `FlexGrid_Read_A1`, `FlexGrid_Read_A3*`, `FlexGrid_Read_I3_1`,
    `FlexGrid_Read_I3_2`, and `COPY_I3_FIELDS` define record selection and
    unit conversion.
- `GCClassic/src/GEOS-Chem/GeosCore/calc_met_mod.F90`
  - `INTERP` linearly interpolates I3 pressure, temperature, and specific
    humidity every dynamic timestep.
- `GCClassic/src/GEOS-Chem/Interfaces/GCClassic/main.F90`
  - Main loop order: HEMCO phase 1 updates met first, then `INTERP` is called,
    then air quantities are updated, then transport/mixing work proceeds.

## Local File Time Axes

For the `2014-09-01` MERRA2 2x2.5 files in `ExtData`:

| Collection | Records/day | NetCDF time values | Meaning |
| --- | ---: | --- | --- |
| `A1` | 24 | `30, 90, 150, ... 1410` minutes | 1-hour averages stamped at hour midpoints |
| `A3dyn` | 8 | `90, 270, 450, ... 1350` minutes | 3-hour averages stamped at 3-hour midpoints |
| `A3mstC` | 8 | `90, 270, 450, ... 1350` minutes | 3-hour averages stamped at 3-hour midpoints |
| `A3mstE` | 8 | `90, 270, 450, ... 1350` minutes | 3-hour averages stamped at 3-hour midpoints |
| `I3` | 8 | `0, 180, 360, ... 1260` minutes | instantaneous 3-hour endpoints |

## GEOS-Chem Cadence

### A1: 1-hour averaged fields

GEOS-Chem reads A1 fields on every model hour. `ITS_TIME_FOR_A1()` checks
`MOD(NHMS, 010000) == 0`. There is a stale comment saying "every 3 hours", but
the code is hourly.

`GET_A1_TIME()` returns `GET_TIME_AHEAD(1800)`, so a model time of `03:00`
requests the `03:30` midpoint record. `FlexGrid_Read_A1` then selects:

```text
t_index = (HHMMSS / 10000) + 1
```

No interpolation is applied to A1 fields after they are read. They are held
until the next hourly read.

Relevant Wombat fields from A1:

| Wombat field | GEOS-Chem/Hemco name | Source variable | Unit handling |
| --- | --- | --- | --- |
| `pbl_height_m` | `PBLH` | `PBLH` | as read |
| `sensible_heat_flux_w_m2` | `HFLUX` | `HFLUX` | as read |
| `latent_heat_flux_w_m2` | `EFLUX` | `EFLUX` | as read |
| `friction_velocity_m_s` | `USTAR` | `USTAR` | as read |
| `convective_precip_mm_day` | `PRECCON` | `PRECCON` | GEOS-Chem multiplies by `86400` from `kg m-2 s-1` to `mm day-1` |

### A3: 3-hour averaged fields

GEOS-Chem reads A3 fields every 3 hours. `ITS_TIME_FOR_A3()` checks
`MOD(NHMS, 030000) == 0`.

`GET_A3_TIME()` returns `GET_TIME_AHEAD(5400)`, so a model time of `03:00`
requests the `04:30` midpoint record. The first read uses
`GET_FIRST_A3_TIME()`, which maps an arbitrary start time to the midpoint of the
containing 3-hour window. The `FlexGrid_Read_A3*` routines select:

```text
t_index = (HHMMSS / 030000) + 1
```

No interpolation is applied to A3 fields after they are read. They are held for
the full 3-hour window.

Relevant Wombat fields from A3:

| Wombat field | GEOS-Chem/Hemco name | File collection | Source variable | Notes |
| --- | --- | --- | --- | --- |
| `u_m_s` | `U` | `A3dyn` | `U` | held 3-hour average |
| `v_m_s` | `V` | `A3dyn` | `V` | held 3-hour average |
| `omega_pa_s` | `OMEGA` | `A3dyn` | `OMEGA` | held 3-hour average |
| `convective_detrainment_kg_m2_s` | `DTRAIN` | `A3dyn` | `DTRAIN` | held 3-hour average |
| `convective_precip_prod_kg_kg_s` | `DQRCU` | `A3mstC` | `DQRCU` | held 3-hour average |
| `convective_precip_reevap_kg_kg_s` | `REEVAPCN` | `A3mstC` | `REEVAPCN` | held 3-hour average |
| `convective_mass_flux_kg_m2_s` | `CMFMC` | `A3mstE` | `CMFMC` | edge field |
| `convective_ice_flux_kg_m2_s` | `PFICU` | `A3mstE` | `PFICU` | edge field |
| `convective_liquid_flux_kg_m2_s` | `PFLCU` | `A3mstE` | `PFLCU` | edge field |

GEOS-Chem also reads A3 cloud and large-scale precip fields, but those are not
currently part of Wombat's transport-only path.

### I3: instantaneous 3-hour fields

GEOS-Chem keeps two I3 endpoints:

- `*_1`: beginning of the current 3-hour window.
- `*_2`: end of the current 3-hour window.

At initialization, `GET_FIRST_I3_TIME()` reads the nearest previous 3-hour
instantaneous record into endpoint 1. At each 3-hour boundary,
`GET_I3_TIME()` reads the next endpoint into endpoint 2. `FlexGrid_Read_I3_1`
and `FlexGrid_Read_I3_2` select:

```text
t_index = (HHMMSS / 030000) + 1
```

For the endpoint at `00:00` on the following day, `FlexGrid_Read_I3_2` uses
the HEMCO names `PS_NEXTDAY`, `SPHU_NEXTDAY`, and `TMPU_NEXTDAY`. Those entries
point at the next day's I3 file and select record 1. This is how GEOS-Chem
brackets the `21:00` to `24:00` window without asking for time index 9 in the
current day's file.

Relevant Wombat fields from I3:

| Wombat field | GEOS-Chem/Hemco name | Source variable | Unit handling |
| --- | --- | --- | --- |
| `surface_pressure_pa` | `PS` / `PS_NEXTDAY` | `PS` | GEOS-Chem converts Pa to hPa internally |
| `specific_humidity_kg_kg` | `SPHU` / `SPHU_NEXTDAY` | `QV` | GEOS-Chem converts kg/kg to g/kg internally |
| `temperature_k` | `TMPU` / `TMPU_NEXTDAY` | `T` | as read |

`INTERP` is called every dynamic timestep. With 600 s transport timesteps it
interpolates:

- surface pressure to the end of the dynamic timestep using `TC2`
- temperature and specific humidity to the midpoint of the dynamic timestep
  using `TM`

For a 3-hour I3 window starting at `NTIME0`, current elapsed time `NTIME1`, and
dynamic timestep `NTDT`:

```text
TM  = (NTIME1 + NTDT / 2 - NTIME0) / 10800
TC2 = (NTIME1 + NTDT     - NTIME0) / 10800
```

After the 3-hour outer timestep has completed, `COPY_I3_FIELDS` copies endpoint
2 into endpoint 1.

## HEMCO `+30minute`, `+90minute`, and `+1day`

The `+30minute` and `+90minute` suffixes in `HEMCO_Config.rc` matter, but they
are timestamp alignment, not an extra physical lag.

- A1 files contain hourly averages stamped at the midpoint. GEOS-Chem asks
  HEMCO for current model hour plus 30 minutes.
- A3 files contain 3-hour averages stamped at the midpoint. GEOS-Chem asks
  HEMCO for current 3-hour boundary plus 90 minutes.
- I3 files are instantaneous and have no midpoint suffix.
- `+1day` on `*_NEXTDAY` I3 entries supplies the first record of the next day
  for the end of the final daily 3-hour bracket.

For a direct Wombat NetCDF reader, the equivalent behavior is to select records
by these actual NetCDF timestamps or by the same deterministic index rules. We
should not treat `+30minute` or `+90minute` as interpolation instructions.

## Current Wombat Behavior

`load_transport_forcing_for_step` maintains separate A1, A3, and I3 clocks for
the runner.

- A1 fields are read hourly and held for that hour.
- A3 fields are read every 3 hours and held for that 3-hour window.
- I3 wet pressure is interpolated to the end of the dynamic timestep.
- I3 temperature and humidity are interpolated to the midpoint of the dynamic
  timestep.
- Dry surface pressure is computed from I3 wet endpoint pressure and endpoint
  humidity, polar averaged, and then interpolated to the dynamic timestep end.
- Transport dry pressure thickness is computed from interpolated dry surface
  pressure, matching the GEOS-Chem `DELP_DRY` transport convention.
- If an initial restart has `Met_DELPDRY`, the runner uses it for the initial
  dry-air mass; otherwise it initializes dry-air mass from computed dry surface
  pressure.

## Behavior We Need

For the current fixed-grid transport target:

1. Maintain separate met clocks for A1, A3, and I3 rather than one 3-hour
   `time_index`.
2. Load A1 fields hourly and hold them constant for that hour.
3. Load A3 fields every 3 hours and hold them constant for that 3-hour window.
4. Maintain I3 bracketing endpoint fields and compute interpolated pressure,
   temperature, and specific humidity for each transport timestep.
5. Handle the last daily I3 bracket by reading `00:00` from the next day's file.
6. Keep the existing direct file-reader cache bounded by collection/date/record;
   it should only need the current A1 record, current A3 record, and the current
   plus next I3 endpoints.
7. Preserve GEOS-Chem unit conventions at the operator boundary:
   - pressure arrays passed to Python operators should be explicit about Pa vs
     hPa;
   - specific humidity should remain kg/kg on the Python side unless a ported
     operator specifically expects GEOS-Chem's internal g/kg convention;
   - `PRECCON` should be converted to `mm day-1` when matching GEOS-Chem
     convection inputs.

These notes intentionally cover only the met fields currently used by the
transport-only runner. Additional GEOS-Chem `AIRQNT` diagnostics should be ported
only when they become operator inputs or comparison targets.
