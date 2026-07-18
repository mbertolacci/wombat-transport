from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np

from wombat_transport.transport.pressure import dry_surface_pressure_hpa, wet_surface_pressure_hpa

from wombat_transport.grid import MODEL_LEVELS, TransportGrid

MERRA2_FILENAME = "MERRA2.{date}.{collection}.{resolution}.nc4"

MERRA2_72_AP_HPA = np.array(
    [
        0.000000e00,
        4.804826e-02,
        6.593752e00,
        1.313480e01,
        1.961311e01,
        2.609201e01,
        3.257081e01,
        3.898201e01,
        4.533901e01,
        5.169611e01,
        5.805321e01,
        6.436264e01,
        7.062198e01,
        7.883422e01,
        8.909992e01,
        9.936521e01,
        1.091817e02,
        1.189586e02,
        1.286959e02,
        1.429100e02,
        1.562600e02,
        1.696090e02,
        1.816190e02,
        1.930970e02,
        2.032590e02,
        2.121500e02,
        2.187760e02,
        2.238980e02,
        2.243630e02,
        2.168650e02,
        2.011920e02,
        1.769300e02,
        1.503930e02,
        1.278370e02,
        1.086630e02,
        9.236572e01,
        7.851231e01,
        6.660341e01,
        5.638791e01,
        4.764391e01,
        4.017541e01,
        3.381001e01,
        2.836781e01,
        2.373041e01,
        1.979160e01,
        1.645710e01,
        1.364340e01,
        1.127690e01,
        9.292942e00,
        7.619842e00,
        6.216801e00,
        5.046801e00,
        4.076571e00,
        3.276431e00,
        2.620211e00,
        2.084970e00,
        1.650790e00,
        1.300510e00,
        1.019440e00,
        7.951341e-01,
        6.167791e-01,
        4.758061e-01,
        3.650411e-01,
        2.785261e-01,
        2.113490e-01,
        1.594950e-01,
        1.197030e-01,
        8.934502e-02,
        6.600001e-02,
        4.758501e-02,
        3.270000e-02,
        2.000000e-02,
        1.000000e-02,
    ],
    dtype=np.float64,
)

MERRA2_72_TO_47_GROUPS = (
    (36, 38),
    (38, 40),
    (40, 42),
    (42, 44),
    (44, 48),
    (48, 52),
    (52, 56),
    (56, 60),
    (60, 64),
    (64, 68),
    (68, 72),
)

MERRA2_72_TO_47_MAPPING = "collapse_72_to_47_pressure_weighted"
MERRA2_A1_RECORDS_PER_DAY = 24
MERRA2_A3_RECORDS_PER_DAY = 8
MERRA2_A1_NATURAL_BLOCK_RECORDS = 24
MERRA2_A3_NATURAL_BLOCK_RECORDS = 4
MERRA2_I3_NATURAL_BLOCK_RECORDS = 4

@dataclass(frozen=True)
class TransportForcing:
    """Meteorological forcing mapped onto the prototype 47-level grid."""

    u_m_s: np.ndarray
    v_m_s: np.ndarray
    omega_pa_s: np.ndarray
    surface_pressure_start_pa: np.ndarray
    surface_pressure_pa: np.ndarray
    restart_surface_pressure_pa: np.ndarray
    wet_surface_pressure_start_hpa: np.ndarray
    wet_surface_pressure_hpa: np.ndarray
    restart_wet_surface_pressure_hpa: np.ndarray
    dry_surface_pressure_start_hpa: np.ndarray
    dry_surface_pressure_hpa: np.ndarray
    restart_dry_surface_pressure_hpa: np.ndarray
    i3_start_wet_surface_pressure_hpa: np.ndarray
    i3_start_dry_surface_pressure_hpa: np.ndarray
    i3_start_specific_humidity_kg_kg: np.ndarray
    specific_humidity_kg_kg: np.ndarray
    restart_specific_humidity_kg_kg: np.ndarray
    i3_start_temperature_k: np.ndarray
    temperature_k: np.ndarray
    restart_temperature_k: np.ndarray
    pbl_height_m: np.ndarray
    sensible_heat_flux_w_m2: np.ndarray
    latent_heat_flux_w_m2: np.ndarray
    friction_velocity_m_s: np.ndarray
    convective_mass_flux_kg_m2_s: np.ndarray
    convective_detrainment_kg_m2_s: np.ndarray
    convective_precip_prod_kg_kg_s: np.ndarray
    convective_precip_reevap_kg_kg_s: np.ndarray
    convective_ice_flux_kg_m2_s: np.ndarray
    convective_liquid_flux_kg_m2_s: np.ndarray
    convective_precip_mm_day: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    vertical_mapping: str
    a1_path: Path
    a3dyn_path: Path
    a3mstc_path: Path
    a3mste_path: Path
    i3_path: Path


@dataclass(frozen=True)
class _A1Fields:
    pblh: np.ndarray
    hflux: np.ndarray
    eflux: np.ndarray
    ustar: np.ndarray
    precccon: np.ndarray
    path: Path


@dataclass(frozen=True)
class _A3Fields:
    u: np.ndarray
    v: np.ndarray
    omega: np.ndarray
    dtrain: np.ndarray
    dqrcu: np.ndarray
    reevapcn: np.ndarray
    cmfmc: np.ndarray
    pficu: np.ndarray
    pflcu: np.ndarray
    a3dyn_path: Path
    a3mstc_path: Path
    a3mste_path: Path


@dataclass(frozen=True)
class _I3Fields:
    surface_pressure: np.ndarray
    qv: np.ndarray
    temperature: np.ndarray
    dry_surface_pressure_hpa: np.ndarray
    wet_surface_pressure_hpa: np.ndarray
    path: Path


@dataclass(frozen=True)
class _A1Block:
    start_index: int
    count: int
    pblh: np.ndarray
    hflux: np.ndarray
    eflux: np.ndarray
    ustar: np.ndarray
    precccon: np.ndarray
    paths: tuple[Path, ...]

    def contains(self, index: int) -> bool:
        return self.start_index <= index < self.start_index + self.count

    def field(self, index: int) -> _A1Fields:
        offset = index - self.start_index
        return _A1Fields(
            pblh=self.pblh[offset : offset + 1],
            hflux=self.hflux[offset : offset + 1],
            eflux=self.eflux[offset : offset + 1],
            ustar=self.ustar[offset : offset + 1],
            precccon=self.precccon[offset : offset + 1],
            path=self.paths[offset],
        )


@dataclass(frozen=True)
class _A3Block:
    start_index: int
    count: int
    u: np.ndarray
    v: np.ndarray
    omega: np.ndarray
    dtrain: np.ndarray
    dqrcu: np.ndarray
    reevapcn: np.ndarray
    cmfmc: np.ndarray
    pficu: np.ndarray
    pflcu: np.ndarray
    a3dyn_paths: tuple[Path, ...]
    a3mstc_paths: tuple[Path, ...]
    a3mste_paths: tuple[Path, ...]

    def contains(self, index: int) -> bool:
        return self.start_index <= index < self.start_index + self.count

    def field(self, index: int) -> _A3Fields:
        offset = index - self.start_index
        return _A3Fields(
            u=self.u[offset : offset + 1],
            v=self.v[offset : offset + 1],
            omega=self.omega[offset : offset + 1],
            dtrain=self.dtrain[offset : offset + 1],
            dqrcu=self.dqrcu[offset : offset + 1],
            reevapcn=self.reevapcn[offset : offset + 1],
            cmfmc=self.cmfmc[offset],
            pficu=self.pficu[offset],
            pflcu=self.pflcu[offset],
            a3dyn_path=self.a3dyn_paths[offset],
            a3mstc_path=self.a3mstc_paths[offset],
            a3mste_path=self.a3mste_paths[offset],
        )


@dataclass(frozen=True)
class _I3Block:
    start_index: int
    count: int
    surface_pressure: np.ndarray
    qv: np.ndarray
    temperature: np.ndarray
    dry_surface_pressure_hpa: np.ndarray
    wet_surface_pressure_hpa: np.ndarray
    paths: tuple[Path, ...]

    def contains_base(self, index: int) -> bool:
        return self.start_index <= index < self.start_index + self.count

    def contains_endpoint(self, index: int) -> bool:
        return self.start_index <= index < self.start_index + self.surface_pressure.shape[0]

    def field(self, index: int) -> _I3Fields:
        offset = index - self.start_index
        return _I3Fields(
            surface_pressure=self.surface_pressure[offset : offset + 1],
            qv=self.qv[offset : offset + 1],
            temperature=self.temperature[offset : offset + 1],
            dry_surface_pressure_hpa=self.dry_surface_pressure_hpa[offset : offset + 1],
            wet_surface_pressure_hpa=self.wet_surface_pressure_hpa[offset : offset + 1],
            path=self.paths[offset],
        )


class TransportForcingProvider:
    """Block-oriented MERRA2 forcing loader for GEOS-Chem-timed steps."""

    def __init__(
        self,
        met_root: str | Path,
        start: datetime,
        grid: TransportGrid,
        *,
        initial_met_time_index: int = 0,
        chunk_multiple: int = 1,
    ) -> None:
        if int(chunk_multiple) < 1:
            raise ValueError("meteorology chunk_multiple must be >= 1")
        self._met_root = Path(met_root)
        self._start = start
        self._start_day = datetime(start.year, start.month, start.day)
        self._grid = grid
        self._initial_met_time_index = int(initial_met_time_index)
        self._chunk_multiple = int(chunk_multiple)
        self._a1_block: _A1Block | None = None
        self._a3_block: _A3Block | None = None
        self._i3_block: _I3Block | None = None

    @property
    def start(self) -> datetime:
        return self._start

    def forcing_for_step(self, current: datetime, *, dt_s: float) -> TransportForcing:
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        elapsed_s = (current - self._start).total_seconds()
        if elapsed_s < 0:
            raise ValueError("current must be at or after start")

        hour_index = self._initial_met_time_index * 3 + int(elapsed_s // 3600.0)
        i3_start_index = self._initial_met_time_index + int(elapsed_s // 10800.0)
        i3_end_index = i3_start_index + 1
        restart_i3_index = self._initial_met_time_index + int((elapsed_s + dt_s) // 10800.0)

        a1 = self._a1_field(hour_index)
        a3 = self._a3_field(i3_start_index)
        i3_start = self._i3_field(i3_start_index, base=True)
        i3_end = self._i3_field(i3_end_index, base=False)
        if restart_i3_index == i3_start_index:
            i3_restart = i3_start
        elif restart_i3_index == i3_end_index:
            i3_restart = i3_end
        else:
            i3_restart = self._i3_field(restart_i3_index, base=False)

        seconds_into_i3_window = elapsed_s % 10800.0
        if seconds_into_i3_window + float(dt_s) > 10800.0 + 1.0e-9:
            raise ValueError(
                "transport step crosses a three-hour meteorology interpolation boundary"
            )
        start_fraction = seconds_into_i3_window / 10800.0
        end_fraction = (seconds_into_i3_window + float(dt_s)) / 10800.0
        midpoint_fraction = (seconds_into_i3_window + float(dt_s) / 2.0) / 10800.0
        dry_surface_start_endpoint = _i3_dry_surface_pressure_hpa(i3_start, self._grid)
        dry_surface_end_endpoint = _i3_dry_surface_pressure_hpa(i3_end, self._grid)
        wet_surface_start_endpoint = _i3_wet_surface_pressure_hpa(i3_start, self._grid)
        wet_surface_end_endpoint = _i3_wet_surface_pressure_hpa(i3_end, self._grid)
        return _assemble_transport_forcing(
            a1,
            a3,
            surface_pressure_start=_interpolate(i3_start.surface_pressure, i3_end.surface_pressure, start_fraction),
            surface_pressure_end=_interpolate(i3_start.surface_pressure, i3_end.surface_pressure, end_fraction),
            restart_surface_pressure=i3_restart.surface_pressure,
            wet_surface_pressure_start=_interpolate(wet_surface_start_endpoint, wet_surface_end_endpoint, start_fraction),
            wet_surface_pressure_end=_interpolate(wet_surface_start_endpoint, wet_surface_end_endpoint, end_fraction),
            restart_wet_surface_pressure=_i3_wet_surface_pressure_hpa(i3_restart, self._grid),
            dry_surface_pressure_start=_interpolate(dry_surface_start_endpoint, dry_surface_end_endpoint, start_fraction),
            dry_surface_pressure_end=_interpolate(dry_surface_start_endpoint, dry_surface_end_endpoint, end_fraction),
            restart_dry_surface_pressure=_i3_dry_surface_pressure_hpa(i3_restart, self._grid),
            i3_start_dry_surface_pressure=dry_surface_start_endpoint,
            i3_start_wet_surface_pressure=wet_surface_start_endpoint,
            i3_start_specific_humidity=i3_start.qv,
            specific_humidity=_interpolate(i3_start.qv, i3_end.qv, midpoint_fraction),
            restart_specific_humidity=i3_restart.qv,
            i3_start_temperature=i3_start.temperature,
            temperature=_interpolate(i3_start.temperature, i3_end.temperature, midpoint_fraction),
            restart_temperature=i3_restart.temperature,
            i3_path=i3_start.path,
            grid=self._grid,
            vertical_mapping=MERRA2_72_TO_47_MAPPING,
        )

    def _a1_field(self, index: int) -> _A1Fields:
        block_size = MERRA2_A1_NATURAL_BLOCK_RECORDS * self._chunk_multiple
        if self._a1_block is None or not self._a1_block.contains(index):
            self._a1_block = _load_a1_block(
                self._met_root,
                self._start_day,
                _block_start(index, block_size),
                block_size,
                self._grid,
            )
        return self._a1_block.field(index)

    def _a3_field(self, index: int) -> _A3Fields:
        block_size = MERRA2_A3_NATURAL_BLOCK_RECORDS * self._chunk_multiple
        if self._a3_block is None or not self._a3_block.contains(index):
            self._a3_block = _load_a3_block(
                self._met_root,
                self._start_day,
                _block_start(index, block_size),
                block_size,
                self._grid,
            )
        return self._a3_block.field(index)

    def _i3_field(self, index: int, *, base: bool) -> _I3Fields:
        block_size = MERRA2_I3_NATURAL_BLOCK_RECORDS * self._chunk_multiple
        in_block = False
        if self._i3_block is not None:
            in_block = self._i3_block.contains_base(index) if base else self._i3_block.contains_endpoint(index)
        if not in_block:
            self._i3_block = _load_i3_block(
                self._met_root,
                self._start_day,
                _block_start(index, block_size),
                block_size,
                self._grid,
            )
        return self._i3_block.field(index)


def load_transport_forcing(
    met_root: str | Path,
    timestamp: datetime,
    grid: TransportGrid,
    *,
    time_index: int = 0,
) -> TransportForcing:
    """Load MERRA2 forcing for one day and map 72 met levels to 47 levels."""

    met_root = Path(met_root)
    timestamp = datetime(timestamp.year, timestamp.month, timestamp.day)
    a1 = _load_a1_fields(met_root, timestamp, grid, int(time_index) * 3, None)
    a3 = _load_a3_fields(met_root, timestamp, grid, int(time_index), None)
    i3 = _load_i3_fields(met_root, timestamp, grid, int(time_index), None)
    dry_surface = _i3_dry_surface_pressure_hpa(i3, grid)
    wet_surface = _i3_wet_surface_pressure_hpa(i3, grid)
    return _assemble_transport_forcing(
        a1,
        a3,
        surface_pressure_start=i3.surface_pressure,
        surface_pressure_end=i3.surface_pressure,
        restart_surface_pressure=i3.surface_pressure,
        wet_surface_pressure_start=wet_surface,
        wet_surface_pressure_end=wet_surface,
        restart_wet_surface_pressure=wet_surface,
        dry_surface_pressure_start=dry_surface,
        dry_surface_pressure_end=dry_surface,
        restart_dry_surface_pressure=dry_surface,
        i3_start_dry_surface_pressure=dry_surface,
        i3_start_wet_surface_pressure=wet_surface,
        i3_start_specific_humidity=i3.qv,
        specific_humidity=i3.qv,
        restart_specific_humidity=i3.qv,
        i3_start_temperature=i3.temperature,
        temperature=i3.temperature,
        restart_temperature=i3.temperature,
        i3_path=i3.path,
        grid=grid,
        vertical_mapping=MERRA2_72_TO_47_MAPPING,
    )


def load_transport_forcing_for_step(
    met_root: str | Path,
    start: datetime,
    current: datetime,
    grid: TransportGrid,
    *,
    dt_s: float,
    initial_met_time_index: int = 0,
    chunk_multiple: int = 1,
) -> TransportForcing:
    """Load GEOS-Chem-timed forcing for one transport step.

    A1 records are hourly averages, A3 records are held for 3 hours, and I3
    endpoint fields are interpolated to the dynamic timestep.
    """

    provider = TransportForcingProvider(
        met_root,
        start,
        grid,
        initial_met_time_index=initial_met_time_index,
        chunk_multiple=chunk_multiple,
    )
    return provider.forcing_for_step(current, dt_s=dt_s)


def _assemble_transport_forcing(
    a1: _A1Fields,
    a3: _A3Fields,
    *,
    surface_pressure_start: np.ndarray,
    surface_pressure_end: np.ndarray,
    restart_surface_pressure: np.ndarray,
    wet_surface_pressure_start: np.ndarray,
    wet_surface_pressure_end: np.ndarray,
    restart_wet_surface_pressure: np.ndarray,
    dry_surface_pressure_start: np.ndarray,
    dry_surface_pressure_end: np.ndarray,
    restart_dry_surface_pressure: np.ndarray,
    i3_start_dry_surface_pressure: np.ndarray,
    i3_start_wet_surface_pressure: np.ndarray,
    i3_start_specific_humidity: np.ndarray,
    specific_humidity: np.ndarray,
    restart_specific_humidity: np.ndarray,
    i3_start_temperature: np.ndarray,
    temperature: np.ndarray,
    restart_temperature: np.ndarray,
    i3_path: Path,
    grid: TransportGrid,
    vertical_mapping: str,
) -> TransportForcing:

    return TransportForcing(
        u_m_s=a3.u,
        v_m_s=a3.v,
        omega_pa_s=a3.omega,
        surface_pressure_start_pa=surface_pressure_start,
        surface_pressure_pa=surface_pressure_end,
        restart_surface_pressure_pa=restart_surface_pressure,
        wet_surface_pressure_start_hpa=wet_surface_pressure_start,
        wet_surface_pressure_hpa=wet_surface_pressure_end,
        restart_wet_surface_pressure_hpa=restart_wet_surface_pressure,
        dry_surface_pressure_start_hpa=dry_surface_pressure_start,
        dry_surface_pressure_hpa=dry_surface_pressure_end,
        restart_dry_surface_pressure_hpa=restart_dry_surface_pressure,
        i3_start_wet_surface_pressure_hpa=i3_start_wet_surface_pressure,
        i3_start_dry_surface_pressure_hpa=i3_start_dry_surface_pressure,
        i3_start_specific_humidity_kg_kg=i3_start_specific_humidity,
        specific_humidity_kg_kg=specific_humidity,
        restart_specific_humidity_kg_kg=restart_specific_humidity,
        i3_start_temperature_k=i3_start_temperature,
        temperature_k=temperature,
        restart_temperature_k=restart_temperature,
        pbl_height_m=a1.pblh,
        sensible_heat_flux_w_m2=a1.hflux,
        latent_heat_flux_w_m2=a1.eflux,
        friction_velocity_m_s=a1.ustar,
        convective_mass_flux_kg_m2_s=a3.cmfmc[np.newaxis, 1:, :, :],
        convective_detrainment_kg_m2_s=a3.dtrain,
        convective_precip_prod_kg_kg_s=a3.dqrcu,
        convective_precip_reevap_kg_kg_s=a3.reevapcn,
        convective_ice_flux_kg_m2_s=a3.pficu[np.newaxis, 1:, :, :],
        convective_liquid_flux_kg_m2_s=a3.pflcu[np.newaxis, 1:, :, :],
        convective_precip_mm_day=a1.precccon * 86400.0,
        lat_deg=grid.lat_deg,
        lon_deg=grid.lon_deg,
        vertical_mapping=vertical_mapping,
        a1_path=a1.path,
        a3dyn_path=a3.a3dyn_path,
        a3mstc_path=a3.a3mstc_path,
        a3mste_path=a3.a3mste_path,
        i3_path=i3_path,
    )


def _record_day_and_index(start_day: datetime, absolute_index: int, records_per_day: int) -> tuple[datetime, int]:
    if absolute_index < 0:
        raise ValueError(f"met record index must be nonnegative, got {absolute_index}")
    return start_day + timedelta(days=absolute_index // records_per_day), absolute_index % records_per_day


def merra2_filename(timestamp: datetime, collection: str, grid: TransportGrid) -> str:
    return MERRA2_FILENAME.format(
        date=timestamp.strftime("%Y%m%d"),
        collection=collection,
        resolution=grid.horizontal_resolution,
    )


def _met_path(met_root: Path, timestamp: datetime, collection: str, grid: TransportGrid) -> Path:
    day_dir = met_root / f"{timestamp.year:04d}" / f"{timestamp.month:02d}"
    return day_dir / merra2_filename(timestamp, collection, grid)


def _block_start(index: int, block_size: int) -> int:
    if index < 0:
        raise ValueError(f"met record index must be nonnegative, got {index}")
    return index // block_size * block_size


def _load_a1_block(
    met_root: Path, start_day: datetime, start_index: int, count: int, grid: TransportGrid
) -> _A1Block:
    pblh = []
    hflux = []
    eflux = []
    ustar = []
    precccon = []
    paths = []
    for day, day_index, records in _record_spans(start_day, start_index, count, MERRA2_A1_RECORDS_PER_DAY):
        a1_path = _met_path(met_root, day, "A1", grid)
        with netCDF4.Dataset(a1_path) as a1:
            _assert_met_horizontal_shape(a1, grid, a1_path)
            selector = slice(day_index, day_index + records)
            pblh.append(np.asarray(a1.variables["PBLH"][selector], dtype=np.float64))
            hflux.append(np.asarray(a1.variables["HFLUX"][selector], dtype=np.float64))
            eflux.append(np.asarray(a1.variables["EFLUX"][selector], dtype=np.float64))
            ustar.append(np.asarray(a1.variables["USTAR"][selector], dtype=np.float64))
            precccon.append(np.asarray(a1.variables["PRECCON"][selector], dtype=np.float64))
        paths.extend([a1_path.resolve()] * records)
    return _A1Block(
        start_index=start_index,
        count=count,
        pblh=np.concatenate(pblh, axis=0),
        hflux=np.concatenate(hflux, axis=0),
        eflux=np.concatenate(eflux, axis=0),
        ustar=np.concatenate(ustar, axis=0),
        precccon=np.concatenate(precccon, axis=0),
        paths=tuple(paths),
    )


def _load_a3_block(
    met_root: Path, start_day: datetime, start_index: int, count: int, grid: TransportGrid
) -> _A3Block:
    u = []
    v = []
    omega = []
    dtrain = []
    dqrcu = []
    reevapcn = []
    cmfmc = []
    pficu = []
    pflcu = []
    a3dyn_paths = []
    a3mstc_paths = []
    a3mste_paths = []
    for day, day_index, records in _record_spans(start_day, start_index, count, MERRA2_A3_RECORDS_PER_DAY):
        a3dyn_path = _met_path(met_root, day, "A3dyn", grid)
        a3mstc_path = _met_path(met_root, day, "A3mstC", grid)
        a3mste_path = _met_path(met_root, day, "A3mstE", grid)
        selector = slice(day_index, day_index + records)
        with (
            netCDF4.Dataset(a3dyn_path) as a3dyn,
            netCDF4.Dataset(a3mstc_path) as a3mstc,
            netCDF4.Dataset(a3mste_path) as a3mste,
        ):
            _assert_met_horizontal_shape(a3dyn, grid, a3dyn_path)
            _assert_met_horizontal_shape(a3mstc, grid, a3mstc_path)
            _assert_met_horizontal_shape(a3mste, grid, a3mste_path)
            u.append(_map_met_levels_to_47(np.asarray(a3dyn.variables["U"][selector], dtype=np.float64)))
            v.append(_map_met_levels_to_47(np.asarray(a3dyn.variables["V"][selector], dtype=np.float64)))
            omega.append(_map_met_levels_to_47(np.asarray(a3dyn.variables["OMEGA"][selector], dtype=np.float64)))
            dtrain.append(_map_met_levels_to_47(np.asarray(a3dyn.variables["DTRAIN"][selector], dtype=np.float64)))
            dqrcu.append(_map_met_levels_to_47(np.asarray(a3mstc.variables["DQRCU"][selector], dtype=np.float64)))
            reevapcn.append(_map_met_levels_to_47(np.asarray(a3mstc.variables["REEVAPCN"][selector], dtype=np.float64)))
            cmfmc.append(_map_met_edges_to_48(np.asarray(a3mste.variables["CMFMC"][selector], dtype=np.float64)))
            pficu.append(_map_met_edges_to_48(np.asarray(a3mste.variables["PFICU"][selector], dtype=np.float64)))
            pflcu.append(_map_met_edges_to_48(np.asarray(a3mste.variables["PFLCU"][selector], dtype=np.float64)))
        a3dyn_paths.extend([a3dyn_path.resolve()] * records)
        a3mstc_paths.extend([a3mstc_path.resolve()] * records)
        a3mste_paths.extend([a3mste_path.resolve()] * records)
    return _A3Block(
        start_index=start_index,
        count=count,
        u=np.concatenate(u, axis=0),
        v=np.concatenate(v, axis=0),
        omega=np.concatenate(omega, axis=0),
        dtrain=np.concatenate(dtrain, axis=0),
        dqrcu=np.concatenate(dqrcu, axis=0),
        reevapcn=np.concatenate(reevapcn, axis=0),
        cmfmc=np.concatenate(cmfmc, axis=0),
        pficu=np.concatenate(pficu, axis=0),
        pflcu=np.concatenate(pflcu, axis=0),
        a3dyn_paths=tuple(a3dyn_paths),
        a3mstc_paths=tuple(a3mstc_paths),
        a3mste_paths=tuple(a3mste_paths),
    )


def _load_i3_block(met_root: Path, start_day: datetime, start_index: int, count: int, grid: TransportGrid) -> _I3Block:
    surface_pressure = []
    qv = []
    temperature = []
    paths = []
    read_count = count + 1
    for day, day_index, records in _record_spans(start_day, start_index, read_count, MERRA2_A3_RECORDS_PER_DAY):
        i3_path = _met_path(met_root, day, "I3", grid)
        selector = slice(day_index, day_index + records)
        with netCDF4.Dataset(i3_path) as i3:
            _assert_met_horizontal_shape(i3, grid, i3_path)
            surface_pressure.append(np.asarray(i3.variables["PS"][selector], dtype=np.float64))
            qv.append(_map_met_levels_to_47(np.asarray(i3.variables["QV"][selector], dtype=np.float64)))
            temperature.append(_map_met_levels_to_47(np.asarray(i3.variables["T"][selector], dtype=np.float64)))
        paths.extend([i3_path.resolve()] * records)
    surface_pressure_array = np.concatenate(surface_pressure, axis=0)
    qv_array = np.concatenate(qv, axis=0)
    return _I3Block(
        start_index=start_index,
        count=count,
        surface_pressure=surface_pressure_array,
        qv=qv_array,
        temperature=np.concatenate(temperature, axis=0),
        dry_surface_pressure_hpa=dry_surface_pressure_hpa(
            surface_pressure_array,
            qv_array,
            grid.hyai_hpa,
            grid.hybi,
            area_m2=grid.area_m2,
        ),
        wet_surface_pressure_hpa=wet_surface_pressure_hpa(surface_pressure_array, area_m2=grid.area_m2),
        paths=tuple(paths),
    )


def _record_spans(
    start_day: datetime,
    start_index: int,
    count: int,
    records_per_day: int,
) -> tuple[tuple[datetime, int, int], ...]:
    spans = []
    remaining = int(count)
    index = int(start_index)
    while remaining > 0:
        day, day_index = _record_day_and_index(start_day, index, records_per_day)
        records = min(remaining, records_per_day - day_index)
        spans.append((day, day_index, records))
        remaining -= records
        index += records
    return tuple(spans)


def _load_a1_fields(
    met_root: Path,
    timestamp: datetime,
    grid: TransportGrid,
    time_index: int,
    cache: dict[tuple[str, datetime, int], Any] | None,
) -> _A1Fields:
    key = ("A1", datetime(timestamp.year, timestamp.month, timestamp.day), int(time_index))
    if cache is not None and key in cache:
        return cache[key]
    a1_path = _met_path(met_root, timestamp, "A1", grid)
    with netCDF4.Dataset(a1_path) as a1:
        _assert_met_horizontal_shape(a1, grid, a1_path)
        fields = _A1Fields(
            pblh=_read_2d_time_slice(a1, "PBLH", time_index),
            hflux=_read_2d_time_slice(a1, "HFLUX", time_index),
            eflux=_read_2d_time_slice(a1, "EFLUX", time_index),
            ustar=_read_2d_time_slice(a1, "USTAR", time_index),
            precccon=np.asarray(a1.variables["PRECCON"][time_index : time_index + 1], dtype=np.float64),
            path=a1_path.resolve(),
        )
    if cache is not None:
        cache[key] = fields
    return fields


def _load_a3_fields(
    met_root: Path,
    timestamp: datetime,
    grid: TransportGrid,
    time_index: int,
    cache: dict[tuple[str, datetime, int], Any] | None,
) -> _A3Fields:
    key = ("A3", datetime(timestamp.year, timestamp.month, timestamp.day), int(time_index))
    if cache is not None and key in cache:
        return cache[key]
    a3dyn_path = _met_path(met_root, timestamp, "A3dyn", grid)
    a3mstc_path = _met_path(met_root, timestamp, "A3mstC", grid)
    a3mste_path = _met_path(met_root, timestamp, "A3mstE", grid)
    with (
        netCDF4.Dataset(a3dyn_path) as a3dyn,
        netCDF4.Dataset(a3mstc_path) as a3mstc,
        netCDF4.Dataset(a3mste_path) as a3mste,
    ):
        _assert_met_horizontal_shape(a3dyn, grid, a3dyn_path)
        _assert_met_horizontal_shape(a3mstc, grid, a3mstc_path)
        _assert_met_horizontal_shape(a3mste, grid, a3mste_path)
        fields = _A3Fields(
            u=_map_met_levels_to_47(_read_3d_time_slice(a3dyn, "U", time_index)),
            v=_map_met_levels_to_47(_read_3d_time_slice(a3dyn, "V", time_index)),
            omega=_map_met_levels_to_47(_read_3d_time_slice(a3dyn, "OMEGA", time_index)),
            dtrain=_map_met_levels_to_47(_read_3d_time_slice(a3dyn, "DTRAIN", time_index)),
            dqrcu=_map_met_levels_to_47(_read_3d_time_slice(a3mstc, "DQRCU", time_index)),
            reevapcn=_map_met_levels_to_47(_read_3d_time_slice(a3mstc, "REEVAPCN", time_index)),
            cmfmc=_map_met_edges_to_48(np.asarray(a3mste.variables["CMFMC"][time_index], dtype=np.float64)),
            pficu=_map_met_edges_to_48(np.asarray(a3mste.variables["PFICU"][time_index], dtype=np.float64)),
            pflcu=_map_met_edges_to_48(np.asarray(a3mste.variables["PFLCU"][time_index], dtype=np.float64)),
            a3dyn_path=a3dyn_path.resolve(),
            a3mstc_path=a3mstc_path.resolve(),
            a3mste_path=a3mste_path.resolve(),
        )
    if cache is not None:
        cache[key] = fields
    return fields


def _load_i3_fields(
    met_root: Path,
    timestamp: datetime,
    grid: TransportGrid,
    time_index: int,
    cache: dict[tuple[str, datetime, int], Any] | None,
) -> _I3Fields:
    key = ("I3", datetime(timestamp.year, timestamp.month, timestamp.day), int(time_index))
    if cache is not None and key in cache:
        return cache[key]
    i3_path = _met_path(met_root, timestamp, "I3", grid)
    with netCDF4.Dataset(i3_path) as i3:
        _assert_met_horizontal_shape(i3, grid, i3_path)
        surface_pressure = np.asarray(i3.variables["PS"][time_index : time_index + 1], dtype=np.float64)
        qv = _map_met_levels_to_47(_read_3d_time_slice(i3, "QV", time_index))
        fields = _I3Fields(
            surface_pressure=surface_pressure,
            qv=qv,
            temperature=_map_met_levels_to_47(_read_3d_time_slice(i3, "T", time_index)),
            dry_surface_pressure_hpa=dry_surface_pressure_hpa(
                surface_pressure,
                qv,
                grid.hyai_hpa,
                grid.hybi,
                area_m2=grid.area_m2,
            ),
            wet_surface_pressure_hpa=wet_surface_pressure_hpa(surface_pressure, area_m2=grid.area_m2),
            path=i3_path.resolve(),
        )
    if cache is not None:
        cache[key] = fields
    return fields


def _interpolate(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    start_array = np.asarray(start, dtype=np.float64)
    end_array = np.asarray(end, dtype=np.float64)
    return start_array + (end_array - start_array) * float(fraction)


def _i3_dry_surface_pressure_hpa(i3: _I3Fields, grid: TransportGrid) -> np.ndarray:
    value = getattr(i3, "dry_surface_pressure_hpa", None)
    if value is not None:
        return value
    return dry_surface_pressure_hpa(i3.surface_pressure, i3.qv, grid.hyai_hpa, grid.hybi, area_m2=grid.area_m2)


def _i3_wet_surface_pressure_hpa(i3: _I3Fields, grid: TransportGrid) -> np.ndarray:
    value = getattr(i3, "wet_surface_pressure_hpa", None)
    if value is not None:
        return value
    return wet_surface_pressure_hpa(i3.surface_pressure, area_m2=grid.area_m2)


def _read_3d_time_slice(dataset: netCDF4.Dataset, variable_name: str, time_index: int) -> np.ndarray:
    return np.asarray(dataset.variables[variable_name][time_index : time_index + 1], dtype=np.float64)

def _read_2d_time_slice(dataset: netCDF4.Dataset, variable_name: str, time_index: int) -> np.ndarray:
    return np.asarray(dataset.variables[variable_name][time_index : time_index + 1], dtype=np.float64)

def _map_met_levels_to_47(data: np.ndarray) -> np.ndarray:
    if data.shape[1] == MODEL_LEVELS:
        return data
    if data.shape[1] == 72:
        mapped = np.empty((data.shape[0], MODEL_LEVELS, data.shape[2], data.shape[3]), dtype=np.float64)
        mapped[:, :36, :, :] = data[:, :36, :, :]
        for target_level, (start, end) in enumerate(MERRA2_72_TO_47_GROUPS, start=36):
            weights = MERRA2_72_AP_HPA[start:end] - MERRA2_72_AP_HPA[start + 1 : end + 1]
            mapped[:, target_level, :, :] = np.sum(
                data[:, start:end, :, :] * weights[np.newaxis, :, np.newaxis, np.newaxis],
                axis=1,
            ) / np.sum(weights)
        return mapped
    raise ValueError(f"cannot map {data.shape[1]} met levels to {MODEL_LEVELS}")


def _map_met_edges_to_48(data: np.ndarray) -> np.ndarray:
    edges = np.asarray(data, dtype=np.float64)
    if edges.ndim == 3:
        edge_axis = 0
    elif edges.ndim == 4:
        edge_axis = 1
    else:
        raise ValueError(f"edge field must be 3-D or 4-D, found {edges.shape}")
    if edges.shape[edge_axis] == MODEL_LEVELS + 1:
        return edges
    if edges.shape[edge_axis] != 73:
        raise ValueError(f"cannot map {edges.shape[edge_axis]} met edges to {MODEL_LEVELS + 1} target edges")
    target_indices = np.array(
        list(range(37)) + [38, 40, 42, 44, 48, 52, 56, 60, 64, 68, 72],
        dtype=np.int64,
    )
    return np.take(edges, target_indices, axis=edge_axis)


def _assert_met_horizontal_shape(
    dataset: netCDF4.Dataset, grid: TransportGrid, path: Path
) -> None:
    expected = (grid.lat_deg.size, grid.lon_deg.size)
    actual = (len(dataset.dimensions["lat"]), len(dataset.dimensions["lon"]))
    if actual != expected:
        raise ValueError(f"{path} horizontal grid {actual} does not match template grid {expected}")
