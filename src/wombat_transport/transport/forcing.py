from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import netCDF4
import numpy as np

from wombat_transport.io import FIXED_GRID

MERRA2_FILENAME = "MERRA2.{date}.{collection}.2x25.nc4"

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

@dataclass(frozen=True)
class TransportForcing:
    """Meteorological forcing mapped onto the prototype 47-level grid."""

    u_m_s: np.ndarray
    v_m_s: np.ndarray
    omega_pa_s: np.ndarray
    surface_pressure_pa: np.ndarray
    specific_humidity_kg_kg: np.ndarray
    temperature_k: np.ndarray
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

def load_transport_forcing(
    met_root: str | Path,
    timestamp: datetime,
    template_path: str | Path,
    *,
    time_index: int = 0,
) -> TransportForcing:
    """Load MERRA2 forcing for one day and map 72 met levels to 47 levels."""

    met_root = Path(met_root)
    day_dir = met_root / f"{timestamp.year:04d}" / f"{timestamp.month:02d}"
    date = timestamp.strftime("%Y%m%d")
    a1_path = day_dir / MERRA2_FILENAME.format(date=date, collection="A1")
    a3dyn_path = day_dir / MERRA2_FILENAME.format(date=date, collection="A3dyn")
    a3mstc_path = day_dir / MERRA2_FILENAME.format(date=date, collection="A3mstC")
    a3mste_path = day_dir / MERRA2_FILENAME.format(date=date, collection="A3mstE")
    i3_path = day_dir / MERRA2_FILENAME.format(date=date, collection="I3")
    a1_convection_time_index = int(time_index) * 3

    with (
        netCDF4.Dataset(a1_path) as a1,
        netCDF4.Dataset(a3dyn_path) as a3dyn,
        netCDF4.Dataset(a3mstc_path) as a3mstc,
        netCDF4.Dataset(a3mste_path) as a3mste,
        netCDF4.Dataset(i3_path) as i3,
        netCDF4.Dataset(template_path) as template,
    ):
        lat = np.asarray(template.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(template.variables["lon"][:], dtype=np.float64)
        u = _read_3d_time_slice(a3dyn, "U", time_index)
        v = _read_3d_time_slice(a3dyn, "V", time_index)
        omega = _read_3d_time_slice(a3dyn, "OMEGA", time_index)
        qv = _read_3d_time_slice(i3, "QV", time_index)
        temperature = _read_3d_time_slice(i3, "T", time_index)
        surface_pressure = np.asarray(i3.variables["PS"][time_index : time_index + 1], dtype=np.float64)
        pblh = _read_2d_time_slice(a1, "PBLH", time_index)
        hflux = _read_2d_time_slice(a1, "HFLUX", time_index)
        eflux = _read_2d_time_slice(a1, "EFLUX", time_index)
        ustar = _read_2d_time_slice(a1, "USTAR", time_index)
        dtrain = _read_3d_time_slice(a3dyn, "DTRAIN", time_index)
        dqrcu = _read_3d_time_slice(a3mstc, "DQRCU", time_index)
        reevapcn = _read_3d_time_slice(a3mstc, "REEVAPCN", time_index)
        cmfmc = _map_met_edges_to_48(np.asarray(a3mste.variables["CMFMC"][time_index], dtype=np.float64))
        pficu = _map_met_edges_to_48(np.asarray(a3mste.variables["PFICU"][time_index], dtype=np.float64))
        pflcu = _map_met_edges_to_48(np.asarray(a3mste.variables["PFLCU"][time_index], dtype=np.float64))
        precccon = np.asarray(
            a1.variables["PRECCON"][a1_convection_time_index : a1_convection_time_index + 1],
            dtype=np.float64,
        )

    return TransportForcing(
        u_m_s=_map_met_levels_to_47(u),
        v_m_s=_map_met_levels_to_47(v),
        omega_pa_s=_map_met_levels_to_47(omega),
        surface_pressure_pa=surface_pressure,
        specific_humidity_kg_kg=_map_met_levels_to_47(qv),
        temperature_k=_map_met_levels_to_47(temperature),
        pbl_height_m=pblh,
        sensible_heat_flux_w_m2=hflux,
        latent_heat_flux_w_m2=eflux,
        friction_velocity_m_s=ustar,
        convective_mass_flux_kg_m2_s=cmfmc[np.newaxis, 1:, :, :],
        convective_detrainment_kg_m2_s=_map_met_levels_to_47(dtrain),
        convective_precip_prod_kg_kg_s=_map_met_levels_to_47(dqrcu),
        convective_precip_reevap_kg_kg_s=_map_met_levels_to_47(reevapcn),
        convective_ice_flux_kg_m2_s=pficu[np.newaxis, 1:, :, :],
        convective_liquid_flux_kg_m2_s=pflcu[np.newaxis, 1:, :, :],
        convective_precip_mm_day=precccon * 86400.0,
        lat_deg=lat,
        lon_deg=lon,
        vertical_mapping=MERRA2_72_TO_47_MAPPING,
        a1_path=a1_path.resolve(),
        a3dyn_path=a3dyn_path.resolve(),
        a3mstc_path=a3mstc_path.resolve(),
        a3mste_path=a3mste_path.resolve(),
        i3_path=i3_path.resolve(),
    )

def _read_3d_time_slice(dataset: netCDF4.Dataset, variable_name: str, time_index: int) -> np.ndarray:
    return np.asarray(dataset.variables[variable_name][time_index : time_index + 1], dtype=np.float64)

def _read_2d_time_slice(dataset: netCDF4.Dataset, variable_name: str, time_index: int) -> np.ndarray:
    return np.asarray(dataset.variables[variable_name][time_index : time_index + 1], dtype=np.float64)

def _map_met_levels_to_47(data: np.ndarray) -> np.ndarray:
    if data.shape[1] == FIXED_GRID["lev"]:
        return data
    if data.shape[1] == 72:
        mapped = np.empty((data.shape[0], FIXED_GRID["lev"], data.shape[2], data.shape[3]), dtype=np.float64)
        mapped[:, :36, :, :] = data[:, :36, :, :]
        for target_level, (start, end) in enumerate(MERRA2_72_TO_47_GROUPS, start=36):
            weights = MERRA2_72_AP_HPA[start:end] - MERRA2_72_AP_HPA[start + 1 : end + 1]
            mapped[:, target_level, :, :] = np.sum(
                data[:, start:end, :, :] * weights[np.newaxis, :, np.newaxis, np.newaxis],
                axis=1,
            ) / np.sum(weights)
        return mapped
    raise ValueError(f"cannot map {data.shape[1]} met levels to {FIXED_GRID['lev']}")


def _map_met_edges_to_48(data: np.ndarray) -> np.ndarray:
    edges = np.asarray(data, dtype=np.float64)
    if edges.ndim != 3:
        raise ValueError(f"edge field must be 3-D (edge, lat, lon), found {edges.shape}")
    if edges.shape[0] == FIXED_GRID["lev"] + 1:
        return edges
    if edges.shape[0] != 73:
        raise ValueError(f"cannot map {edges.shape[0]} met edges to {FIXED_GRID['lev'] + 1} target edges")
    target_indices = np.array(
        list(range(37)) + [38, 40, 42, 44, 48, 52, 56, 60, 64, 68, 72],
        dtype=np.int64,
    )
    return edges[target_indices]
