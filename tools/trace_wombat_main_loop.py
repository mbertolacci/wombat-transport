#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np

from wombat_transport.constants import AIRMW_G_PER_MOL
from wombat_transport.emissions import EmissionsOperator
from wombat_transport.fields import canonical_time_slice
from wombat_transport.grid import load_transport_grid
from wombat_transport.io import initialize_tracers
from wombat_transport.run_config import (
    emissions_timestep_s,
    load_run_config,
    meteorology_initial_time_index,
    meteorology_root,
    simulation_start,
    transport_timestep_s,
)
from wombat_transport.runner import (
    _initial_dry_air_mass,
    _is_time_for_emissions,
    emitted_mass_by_tracer_for_step,
    has_invalid_emissions,
)
from wombat_transport.species import load_species_database
from wombat_transport.transport.forcing import ForcingRecordCache, load_transport_forcing_for_step, prune_forcing_record_cache
from wombat_transport.transport.driver import TransportStepDiagnostics, trace_transport_one_step


BOUNDARIES = (
    "step_start",
    "before_do_transport",
    "before_tpcore_fvdas",
    "after_do_transport",
    "after_setup_wetscav",
    "after_compute_pbl_height",
    "after_emissions_run_phase2",
    "after_compute_sflx_for_vdiff",
    "before_do_mixing",
    "before_do_vdiff",
    "after_do_vdiff",
    "before_do_tend",
    "after_do_tend",
    "after_do_mixing",
    "before_do_convection",
    "after_do_convection",
    "after_history_record",
)


@dataclass(frozen=True)
class TraceColumns:
    lat_indices: tuple[int, ...]
    lon_indices: tuple[int, ...]
    lat_values: tuple[float, ...]
    lon_values: tuple[float, ...]
    nlat: int
    nlon: int


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write a Wombat transport trace with GEOS-Chem main-loop-style boundary names."
    )
    parser.add_argument("run_config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Optional directory for full-grid tracer concentration snapshots.",
    )
    parser.add_argument(
        "--snapshot-step",
        action="append",
        type=int,
        default=[],
        help="0-based transport step to snapshot. May be repeated.",
    )
    parser.add_argument(
        "--column",
        action="append",
        default=[],
        metavar="LAT,LON",
        help="Column center to trace, e.g. 26,87.5. May be repeated.",
    )
    parser.add_argument("--max-tracers", type=int, default=None)
    args = parser.parse_args()

    config = load_run_config(args.run_config)
    grid = load_transport_grid(config.grid_template)
    species = load_species_database(config.species_database)
    state = initialize_tracers(config.initial_restart, config.species_database, template_path=config.grid_template)
    if args.max_tracers is not None:
        count = int(args.max_tracers)
        state = state.__class__(
            names=state.names[:count],
            data=state.data[..., :count],
            units=state.units[:count],
            coords=state.coords,
        )
        species = species[:count]
    surface_flux_to_vmr_factor = _surface_flux_to_vmr_factor(state.names, species)

    columns = _resolve_columns(args.column, grid.lat_deg, grid.lon_deg)
    met_root = meteorology_root(config)
    start = simulation_start(config)
    transport_dt_s = float(transport_timestep_s(config))
    emissions_dt_s = float(emissions_timestep_s(config))
    emissions_operator = _load_emissions_operator(config, species, grid)

    forcing_cache: ForcingRecordCache = {}
    first_forcing = load_transport_foring_cached(
        forcing_cache,
        met_root,
        start,
        start,
        grid,
        transport_dt_s,
        meteorology_initial_time_index(config),
    )
    dry_air_mass = _initial_dry_air_mass(config, first_forcing, grid)
    active_emissions = None
    emitted_mass = np.zeros(len(species), dtype=np.float64)
    records: list[dict[str, np.ndarray | datetime | int | str]] = []

    for step in range(args.steps):
        current = start + timedelta(seconds=step * transport_dt_s)
        forcing = load_transport_foring_cached(
            forcing_cache,
            met_root,
            start,
            current,
            grid,
            transport_dt_s,
            meteorology_initial_time_index(config),
        )
        elapsed_s = int(round((current - start).total_seconds()))
        emissions_was_refreshed = False
        if _is_time_for_emissions(elapsed_s, transport_dt_s, emissions_dt_s):
            midpoint = current + timedelta(seconds=emissions_dt_s / 2.0)
            active_emissions = emissions_operator.evaluate(midpoint)
            if args.max_tracers is not None:
                active_emissions = active_emissions.__class__(
                    names=active_emissions.names[: len(species)],
                    data=active_emissions.data[..., : len(species)],
                    units=active_emissions.units[: len(species)],
                    coords=active_emissions.coords,
                )
            if has_invalid_emissions(active_emissions):
                raise ValueError(f"configured emissions contain invalid values at {midpoint:%Y-%m-%d %H:%M}")
            emitted_mass += emitted_mass_by_tracer_for_step(active_emissions, emissions_dt_s)
            emissions_was_refreshed = True

        trace = trace_transport_one_step(
            state,
            forcing,
            grid,
            dt_s=transport_dt_s,
            active_emissions=active_emissions,
            surface_flux_to_vmr_factor=surface_flux_to_vmr_factor,
            dry_air_mass_kg=dry_air_mass,
        )
        if args.snapshot_dir is not None and step in set(args.snapshot_step):
            _write_snapshots_for_step(args.snapshot_dir, step, state, trace, forcing.dry_surface_pressure_hpa[0])
        records.extend(
            _records_for_step(
                step,
                current,
                state,
                trace,
                columns,
                emissions_was_refreshed=emissions_was_refreshed,
                tpcore_p2_hpa=forcing.dry_surface_pressure_hpa[0],
            )
        )
        state = trace.result.state
        dry_air_mass = trace.result.dry_air_mass_kg

    _write_trace(args.output, records, columns, state.names, grid.lev, emitted_mass)
    print(f"wrote_wombat_main_loop_trace: {args.output}")
    return 0


def load_transport_foring_cached(
    cache: ForcingRecordCache,
    met_root: Path,
    start: datetime,
    current: datetime,
    grid,
    dt_s: float,
    initial_met_time_index: int,
):
    forcing = load_transport_forcing_for_step(
        met_root,
        start,
        current,
        grid,
        dt_s=dt_s,
        initial_met_time_index=initial_met_time_index,
        cache=cache,
    )
    prune_forcing_record_cache(cache)
    return forcing


def _load_emissions_operator(config, species, grid) -> EmissionsOperator:
    if isinstance(config.emissions, str):
        return EmissionsOperator.from_yaml(config.emissions, root=config.root, species=species, grid=grid)
    return EmissionsOperator.from_mapping(dict(config.emissions), root=config.root, species=species, grid=grid)


def _surface_flux_to_vmr_factor(names: tuple[str, ...], species) -> np.ndarray:
    species_by_name = {item.name: item for item in species}
    return np.asarray([AIRMW_G_PER_MOL / species_by_name[name].molecular_weight_g for name in names], dtype=np.float64)


def _resolve_columns(raw_columns: list[str], lat: np.ndarray, lon: np.ndarray) -> TraceColumns:
    if not raw_columns:
        raw_columns = ["26,87.5", "30,120", "34,112.5", "40,117.5", "-12,12.5", "-6,142.5"]
    lat_indices: list[int] = []
    lon_indices: list[int] = []
    lat_values: list[float] = []
    lon_values: list[float] = []
    for raw in raw_columns:
        try:
            lat_value, lon_value = (float(item.strip()) for item in raw.split(",", 1))
        except ValueError as exc:
            raise ValueError(f"--column must be LAT,LON, got {raw!r}") from exc
        lat_index = int(np.argmin(np.abs(lat - lat_value)))
        lon_index = int(np.argmin(np.abs(lon - lon_value)))
        lat_indices.append(lat_index)
        lon_indices.append(lon_index)
        lat_values.append(float(lat[lat_index]))
        lon_values.append(float(lon[lon_index]))
    return TraceColumns(tuple(lat_indices), tuple(lon_indices), tuple(lat_values), tuple(lon_values), len(lat), len(lon))


def _records_for_step(
    step: int,
    timestamp: datetime,
    initial_state,
    trace: TransportStepDiagnostics,
    columns: TraceColumns,
    *,
    emissions_was_refreshed: bool,
    tpcore_p2_hpa: np.ndarray,
) -> list[dict[str, np.ndarray | datetime | int | str]]:
    records = [
        _record(step, timestamp, "step_start", canonical_time_slice(initial_state.data), columns),
        _record(step, timestamp, "before_do_transport", canonical_time_slice(initial_state.data), columns),
        _record(
            step,
            timestamp,
            "before_tpcore_fvdas",
            canonical_time_slice(initial_state.data),
            columns,
            p_tp1_hpa=_tpcore_initial_surface_pressure_hpa(trace.tpcore_state),
            p_tp2_hpa=tpcore_p2_hpa,
            xmass_hpa=trace.tpcore_state.xmass_hpa,
            ymass_hpa=trace.tpcore_state.ymass_hpa,
        ),
        _record(step, timestamp, "after_do_transport", trace.tpcore_state.tracer_conc_after, columns),
        _record(step, timestamp, "after_setup_wetscav", trace.tpcore_state.tracer_conc_after, columns),
        _record(
            step,
            timestamp,
            "after_compute_pbl_height",
            trace.tpcore_state.tracer_conc_after,
            columns,
            bxheight=trace.vdiff_input.bxheight_m,
            pbl_top_m=trace.vdiff_input.pbl_top_m,
        ),
        _record(
            step,
            timestamp,
            "after_compute_sflx_for_vdiff",
            trace.vdiff_input.tracer_conc,
            columns,
            surface_flux=trace.vdiff_input.surface_flux_kg_m2_s,
        ),
        _record(
            step,
            timestamp,
            "before_do_mixing",
            trace.vdiff_input.tracer_conc,
            columns,
            surface_flux=trace.vdiff_input.surface_flux_kg_m2_s,
            delp_dry_hpa=trace.result.delp_dry_hpa,
            dry_air_mass_kg=trace.vdiff_input.dry_air_mass_kg,
            bxheight=trace.vdiff_input.bxheight_m,
            sphu=trace.vdiff_input.specific_humidity_kg_kg,
            temperature=trace.vdiff_input.temperature_k,
            pbl_top_m=trace.vdiff_input.pbl_top_m,
            **_vdiff_input_trace_fields(trace.vdiff_input),
        ),
        _record(
            step,
            timestamp,
            "before_do_vdiff",
            trace.vdiff_input.tracer_conc,
            columns,
            surface_flux=trace.vdiff_input.surface_flux_kg_m2_s,
            delp_dry_hpa=trace.result.delp_dry_hpa,
            dry_air_mass_kg=trace.vdiff_input.dry_air_mass_kg,
            bxheight=trace.vdiff_input.bxheight_m,
            sphu=trace.vdiff_input.specific_humidity_kg_kg,
            temperature=trace.vdiff_input.temperature_k,
            pbl_top_m=trace.vdiff_input.pbl_top_m,
            **_vdiff_input_trace_fields(trace.vdiff_input),
        ),
        _record(
            step,
            timestamp,
            "after_do_vdiff",
            trace.vdiff_output.tracer_conc,
            columns,
            surface_flux=trace.vdiff_input.surface_flux_kg_m2_s,
            bxheight=trace.vdiff_input.bxheight_m,
            sphu=trace.vdiff_output.specific_humidity_kg_kg,
            temperature=trace.vdiff_input.temperature_k,
            pbl_top_m=trace.vdiff_input.pbl_top_m,
            **_vdiff_input_trace_fields(trace.vdiff_input),
        ),
        _record(step, timestamp, "before_do_tend", trace.vdiff_output.tracer_conc, columns),
        _record(step, timestamp, "after_do_tend", trace.vdiff_output.tracer_conc, columns),
        _record(step, timestamp, "after_do_mixing", trace.vdiff_output.tracer_conc, columns),
        _record(
            step,
            timestamp,
            "before_do_convection",
            trace.convection_input.tracer_conc,
            columns,
            bxheight=trace.convection_input.bxheight_m,
            temperature=trace.convection_input.temperature_k,
            cmfmc=trace.convection_input.cmfmc_kg_m2_s,
            dtrain=trace.convection_input.dtrain_kg_m2_s,
            dqrcu=trace.convection_input.dqrcu_kg_kg_s,
            reevapcn=trace.convection_input.reevapcn_kg_kg_s,
            pficu=trace.convection_input.pficu_kg_m2_s,
            pflcu=trace.convection_input.pflcu_kg_m2_s,
            precccon=trace.convection_input.precccon_mm_day,
        ),
        _record(step, timestamp, "after_do_convection", trace.convection_output.tracer_conc, columns),
        _record(step, timestamp, "after_history_record", canonical_time_slice(trace.result.state.data), columns),
    ]
    if emissions_was_refreshed:
        records.insert(
            5,
            _record(
                step,
                timestamp,
                "after_emissions_run_phase2",
                trace.tpcore_state.tracer_conc_after,
                columns,
                surface_flux=trace.vdiff_input.surface_flux_kg_m2_s,
            ),
        )
    return records


def _write_snapshots_for_step(
    snapshot_dir: Path,
    step: int,
    initial_state,
    trace: TransportStepDiagnostics,
    tpcore_p2_hpa: np.ndarray,
) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshots = {
        "step_start": canonical_time_slice(initial_state.data),
        "before_do_transport": canonical_time_slice(initial_state.data),
        "before_tpcore_fvdas": canonical_time_slice(initial_state.data),
        "after_do_transport": trace.tpcore_state.tracer_conc_after,
        "after_setup_wetscav": trace.tpcore_state.tracer_conc_after,
        "after_compute_pbl_height": trace.tpcore_state.tracer_conc_after,
        "after_compute_sflx_for_vdiff": trace.vdiff_input.tracer_conc,
        "before_do_mixing": trace.vdiff_input.tracer_conc,
        "before_do_vdiff": trace.vdiff_input.tracer_conc,
        "after_do_vdiff": trace.vdiff_output.tracer_conc,
        "before_do_tend": trace.vdiff_output.tracer_conc,
        "after_do_tend": trace.vdiff_output.tracer_conc,
        "after_do_mixing": trace.vdiff_output.tracer_conc,
        "before_do_convection": trace.convection_input.tracer_conc,
        "after_do_convection": trace.convection_output.tracer_conc,
        "after_history_record": canonical_time_slice(trace.result.state.data),
    }
    for boundary, tracer_top in snapshots.items():
        _write_snapshot(snapshot_dir / f"{boundary}_{step:06d}.bin", tracer_top)
    _write_tpcore_input_snapshot(
        snapshot_dir / f"before_tpcore_fvdas_inputs_{step:06d}.bin",
        _tpcore_initial_surface_pressure_hpa(trace.tpcore_state),
        tpcore_p2_hpa,
        trace.tpcore_state.xmass_hpa,
        trace.tpcore_state.ymass_hpa,
    )


def _write_snapshot(path: Path, tracer_top: np.ndarray) -> None:
    array = np.asarray(tracer_top, dtype=np.float64)
    if array.ndim != 4:
        raise ValueError(f"expected tracer field (lev_top, lat, lon, tracer), got {array.shape}")
    lon_lat_lev_tracer = np.transpose(array[::-1, :, :, :], (2, 1, 0, 3))
    dims = np.asarray(lon_lat_lev_tracer.shape, dtype=np.int32)
    with path.open("wb") as handle:
        dims.tofile(handle)
        np.asarray(lon_lat_lev_tracer, dtype="<f8").ravel(order="F").tofile(handle)


def _write_tpcore_input_snapshot(
    path: Path,
    p_tp1_hpa: np.ndarray,
    p_tp2_hpa: np.ndarray,
    xmass_hpa: np.ndarray,
    ymass_hpa: np.ndarray,
) -> None:
    p1 = np.transpose(np.asarray(p_tp1_hpa, dtype=np.float64), (1, 0))
    p2 = np.transpose(np.asarray(p_tp2_hpa, dtype=np.float64), (1, 0))
    xmass = np.transpose(np.asarray(xmass_hpa, dtype=np.float64)[::-1], (2, 1, 0))
    ymass = np.transpose(np.asarray(ymass_hpa, dtype=np.float64)[::-1], (2, 1, 0))
    dims = np.asarray((p1.shape[0], p1.shape[1], xmass.shape[2]), dtype=np.int32)
    with path.open("wb") as handle:
        dims.tofile(handle)
        for values in (p1, p2, xmass, ymass):
            np.asarray(values, dtype="<f8").ravel(order="F").tofile(handle)


def _vdiff_input_trace_fields(vdiff_input) -> dict[str, np.ndarray]:
    return {
        "u_m_s": vdiff_input.u_m_s,
        "v_m_s": vdiff_input.v_m_s,
        "pmid_hpa": vdiff_input.pmid_hpa,
        "pedge_lower_hpa": _surfaceward_edge_by_bottom_level(vdiff_input.pedge_hpa),
        "pedge_upper_hpa": _topward_edge_by_bottom_level(vdiff_input.pedge_hpa),
        "tv_k": vdiff_input.virtual_temperature_k,
        "hflux_w_m2": vdiff_input.hflux_w_m2,
        "eflux_w_m2": vdiff_input.eflux_w_m2,
        "ustar_m_s": vdiff_input.ustar_m_s,
    }


def _tpcore_initial_surface_pressure_hpa(tpcore_state) -> np.ndarray:
    top_edge = tpcore_state.surface_pressure_hpa - np.sum(tpcore_state.delp2_hpa, axis=0)
    return top_edge + np.sum(tpcore_state.delp1_hpa, axis=0)


def _surfaceward_edge_by_bottom_level(pedge_top: np.ndarray) -> np.ndarray:
    edge = np.asarray(pedge_top, dtype=np.float64)
    return np.ascontiguousarray(edge[1:])


def _topward_edge_by_bottom_level(pedge_top: np.ndarray) -> np.ndarray:
    edge = np.asarray(pedge_top, dtype=np.float64)
    return np.ascontiguousarray(edge[:-1])


def _record(
    step: int,
    timestamp: datetime,
    boundary: str,
    tracer_top: np.ndarray,
    columns: TraceColumns,
    **fields,
) -> dict[str, np.ndarray | datetime | int | str]:
    return {
        "step": step,
        "timestamp": timestamp,
        "boundary": boundary,
        "tracer": _sample_top4(tracer_top, columns),
        **{name: _sample_optional(value, columns) for name, value in fields.items()},
    }


def _sample_top4(values: np.ndarray, columns: TraceColumns) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 4:
        raise ValueError(f"expected tracer field (lev_top, lat, lon, tracer), got {array.shape}")
    return np.stack(
        [array[::-1, lat_index, lon_index, :] for lat_index, lon_index in zip(columns.lat_indices, columns.lon_indices)],
        axis=1,
    )


def _sample_optional(values, columns: TraceColumns) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 4:
        return _sample_top4(array, columns)
    if array.ndim == 3:
        if array.shape[:2] == (columns.nlat, columns.nlon):
            return np.stack(
                [array[lat_index, lon_index, :] for lat_index, lon_index in zip(columns.lat_indices, columns.lon_indices)],
                axis=0,
            )
        if array.shape[1:] != (columns.nlat, columns.nlon):
            raise ValueError(f"cannot infer 3-D field layout for shape {array.shape}")
        return np.stack(
            [array[::-1, lat_index, lon_index] for lat_index, lon_index in zip(columns.lat_indices, columns.lon_indices)],
            axis=1,
        )
    if array.ndim == 2:
        if array.shape != (columns.nlat, columns.nlon):
            raise ValueError(f"cannot infer 2-D field layout for shape {array.shape}")
        return np.asarray([array[lat_index, lon_index] for lat_index, lon_index in zip(columns.lat_indices, columns.lon_indices)])
    raise ValueError(f"cannot sample array with shape {array.shape}")


def _write_trace(
    path: Path,
    records: list[dict[str, np.ndarray | datetime | int | str]],
    columns: TraceColumns,
    tracer_names: tuple[str, ...],
    lev: np.ndarray,
    emitted_mass: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nrecord = len(records)
    nlev = len(lev)
    ncol = len(columns.lat_indices)
    ntracer = len(tracer_names)
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("record", nrecord)
        dataset.createDimension("lev", nlev)
        dataset.createDimension("column", ncol)
        dataset.createDimension("tracer", ntracer)
        dataset.createDimension("name_strlen", max(max(map(len, tracer_names), default=1), 1))
        boundary_strlen = max(len(item) for item in BOUNDARIES)
        dataset.createDimension("boundary_strlen", boundary_strlen)

        dataset.harness = "wombat-main-loop-trace-v1"
        dataset.level_order = "lev_bottom"
        dataset.tracer_layout = "(record, lev_bottom, column, tracer)"
        dataset.total_emitted_mass_kg = float(np.sum(emitted_mass))

        dataset.createVariable("record_step", "i4", ("record",))[:] = [int(item["step"]) for item in records]
        time_var = dataset.createVariable("record_time", "f8", ("record",))
        base = records[0]["timestamp"] if records else datetime(2000, 1, 1)
        assert isinstance(base, datetime)
        time_var.units = f"seconds since {base:%Y-%m-%d %H:%M:%S}"
        time_var.calendar = "standard"
        time_var[:] = [float(((item["timestamp"]) - base).total_seconds()) for item in records]  # type: ignore[operator]
        boundary_var = dataset.createVariable("boundary", "S1", ("record", "boundary_strlen"))
        boundary_var[:] = netCDF4.stringtochar(
            np.asarray([str(item["boundary"]) for item in records], dtype=f"S{boundary_strlen}")
        )

        dataset.createVariable("lev", "f8", ("lev",))[:] = np.asarray(lev[::-1], dtype=np.float64)
        dataset.createVariable("lat", "f8", ("column",))[:] = columns.lat_values
        dataset.createVariable("lon", "f8", ("column",))[:] = columns.lon_values
        dataset.createVariable("lat_index", "i4", ("column",))[:] = np.asarray(columns.lat_indices, dtype=np.int32)
        dataset.createVariable("lon_index", "i4", ("column",))[:] = np.asarray(columns.lon_indices, dtype=np.int32)
        dataset.createVariable("tracer_name", "S1", ("tracer", "name_strlen"))[:] = netCDF4.stringtochar(
            np.asarray(tracer_names, dtype=f"S{dataset.dimensions['name_strlen'].size}")
        )
        dataset.createVariable("emitted_mass_kg", "f8", ("tracer",))[:] = emitted_mass

        _write_field(dataset, records, "tracer", ("record", "lev", "column", "tracer"))
        for name in (
            "surface_flux",
            "delp_dry_hpa",
            "dry_air_mass_kg",
            "bxheight",
            "sphu",
            "temperature",
            "pbl_top_m",
            "cmfmc",
            "dtrain",
            "dqrcu",
            "reevapcn",
            "pficu",
            "pflcu",
            "precccon",
            "u_m_s",
            "v_m_s",
            "pmid_hpa",
            "pedge_lower_hpa",
            "pedge_upper_hpa",
            "tv_k",
            "hflux_w_m2",
            "eflux_w_m2",
            "ustar_m_s",
            "p_tp1_hpa",
            "p_tp2_hpa",
            "xmass_hpa",
            "ymass_hpa",
        ):
            if any(name in item for item in records):
                shape = _field_shape(records, name)
                dims = _dims_for_sample_shape(shape, nlev=nlev, ncol=ncol, ntracer=ntracer)
                _write_field(dataset, records, name, dims)


def _field_shape(records: list[dict[str, np.ndarray | datetime | int | str]], name: str) -> tuple[int, ...]:
    for item in records:
        if name in item:
            return np.asarray(item[name]).shape
    raise KeyError(name)


def _dims_for_sample_shape(shape: tuple[int, ...], *, nlev: int, ncol: int, ntracer: int) -> tuple[str, ...]:
    if shape == (ncol,):
        return ("record", "column")
    if shape == (ncol, ntracer):
        return ("record", "column", "tracer")
    if shape == (nlev, ncol):
        return ("record", "lev", "column")
    if shape == (nlev, ncol, ntracer):
        return ("record", "lev", "column", "tracer")
    raise ValueError(f"unsupported sampled trace field shape {shape}")


def _write_field(dataset: netCDF4.Dataset, records: list[dict[str, np.ndarray | datetime | int | str]], name: str, dims):
    shape = tuple(dataset.dimensions[dim].size for dim in dims)
    values = np.full(shape, np.nan, dtype=np.float64)
    for index, item in enumerate(records):
        if name in item:
            values[(index, *([slice(None)] * (values.ndim - 1)))] = item[name]
    var = dataset.createVariable(name, "f8", dims)
    var[:] = values


if __name__ == "__main__":
    raise SystemExit(main())
