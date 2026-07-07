from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np

from wombat_transport.fields import TracerField
from wombat_transport.transport.convection import run_cloud_convection_one_step
from wombat_transport.transport.forcing import TransportForcing, load_transport_forcing
from wombat_transport.transport.metrics import scalar_mass_by_tracer
from wombat_transport.transport.pbl import RD_J_PER_KG_K, ZVIR, G0_M_PER_S2, run_vdiffdr_one_step
from wombat_transport.transport.pressure import (
    _dry_air_mass_to_pressure,
    dry_air_mass_from_pressure,
    dry_pressure_edges_from_thickness_hpa,
    dry_pressure_thickness_hpa,
)
from wombat_transport.transport.tpcore import (
    analyze_tpcore_branches,
    run_tpcore_one_step,
    setup_tpcore_terms,
    validate_tpcore_branch_support,
)


@dataclass(frozen=True)
class TransportStageMass:
    operator: str
    initial_scalar_mass: np.ndarray
    final_scalar_mass: np.ndarray


@dataclass(frozen=True)
class TransportStepResult:
    state: TracerField
    dry_air_mass_kg: np.ndarray
    delp_dry_hpa: np.ndarray
    xmass_hpa: np.ndarray
    ymass_hpa: np.ndarray
    zmass_hpa: np.ndarray
    initial_scalar_mass: np.ndarray
    final_scalar_mass: np.ndarray
    transport_operators: tuple[str, ...]
    stage_masses: tuple[TransportStageMass, ...]

@dataclass(frozen=True)
class TransportWindowResult:
    state: TracerField
    average_state: TracerField
    dry_air_mass_kg: np.ndarray
    average_dry_air_mass_kg: np.ndarray
    delp_dry_hpa: np.ndarray
    average_delp_dry_hpa: np.ndarray
    initial_scalar_mass: np.ndarray
    final_scalar_mass: np.ndarray
    steps: int
    dt_s: float
    transport_operators: tuple[str, ...]
    stage_masses: tuple[TransportStageMass, ...]

def run_transport_one_step(
    tracer_field: TracerField,
    forcing: TransportForcing,
    template_path: str | Path,
    *,
    dt_s: float = 600.0,
    max_courant: float = 0.95,
) -> TransportStepResult:
    """Run one GEOS-Chem-oriented TPCORE + VDIFF + convection transport step."""

    with netCDF4.Dataset(template_path) as template:
        hyai = np.asarray(template.variables["hyai"][:], dtype=np.float64)
        hybi = np.asarray(template.variables["hybi"][:], dtype=np.float64)
        area = np.asarray(template.variables["AREA"][:], dtype=np.float64)

    surface_pressure_hpa = forcing.surface_pressure_pa[0] / 100.0
    delp = dry_pressure_thickness_hpa(forcing.surface_pressure_pa, hyai, hybi)
    dry_air_mass = dry_air_mass_from_pressure(delp, area)
    return _run_tpcore_one_step_from_mass(
        tracer_field,
        forcing,
        dry_air_mass,
        area,
        hyai,
        hybi,
        p2_hpa=surface_pressure_hpa,
        p1_hpa=surface_pressure_hpa,
        dt_s=dt_s,
    )

def run_transport_window(
    tracer_field: TracerField,
    met_root: str | Path,
    start: datetime,
    template_path: str | Path,
    *,
    steps: int = 18,
    dt_s: float = 600.0,
    initial_met_time_index: int = 0,
    max_courant: float = 0.95,
) -> TransportWindowResult:
    """Run a short transport window and accumulate arithmetic mean state."""

    if steps <= 0:
        raise ValueError("steps must be positive")

    state = tracer_field
    dry_mass_sum = None
    state_sum = None
    delp_sum = None
    initial_scalar_mass = None
    final_scalar_mass = None
    forcing_cache: dict[tuple[datetime, int], TransportForcing] = {}
    with netCDF4.Dataset(template_path) as template:
        hyai = np.asarray(template.variables["hyai"][:], dtype=np.float64)
        hybi = np.asarray(template.variables["hybi"][:], dtype=np.float64)
        area = np.asarray(template.variables["AREA"][:], dtype=np.float64)
    first_forcing = _load_window_forcing(
        forcing_cache,
        met_root,
        start,
        template_path,
        step=0,
        dt_s=dt_s,
        initial_met_time_index=initial_met_time_index,
    )
    dry_air_mass = dry_air_mass_from_pressure(
        dry_pressure_thickness_hpa(first_forcing.surface_pressure_pa, hyai, hybi),
        area,
    )

    for step in range(steps):
        forcing = _load_window_forcing(
            forcing_cache,
            met_root,
            start,
            template_path,
            step=step,
            dt_s=dt_s,
            initial_met_time_index=initial_met_time_index,
        )
        step_result = _run_transport_one_step_with_mass(
            state,
            forcing,
            dry_air_mass,
            area,
            hyai,
            hybi,
            dt_s=dt_s,
            max_courant=max_courant,
        )
        if initial_scalar_mass is None:
            initial_scalar_mass = step_result.initial_scalar_mass
        final_scalar_mass = step_result.final_scalar_mass
        state = step_result.state
        dry_air_mass = step_result.dry_air_mass_kg

        if state_sum is None:
            state_sum = np.zeros_like(state.data)
            dry_mass_sum = np.zeros_like(step_result.dry_air_mass_kg)
            delp_sum = np.zeros_like(step_result.delp_dry_hpa)
        state_sum += state.data
        dry_mass_sum += step_result.dry_air_mass_kg
        delp_sum += step_result.delp_dry_hpa

    assert state_sum is not None
    assert dry_mass_sum is not None
    assert delp_sum is not None
    assert initial_scalar_mass is not None
    assert final_scalar_mass is not None

    average_state = TracerField(
        names=state.names,
        data=state_sum / float(steps),
        units=state.units,
        coords=state.coords,
    )
    return TransportWindowResult(
        state=state,
        average_state=average_state,
        dry_air_mass_kg=step_result.dry_air_mass_kg,
        average_dry_air_mass_kg=dry_mass_sum / float(steps),
        delp_dry_hpa=step_result.delp_dry_hpa,
        average_delp_dry_hpa=delp_sum / float(steps),
        initial_scalar_mass=initial_scalar_mass,
        final_scalar_mass=final_scalar_mass,
        steps=steps,
        dt_s=dt_s,
        transport_operators=step_result.transport_operators,
        stage_masses=step_result.stage_masses,
    )

def _run_transport_one_step_with_mass(
    tracer_field: TracerField,
    forcing: TransportForcing,
    dry_air_mass: np.ndarray,
    area: np.ndarray,
    hyai: np.ndarray,
    hybi: np.ndarray,
    *,
    dt_s: float,
    max_courant: float,
) -> TransportStepResult:
    p1_hpa = np.sum(_dry_air_mass_to_pressure(dry_air_mass, area), axis=1)[0] + float(hyai[-1])
    p2_hpa = forcing.surface_pressure_pa[0] / 100.0
    return _run_tpcore_one_step_from_mass(
        tracer_field,
        forcing,
        dry_air_mass,
        area,
        hyai,
        hybi,
        p2_hpa=p2_hpa,
        p1_hpa=p1_hpa,
        dt_s=dt_s,
    )


def _run_tpcore_one_step_from_mass(
    tracer_field: TracerField,
    forcing: TransportForcing,
    dry_air_mass: np.ndarray,
    area: np.ndarray,
    hyai: np.ndarray,
    hybi: np.ndarray,
    *,
    p2_hpa: np.ndarray,
    dt_s: float,
    p1_hpa: np.ndarray | None = None,
) -> TransportStepResult:
    if tracer_field.data.shape[1] != 1:
        raise ValueError(f"TPCORE driver expects one time slice, found shape {tracer_field.data.shape}")
    if p1_hpa is None:
        p1_hpa = np.sum(_dry_air_mass_to_pressure(dry_air_mass, area), axis=1)[0] + float(hyai[-1])

    setup = setup_tpcore_terms(
        p1_hpa=p1_hpa,
        p2_hpa=p2_hpa,
        u_m_s=forcing.u_m_s[0],
        v_m_s=forcing.v_m_s[0],
        area_m2=area,
        hyai_hpa=hyai,
        hybi=hybi,
        lat_deg=forcing.lat_deg,
        dt_s=dt_s,
    )
    try:
        validate_tpcore_branch_support(setup)
    except NotImplementedError as exc:
        report = analyze_tpcore_branches(setup)
        raise NotImplementedError(_format_tpcore_branch_preflight_error(report)) from exc

    tpcore = run_tpcore_one_step(
        tracer_conc=np.asarray(tracer_field.data[:, 0, :, :, :], dtype=np.float64),
        p1_hpa=p1_hpa,
        p2_hpa=p2_hpa,
        u_m_s=forcing.u_m_s[0],
        v_m_s=forcing.v_m_s[0],
        area_m2=area,
        hyai_hpa=hyai,
        hybi=hybi,
        lat_deg=forcing.lat_deg,
        dt_s=dt_s,
    )
    next_delp = tpcore.delp2_hpa[np.newaxis, ...]
    next_dry_air_mass = dry_air_mass_from_pressure(next_delp, area)
    tpcore_state = TracerField(
        names=tracer_field.names,
        data=tpcore.tracer_conc_after[:, np.newaxis, :, :, :],
        units=tracer_field.units,
        coords=tracer_field.coords,
    )
    initial_scalar_mass = scalar_mass_by_tracer(tracer_field.data, dry_air_mass)
    tpcore_scalar_mass = scalar_mass_by_tracer(tpcore_state.data, next_dry_air_mass)
    vdiff = _run_vdiff_after_tpcore(
        tpcore_state,
        forcing,
        next_dry_air_mass,
        next_delp,
        area,
        top_edge_hpa=float(hyai[-1]),
        dt_s=dt_s,
    )
    state = TracerField(
        names=tracer_field.names,
        data=vdiff.tracer_conc[:, np.newaxis, :, :, :],
        units=tracer_field.units,
        coords=tracer_field.coords,
    )
    vdiff_scalar_mass = scalar_mass_by_tracer(state.data, next_dry_air_mass)
    convection = _run_convection_after_vdiff(
        state,
        forcing,
        next_delp,
        area,
        top_edge_hpa=float(hyai[-1]),
        dt_s=dt_s,
    )
    state = TracerField(
        names=tracer_field.names,
        data=convection.tracer_conc[:, np.newaxis, :, :, :],
        units=tracer_field.units,
        coords=tracer_field.coords,
    )
    convection_scalar_mass = scalar_mass_by_tracer(state.data, next_dry_air_mass)
    return TransportStepResult(
        state=state,
        dry_air_mass_kg=next_dry_air_mass,
        delp_dry_hpa=next_delp,
        xmass_hpa=tpcore.xmass_hpa[np.newaxis, ...],
        ymass_hpa=tpcore.ymass_hpa[np.newaxis, ...],
        zmass_hpa=_tpcore_vertical_flux_edges(tpcore.vertical_mass_flux_hpa),
        initial_scalar_mass=initial_scalar_mass,
        final_scalar_mass=convection_scalar_mass,
        transport_operators=("tpcore", "vdiff", "convection"),
        stage_masses=(
            TransportStageMass("tpcore", initial_scalar_mass=initial_scalar_mass, final_scalar_mass=tpcore_scalar_mass),
            TransportStageMass("vdiff", initial_scalar_mass=tpcore_scalar_mass, final_scalar_mass=vdiff_scalar_mass),
            TransportStageMass(
                "convection",
                initial_scalar_mass=vdiff_scalar_mass,
                final_scalar_mass=convection_scalar_mass,
            ),
        ),
    )


def _run_vdiff_after_tpcore(
    tracer_field: TracerField,
    forcing: TransportForcing,
    dry_air_mass: np.ndarray,
    delp_dry_hpa: np.ndarray,
    area: np.ndarray,
    *,
    top_edge_hpa: float,
    dt_s: float,
):
    pedge = dry_pressure_edges_from_thickness_hpa(delp_dry_hpa, top_edge_hpa=top_edge_hpa)[0]
    pmid = 0.5 * (pedge[:-1] + pedge[1:])
    temperature = np.asarray(forcing.temperature_k[0], dtype=np.float64)
    sphu = np.asarray(forcing.specific_humidity_kg_kg[0], dtype=np.float64)
    virtual_temperature = temperature * (1.0 + ZVIR * sphu)
    bxheight = _hydrostatic_box_height_m(pedge, virtual_temperature)
    return run_vdiffdr_one_step(
        tracer_conc=np.asarray(tracer_field.data[:, 0, :, :, :], dtype=np.float64),
        u_m_s=np.asarray(forcing.u_m_s[0], dtype=np.float64),
        v_m_s=np.asarray(forcing.v_m_s[0], dtype=np.float64),
        temperature_k=temperature,
        specific_humidity_kg_kg=sphu,
        pmid_hpa=pmid,
        pedge_hpa=pedge,
        virtual_temperature_k=virtual_temperature,
        bxheight_m=bxheight,
        dry_air_mass_kg=np.asarray(dry_air_mass[0], dtype=np.float64),
        pbl_top_m=np.asarray(forcing.pbl_height_m[0], dtype=np.float64),
        hflux_w_m2=np.asarray(forcing.sensible_heat_flux_w_m2[0], dtype=np.float64),
        eflux_w_m2=np.asarray(forcing.latent_heat_flux_w_m2[0], dtype=np.float64),
        ustar_m_s=np.asarray(forcing.friction_velocity_m_s[0], dtype=np.float64),
        area_m2=area,
        dt_s=dt_s,
        surface_flux_kg_m2_s=np.zeros(
            (tracer_field.data.shape[0], tracer_field.data.shape[3], tracer_field.data.shape[4]),
            dtype=np.float64,
        ),
    )


def _run_convection_after_vdiff(
    tracer_field: TracerField,
    forcing: TransportForcing,
    delp_dry_hpa: np.ndarray,
    area: np.ndarray,
    *,
    top_edge_hpa: float,
    dt_s: float,
):
    pedge = dry_pressure_edges_from_thickness_hpa(delp_dry_hpa, top_edge_hpa=top_edge_hpa)[0]
    temperature = np.asarray(forcing.temperature_k[0], dtype=np.float64)
    sphu = np.asarray(forcing.specific_humidity_kg_kg[0], dtype=np.float64)
    virtual_temperature = temperature * (1.0 + ZVIR * sphu)
    bxheight = _hydrostatic_box_height_m(pedge, virtual_temperature)
    delp = np.asarray(delp_dry_hpa[0], dtype=np.float64)
    return run_cloud_convection_one_step(
        tracer_conc=np.asarray(tracer_field.data[:, 0, :, :, :], dtype=np.float64),
        cmfmc_kg_m2_s=np.asarray(forcing.convective_mass_flux_kg_m2_s[0], dtype=np.float64),
        dtrain_kg_m2_s=np.asarray(forcing.convective_detrainment_kg_m2_s[0], dtype=np.float64),
        dqrcu_kg_kg_s=np.asarray(forcing.convective_precip_prod_kg_kg_s[0], dtype=np.float64),
        reevapcn_kg_kg_s=np.asarray(forcing.convective_precip_reevap_kg_kg_s[0], dtype=np.float64),
        delp_dry_hpa=delp,
        delp_hpa=delp.copy(),
        area_m2=area,
        bxheight_m=bxheight,
        pficu_kg_m2_s=np.asarray(forcing.convective_ice_flux_kg_m2_s[0], dtype=np.float64),
        pflcu_kg_m2_s=np.asarray(forcing.convective_liquid_flux_kg_m2_s[0], dtype=np.float64),
        temperature_k=temperature,
        precccon_mm_day=np.asarray(forcing.convective_precip_mm_day[0], dtype=np.float64),
        dt_s=dt_s,
    )


def _hydrostatic_box_height_m(pedge_hpa: np.ndarray, virtual_temperature_k: np.ndarray) -> np.ndarray:
    pedge = np.asarray(pedge_hpa, dtype=np.float64)
    tv = np.asarray(virtual_temperature_k, dtype=np.float64)
    return (RD_J_PER_KG_K / G0_M_PER_S2) * tv * np.log(pedge[:-1] / pedge[1:])


def _tpcore_vertical_flux_edges(vertical_flux_hpa: np.ndarray) -> np.ndarray:
    flux = np.asarray(vertical_flux_hpa, dtype=np.float64)
    edges = np.zeros((1, flux.shape[0] + 1, flux.shape[1], flux.shape[2]), dtype=np.float64)
    edges[:, :-1, :, :] = flux[np.newaxis, ...]
    return edges


def _format_tpcore_branch_preflight_error(report) -> str:
    reasons = " | ".join(report.unsupported_reasons) or "unknown unsupported branch"
    return (
        "TPCORE branch preflight failed: "
        f"{reasons}. shape={report.shape}, max_abs_cx={report.max_abs_cx:.8e}, "
        f"max_abs_cy={report.max_abs_cy:.8e}, has_large_cx={report.has_large_cx}, "
        f"has_large_cy={report.has_large_cy}, needs_fxppm={report.needs_fxppm}"
    )

def _load_window_forcing(
    cache: dict[tuple[datetime, int], TransportForcing],
    met_root: str | Path,
    start: datetime,
    template_path: str | Path,
    *,
    step: int,
    dt_s: float,
    initial_met_time_index: int,
) -> TransportForcing:
    met_step = int((step * float(dt_s)) // (3.0 * 60.0 * 60.0))
    absolute_index = int(initial_met_time_index) + met_step
    timestamp = start + timedelta(days=absolute_index // 8)
    time_index = absolute_index % 8
    key = (datetime(timestamp.year, timestamp.month, timestamp.day), time_index)
    if key not in cache:
        cache[key] = load_transport_forcing(met_root, key[0], template_path, time_index=time_index)
    return cache[key]
