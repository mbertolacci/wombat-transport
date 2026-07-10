from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from wombat_transport.fields import (
    TracerField,
    canonical_time_slice,
    transport_tracer_to_canonical,
)
from wombat_transport.grid import TransportGrid
from wombat_transport.transport.convection import ConvectionResult, run_cloud_convection_one_step
from wombat_transport.transport.forcing import TransportForcing, load_transport_forcing
from wombat_transport.transport.metrics import scalar_mass_by_tracer
from wombat_transport.transport.pbl import RD_J_PER_KG_K, ZVIR, G0_M_PER_S2, VdiffDrResult, run_vdiffdr_one_step
from wombat_transport.transport.pressure import (
    _dry_air_mass_to_pressure,
    dry_air_mass_from_pressure,
    dry_pressure_edges_from_thickness_hpa,
    dry_pressure_thickness_hpa,
)
from wombat_transport.transport.tpcore import (
    TpcoreState,
    _average_const_poles_batch,
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
    transport_operators: tuple[str, ...]


@dataclass(frozen=True)
class VdiffInputState:
    tracer_conc: np.ndarray
    u_m_s: np.ndarray
    v_m_s: np.ndarray
    temperature_k: np.ndarray
    specific_humidity_kg_kg: np.ndarray
    pmid_hpa: np.ndarray
    pedge_hpa: np.ndarray
    virtual_temperature_k: np.ndarray
    bxheight_m: np.ndarray
    dry_air_mass_kg: np.ndarray
    pbl_top_m: np.ndarray
    hflux_w_m2: np.ndarray
    eflux_w_m2: np.ndarray
    ustar_m_s: np.ndarray
    area_m2: np.ndarray
    surface_flux_kg_m2_s: np.ndarray
    dt_s: float


@dataclass(frozen=True)
class ConvectionInputState:
    tracer_conc: np.ndarray
    cmfmc_kg_m2_s: np.ndarray
    dtrain_kg_m2_s: np.ndarray
    dqrcu_kg_kg_s: np.ndarray
    reevapcn_kg_kg_s: np.ndarray
    delp_dry_hpa: np.ndarray
    delp_hpa: np.ndarray
    area_m2: np.ndarray
    bxheight_m: np.ndarray
    pficu_kg_m2_s: np.ndarray
    pflcu_kg_m2_s: np.ndarray
    temperature_k: np.ndarray
    precccon_mm_day: np.ndarray
    dt_s: float


@dataclass(frozen=True)
class TransportStepDiagnostics:
    result: TransportStepResult
    tpcore_state: TpcoreState
    vdiff_input: VdiffInputState
    vdiff_output: VdiffDrResult
    convection_input: ConvectionInputState
    convection_output: ConvectionResult


@dataclass(frozen=True)
class TransportWindowResult:
    state: TracerField
    average_state: TracerField
    dry_air_mass_kg: np.ndarray
    average_dry_air_mass_kg: np.ndarray
    delp_dry_hpa: np.ndarray
    average_delp_dry_hpa: np.ndarray
    steps: int
    dt_s: float
    transport_operators: tuple[str, ...]

def run_transport_one_step(
    tracer_field: TracerField,
    forcing: TransportForcing,
    grid: TransportGrid,
    *,
    dt_s: float = 600.0,
    max_courant: float = 0.95,
) -> TransportStepResult:
    """Run one GEOS-Chem-oriented TPCORE + VDIFF + convection transport step."""

    surface_pressure_hpa = forcing.surface_pressure_pa[0] / 100.0
    delp = dry_pressure_thickness_hpa(
        forcing.surface_pressure_pa,
        grid.hyai_hpa,
        grid.hybi,
    )
    dry_air_mass = dry_air_mass_from_pressure(delp, grid.area_m2)
    return _run_tpcore_one_step_from_mass(
        tracer_field,
        forcing,
        dry_air_mass,
        grid.area_m2,
        grid.hyai_hpa,
        grid.hybi,
        p2_hpa=surface_pressure_hpa,
        p1_hpa=surface_pressure_hpa,
        dt_s=dt_s,
    )


def trace_transport_one_step(
    tracer_field: TracerField,
    forcing: TransportForcing,
    grid: TransportGrid,
    *,
    dt_s: float = 600.0,
    max_courant: float = 0.95,
) -> TransportStepDiagnostics:
    """Run one transport step and retain exact operator handoff arrays."""

    surface_pressure_hpa = forcing.surface_pressure_pa[0] / 100.0
    delp = dry_pressure_thickness_hpa(
        forcing.surface_pressure_pa,
        grid.hyai_hpa,
        grid.hybi,
    )
    dry_air_mass = dry_air_mass_from_pressure(delp, grid.area_m2)
    return _trace_tpcore_one_step_from_mass(
        tracer_field,
        forcing,
        dry_air_mass,
        grid.area_m2,
        grid.hyai_hpa,
        grid.hybi,
        p2_hpa=surface_pressure_hpa,
        p1_hpa=surface_pressure_hpa,
        dt_s=dt_s,
    )


def compute_transport_stage_masses(
    trace: TransportStepDiagnostics,
    initial_field: TracerField,
    area_m2: np.ndarray,
) -> tuple[TransportStageMass, ...]:
    """Compute per-stage scalar mass diagnostics from a traced transport step."""

    dry_air_mass_top = trace.result.dry_air_mass_kg[:, ::-1, :, :]
    initial_scalar_mass = _tpcore_initial_scalar_mass(
        initial_field.data,
        trace.tpcore_state.delp1_hpa,
        area_m2,
    )
    tpcore_scalar_mass = scalar_mass_by_tracer(
        transport_tracer_to_canonical(trace.tpcore_state.tracer_conc_after),
        dry_air_mass_top,
    )
    vdiff_scalar_mass = scalar_mass_by_tracer(
        transport_tracer_to_canonical(trace.vdiff_output.tracer_conc),
        dry_air_mass_top,
    )
    convection_scalar_mass = scalar_mass_by_tracer(
        transport_tracer_to_canonical(trace.convection_output.tracer_conc),
        dry_air_mass_top,
    )
    return (
        TransportStageMass("tpcore", initial_scalar_mass=initial_scalar_mass, final_scalar_mass=tpcore_scalar_mass),
        TransportStageMass("vdiff", initial_scalar_mass=tpcore_scalar_mass, final_scalar_mass=vdiff_scalar_mass),
        TransportStageMass(
            "convection",
            initial_scalar_mass=vdiff_scalar_mass,
            final_scalar_mass=convection_scalar_mass,
        ),
    )


def run_transport_window(
    tracer_field: TracerField,
    met_root: str | Path,
    start: datetime,
    grid: TransportGrid,
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
    forcing_cache: dict[tuple[datetime, int], TransportForcing] = {}
    first_forcing = _load_window_forcing(
        forcing_cache,
        met_root,
        start,
        grid,
        step=0,
        dt_s=dt_s,
        initial_met_time_index=initial_met_time_index,
    )
    dry_air_mass = dry_air_mass_from_pressure(
        dry_pressure_thickness_hpa(
            first_forcing.surface_pressure_pa,
            grid.hyai_hpa,
            grid.hybi,
        ),
        grid.area_m2,
    )

    for step in range(steps):
        forcing = _load_window_forcing(
            forcing_cache,
            met_root,
            start,
            grid,
            step=step,
            dt_s=dt_s,
            initial_met_time_index=initial_met_time_index,
        )
        step_result = _run_transport_one_step_with_mass(
            state,
            forcing,
            dry_air_mass,
            grid.area_m2,
            grid.hyai_hpa,
            grid.hybi,
            dt_s=dt_s,
            max_courant=max_courant,
        )
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
        steps=steps,
        dt_s=dt_s,
        transport_operators=step_result.transport_operators,
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
    if tracer_field.data.shape[0] != 1:
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
        tracer_conc=canonical_time_slice(tracer_field.data),
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
    next_delp = tpcore.delp2_hpa[np.newaxis, ::-1, :, :]
    next_dry_air_mass = dry_air_mass_from_pressure(next_delp, area)
    tpcore_state = TracerField(
        names=tracer_field.names,
        data=transport_tracer_to_canonical(tpcore.tracer_conc_after),
        units=tracer_field.units,
        coords=tracer_field.coords,
    )
    vdiff_input = _build_vdiff_input_after_tpcore(
        tpcore_state,
        forcing,
        next_dry_air_mass,
        next_delp,
        area,
        top_edge_hpa=float(hyai[-1]),
        dt_s=dt_s,
    )
    vdiff = _run_vdiff_input(vdiff_input, diagnostics=False)
    state = TracerField(
        names=tracer_field.names,
        data=transport_tracer_to_canonical(vdiff.tracer_conc),
        units=tracer_field.units,
        coords=tracer_field.coords,
    )
    convection_input = _build_convection_input_after_vdiff(
        state,
        forcing,
        next_delp,
        area,
        top_edge_hpa=float(hyai[-1]),
        dt_s=dt_s,
    )
    convection = _run_convection_input(convection_input, diagnostics=False)
    state = TracerField(
        names=tracer_field.names,
        data=transport_tracer_to_canonical(convection.tracer_conc),
        units=tracer_field.units,
        coords=tracer_field.coords,
    )
    return TransportStepResult(
        state=state,
        dry_air_mass_kg=next_dry_air_mass,
        delp_dry_hpa=next_delp,
        xmass_hpa=tpcore.xmass_hpa[np.newaxis, ::-1, :, :],
        ymass_hpa=tpcore.ymass_hpa[np.newaxis, ::-1, :, :],
        zmass_hpa=_tpcore_vertical_flux_edges(tpcore.vertical_mass_flux_hpa[::-1]),
        transport_operators=("tpcore", "vdiff", "convection"),
    )


def _trace_tpcore_one_step_from_mass(
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
) -> TransportStepDiagnostics:
    if tracer_field.data.shape[0] != 1:
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
        tracer_conc=canonical_time_slice(tracer_field.data),
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
    next_delp = tpcore.delp2_hpa[np.newaxis, ::-1, :, :]
    next_dry_air_mass = dry_air_mass_from_pressure(next_delp, area)
    tpcore_state = TracerField(
        names=tracer_field.names,
        data=transport_tracer_to_canonical(tpcore.tracer_conc_after),
        units=tracer_field.units,
        coords=tracer_field.coords,
    )
    vdiff_input = _build_vdiff_input_after_tpcore(
        tpcore_state,
        forcing,
        next_dry_air_mass,
        next_delp,
        area,
        top_edge_hpa=float(hyai[-1]),
        dt_s=dt_s,
    )
    vdiff = _run_vdiff_input(vdiff_input, diagnostics=True)
    state = TracerField(
        names=tracer_field.names,
        data=transport_tracer_to_canonical(vdiff.tracer_conc),
        units=tracer_field.units,
        coords=tracer_field.coords,
    )
    convection_input = _build_convection_input_after_vdiff(
        state,
        forcing,
        next_delp,
        area,
        top_edge_hpa=float(hyai[-1]),
        dt_s=dt_s,
    )
    convection = _run_convection_input(convection_input, diagnostics=True)
    state = TracerField(
        names=tracer_field.names,
        data=transport_tracer_to_canonical(convection.tracer_conc),
        units=tracer_field.units,
        coords=tracer_field.coords,
    )
    result = TransportStepResult(
        state=state,
        dry_air_mass_kg=next_dry_air_mass,
        delp_dry_hpa=next_delp,
        xmass_hpa=tpcore.xmass_hpa[np.newaxis, ::-1, :, :],
        ymass_hpa=tpcore.ymass_hpa[np.newaxis, ::-1, :, :],
        zmass_hpa=_tpcore_vertical_flux_edges(tpcore.vertical_mass_flux_hpa[::-1]),
        transport_operators=("tpcore", "vdiff", "convection"),
    )
    return TransportStepDiagnostics(
        result=result,
        tpcore_state=tpcore,
        vdiff_input=vdiff_input,
        vdiff_output=vdiff,
        convection_input=convection_input,
        convection_output=convection,
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
    return _run_vdiff_input(
        _build_vdiff_input_after_tpcore(
            tracer_field,
            forcing,
            dry_air_mass,
            delp_dry_hpa,
            area,
            top_edge_hpa=top_edge_hpa,
            dt_s=dt_s,
        ),
        diagnostics=False,
    )


def _build_vdiff_input_after_tpcore(
    tracer_field: TracerField,
    forcing: TransportForcing,
    dry_air_mass: np.ndarray,
    delp_dry_hpa: np.ndarray,
    area: np.ndarray,
    *,
    top_edge_hpa: float,
    dt_s: float,
) -> VdiffInputState:
    pedge = dry_pressure_edges_from_thickness_hpa(delp_dry_hpa, top_edge_hpa=top_edge_hpa)[0]
    pmid = 0.5 * (pedge[:-1] + pedge[1:])
    temperature = np.asarray(forcing.temperature_k[0], dtype=np.float64)
    sphu = np.asarray(forcing.specific_humidity_kg_kg[0], dtype=np.float64)
    virtual_temperature = temperature * (1.0 + ZVIR * sphu)
    bxheight = _hydrostatic_box_height_m(pedge, virtual_temperature)
    ntracer = len(tracer_field.names)
    return VdiffInputState(
        tracer_conc=canonical_time_slice(tracer_field.data),
        u_m_s=np.asarray(forcing.u_m_s[0], dtype=np.float64)[::-1],
        v_m_s=np.asarray(forcing.v_m_s[0], dtype=np.float64)[::-1],
        temperature_k=temperature[::-1],
        specific_humidity_kg_kg=sphu[::-1],
        pmid_hpa=pmid[::-1],
        pedge_hpa=pedge[::-1],
        virtual_temperature_k=virtual_temperature[::-1],
        bxheight_m=bxheight[::-1],
        dry_air_mass_kg=np.asarray(dry_air_mass[0], dtype=np.float64)[::-1],
        pbl_top_m=np.asarray(forcing.pbl_height_m[0], dtype=np.float64),
        hflux_w_m2=np.asarray(forcing.sensible_heat_flux_w_m2[0], dtype=np.float64),
        eflux_w_m2=np.asarray(forcing.latent_heat_flux_w_m2[0], dtype=np.float64),
        ustar_m_s=np.asarray(forcing.friction_velocity_m_s[0], dtype=np.float64),
        area_m2=area,
        dt_s=dt_s,
        surface_flux_kg_m2_s=np.zeros((tracer_field.data.shape[2], tracer_field.data.shape[3], ntracer), dtype=np.float64),
    )


def _run_vdiff_input(state: VdiffInputState, *, diagnostics: bool = False) -> VdiffDrResult:
    return run_vdiffdr_one_step(
        tracer_conc=state.tracer_conc,
        u_m_s=state.u_m_s,
        v_m_s=state.v_m_s,
        temperature_k=state.temperature_k,
        specific_humidity_kg_kg=state.specific_humidity_kg_kg,
        pmid_hpa=state.pmid_hpa,
        pedge_hpa=state.pedge_hpa,
        virtual_temperature_k=state.virtual_temperature_k,
        bxheight_m=state.bxheight_m,
        dry_air_mass_kg=state.dry_air_mass_kg,
        pbl_top_m=state.pbl_top_m,
        hflux_w_m2=state.hflux_w_m2,
        eflux_w_m2=state.eflux_w_m2,
        ustar_m_s=state.ustar_m_s,
        area_m2=state.area_m2,
        dt_s=state.dt_s,
        surface_flux_kg_m2_s=state.surface_flux_kg_m2_s,
        diagnostics=diagnostics,
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
    return _run_convection_input(
        _build_convection_input_after_vdiff(
            tracer_field,
            forcing,
            delp_dry_hpa,
            area,
            top_edge_hpa=top_edge_hpa,
            dt_s=dt_s,
        )
    )


def _build_convection_input_after_vdiff(
    tracer_field: TracerField,
    forcing: TransportForcing,
    delp_dry_hpa: np.ndarray,
    area: np.ndarray,
    *,
    top_edge_hpa: float,
    dt_s: float,
) -> ConvectionInputState:
    pedge = dry_pressure_edges_from_thickness_hpa(delp_dry_hpa, top_edge_hpa=top_edge_hpa)[0]
    temperature = np.asarray(forcing.temperature_k[0], dtype=np.float64)
    sphu = np.asarray(forcing.specific_humidity_kg_kg[0], dtype=np.float64)
    virtual_temperature = temperature * (1.0 + ZVIR * sphu)
    bxheight = _hydrostatic_box_height_m(pedge, virtual_temperature)
    delp = np.asarray(delp_dry_hpa[0], dtype=np.float64)
    return ConvectionInputState(
        tracer_conc=canonical_time_slice(tracer_field.data),
        cmfmc_kg_m2_s=np.asarray(forcing.convective_mass_flux_kg_m2_s[0], dtype=np.float64)[::-1],
        dtrain_kg_m2_s=np.asarray(forcing.convective_detrainment_kg_m2_s[0], dtype=np.float64)[::-1],
        dqrcu_kg_kg_s=np.asarray(forcing.convective_precip_prod_kg_kg_s[0], dtype=np.float64)[::-1],
        reevapcn_kg_kg_s=np.asarray(forcing.convective_precip_reevap_kg_kg_s[0], dtype=np.float64)[::-1],
        delp_dry_hpa=delp[::-1],
        delp_hpa=delp[::-1].copy(),
        area_m2=area,
        bxheight_m=bxheight[::-1],
        pficu_kg_m2_s=np.asarray(forcing.convective_ice_flux_kg_m2_s[0], dtype=np.float64)[::-1],
        pflcu_kg_m2_s=np.asarray(forcing.convective_liquid_flux_kg_m2_s[0], dtype=np.float64)[::-1],
        temperature_k=temperature[::-1],
        precccon_mm_day=np.asarray(forcing.convective_precip_mm_day[0], dtype=np.float64),
        dt_s=dt_s,
    )


def _run_convection_input(state: ConvectionInputState, *, diagnostics: bool = False) -> ConvectionResult:
    return run_cloud_convection_one_step(
        tracer_conc=state.tracer_conc,
        cmfmc_kg_m2_s=state.cmfmc_kg_m2_s,
        dtrain_kg_m2_s=state.dtrain_kg_m2_s,
        dqrcu_kg_kg_s=state.dqrcu_kg_kg_s,
        reevapcn_kg_kg_s=state.reevapcn_kg_kg_s,
        delp_dry_hpa=state.delp_dry_hpa,
        delp_hpa=state.delp_hpa,
        area_m2=state.area_m2,
        bxheight_m=state.bxheight_m,
        pficu_kg_m2_s=state.pficu_kg_m2_s,
        pflcu_kg_m2_s=state.pflcu_kg_m2_s,
        temperature_k=state.temperature_k,
        precccon_mm_day=state.precccon_mm_day,
        dt_s=state.dt_s,
        diagnostics=diagnostics,
        reuse_output=not diagnostics,
    )


def _tpcore_initial_scalar_mass(field_data: np.ndarray, delp1_hpa: np.ndarray, area_m2: np.ndarray) -> np.ndarray:
    tracer = canonical_time_slice(field_data).copy()
    area_1d = area_m2[:, 0]
    for level in range(tracer.shape[0]):
        _average_const_poles_batch(tracer[level], delp1_hpa[level], area_1d)
    dry_mass_top = dry_air_mass_from_pressure(delp1_hpa[np.newaxis, ::-1, :, :], area_m2)[:, ::-1, :, :]
    return scalar_mass_by_tracer(transport_tracer_to_canonical(tracer), dry_mass_top)


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
    grid: TransportGrid,
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
        cache.clear()
        cache[key] = load_transport_forcing(met_root, key[0], grid, time_index=time_index)
    return cache[key]
