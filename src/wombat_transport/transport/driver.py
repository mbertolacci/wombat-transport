from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from wombat_transport.constants import AIRMW_G_PER_MOL, H2OMW_G_PER_MOL
from wombat_transport.fields import (
    TracerField,
    canonical_time_slice,
    transport_tracer_to_canonical,
)
from wombat_transport.emissions import SurfaceEmissions
from wombat_transport.grid import TransportGrid
from wombat_transport.transport.convection import ConvectionResult, run_cloud_convection_one_step
from wombat_transport.transport.forcing import (
    TransportForcing,
    TransportForcingProvider,
    load_transport_forcing_for_step,
)
from wombat_transport.transport.metrics import scalar_mass_by_tracer
from wombat_transport.transport.pbl import RD_J_PER_KG_K, G0_M_PER_S2, VdiffDrResult, run_vdiffdr_one_step
from wombat_transport.transport.pressure import (
    _dry_air_mass_to_pressure,
    dry_air_mass_from_pressure,
    dry_pressure_edges_from_thickness_hpa,
    dry_pressure_thickness_from_surface_hpa,
    pressure_edges_hpa,
    pressure_edges_from_surface_hpa,
)
from wombat_transport.transport.tpcore import (
    TpcoreState,
    TpcoreStaticTerms,
    _average_const_poles_batch,
    analyze_tpcore_branches,
    build_tpcore_static_terms,
    run_tpcore_one_step_with_setup,
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
    xmass_hpa: np.ndarray | None
    ymass_hpa: np.ndarray | None
    zmass_hpa: np.ndarray | None
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
    surface_flux_for_vdiff: np.ndarray
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
    bxheight_m: np.ndarray | None
    pficu_kg_m2_s: np.ndarray | None
    pflcu_kg_m2_s: np.ndarray | None
    temperature_k: np.ndarray | None
    precccon_mm_day: np.ndarray | None
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
    active_emissions: TracerField | SurfaceEmissions | None = None,
    surface_flux_to_vmr_factor: np.ndarray | None = None,
    dry_air_mass_kg: np.ndarray | None = None,
    tpcore_static_terms: TpcoreStaticTerms | None = None,
    include_flux_diagnostics: bool = False,
    validate_tpcore_branches: bool = True,
) -> TransportStepResult:
    """Run one GEOS-Chem-oriented TPCORE + VDIFF + convection transport step."""

    dry_air_mass = (
        _dry_air_mass_from_forcing_start(forcing, grid)
        if dry_air_mass_kg is None
        else np.asarray(dry_air_mass_kg, dtype=np.float64)
    )
    return _run_tpcore_one_step_from_mass(
        tracer_field,
        forcing,
        dry_air_mass,
        grid.area_m2,
        grid.hyai_hpa,
        grid.hybi,
        p2_hpa=forcing.dry_surface_pressure_hpa[0],
        p1_hpa=None if dry_air_mass_kg is not None else forcing.dry_surface_pressure_start_hpa[0],
        dt_s=dt_s,
        active_emissions=active_emissions,
        surface_flux_to_vmr_factor=surface_flux_to_vmr_factor,
        tpcore_static_terms=tpcore_static_terms,
        include_flux_diagnostics=include_flux_diagnostics,
        validate_tpcore_branches=validate_tpcore_branches,
    )


def trace_transport_one_step(
    tracer_field: TracerField,
    forcing: TransportForcing,
    grid: TransportGrid,
    *,
    dt_s: float = 600.0,
    max_courant: float = 0.95,
    active_emissions: TracerField | SurfaceEmissions | None = None,
    surface_flux_to_vmr_factor: np.ndarray | None = None,
    dry_air_mass_kg: np.ndarray | None = None,
    tpcore_static_terms: TpcoreStaticTerms | None = None,
) -> TransportStepDiagnostics:
    """Run one transport step and retain exact operator handoff arrays."""

    dry_air_mass = (
        _dry_air_mass_from_forcing_start(forcing, grid)
        if dry_air_mass_kg is None
        else np.asarray(dry_air_mass_kg, dtype=np.float64)
    )
    return _trace_tpcore_one_step_from_mass(
        tracer_field,
        forcing,
        dry_air_mass,
        grid.area_m2,
        grid.hyai_hpa,
        grid.hybi,
        p2_hpa=forcing.dry_surface_pressure_hpa[0],
        p1_hpa=None if dry_air_mass_kg is not None else forcing.dry_surface_pressure_start_hpa[0],
        dt_s=dt_s,
        active_emissions=active_emissions,
        surface_flux_to_vmr_factor=surface_flux_to_vmr_factor,
        tpcore_static_terms=tpcore_static_terms,
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
    chunk_multiple: int = 1,
    max_courant: float = 0.95,
) -> TransportWindowResult:
    """Run a short transport window and accumulate arithmetic mean state."""

    if steps <= 0:
        raise ValueError("steps must be positive")

    state = tracer_field
    dry_mass_sum = None
    state_sum = None
    delp_sum = None
    forcing_provider = TransportForcingProvider(
        met_root,
        start,
        grid,
        initial_met_time_index=initial_met_time_index,
        chunk_multiple=chunk_multiple,
    )
    tpcore_static_terms = build_tpcore_static_terms(
        area_m2=grid.area_m2,
        hyai_hpa=grid.hyai_hpa,
        hybi=grid.hybi,
        lat_deg=grid.lat_deg,
    )
    first_forcing = _load_window_forcing(
        forcing_provider,
        step=0,
        dt_s=dt_s,
    )
    dry_air_mass = _dry_air_mass_from_forcing_start(first_forcing, grid)

    for step in range(steps):
        forcing = _load_window_forcing(
            forcing_provider,
            step=step,
            dt_s=dt_s,
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
            tpcore_static_terms=tpcore_static_terms,
            validate_tpcore_branches=step == 0,
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
    active_emissions: TracerField | SurfaceEmissions | None = None,
    surface_flux_to_vmr_factor: np.ndarray | None = None,
    tpcore_static_terms: TpcoreStaticTerms | None = None,
    include_flux_diagnostics: bool = False,
    validate_tpcore_branches: bool = True,
) -> TransportStepResult:
    p1_hpa = np.sum(_dry_air_mass_to_pressure(dry_air_mass, area), axis=1)[0] + float(hyai[-1])
    p2_hpa = forcing.dry_surface_pressure_hpa[0]
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
        active_emissions=active_emissions,
        surface_flux_to_vmr_factor=surface_flux_to_vmr_factor,
        tpcore_static_terms=tpcore_static_terms,
        include_flux_diagnostics=include_flux_diagnostics,
        validate_tpcore_branches=validate_tpcore_branches,
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
    active_emissions: TracerField | SurfaceEmissions | None = None,
    surface_flux_to_vmr_factor: np.ndarray | None = None,
    tpcore_static_terms: TpcoreStaticTerms | None = None,
    include_flux_diagnostics: bool = False,
    validate_tpcore_branches: bool = True,
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
        static_terms=tpcore_static_terms,
    )
    if validate_tpcore_branches:
        try:
            validate_tpcore_branch_support(setup)
        except NotImplementedError as exc:
            report = analyze_tpcore_branches(setup)
            raise NotImplementedError(_format_tpcore_branch_preflight_error(report)) from exc

    tpcore = run_tpcore_one_step_with_setup(
        tracer_conc=canonical_time_slice(tracer_field.data),
        setup=setup,
        area_m2=area,
        validate_branches=False,
        reuse_output=True,
    )
    next_delp = dry_pressure_thickness_from_surface_hpa(
        forcing.dry_surface_pressure_hpa,
        hyai,
        hybi,
    )
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
        hyai_hpa=hyai,
        hybi=hybi,
        top_edge_hpa=float(hyai[-1]),
        dt_s=dt_s,
        active_emissions=active_emissions,
        surface_flux_to_vmr_factor=surface_flux_to_vmr_factor,
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
        hyai_hpa=hyai,
        hybi=hybi,
        top_edge_hpa=float(hyai[-1]),
        dt_s=dt_s,
        specific_humidity_top=vdiff.specific_humidity_kg_kg,
        include_diagnostics_fields=False,
    )
    convection = _run_convection_input(convection_input, diagnostics=False)
    state = TracerField(
        names=tracer_field.names,
        data=transport_tracer_to_canonical(convection.tracer_conc),
        units=tracer_field.units,
        coords=tracer_field.coords,
    )
    xmass_hpa = None
    ymass_hpa = None
    zmass_hpa = None
    if include_flux_diagnostics:
        xmass_hpa = tpcore.xmass_hpa[np.newaxis, ::-1, :, :]
        ymass_hpa = tpcore.ymass_hpa[np.newaxis, ::-1, :, :]
        zmass_hpa = _tpcore_vertical_flux_edges(tpcore.vertical_mass_flux_hpa[::-1])
    return TransportStepResult(
        state=state,
        dry_air_mass_kg=next_dry_air_mass,
        delp_dry_hpa=next_delp,
        xmass_hpa=xmass_hpa,
        ymass_hpa=ymass_hpa,
        zmass_hpa=zmass_hpa,
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
    active_emissions: TracerField | SurfaceEmissions | None = None,
    surface_flux_to_vmr_factor: np.ndarray | None = None,
    tpcore_static_terms: TpcoreStaticTerms | None = None,
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
        static_terms=tpcore_static_terms,
    )
    try:
        validate_tpcore_branch_support(setup)
    except NotImplementedError as exc:
        report = analyze_tpcore_branches(setup)
        raise NotImplementedError(_format_tpcore_branch_preflight_error(report)) from exc

    tpcore = run_tpcore_one_step_with_setup(
        tracer_conc=canonical_time_slice(tracer_field.data),
        setup=setup,
        area_m2=area,
        validate_branches=False,
    )
    next_delp = dry_pressure_thickness_from_surface_hpa(
        forcing.dry_surface_pressure_hpa,
        hyai,
        hybi,
    )
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
        hyai_hpa=hyai,
        hybi=hybi,
        top_edge_hpa=float(hyai[-1]),
        dt_s=dt_s,
        active_emissions=active_emissions,
        surface_flux_to_vmr_factor=surface_flux_to_vmr_factor,
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
        hyai_hpa=hyai,
        hybi=hybi,
        top_edge_hpa=float(hyai[-1]),
        dt_s=dt_s,
        specific_humidity_top=vdiff.specific_humidity_kg_kg,
        include_diagnostics_fields=True,
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
    hyai_hpa: np.ndarray,
    hybi: np.ndarray,
    *,
    top_edge_hpa: float,
    dt_s: float,
    active_emissions: TracerField | SurfaceEmissions | None = None,
    surface_flux_to_vmr_factor: np.ndarray | None = None,
):
    return _run_vdiff_input(
        _build_vdiff_input_after_tpcore(
            tracer_field,
            forcing,
            dry_air_mass,
            delp_dry_hpa,
            area,
            hyai_hpa,
            hybi,
            top_edge_hpa=top_edge_hpa,
            dt_s=dt_s,
            active_emissions=active_emissions,
            surface_flux_to_vmr_factor=surface_flux_to_vmr_factor,
        ),
        diagnostics=False,
    )


def _build_vdiff_input_after_tpcore(
    tracer_field: TracerField,
    forcing: TransportForcing,
    dry_air_mass: np.ndarray,
    delp_dry_hpa: np.ndarray,
    area: np.ndarray,
    hyai_hpa: np.ndarray,
    hybi: np.ndarray,
    *,
    top_edge_hpa: float,
    dt_s: float,
    active_emissions: TracerField | SurfaceEmissions | None = None,
    surface_flux_to_vmr_factor: np.ndarray | None = None,
) -> VdiffInputState:
    pedge = pressure_edges_from_surface_hpa(forcing.wet_surface_pressure_hpa, hyai_hpa, hybi)[0]
    pmid = 0.5 * (pedge[:-1] + pedge[1:])
    temperature = np.asarray(forcing.temperature_k[0], dtype=np.float64)
    sphu = np.asarray(forcing.specific_humidity_kg_kg[0], dtype=np.float64)
    virtual_temperature = _virtual_temperature_k(temperature, sphu)
    bxheight = _hydrostatic_box_height_m(pedge, virtual_temperature)
    ntracer = len(tracer_field.names)
    surface_flux = _surface_flux_from_active_emissions(
        tracer_field,
        active_emissions,
        nlat=tracer_field.data.shape[2],
        nlon=tracer_field.data.shape[3],
        ntracer=ntracer,
    )
    surface_flux_for_vdiff = _scale_surface_flux_for_vdiff(
        surface_flux,
        surface_flux_to_vmr_factor,
        ntracer=ntracer,
    )
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
        surface_flux_kg_m2_s=surface_flux,
        surface_flux_for_vdiff=surface_flux_for_vdiff,
    )


def _scale_surface_flux_for_vdiff(
    surface_flux: np.ndarray,
    surface_flux_to_vmr_factor: np.ndarray | None,
    *,
    ntracer: int,
) -> np.ndarray:
    if surface_flux_to_vmr_factor is None:
        return surface_flux
    factor = np.asarray(surface_flux_to_vmr_factor, dtype=np.float64)
    if factor.shape != (ntracer,):
        raise ValueError(f"surface_flux_to_vmr_factor shape {factor.shape} does not match {(ntracer,)}")
    return np.ascontiguousarray(surface_flux * factor[np.newaxis, np.newaxis, :])


def _surface_flux_from_active_emissions(
    tracer_field: TracerField,
    active_emissions: TracerField | SurfaceEmissions | None,
    *,
    nlat: int,
    nlon: int,
    ntracer: int,
) -> np.ndarray:
    if active_emissions is None:
        return np.zeros((nlat, nlon, ntracer), dtype=np.float64)
    if active_emissions.names != tracer_field.names:
        raise ValueError("active emissions names do not match tracer field names")
    if isinstance(active_emissions, SurfaceEmissions):
        if active_emissions.data.shape != (nlat, nlon, ntracer):
            raise ValueError(
                f"active surface emissions shape {active_emissions.data.shape} does not match {(nlat, nlon, ntracer)}"
            )
        return np.ascontiguousarray(active_emissions.data)
    if active_emissions.data.shape != tracer_field.data.shape:
        raise ValueError(
            f"active emissions shape {active_emissions.data.shape} does not match tracer field shape {tracer_field.data.shape}"
        )

    data = np.asarray(active_emissions.data, dtype=np.float64)
    if data.shape[0] != 1:
        raise ValueError(f"active emissions must contain one time slice, found shape {data.shape}")
    above_surface = data[:, :-1, :, :, :]
    if np.any(above_surface != 0.0):
        raise ValueError("vertically distributed emissions are not yet supported for non-local PBL mixing")
    return np.ascontiguousarray(data[0, -1, :, :, :])


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
        surface_flux_kg_m2_s=state.surface_flux_for_vdiff,
        diagnostics=diagnostics,
        reuse_output=not diagnostics,
    )


def _run_convection_after_vdiff(
    tracer_field: TracerField,
    forcing: TransportForcing,
    delp_dry_hpa: np.ndarray,
    area: np.ndarray,
    hyai_hpa: np.ndarray,
    hybi: np.ndarray,
    *,
    top_edge_hpa: float,
    dt_s: float,
    include_diagnostics_fields: bool = True,
):
    return _run_convection_input(
        _build_convection_input_after_vdiff(
            tracer_field,
            forcing,
            delp_dry_hpa,
            area,
            hyai_hpa,
            hybi,
            top_edge_hpa=top_edge_hpa,
            dt_s=dt_s,
            include_diagnostics_fields=include_diagnostics_fields,
        )
    )


def _build_convection_input_after_vdiff(
    tracer_field: TracerField,
    forcing: TransportForcing,
    delp_dry_hpa: np.ndarray,
    area: np.ndarray,
    hyai_hpa: np.ndarray,
    hybi: np.ndarray,
    *,
    top_edge_hpa: float,
    dt_s: float,
    specific_humidity_top: np.ndarray | None = None,
    include_diagnostics_fields: bool = True,
) -> ConvectionInputState:
    delp = np.asarray(delp_dry_hpa[0], dtype=np.float64)
    delp_top = delp[::-1]
    bxheight = None
    pficu = None
    pflcu = None
    temperature_top = None
    precccon = None
    if include_diagnostics_fields:
        pedge = pressure_edges_from_surface_hpa(forcing.wet_surface_pressure_hpa, hyai_hpa, hybi)[0]
        temperature = np.asarray(forcing.temperature_k[0], dtype=np.float64)
        if specific_humidity_top is None:
            sphu = np.asarray(forcing.specific_humidity_kg_kg[0], dtype=np.float64)
        else:
            sphu_top = np.asarray(specific_humidity_top, dtype=np.float64)
            if sphu_top.shape != temperature.shape:
                raise ValueError(
                    f"specific_humidity_top shape {sphu_top.shape} does not match temperature {temperature.shape}"
                )
            sphu = sphu_top[::-1]
        virtual_temperature = _virtual_temperature_k(temperature, sphu)
        bxheight = _hydrostatic_box_height_m(pedge, virtual_temperature)[::-1]
        pficu = np.asarray(forcing.convective_ice_flux_kg_m2_s[0], dtype=np.float64)[::-1]
        pflcu = np.asarray(forcing.convective_liquid_flux_kg_m2_s[0], dtype=np.float64)[::-1]
        temperature_top = temperature[::-1]
        precccon = np.asarray(forcing.convective_precip_mm_day[0], dtype=np.float64)
    return ConvectionInputState(
        tracer_conc=canonical_time_slice(tracer_field.data),
        cmfmc_kg_m2_s=np.asarray(forcing.convective_mass_flux_kg_m2_s[0], dtype=np.float64)[::-1],
        dtrain_kg_m2_s=np.asarray(forcing.convective_detrainment_kg_m2_s[0], dtype=np.float64)[::-1],
        dqrcu_kg_kg_s=np.asarray(forcing.convective_precip_prod_kg_kg_s[0], dtype=np.float64)[::-1],
        reevapcn_kg_kg_s=np.asarray(forcing.convective_precip_reevap_kg_kg_s[0], dtype=np.float64)[::-1],
        delp_dry_hpa=delp_top,
        delp_hpa=delp_top,
        area_m2=area,
        bxheight_m=bxheight,
        pficu_kg_m2_s=pficu,
        pflcu_kg_m2_s=pflcu,
        temperature_k=temperature_top,
        precccon_mm_day=precccon,
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


def _virtual_temperature_k(temperature_k: np.ndarray, specific_humidity_kg_kg: np.ndarray) -> np.ndarray:
    temperature = np.asarray(temperature_k, dtype=np.float64)
    sphu = np.asarray(specific_humidity_kg_kg, dtype=np.float64)
    water_vapor_vv_dry = AIRMW_G_PER_MOL * sphu / (H2OMW_G_PER_MOL * (1.0 - sphu))
    xh2o = water_vapor_vv_dry / (1.0 + water_vapor_vv_dry)
    return temperature / (1.0 - xh2o * (1.0 - H2OMW_G_PER_MOL / AIRMW_G_PER_MOL))


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

def _dry_air_mass_from_forcing_start(forcing: TransportForcing, grid: TransportGrid) -> np.ndarray:
    delp = dry_pressure_thickness_from_surface_hpa(
        forcing.dry_surface_pressure_start_hpa,
        grid.hyai_hpa,
        grid.hybi,
    )
    return dry_air_mass_from_pressure(delp, grid.area_m2)

def _load_window_forcing(
    forcing_provider: TransportForcingProvider,
    *,
    step: int,
    dt_s: float,
) -> TransportForcing:
    current = forcing_provider.start + timedelta(seconds=int(step) * float(dt_s))
    return forcing_provider.forcing_for_step(current, dt_s=dt_s)
