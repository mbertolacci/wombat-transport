from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

try:  # Optional acceleration path; NumPy remains the reference fallback.
    from numba import njit
except ImportError:  # pragma: no cover - exercised in environments without numba.
    njit = None


G0_M_PER_S2 = 9.80665
RD_J_PER_KG_K = 287.0
RV_J_PER_KG_K = 461.0
CPAIR_J_PER_KG_K = 1004.64
LATVAP_J_PER_KG = 2.5104e6
ZVIR = RV_J_PER_KG_K / RD_J_PER_KG_K - 1.0
CAPPA = RD_J_PER_KG_K / CPAIR_J_PER_KG_K
VON_KARMAN = 0.4
ZKMIN_M2_S = 0.01

_PBL_PRESSURE_CAP_HPA = 400.0
_PBLH_ARCHIVED = True
_BETAM = 15.0
_BETAS = 5.0
_BETAH = 15.0
_FAK = 8.50
_FAKN = 7.20
_SFFRAC = 0.1
_CCON = _FAK * _SFFRAC * VON_KARMAN
_BINM = _BETAM * _SFFRAC
_BINH = _BETAH * _SFFRAC
_ONET = 1.0 / 3.0
_NUMBA_AVAILABLE = njit is not None


@dataclass(frozen=True)
class PblHeightState:
    """GEOS-Chem PBL-top bookkeeping in bottom-to-top vertical order."""

    pbl_top_m: np.ndarray
    pbl_top_hpa: np.ndarray
    pbl_top_l: np.ndarray
    pbl_thick_hpa: np.ndarray
    in_pbl: np.ndarray
    f_of_pbl: np.ndarray
    f_under_pbl_top: np.ndarray
    pbl_max_l: int


@dataclass(frozen=True)
class VdiffDrResult:
    """One-step GEOS-Chem non-local VDIFFDR result in canonical vertical order."""

    tracer_conc: np.ndarray
    specific_humidity_kg_kg: np.ndarray
    kvh_m2_s: np.ndarray
    kvm_m2_s: np.ndarray
    pbl_top_m: np.ndarray
    tpert_k: np.ndarray
    qpert_kg_kg: np.ndarray
    negative_count_before_clip: int
    negative_count_after_clip: int
    initial_tracer_mass: np.ndarray
    final_tracer_mass: np.ndarray


@dataclass(frozen=True)
class _VdiffFullGridWorkspace:
    pmid: np.ndarray
    pint: np.ndarray
    rpdel: np.ndarray
    rpdeli: np.ndarray
    zm: np.ndarray
    thp: np.ndarray
    kvf: np.ndarray
    kvh: np.ndarray
    kvm: np.ndarray
    cgsh: np.ndarray
    cgs: np.ndarray
    tpert: np.ndarray
    qpert: np.ndarray
    tmp1: np.ndarray
    dshbot: np.ndarray
    rrho: np.ndarray
    khfs: np.ndarray
    kshfs: np.ndarray
    thvsrf: np.ndarray
    heatv: np.ndarray
    obklen: np.ndarray
    fak1: np.ndarray
    phiminv: np.ndarray
    phihinv: np.ndarray
    wm: np.ndarray
    fak2: np.ndarray
    fak3: np.ndarray
    pblk: np.ndarray
    pr: np.ndarray
    potbar: np.ndarray
    cah: np.ndarray
    cch: np.ndarray
    zeh: np.ndarray
    termh: np.ndarray
    tracer_diffused: np.ndarray
    tracer_ratio: np.ndarray
    tracer_after_mass: np.ndarray
    shmx: np.ndarray
    zfq_scalar: np.ndarray
    sphu_diffused: np.ndarray


_VDIFF_FULLGRID_WORKSPACE: _VdiffFullGridWorkspace | None = None


def _get_vdiff_fullgrid_workspace(nlev: int, nlon: int, ntracer: int) -> _VdiffFullGridWorkspace:
    global _VDIFF_FULLGRID_WORKSPACE
    existing = _VDIFF_FULLGRID_WORKSPACE
    if (
        existing is not None
        and existing.pmid.shape == (nlon, nlev)
        and existing.tracer_diffused.shape == (nlon, nlev, ntracer)
        and existing.tracer_ratio.shape == (nlon, ntracer)
        and existing.tracer_after_mass.shape == (nlon, ntracer)
    ):
        return existing

    lev_shape = (nlon, nlev)
    edge_shape = (nlon, nlev + 1)
    tracer_shape = (nlon, ntracer)
    _VDIFF_FULLGRID_WORKSPACE = _VdiffFullGridWorkspace(
        pmid=np.empty(lev_shape, dtype=np.float64),
        pint=np.empty(edge_shape, dtype=np.float64),
        rpdel=np.empty(lev_shape, dtype=np.float64),
        rpdeli=np.empty(lev_shape, dtype=np.float64),
        zm=np.empty(lev_shape, dtype=np.float64),
        thp=np.empty(lev_shape, dtype=np.float64),
        kvf=np.empty(edge_shape, dtype=np.float64),
        kvh=np.empty(edge_shape, dtype=np.float64),
        kvm=np.empty(edge_shape, dtype=np.float64),
        cgsh=np.empty(edge_shape, dtype=np.float64),
        cgs=np.empty(edge_shape, dtype=np.float64),
        tpert=np.empty(nlon, dtype=np.float64),
        qpert=np.empty(nlon, dtype=np.float64),
        tmp1=np.empty(nlon, dtype=np.float64),
        dshbot=np.empty(nlon, dtype=np.float64),
        rrho=np.empty(nlon, dtype=np.float64),
        khfs=np.empty(nlon, dtype=np.float64),
        kshfs=np.empty(nlon, dtype=np.float64),
        thvsrf=np.empty(nlon, dtype=np.float64),
        heatv=np.empty(nlon, dtype=np.float64),
        obklen=np.empty(nlon, dtype=np.float64),
        fak1=np.empty(nlon, dtype=np.float64),
        phiminv=np.empty(nlon, dtype=np.float64),
        phihinv=np.empty(nlon, dtype=np.float64),
        wm=np.empty(nlon, dtype=np.float64),
        fak2=np.empty(nlon, dtype=np.float64),
        fak3=np.empty(nlon, dtype=np.float64),
        pblk=np.empty(nlon, dtype=np.float64),
        pr=np.empty(nlon, dtype=np.float64),
        potbar=np.empty(edge_shape, dtype=np.float64),
        cah=np.empty(lev_shape, dtype=np.float64),
        cch=np.empty(lev_shape, dtype=np.float64),
        zeh=np.empty(lev_shape, dtype=np.float64),
        termh=np.empty(lev_shape, dtype=np.float64),
        tracer_diffused=np.empty((nlon, nlev, ntracer), dtype=np.float64),
        tracer_ratio=np.empty(tracer_shape, dtype=np.float64),
        tracer_after_mass=np.empty(tracer_shape, dtype=np.float64),
        shmx=np.empty(lev_shape, dtype=np.float64),
        zfq_scalar=np.empty(lev_shape, dtype=np.float64),
        sphu_diffused=np.empty(lev_shape, dtype=np.float64),
    )
    return _VDIFF_FULLGRID_WORKSPACE


def _tracer_working_mass(tracer_conc: np.ndarray, dry_air_mass_top: np.ndarray) -> np.ndarray:
    """Return total tracer mass from VDIFF working layout."""

    if _numba_vdiff_enabled():
        return _tracer_working_mass_numba(tracer_conc, dry_air_mass_top)
    return np.sum(tracer_conc * dry_air_mass_top[:, :, :, np.newaxis], axis=(0, 1, 2))


def _tracer_working_mass_numba(tracer_conc: np.ndarray, dry_air_mass_top: np.ndarray) -> np.ndarray:
    if not _NUMBA_AVAILABLE:
        return np.sum(tracer_conc * dry_air_mass_top[:, :, :, np.newaxis], axis=(0, 1, 2))
    return _tracer_working_mass_numba_kernel(tracer_conc, dry_air_mass_top)


def compute_pbl_height(
    *,
    pbl_height_m: np.ndarray,
    bxheight_m: np.ndarray,
    pressure_edges_hpa: np.ndarray,
    virtual_temperature_k: np.ndarray,
) -> PblHeightState:
    """Port GEOS-Chem ``Compute_Pbl_Height`` for the fixed-grid array layout.

    Arrays use Wombat/NetCDF order with vertical level 0 nearest the surface.
    ``pressure_edges_hpa`` has one more vertical edge than ``bxheight_m``.
    """

    pbl_height = np.asarray(pbl_height_m, dtype=np.float64)
    bxheight = np.asarray(bxheight_m, dtype=np.float64)
    pedge = np.asarray(pressure_edges_hpa, dtype=np.float64)
    tv = np.asarray(virtual_temperature_k, dtype=np.float64)
    if bxheight.ndim != 3:
        raise ValueError(f"bxheight_m must be 3-D (lev, lat, lon), found {bxheight.shape}")
    if tv.shape != bxheight.shape:
        raise ValueError(f"virtual_temperature_k shape {tv.shape} does not match bxheight_m {bxheight.shape}")
    if pedge.shape != (bxheight.shape[0] + 1, bxheight.shape[1], bxheight.shape[2]):
        raise ValueError(f"pressure_edges_hpa shape {pedge.shape} is incompatible with bxheight_m {bxheight.shape}")
    if pbl_height.shape != bxheight.shape[1:]:
        raise ValueError(f"pbl_height_m shape {pbl_height.shape} does not match horizontal grid {bxheight.shape[1:]}")
    if np.any(pbl_height <= 0.0):
        raise ValueError("pbl_height_m must be positive")

    nlev, nlat, nlon = bxheight.shape
    in_pbl = np.zeros((nlev, nlat, nlon), dtype=bool)
    f_of_pbl = np.zeros((nlev, nlat, nlon), dtype=np.float64)
    f_under = np.zeros((nlev, nlat, nlon), dtype=np.float64)
    pbl_top_hpa = np.empty((nlat, nlon), dtype=np.float64)
    pbl_top_l = np.empty((nlat, nlon), dtype=np.float64)
    pbl_thick = np.empty((nlat, nlon), dtype=np.float64)

    for j in range(nlat):
        for i in range(nlon):
            lower_edge_height = 0.0
            found = False
            for lev in range(nlev):
                next_edge_height = lower_edge_height + bxheight[lev, j, i]
                if next_edge_height >= pbl_height[j, i]:
                    top_hpa = pedge[lev, j, i] * np.exp(
                        -(pbl_height[j, i] - lower_edge_height) * G0_M_PER_S2 / (RD_J_PER_KG_K * tv[lev, j, i])
                    )
                    layer_thick = pedge[lev, j, i] - pedge[lev + 1, j, i]
                    pbl_mass_thick = pedge[lev, j, i] - top_hpa
                    pbl_top_hpa[j, i] = top_hpa
                    pbl_thick[j, i] = pedge[0, j, i] - top_hpa
                    f_of_pbl[lev, j, i] = pbl_mass_thick
                    f_under[lev, j, i] = pbl_mass_thick / layer_thick
                    pbl_top_l[j, i] = float(lev) + f_under[lev, j, i]
                    found = True
                    break

                in_pbl[lev, j, i] = True
                f_under[lev, j, i] = 1.0
                f_of_pbl[lev, j, i] = pedge[lev, j, i] - pedge[lev + 1, j, i]
                lower_edge_height = next_edge_height

            if not found:
                raise ValueError(f"PBL height {pbl_height[j, i]} m exceeds modeled column height at lat/lon index {j}/{i}")
            f_of_pbl[:, j, i] /= pbl_thick[j, i]

    sums = np.sum(f_of_pbl, axis=0)
    if np.any(np.abs(sums - 1.0) > 1.0e-3):
        raise ValueError("computed F_of_PBL does not sum to 1 within GEOS-Chem tolerance")

    return PblHeightState(
        pbl_top_m=pbl_height.copy(),
        pbl_top_hpa=pbl_top_hpa,
        pbl_top_l=pbl_top_l,
        pbl_thick_hpa=pbl_thick,
        in_pbl=in_pbl,
        f_of_pbl=f_of_pbl,
        f_under_pbl_top=f_under,
        pbl_max_l=int(np.max(np.ceil(pbl_top_l))),
    )


def mix_full_pbl(
    tracer_conc: np.ndarray,
    dry_air_mass_kg: np.ndarray,
    pbl_top_l: np.ndarray,
) -> np.ndarray:
    """Port the compact mass-weighted mixing core used by GEOS-Chem ``TurbDay``.

    This is the full-PBL mixer, not the configured non-local VDIFF scheme. It is
    useful as the first PBL bookkeeping/mass-conservation target and as a small
    oracle surface before porting the larger non-local path.
    """

    tracer = np.asarray(tracer_conc, dtype=np.float64)
    dry_mass = np.asarray(dry_air_mass_kg, dtype=np.float64)
    top_l = np.asarray(pbl_top_l, dtype=np.float64)
    if tracer.ndim != 4:
        raise ValueError(f"tracer_conc must be 4-D (tracer, lev, lat, lon), found {tracer.shape}")
    if dry_mass.shape != tracer.shape[1:]:
        raise ValueError(f"dry_air_mass_kg shape {dry_mass.shape} does not match tracer grid {tracer.shape[1:]}")
    if top_l.shape != tracer.shape[2:]:
        raise ValueError(f"pbl_top_l shape {top_l.shape} does not match horizontal grid {tracer.shape[2:]}")

    mixed = tracer.copy()
    ntracer, nlev, nlat, nlon = tracer.shape
    for j in range(nlat):
        for i in range(nlon):
            imix = int(np.ceil(top_l[j, i]))
            if imix < 1 or imix > nlev:
                raise ValueError(f"pbl_top_l at lat/lon index {j}/{i} is outside model levels: {top_l[j, i]}")
            top_index = imix - 1
            fpbl = top_l[j, i] - float(imix - 1)
            if fpbl <= 0.0 or fpbl > 1.0:
                raise ValueError(f"invalid fractional PBL top at lat/lon index {j}/{i}: {top_l[j, i]}")

            full_mass = dry_mass[:top_index, j, i]
            top_mass = dry_mass[top_index, j, i] * fpbl
            air_mass = np.sum(full_mass) + top_mass
            if air_mass <= 0.0:
                raise ValueError(f"non-positive PBL air mass at lat/lon index {j}/{i}")

            tracer_mass = np.sum(tracer[:, :top_index, j, i] * full_mass[np.newaxis, :], axis=1)
            tracer_mass = tracer_mass + tracer[:, top_index, j, i] * top_mass
            mean = tracer_mass / air_mass
            if top_index > 0:
                mixed[:, :top_index, j, i] = mean[:, np.newaxis]
            mixed[:, top_index, j, i] = tracer[:, top_index, j, i] + fpbl * (mean - tracer[:, top_index, j, i])

    return mixed


def run_vdiffdr_one_step(
    *,
    tracer_conc: np.ndarray,
    u_m_s: np.ndarray,
    v_m_s: np.ndarray,
    temperature_k: np.ndarray,
    specific_humidity_kg_kg: np.ndarray,
    pmid_hpa: np.ndarray,
    pedge_hpa: np.ndarray,
    virtual_temperature_k: np.ndarray,
    bxheight_m: np.ndarray,
    dry_air_mass_kg: np.ndarray,
    pbl_top_m: np.ndarray,
    hflux_w_m2: np.ndarray,
    eflux_w_m2: np.ndarray,
    ustar_m_s: np.ndarray,
    area_m2: np.ndarray,
    dt_s: float = 600.0,
    surface_flux_kg_m2_s: np.ndarray | None = None,
    diagnostics: bool = True,
) -> VdiffDrResult:
    """Port GEOS-Chem ``VDIFFDR`` for one non-local PBL mixing step.

    Inputs and outputs use canonical transport order: level 0 is the model top
    and tracer is the last axis.
    """

    tracer = np.asarray(tracer_conc, dtype=np.float64)
    u = np.asarray(u_m_s, dtype=np.float64)
    v = np.asarray(v_m_s, dtype=np.float64)
    temperature = np.asarray(temperature_k, dtype=np.float64)
    sphu = np.asarray(specific_humidity_kg_kg, dtype=np.float64)
    pmid = np.asarray(pmid_hpa, dtype=np.float64)
    pedge = np.asarray(pedge_hpa, dtype=np.float64)
    tv = np.asarray(virtual_temperature_k, dtype=np.float64)
    bxheight = np.asarray(bxheight_m, dtype=np.float64)
    dry_mass = np.asarray(dry_air_mass_kg, dtype=np.float64)
    pblh = np.asarray(pbl_top_m, dtype=np.float64)
    hflux = np.asarray(hflux_w_m2, dtype=np.float64)
    eflux = np.asarray(eflux_w_m2, dtype=np.float64)
    ustar = np.asarray(ustar_m_s, dtype=np.float64)
    area = np.asarray(area_m2, dtype=np.float64)

    if tracer.ndim != 4:
        raise ValueError(f"tracer_conc must be 4-D (lev, lat, lon, tracer), found {tracer.shape}")
    nlev, nlat, nlon, ntracer = tracer.shape
    grid_shape = (nlev, nlat, nlon)
    edge_shape = (nlev + 1, nlat, nlon)
    horizontal_shape = (nlat, nlon)
    for name, value in (
        ("u_m_s", u),
        ("v_m_s", v),
        ("temperature_k", temperature),
        ("specific_humidity_kg_kg", sphu),
        ("pmid_hpa", pmid),
        ("virtual_temperature_k", tv),
        ("bxheight_m", bxheight),
        ("dry_air_mass_kg", dry_mass),
    ):
        if value.shape != grid_shape:
            raise ValueError(f"{name} shape {value.shape} does not match tracer grid {grid_shape}")
    if pedge.shape != edge_shape:
        raise ValueError(f"pedge_hpa shape {pedge.shape} does not match tracer edge grid {edge_shape}")
    for name, value in (
        ("pbl_top_m", pblh),
        ("hflux_w_m2", hflux),
        ("eflux_w_m2", eflux),
        ("ustar_m_s", ustar),
        ("area_m2", area),
    ):
        if value.shape != horizontal_shape:
            raise ValueError(f"{name} shape {value.shape} does not match horizontal grid {horizontal_shape}")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    if surface_flux_kg_m2_s is None:
        surface_flux = np.zeros((nlat, nlon, ntracer), dtype=np.float64)
    else:
        surface_flux = np.asarray(surface_flux_kg_m2_s, dtype=np.float64)
        if surface_flux.shape != (nlat, nlon, ntracer):
            raise ValueError(
                f"surface_flux_kg_m2_s shape {surface_flux.shape} does not match {(nlat, nlon, ntracer)}"
            )

    numba_vdiff = _numba_vdiff_enabled()
    surface_flux_is_zero = bool(not np.any(surface_flux != 0.0)) if numba_vdiff else False
    if numba_vdiff and surface_flux_is_zero and not diagnostics:
        return _run_vdiffdr_one_step_fullgrid_numba(
            tracer_top=tracer,
            u_top=u,
            v_top=v,
            temperature_top=temperature,
            sphu_top=sphu,
            pmid_hpa=pmid,
            pint_hpa=pedge,
            virtual_temperature_top=tv,
            bxheight_top=bxheight,
            dry_mass_top=dry_mass,
            pblh_m=pblh,
            hflux_w_m2=hflux,
            water_flux_kg_m2_s=eflux / LATVAP_J_PER_KG,
            ustar_m_s=ustar,
            dt_s=float(dt_s),
            npbl=_max_pbl_levels_from_pressure(pmid),
        )

    pmid_pa = pmid * 100.0
    pedge_pa = pedge * 100.0
    thp_top = temperature * (1.0e5 / pmid_pa) ** CAPPA
    zm_top = _vdiff_midpoint_height_top_m(
        bxheight_m=bxheight,
        pmid_pa=pmid_pa,
        pedge_pa=pedge_pa,
        virtual_temperature_k=tv,
    )
    rpdel_top = 1.0 / (pedge_pa[1:] - pedge_pa[:-1])
    rpdeli_top = np.zeros_like(rpdel_top)
    rpdeli_top[:-1] = 1.0 / (pmid_pa[1:] - pmid_pa[:-1])

    ntopfl = 0
    npbl = _max_pbl_levels_from_pressure(pmid_hpa)
    ml2 = np.zeros(nlev + 1, dtype=np.float64)
    if nlev > 1:
        ml2[1:nlev] = 900.0

    tracer_after_top = np.empty_like(tracer)
    sphu_after_top = sphu.copy()
    kvh_top = np.zeros((nlev + 1, nlat, nlon), dtype=np.float64)
    kvm_top = np.zeros_like(kvh_top)
    tpert = np.zeros((nlat, nlon), dtype=np.float64)
    qpert = np.zeros((nlat, nlon), dtype=np.float64)
    negative_before = 0

    initial_mass = _tracer_working_mass(tracer, dry_mass)

    for lat_index in range(nlat):
        latitude_args = dict(
            tracer_top=tracer[:, lat_index, :, :].transpose(1, 0, 2),
            u_top=u[:, lat_index, :].T,
            v_top=v[:, lat_index, :].T,
            temperature_top=temperature[:, lat_index, :].T,
            sphu_top=sphu[:, lat_index, :].T,
            pmid_top=pmid_pa[:, lat_index, :].T,
            pint_top=pedge_pa[:, lat_index, :].T,
            rpdel_top=rpdel_top[:, lat_index, :].T,
            rpdeli_top=rpdeli_top[:, lat_index, :].T,
            zm_top=zm_top[:, lat_index, :].T,
            thp_top=thp_top[:, lat_index, :].T,
            dry_mass_top=dry_mass[:, lat_index, :].T,
            pblh_m=pblh[lat_index, :],
            hflux_w_m2=hflux[lat_index, :],
            water_flux_kg_m2_s=eflux[lat_index, :] / LATVAP_J_PER_KG,
            surface_flux_kg_m2_s=surface_flux[lat_index],
            ustar_m_s=ustar[lat_index, :],
            area_m2=area[lat_index, :],
            dt_s=float(dt_s),
            ntopfl=ntopfl,
            npbl=npbl,
            ml2=ml2,
        )
        if numba_vdiff:
            column = _run_vdiff_latitude_numba(**latitude_args, surface_flux_is_zero=surface_flux_is_zero)
        else:
            column = _run_vdiff_latitude(**latitude_args)
        tracer_after_top[:, lat_index, :, :] = column.tracer_top.transpose(1, 0, 2)
        sphu_after_top[:, lat_index, :] = column.sphu_top.T
        kvh_top[:, lat_index, :] = column.kvh_top.T
        kvm_top[:, lat_index, :] = column.kvm_top.T
        tpert[lat_index, :] = column.tpert
        qpert[lat_index, :] = column.qpert
        negative_before += column.negative_count_before_clip

    negative_after = int(np.count_nonzero(tracer_after_top < 0.0))
    final_mass = _tracer_working_mass(tracer_after_top, dry_mass)

    return VdiffDrResult(
        tracer_conc=tracer_after_top,
        specific_humidity_kg_kg=sphu_after_top,
        kvh_m2_s=kvh_top,
        kvm_m2_s=kvm_top,
        pbl_top_m=pblh.copy(),
        tpert_k=tpert,
        qpert_kg_kg=qpert,
        negative_count_before_clip=int(negative_before),
        negative_count_after_clip=negative_after,
        initial_tracer_mass=initial_mass,
        final_tracer_mass=final_mass,
    )


@dataclass(frozen=True)
class _VdiffLatitudeResult:
    tracer_top: np.ndarray
    sphu_top: np.ndarray
    kvh_top: np.ndarray
    kvm_top: np.ndarray
    tpert: np.ndarray
    qpert: np.ndarray
    negative_count_before_clip: int


def _vdiff_midpoint_height_m(
    *,
    bxheight_m: np.ndarray,
    pmid_pa: np.ndarray,
    pedge_pa: np.ndarray,
    virtual_temperature_k: np.ndarray,
) -> np.ndarray:
    midpoint = np.empty_like(bxheight_m)
    cumulative_top = np.cumsum(bxheight_m, axis=0)
    for lev in range(bxheight_m.shape[0]):
        midpoint[lev] = cumulative_top[lev] - np.log(pmid_pa[lev] / pedge_pa[lev + 1]) * (
            RD_J_PER_KG_K / G0_M_PER_S2
        ) * virtual_temperature_k[lev]
    return midpoint


def _vdiff_midpoint_height_top_m(
    *,
    bxheight_m: np.ndarray,
    pmid_pa: np.ndarray,
    pedge_pa: np.ndarray,
    virtual_temperature_k: np.ndarray,
) -> np.ndarray:
    top_edge_height = np.flip(np.cumsum(np.flip(bxheight_m, axis=0), axis=0), axis=0)
    return top_edge_height - np.log(pmid_pa / pedge_pa[:-1]) * (
        RD_J_PER_KG_K / G0_M_PER_S2
    ) * virtual_temperature_k


def _max_pbl_levels_from_pressure(pmid_hpa: np.ndarray) -> int:
    ref_pmid_top = np.mean(pmid_hpa, axis=(1, 2))
    nlev = ref_pmid_top.size
    break_index = 0
    for index in range(nlev - 1, -1, -1):
        if ref_pmid_top[index] < _PBL_PRESSURE_CAP_HPA:
            break_index = index
            break
    return max(1, nlev - (break_index + 1))


def _run_vdiff_latitude(
    *,
    tracer_top: np.ndarray,
    u_top: np.ndarray,
    v_top: np.ndarray,
    temperature_top: np.ndarray,
    sphu_top: np.ndarray,
    pmid_top: np.ndarray,
    pint_top: np.ndarray,
    rpdel_top: np.ndarray,
    rpdeli_top: np.ndarray,
    zm_top: np.ndarray,
    thp_top: np.ndarray,
    dry_mass_top: np.ndarray,
    pblh_m: np.ndarray,
    hflux_w_m2: np.ndarray,
    water_flux_kg_m2_s: np.ndarray,
    surface_flux_kg_m2_s: np.ndarray,
    ustar_m_s: np.ndarray,
    area_m2: np.ndarray,
    dt_s: float,
    ntopfl: int,
    npbl: int,
    ml2: np.ndarray,
) -> _VdiffLatitudeResult:
    nlon, nlev, ntracer = tracer_top.shape
    kvf = np.zeros((nlon, nlev + 1), dtype=np.float64)
    kvh = np.zeros_like(kvf)
    kvm = np.zeros_like(kvf)
    tpert = np.zeros(nlon, dtype=np.float64)
    qpert = np.zeros(nlon, dtype=np.float64)
    qmincg = np.zeros(ntracer, dtype=np.float64)

    tmp1 = dt_s * G0_M_PER_S2 * rpdel_top[:, nlev - 1]
    dshbot = water_flux_kg_m2_s * tmp1
    dtbot = hflux_w_m2 * tmp1 / CPAIR_J_PER_KG_K
    dqbot = surface_flux_kg_m2_s * tmp1[:, np.newaxis]
    kvf[:, nlev] = 0.0

    kvf[:, : ntopfl + 1] = 0.0
    thv = thp_top * (1.0 + ZVIR * sphu_top)
    for lev in range(ntopfl, nlev - 1):
        dvdz2 = (u_top[:, lev] - u_top[:, lev + 1]) ** 2 + (v_top[:, lev] - v_top[:, lev + 1]) ** 2
        dvdz2 = np.maximum(dvdz2, 1.0e-36)
        dz = zm_top[:, lev] - zm_top[:, lev + 1]
        dvdz2 = dvdz2 / dz**2
        sstab = G0_M_PER_S2 * 2.0 * (thv[:, lev] - thv[:, lev + 1]) / (
            (thv[:, lev] + thv[:, lev + 1]) * dz
        )
        rinub = sstab / dvdz2
        fstab = 1.0 / (1.0 + 10.0 * rinub * (1.0 + 8.0 * rinub))
        funst = np.maximum(1.0 - 18.0 * rinub, 0.0)
        fstab = np.where(rinub < 0.0, np.sqrt(funst), fstab)
        kvn = ml2[lev] * np.sqrt(dvdz2)
        kvf[:, lev + 1] = np.maximum(ZKMIN_M2_S, kvn * fstab)

    cgh, cgq, cgsh, cgs, tpert, qpert = _pbldif_archived_pblh(
        th=thp_top,
        q=sphu_top,
        z=zm_top,
        temperature=temperature_top,
        pmid=pmid_top,
        kvf=kvf,
        surface_flux=surface_flux_kg_m2_s,
        water_flux=water_flux_kg_m2_s,
        heat_flux=hflux_w_m2,
        pblh=pblh_m,
        ustar=ustar_m_s,
        npbl=npbl,
    )
    kvh[:] = kvf
    kvm[:] = kvf
    _apply_pbldif_diffusivity(
        kvh=kvh,
        kvm=kvm,
        cgs=cgs,
        cgh=cgh,
        cgq=cgq,
        cgsh=cgsh,
        tpert=tpert,
        qpert=qpert,
        th=thp_top,
        q=sphu_top,
        z=zm_top,
        temperature=temperature_top,
        pmid=pmid_top,
        kvf=kvf,
        surface_flux=surface_flux_kg_m2_s,
        water_flux=water_flux_kg_m2_s,
        heat_flux=hflux_w_m2,
        pblh=pblh_m,
        ustar=ustar_m_s,
        npbl=npbl,
    )

    qmx = _countergradient_adjust_tracers(
        tracer_top,
        rpdel_top,
        pint_top,
        temperature_top,
        kvh,
        cgq,
        dt_s,
        ntopfl,
        npbl,
        qmincg,
    )
    shmx = _countergradient_adjust_scalar(
        sphu_top,
        rpdel_top,
        pint_top,
        temperature_top,
        kvh,
        cgsh,
        dt_s,
        ntopfl,
        npbl,
        minimum=1.0e-12,
    )
    thx = _countergradient_adjust_scalar(
        thp_top,
        rpdel_top,
        pint_top,
        temperature_top,
        kvh,
        cgh,
        dt_s,
        ntopfl,
        npbl,
        minimum=None,
    )

    cch, zeh, termh = _diffusion_coefficients(kvh, rpdel_top, rpdeli_top, pint_top, temperature_top, dt_s, ntopfl)
    tracer_diffused = _qvdiff(qmx, dqbot, cch, zeh, termh, ntopfl)
    negative_count = int(np.count_nonzero(tracer_diffused < 0.0))
    tracer_diffused = np.where(tracer_diffused < 0.0, 0.0, tracer_diffused)
    tracer_diffused = _rescale_long_lived_mass(
        before=tracer_top,
        after=tracer_diffused,
        dry_mass_top=dry_mass_top,
        surface_flux=surface_flux_kg_m2_s,
        area_m2=area_m2,
        dt_s=dt_s,
        ntopfl=ntopfl,
    )

    sphu_diffused = _qvdiff(shmx[:, :, np.newaxis], dshbot[:, np.newaxis], cch, zeh, termh, ntopfl)[:, :, 0]
    sphu_diffused = np.where(sphu_diffused < 1.0e-12, 0.0, sphu_diffused)
    _ = _qvdiff(thx[:, :, np.newaxis], dtbot[:, np.newaxis], cch, zeh, termh, ntopfl)

    return _VdiffLatitudeResult(
        tracer_top=tracer_diffused,
        sphu_top=sphu_diffused,
        kvh_top=kvh,
        kvm_top=kvm,
        tpert=tpert,
        qpert=qpert,
        negative_count_before_clip=negative_count,
    )


def _numba_vdiff_mode() -> str:
    return os.environ.get("WOMBAT_VDIFF_NUMBA", "1").lower()


def _numba_vdiff_enabled() -> bool:
    if not _NUMBA_AVAILABLE:
        return False
    return _numba_vdiff_mode() not in {"0", "false", "no", "off", "none"}


def _run_vdiff_latitude_numba(
    *,
    tracer_top: np.ndarray,
    u_top: np.ndarray,
    v_top: np.ndarray,
    temperature_top: np.ndarray,
    sphu_top: np.ndarray,
    pmid_top: np.ndarray,
    pint_top: np.ndarray,
    rpdel_top: np.ndarray,
    rpdeli_top: np.ndarray,
    zm_top: np.ndarray,
    thp_top: np.ndarray,
    dry_mass_top: np.ndarray,
    pblh_m: np.ndarray,
    hflux_w_m2: np.ndarray,
    water_flux_kg_m2_s: np.ndarray,
    surface_flux_kg_m2_s: np.ndarray,
    ustar_m_s: np.ndarray,
    area_m2: np.ndarray,
    dt_s: float,
    ntopfl: int,
    npbl: int,
    ml2: np.ndarray,
    surface_flux_is_zero: bool,
) -> _VdiffLatitudeResult:
    if not _NUMBA_AVAILABLE:
        return _run_vdiff_latitude(
            tracer_top=tracer_top,
            u_top=u_top,
            v_top=v_top,
            temperature_top=temperature_top,
            sphu_top=sphu_top,
            pmid_top=pmid_top,
            pint_top=pint_top,
            rpdel_top=rpdel_top,
            rpdeli_top=rpdeli_top,
            zm_top=zm_top,
            thp_top=thp_top,
            dry_mass_top=dry_mass_top,
            pblh_m=pblh_m,
            hflux_w_m2=hflux_w_m2,
            water_flux_kg_m2_s=water_flux_kg_m2_s,
            surface_flux_kg_m2_s=surface_flux_kg_m2_s,
            ustar_m_s=ustar_m_s,
            area_m2=area_m2,
            dt_s=dt_s,
            ntopfl=ntopfl,
            npbl=npbl,
            ml2=ml2,
        )
    tracer_out, sphu_out, kvh, kvm, tpert, qpert, negative_count = _run_vdiff_latitude_numba_kernel(
        tracer_top,
        u_top,
        v_top,
        temperature_top,
        sphu_top,
        pmid_top,
        pint_top,
        rpdel_top,
        rpdeli_top,
        zm_top,
        thp_top,
        dry_mass_top,
        pblh_m,
        hflux_w_m2,
        water_flux_kg_m2_s,
        surface_flux_kg_m2_s,
        ustar_m_s,
        area_m2,
        dt_s,
        ntopfl,
        npbl,
        ml2,
        surface_flux_is_zero,
    )
    return _VdiffLatitudeResult(
        tracer_top=tracer_out,
        sphu_top=sphu_out,
        kvh_top=kvh,
        kvm_top=kvm,
        tpert=tpert,
        qpert=qpert,
        negative_count_before_clip=int(negative_count),
    )


def _run_vdiffdr_one_step_fullgrid_numba(
    *,
    tracer_top: np.ndarray,
    u_top: np.ndarray,
    v_top: np.ndarray,
    temperature_top: np.ndarray,
    sphu_top: np.ndarray,
    pmid_hpa: np.ndarray,
    pint_hpa: np.ndarray,
    virtual_temperature_top: np.ndarray,
    bxheight_top: np.ndarray,
    dry_mass_top: np.ndarray,
    pblh_m: np.ndarray,
    hflux_w_m2: np.ndarray,
    water_flux_kg_m2_s: np.ndarray,
    ustar_m_s: np.ndarray,
    dt_s: float,
    npbl: int,
) -> VdiffDrResult:
    if not _NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    nlev, _, nlon, ntracer = tracer_top.shape
    tracer_out = np.empty_like(tracer_top)
    sphu_out = np.empty_like(sphu_top)
    workspace = _get_vdiff_fullgrid_workspace(nlev, nlon, ntracer)
    negative_count = _run_vdiffdr_fullgrid_zero_flux_numba_kernel(
        tracer_top,
        u_top,
        v_top,
        temperature_top,
        sphu_top,
        pmid_hpa,
        pint_hpa,
        virtual_temperature_top,
        bxheight_top,
        dry_mass_top,
        pblh_m,
        hflux_w_m2,
        water_flux_kg_m2_s,
        ustar_m_s,
        dt_s,
        npbl,
        tracer_out,
        sphu_out,
        workspace.pmid,
        workspace.pint,
        workspace.rpdel,
        workspace.rpdeli,
        workspace.zm,
        workspace.thp,
        workspace.kvf,
        workspace.kvh,
        workspace.kvm,
        workspace.cgsh,
        workspace.cgs,
        workspace.tpert,
        workspace.qpert,
        workspace.tmp1,
        workspace.dshbot,
        workspace.rrho,
        workspace.khfs,
        workspace.kshfs,
        workspace.thvsrf,
        workspace.heatv,
        workspace.obklen,
        workspace.fak1,
        workspace.phiminv,
        workspace.phihinv,
        workspace.wm,
        workspace.fak2,
        workspace.fak3,
        workspace.pblk,
        workspace.pr,
        workspace.potbar,
        workspace.cah,
        workspace.cch,
        workspace.zeh,
        workspace.termh,
        workspace.tracer_diffused,
        workspace.tracer_ratio,
        workspace.tracer_after_mass,
        workspace.shmx,
        workspace.zfq_scalar,
        workspace.sphu_diffused,
    )
    empty = np.empty(0, dtype=np.float64)
    return VdiffDrResult(
        tracer_conc=tracer_out,
        specific_humidity_kg_kg=sphu_out,
        kvh_m2_s=empty,
        kvm_m2_s=empty,
        pbl_top_m=pblh_m.copy(),
        tpert_k=empty,
        qpert_kg_kg=empty,
        negative_count_before_clip=int(negative_count),
        negative_count_after_clip=0,
        initial_tracer_mass=empty,
        final_tracer_mass=empty,
    )


if njit is not None:

    @njit(cache=True)
    def _tracer_working_mass_numba_kernel(tracer_conc: np.ndarray, dry_air_mass_top: np.ndarray) -> np.ndarray:
        nlev = tracer_conc.shape[0]
        nlat = tracer_conc.shape[1]
        nlon = tracer_conc.shape[2]
        ntracer = tracer_conc.shape[3]
        total = np.zeros(ntracer, dtype=np.float64)
        for lev in range(nlev):
            for lat in range(nlat):
                for lon in range(nlon):
                    mass = dry_air_mass_top[lev, lat, lon]
                    for tracer in range(ntracer):
                        total[tracer] += tracer_conc[lev, lat, lon, tracer] * mass
        return total


    @njit(cache=True)
    def _run_vdiffdr_fullgrid_zero_flux_numba_kernel(
        tracer_top: np.ndarray,
        u_top: np.ndarray,
        v_top: np.ndarray,
        temperature_top: np.ndarray,
        sphu_top: np.ndarray,
        pmid_hpa: np.ndarray,
        pint_hpa: np.ndarray,
        virtual_temperature_top: np.ndarray,
        bxheight_top: np.ndarray,
        dry_mass_top: np.ndarray,
        pblh_m: np.ndarray,
        hflux_w_m2: np.ndarray,
        water_flux_kg_m2_s: np.ndarray,
        ustar_m_s: np.ndarray,
        dt_s: float,
        npbl: int,
        tracer_out: np.ndarray,
        sphu_out: np.ndarray,
        pmid: np.ndarray,
        pint: np.ndarray,
        rpdel: np.ndarray,
        rpdeli: np.ndarray,
        zm: np.ndarray,
        thp: np.ndarray,
        kvf: np.ndarray,
        kvh: np.ndarray,
        kvm: np.ndarray,
        cgsh: np.ndarray,
        cgs: np.ndarray,
        tpert: np.ndarray,
        qpert: np.ndarray,
        tmp1: np.ndarray,
        dshbot: np.ndarray,
        rrho: np.ndarray,
        khfs: np.ndarray,
        kshfs: np.ndarray,
        thvsrf: np.ndarray,
        heatv: np.ndarray,
        obklen: np.ndarray,
        fak1: np.ndarray,
        phiminv: np.ndarray,
        phihinv: np.ndarray,
        wm: np.ndarray,
        fak2: np.ndarray,
        fak3: np.ndarray,
        pblk: np.ndarray,
        pr: np.ndarray,
        potbar: np.ndarray,
        cah: np.ndarray,
        cch: np.ndarray,
        zeh: np.ndarray,
        termh: np.ndarray,
        tracer_diffused: np.ndarray,
        tracer_ratio: np.ndarray,
        tracer_after_mass: np.ndarray,
        shmx: np.ndarray,
        zfq_scalar: np.ndarray,
        sphu_diffused: np.ndarray,
    ) -> int:
        nlev = tracer_top.shape[0]
        nlat = tracer_top.shape[1]
        nlon = tracer_top.shape[2]
        ntracer = tracer_top.shape[3]
        ntopfl = 0

        negative_count = 0
        start = ntopfl
        if nlev - npbl > start:
            start = nlev - npbl
        ztodtgor = dt_s * G0_M_PER_S2 / RD_J_PER_KG_K
        gorsq = (G0_M_PER_S2 / RD_J_PER_KG_K) ** 2

        for lat in range(nlat):
            for lon in range(nlon):
                cumulative_height = 0.0
                for lev_down in range(nlev - 1, -1, -1):
                    cumulative_height += bxheight_top[lev_down, lat, lon]
                    pmid_value = pmid_hpa[lev_down, lat, lon] * 100.0
                    pint_value = pint_hpa[lev_down, lat, lon] * 100.0
                    pmid[lon, lev_down] = pmid_value
                    thp[lon, lev_down] = temperature_top[lev_down, lat, lon] * (1.0e5 / pmid_value) ** CAPPA
                    zm[lon, lev_down] = cumulative_height - np.log(pmid_value / pint_value) * (
                        RD_J_PER_KG_K / G0_M_PER_S2
                    ) * virtual_temperature_top[lev_down, lat, lon]
                for edge in range(nlev + 1):
                    pint[lon, edge] = pint_hpa[edge, lat, lon] * 100.0
                for lev in range(nlev):
                    rpdel[lon, lev] = 1.0 / (pint[lon, lev + 1] - pint[lon, lev])
                    if lev < nlev - 1:
                        rpdeli[lon, lev] = 1.0 / (pmid[lon, lev + 1] - pmid[lon, lev])
                    else:
                        rpdeli[lon, lev] = 0.0

            for lon in range(nlon):
                for edge in range(nlev + 1):
                    kvf[lon, edge] = 0.0
                    kvh[lon, edge] = 0.0
                    kvm[lon, edge] = 0.0
                    cgsh[lon, edge] = 0.0
                    cgs[lon, edge] = 0.0
                tpert[lon] = 0.0
                qpert[lon] = 0.0
                tmp1[lon] = dt_s * G0_M_PER_S2 * rpdel[lon, nlev - 1]
                dshbot[lon] = water_flux_kg_m2_s[lat, lon] * tmp1[lon]

            for lev in range(ntopfl, nlev - 1):
                ml2_value = 0.0
                if lev > 0:
                    ml2_value = 900.0
                for lon in range(nlon):
                    dvdz2 = (u_top[lev, lat, lon] - u_top[lev + 1, lat, lon]) ** 2
                    dvdz2 += (v_top[lev, lat, lon] - v_top[lev + 1, lat, lon]) ** 2
                    if dvdz2 < 1.0e-36:
                        dvdz2 = 1.0e-36
                    dz = zm[lon, lev] - zm[lon, lev + 1]
                    dvdz2 = dvdz2 / (dz * dz)
                    thv_lev = thp[lon, lev] * (1.0 + ZVIR * sphu_top[lev, lat, lon])
                    thv_next = thp[lon, lev + 1] * (1.0 + ZVIR * sphu_top[lev + 1, lat, lon])
                    sstab = G0_M_PER_S2 * 2.0 * (thv_lev - thv_next) / ((thv_lev + thv_next) * dz)
                    rinub = sstab / dvdz2
                    fstab = 1.0 / (1.0 + 10.0 * rinub * (1.0 + 8.0 * rinub))
                    if rinub < 0.0:
                        funst = 1.0 - 18.0 * rinub
                        if funst < 0.0:
                            funst = 0.0
                        fstab = np.sqrt(funst)
                    value = ml2_value * np.sqrt(dvdz2) * fstab
                    if value < ZKMIN_M2_S:
                        value = ZKMIN_M2_S
                    kvf[lon, lev + 1] = value

            for lon in range(nlon):
                for edge in range(nlev + 1):
                    kvh[lon, edge] = kvf[lon, edge]
                    kvm[lon, edge] = kvf[lon, edge]

            for lon in range(nlon):
                rrho[lon] = RD_J_PER_KG_K * temperature_top[nlev - 1, lat, lon] / pmid[lon, nlev - 1]
                khfs[lon] = hflux_w_m2[lat, lon] * rrho[lon] / CPAIR_J_PER_KG_K
                kshfs[lon] = water_flux_kg_m2_s[lat, lon] * rrho[lon]
                thvsrf[lon] = thp[lon, nlev - 1] * (1.0 + 0.61 * sphu_top[nlev - 1, lat, lon])
                heatv[lon] = khfs[lon] + 0.61 * thp[lon, nlev - 1] * kshfs[lon]
                sign_heat = 1.0e-10
                if heatv[lon] < 0.0:
                    sign_heat = -1.0e-10
                obklen[lon] = -thvsrf[lon] * ustar_m_s[lat, lon] ** 3 / (
                    G0_M_PER_S2 * VON_KARMAN * (heatv[lon] + sign_heat)
                )
                fak1[lon] = ustar_m_s[lat, lon] * pblh_m[lat, lon] * VON_KARMAN
                phiminv[lon] = 0.0
                phihinv[lon] = 0.0
                wm[lon] = 0.0
                fak2[lon] = 0.0
                fak3[lon] = 0.0
                pblk[lon] = 0.0
                pr[lon] = 0.0
                if heatv[lon] > 0.0:
                    phiminv[lon] = (1.0 - _BINM * pblh_m[lat, lon] / obklen[lon]) ** _ONET
                    phihinv[lon] = np.sqrt(1.0 - _BINH * pblh_m[lat, lon] / obklen[lon])
                    wm[lon] = ustar_m_s[lat, lon] * phiminv[lon]
                    fak2[lon] = wm[lon] * pblh_m[lat, lon] * VON_KARMAN
                    wstr = (heatv[lon] * G0_M_PER_S2 * pblh_m[lat, lon] / thvsrf[lon]) ** _ONET
                    fak3[lon] = _FAKN * wstr / wm[lon]
                    t_val = khfs[lon] * _FAK / wm[lon]
                    q_val = kshfs[lon] * _FAK / wm[lon]
                else:
                    t_val = khfs[lon] * _FAK / ustar_m_s[lat, lon]
                    q_val = kshfs[lon] * _FAK / ustar_m_s[lat, lon]
                if t_val < 0.0:
                    t_val = 0.0
                if q_val < 0.0:
                    q_val = 0.0
                tpert[lon] = t_val
                qpert[lon] = q_val

            for lev in range(nlev - 1, nlev - npbl, -1):
                if lev <= 0:
                    break
                for lon in range(nlon):
                    zm_value = zm[lon, lev]
                    if zm_value >= pblh_m[lat, lon]:
                        continue
                    zp = zm[lon, lev - 1]
                    zmzp = 0.5 * (zm_value + zp)
                    zh = zmzp / pblh_m[lat, lon]
                    zl = zmzp / obklen[lon]
                    zzh = 0.0
                    if zh <= 1.0:
                        zzh = (1.0 - zh) ** 2

                    if heatv[lon] <= 0.0:
                        if zl <= 1.0:
                            pblk[lon] = fak1[lon] * zh * zzh / (1.0 + _BETAS * zl)
                        else:
                            pblk[lon] = fak1[lon] * zh * zzh / (_BETAS + zl)
                        if pblk[lon] > kvf[lon, lev]:
                            kvm[lon, lev] = pblk[lon]
                            kvh[lon, lev] = pblk[lon]
                        else:
                            kvm[lon, lev] = kvf[lon, lev]
                            kvh[lon, lev] = kvf[lon, lev]
                    else:
                        if zh < _SFFRAC:
                            term = (1.0 - _BETAM * zl) ** _ONET
                            pblk[lon] = fak1[lon] * zh * zzh * term
                            pr[lon] = term / np.sqrt(1.0 - _BETAH * zl)
                        else:
                            pblk[lon] = fak2[lon] * zh * zzh
                            cgs[lon, lev] = fak3[lon] / (pblh_m[lat, lon] * wm[lon])
                            pr[lon] = phiminv[lon] / phihinv[lon] + _CCON * fak3[lon] / _FAK
                            cgsh[lon, lev] = kshfs[lon] * cgs[lon, lev]
                        if pblk[lon] > kvf[lon, lev]:
                            kvm[lon, lev] = pblk[lon]
                        else:
                            kvm[lon, lev] = kvf[lon, lev]
                        kh = pblk[lon] / pr[lon]
                        if kh > kvf[lon, lev]:
                            kvh[lon, lev] = kh
                        else:
                            kvh[lon, lev] = kvf[lon, lev]

            for lon in range(nlon):
                for edge in range(nlev + 1):
                    potbar[lon, edge] = 0.0
                for lev in range(1, nlev):
                    potbar[lon, lev] = pint[lon, lev] / (
                        0.5 * (temperature_top[lev, lat, lon] + temperature_top[lev - 1, lat, lon])
                    )
                potbar[lon, nlev] = pint[lon, nlev] / temperature_top[nlev - 1, lat, lon]

            for lon in range(nlon):
                restore = False
                for lev in range(nlev):
                    shmx[lon, lev] = sphu_top[lev, lat, lon]
                if npbl > 1:
                    for lev in range(start, nlev):
                        scale = ztodtgor * rpdel[lon, lev]
                        shmx[lon, lev] = sphu_top[lev, lat, lon] + scale * (
                            potbar[lon, lev + 1] * kvh[lon, lev + 1] * cgsh[lon, lev + 1]
                            - potbar[lon, lev] * kvh[lon, lev] * cgsh[lon, lev]
                        )
                        if shmx[lon, lev] < 1.0e-12:
                            restore = True
                    if restore:
                        for lev in range(start, nlev):
                            shmx[lon, lev] = sphu_top[lev, lat, lon]

            for lon in range(nlon):
                for lev in range(nlev):
                    cah[lon, lev] = 0.0
                    cch[lon, lev] = 0.0
                    zeh[lon, lev] = 0.0
                    termh[lon, lev] = 0.0
            for lev in range(ntopfl, nlev - 1):
                for lon in range(nlon):
                    tmp2 = dt_s * gorsq * rpdeli[lon, lev] * potbar[lon, lev + 1] ** 2
                    cah[lon, lev] = kvh[lon, lev + 1] * tmp2 * rpdel[lon, lev]
                    cch[lon, lev + 1] = kvh[lon, lev + 1] * tmp2 * rpdel[lon, lev + 1]
            for lon in range(nlon):
                termh[lon, ntopfl] = 1.0 / (1.0 + cah[lon, ntopfl])
                zeh[lon, ntopfl] = cah[lon, ntopfl] * termh[lon, ntopfl]
            for lev in range(ntopfl + 1, nlev - 1):
                for lon in range(nlon):
                    termh[lon, lev] = 1.0 / (
                        1.0 + cah[lon, lev] + cch[lon, lev] * (1.0 - zeh[lon, lev - 1])
                    )
                    zeh[lon, lev] = cah[lon, lev] * termh[lon, lev]

            for lon in range(nlon):
                dry_mass = dry_mass_top[ntopfl, lat, lon]
                for tracer in range(ntracer):
                    tracer_value = tracer_top[ntopfl, lat, lon, tracer]
                    tracer_ratio[lon, tracer] = tracer_value * dry_mass
                    tracer_diffused[lon, ntopfl, tracer] = (
                        tracer_value * termh[lon, ntopfl]
                    )
            for lev in range(ntopfl + 1, nlev - 1):
                for lon in range(nlon):
                    dry_mass = dry_mass_top[lev, lat, lon]
                    cch_value = cch[lon, lev]
                    termh_value = termh[lon, lev]
                    for tracer in range(ntracer):
                        tracer_value = tracer_top[lev, lat, lon, tracer]
                        tracer_ratio[lon, tracer] += tracer_value * dry_mass
                        tracer_diffused[lon, lev, tracer] = (
                            tracer_value
                            + cch_value * tracer_diffused[lon, lev - 1, tracer]
                        ) * termh_value
            for lon in range(nlon):
                tmp1d = 1.0 / (1.0 + cch[lon, nlev - 1] * (1.0 - zeh[lon, nlev - 2]))
                dry_mass = dry_mass_top[nlev - 1, lat, lon]
                cch_bottom = cch[lon, nlev - 1]
                for tracer in range(ntracer):
                    tracer_value = tracer_top[nlev - 1, lat, lon, tracer]
                    tracer_ratio[lon, tracer] += tracer_value * dry_mass
                    tracer_diffused[lon, nlev - 1, tracer] = (
                        tracer_value
                        + cch_bottom * tracer_diffused[lon, nlev - 2, tracer]
                    ) * tmp1d
            for lev in range(nlev - 2, ntopfl - 1, -1):
                for lon in range(nlon):
                    for tracer in range(ntracer):
                        tracer_diffused[lon, lev, tracer] = (
                            tracer_diffused[lon, lev, tracer]
                            + zeh[lon, lev] * tracer_diffused[lon, lev + 1, tracer]
                        )

            for lon in range(nlon):
                for tracer in range(ntracer):
                    tracer_after_mass[lon, tracer] = 0.0
                for lev in range(ntopfl, nlev):
                    dry_mass = dry_mass_top[lev, lat, lon]
                    for tracer in range(ntracer):
                        value = tracer_diffused[lon, lev, tracer]
                        if value < 0.0:
                            negative_count += 1
                            value = 0.0
                            tracer_diffused[lon, lev, tracer] = 0.0
                        tracer_after_mass[lon, tracer] += value * dry_mass
                for tracer in range(ntracer):
                    ratio = 1.0
                    before_mass = tracer_ratio[lon, tracer]
                    after_mass = tracer_after_mass[lon, tracer]
                    if abs(before_mass) > 0.0 and abs(after_mass) > 0.0:
                        ratio = before_mass / after_mass
                    tracer_ratio[lon, tracer] = ratio
                for lev in range(ntopfl, nlev):
                    for tracer in range(ntracer):
                        tracer_out[lev, lat, lon, tracer] = tracer_diffused[lon, lev, tracer] * tracer_ratio[
                            lon, tracer
                        ]

            for lon in range(nlon):
                for lev in range(nlev):
                    zfq_scalar[lon, lev] = 0.0
                    sphu_diffused[lon, lev] = 0.0
                zfq_scalar[lon, ntopfl] = shmx[lon, ntopfl] * termh[lon, ntopfl]
            for lev in range(ntopfl + 1, nlev - 1):
                for lon in range(nlon):
                    zfq_scalar[lon, lev] = (
                        shmx[lon, lev] + cch[lon, lev] * zfq_scalar[lon, lev - 1]
                    ) * termh[lon, lev]
            for lon in range(nlon):
                tmp1d = 1.0 / (1.0 + cch[lon, nlev - 1] * (1.0 - zeh[lon, nlev - 2]))
                zfq_scalar[lon, nlev - 1] = (
                    shmx[lon, nlev - 1] + dshbot[lon] + cch[lon, nlev - 1] * zfq_scalar[lon, nlev - 2]
                ) * tmp1d
                sphu_diffused[lon, nlev - 1] = zfq_scalar[lon, nlev - 1]
            for lev in range(nlev - 2, ntopfl - 1, -1):
                for lon in range(nlon):
                    sphu_diffused[lon, lev] = zfq_scalar[lon, lev] + zeh[lon, lev] * sphu_diffused[lon, lev + 1]
            for lon in range(nlon):
                for lev in range(nlev):
                    value = sphu_diffused[lon, lev]
                    if value < 1.0e-12:
                        value = 0.0
                    sphu_out[lev, lat, lon] = value

        return negative_count


    @njit(cache=True)
    def _run_vdiff_latitude_numba_kernel(
        tracer_top: np.ndarray,
        u_top: np.ndarray,
        v_top: np.ndarray,
        temperature_top: np.ndarray,
        sphu_top: np.ndarray,
        pmid_top: np.ndarray,
        pint_top: np.ndarray,
        rpdel_top: np.ndarray,
        rpdeli_top: np.ndarray,
        zm_top: np.ndarray,
        thp_top: np.ndarray,
        dry_mass_top: np.ndarray,
        pblh_m: np.ndarray,
        hflux_w_m2: np.ndarray,
        water_flux_kg_m2_s: np.ndarray,
        surface_flux_kg_m2_s: np.ndarray,
        ustar_m_s: np.ndarray,
        area_m2: np.ndarray,
        dt_s: float,
        ntopfl: int,
        npbl: int,
        ml2: np.ndarray,
        surface_flux_is_zero: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        nlon = tracer_top.shape[0]
        nlev = tracer_top.shape[1]
        ntracer = tracer_top.shape[2]

        kvf = np.zeros((nlon, nlev + 1), dtype=np.float64)
        kvh = np.zeros((nlon, nlev + 1), dtype=np.float64)
        kvm = np.zeros((nlon, nlev + 1), dtype=np.float64)
        cgsh = np.zeros((nlon, nlev + 1), dtype=np.float64)
        cgs = np.zeros((nlon, nlev + 1), dtype=np.float64)
        cgq = np.zeros((nlon, nlev + 1, ntracer), dtype=np.float64)
        tpert = np.zeros(nlon, dtype=np.float64)
        qpert = np.zeros(nlon, dtype=np.float64)

        tmp1 = np.empty(nlon, dtype=np.float64)
        dshbot = np.empty(nlon, dtype=np.float64)
        dqbot = np.empty((nlon, ntracer), dtype=np.float64)

        for lon in range(nlon):
            tmp1[lon] = dt_s * G0_M_PER_S2 * rpdel_top[lon, nlev - 1]
            dshbot[lon] = water_flux_kg_m2_s[lon] * tmp1[lon]
            if not surface_flux_is_zero:
                for tracer in range(ntracer):
                    dqbot[lon, tracer] = surface_flux_kg_m2_s[lon, tracer] * tmp1[lon]

        for lev in range(ntopfl, nlev - 1):
            for lon in range(nlon):
                dvdz2 = (u_top[lon, lev] - u_top[lon, lev + 1]) ** 2
                dvdz2 += (v_top[lon, lev] - v_top[lon, lev + 1]) ** 2
                if dvdz2 < 1.0e-36:
                    dvdz2 = 1.0e-36
                dz = zm_top[lon, lev] - zm_top[lon, lev + 1]
                dvdz2 = dvdz2 / (dz * dz)
                thv_lev = thp_top[lon, lev] * (1.0 + ZVIR * sphu_top[lon, lev])
                thv_next = thp_top[lon, lev + 1] * (1.0 + ZVIR * sphu_top[lon, lev + 1])
                sstab = G0_M_PER_S2 * 2.0 * (thv_lev - thv_next) / ((thv_lev + thv_next) * dz)
                rinub = sstab / dvdz2
                fstab = 1.0 / (1.0 + 10.0 * rinub * (1.0 + 8.0 * rinub))
                if rinub < 0.0:
                    funst = 1.0 - 18.0 * rinub
                    if funst < 0.0:
                        funst = 0.0
                    fstab = np.sqrt(funst)
                kvn = ml2[lev] * np.sqrt(dvdz2)
                value = kvn * fstab
                if value < ZKMIN_M2_S:
                    value = ZKMIN_M2_S
                kvf[lon, lev + 1] = value

        for lon in range(nlon):
            for edge in range(nlev + 1):
                kvh[lon, edge] = kvf[lon, edge]
                kvm[lon, edge] = kvf[lon, edge]

        rrho = np.empty(nlon, dtype=np.float64)
        khfs = np.empty(nlon, dtype=np.float64)
        kshfs = np.empty(nlon, dtype=np.float64)
        thvsrf = np.empty(nlon, dtype=np.float64)
        heatv = np.empty(nlon, dtype=np.float64)
        obklen = np.empty(nlon, dtype=np.float64)
        fak1 = np.empty(nlon, dtype=np.float64)
        phiminv = np.zeros(nlon, dtype=np.float64)
        phihinv = np.zeros(nlon, dtype=np.float64)
        wm = np.zeros(nlon, dtype=np.float64)
        fak2 = np.zeros(nlon, dtype=np.float64)
        fak3 = np.zeros(nlon, dtype=np.float64)
        pblk = np.zeros(nlon, dtype=np.float64)
        pr = np.zeros(nlon, dtype=np.float64)

        for lon in range(nlon):
            rrho[lon] = RD_J_PER_KG_K * temperature_top[lon, nlev - 1] / pmid_top[lon, nlev - 1]
            khfs[lon] = hflux_w_m2[lon] * rrho[lon] / CPAIR_J_PER_KG_K
            kshfs[lon] = water_flux_kg_m2_s[lon] * rrho[lon]
            thvsrf[lon] = thp_top[lon, nlev - 1] * (1.0 + 0.61 * sphu_top[lon, nlev - 1])
            heatv[lon] = khfs[lon] + 0.61 * thp_top[lon, nlev - 1] * kshfs[lon]
            sign_heat = 1.0e-10
            if heatv[lon] < 0.0:
                sign_heat = -1.0e-10
            obklen[lon] = -thvsrf[lon] * ustar_m_s[lon] ** 3 / (
                G0_M_PER_S2 * VON_KARMAN * (heatv[lon] + sign_heat)
            )
            fak1[lon] = ustar_m_s[lon] * pblh_m[lon] * VON_KARMAN
            if heatv[lon] > 0.0:
                phiminv[lon] = (1.0 - _BINM * pblh_m[lon] / obklen[lon]) ** _ONET
                phihinv[lon] = np.sqrt(1.0 - _BINH * pblh_m[lon] / obklen[lon])
                wm[lon] = ustar_m_s[lon] * phiminv[lon]
                fak2[lon] = wm[lon] * pblh_m[lon] * VON_KARMAN
                wstr = (heatv[lon] * G0_M_PER_S2 * pblh_m[lon] / thvsrf[lon]) ** _ONET
                fak3[lon] = _FAKN * wstr / wm[lon]
                t_val = khfs[lon] * _FAK / wm[lon]
                q_val = kshfs[lon] * _FAK / wm[lon]
            else:
                t_val = khfs[lon] * _FAK / ustar_m_s[lon]
                q_val = kshfs[lon] * _FAK / ustar_m_s[lon]
            if t_val < 0.0:
                t_val = 0.0
            if q_val < 0.0:
                q_val = 0.0
            tpert[lon] = t_val
            qpert[lon] = q_val

        for lev in range(nlev - 1, nlev - npbl, -1):
            if lev <= 0:
                break
            for lon in range(nlon):
                zm = zm_top[lon, lev]
                if zm >= pblh_m[lon]:
                    continue
                zp = zm_top[lon, lev - 1]
                zmzp = 0.5 * (zm + zp)
                zh = zmzp / pblh_m[lon]
                zl = zmzp / obklen[lon]
                zzh = 0.0
                if zh <= 1.0:
                    zzh = (1.0 - zh) ** 2

                if heatv[lon] <= 0.0:
                    if zl <= 1.0:
                        pblk[lon] = fak1[lon] * zh * zzh / (1.0 + _BETAS * zl)
                    else:
                        pblk[lon] = fak1[lon] * zh * zzh / (_BETAS + zl)
                    if pblk[lon] > kvf[lon, lev]:
                        kvm[lon, lev] = pblk[lon]
                        kvh[lon, lev] = pblk[lon]
                    else:
                        kvm[lon, lev] = kvf[lon, lev]
                        kvh[lon, lev] = kvf[lon, lev]
                else:
                    if zh < _SFFRAC:
                        term = (1.0 - _BETAM * zl) ** _ONET
                        pblk[lon] = fak1[lon] * zh * zzh * term
                        pr[lon] = term / np.sqrt(1.0 - _BETAH * zl)
                    else:
                        pblk[lon] = fak2[lon] * zh * zzh
                        cgs[lon, lev] = fak3[lon] / (pblh_m[lon] * wm[lon])
                        pr[lon] = phiminv[lon] / phihinv[lon] + _CCON * fak3[lon] / _FAK
                        cgsh[lon, lev] = kshfs[lon] * cgs[lon, lev]
                        if not surface_flux_is_zero:
                            for tracer in range(ntracer):
                                cgq[lon, lev, tracer] = (
                                    surface_flux_kg_m2_s[lon, tracer] * rrho[lon] * cgs[lon, lev]
                                )
                    if pblk[lon] > kvf[lon, lev]:
                        kvm[lon, lev] = pblk[lon]
                    else:
                        kvm[lon, lev] = kvf[lon, lev]
                    kh = pblk[lon] / pr[lon]
                    if kh > kvf[lon, lev]:
                        kvh[lon, lev] = kh
                    else:
                        kvh[lon, lev] = kvf[lon, lev]

        potbar = np.zeros((nlon, nlev + 1), dtype=np.float64)
        for lev in range(1, nlev):
            for lon in range(nlon):
                potbar[lon, lev] = pint_top[lon, lev] / (
                    0.5 * (temperature_top[lon, lev] + temperature_top[lon, lev - 1])
                )
        for lon in range(nlon):
            potbar[lon, nlev] = pint_top[lon, nlev] / temperature_top[lon, nlev - 1]

        start = ntopfl
        if nlev - npbl > start:
            start = nlev - npbl
        ztodtgor = dt_s * G0_M_PER_S2 / RD_J_PER_KG_K

        if surface_flux_is_zero:
            qmx = tracer_top
        else:
            qmx = tracer_top.copy()
        if npbl > 1 and not surface_flux_is_zero:
            adjust = np.zeros((nlon, ntracer), dtype=np.bool_)
            for lev in range(start, nlev):
                for lon in range(nlon):
                    scale = ztodtgor * rpdel_top[lon, lev]
                    term_next = potbar[lon, lev + 1] * kvh[lon, lev + 1]
                    term_now = potbar[lon, lev] * kvh[lon, lev]
                    for tracer in range(ntracer):
                        qmx[lon, lev, tracer] = tracer_top[lon, lev, tracer] + scale * (
                            term_next * cgq[lon, lev + 1, tracer] - term_now * cgq[lon, lev, tracer]
                        )
                        if qmx[lon, lev, tracer] < 0.0:
                            adjust[lon, tracer] = True
            for lon in range(nlon):
                for tracer in range(ntracer):
                    if adjust[lon, tracer]:
                        for lev in range(start, nlev):
                            qmx[lon, lev, tracer] = tracer_top[lon, lev, tracer]

        shmx = sphu_top.copy()
        if npbl > 1:
            for lev in range(start, nlev):
                for lon in range(nlon):
                    scale = ztodtgor * rpdel_top[lon, lev]
                    shmx[lon, lev] = sphu_top[lon, lev] + scale * (
                        potbar[lon, lev + 1] * kvh[lon, lev + 1] * cgsh[lon, lev + 1]
                        - potbar[lon, lev] * kvh[lon, lev] * cgsh[lon, lev]
                    )
            for lon in range(nlon):
                restore = False
                for lev in range(start, nlev):
                    if shmx[lon, lev] < 1.0e-12:
                        restore = True
                if restore:
                    for lev in range(start, nlev):
                        shmx[lon, lev] = sphu_top[lon, lev]

        cah = np.zeros((nlon, nlev), dtype=np.float64)
        cch = np.zeros((nlon, nlev), dtype=np.float64)
        zeh = np.zeros((nlon, nlev), dtype=np.float64)
        termh = np.zeros((nlon, nlev), dtype=np.float64)
        gorsq = (G0_M_PER_S2 / RD_J_PER_KG_K) ** 2
        for lev in range(ntopfl, nlev - 1):
            for lon in range(nlon):
                tmp2 = dt_s * gorsq * rpdeli_top[lon, lev] * potbar[lon, lev + 1] ** 2
                cah[lon, lev] = kvh[lon, lev + 1] * tmp2 * rpdel_top[lon, lev]
                cch[lon, lev + 1] = kvh[lon, lev + 1] * tmp2 * rpdel_top[lon, lev + 1]
        for lon in range(nlon):
            termh[lon, ntopfl] = 1.0 / (1.0 + cah[lon, ntopfl])
            zeh[lon, ntopfl] = cah[lon, ntopfl] * termh[lon, ntopfl]
        for lev in range(ntopfl + 1, nlev - 1):
            for lon in range(nlon):
                termh[lon, lev] = 1.0 / (
                    1.0 + cah[lon, lev] + cch[lon, lev] * (1.0 - zeh[lon, lev - 1])
                )
                zeh[lon, lev] = cah[lon, lev] * termh[lon, lev]

        tracer_diffused = np.zeros((nlon, nlev, ntracer), dtype=np.float64)
        for lon in range(nlon):
            for tracer in range(ntracer):
                tracer_diffused[lon, ntopfl, tracer] = qmx[lon, ntopfl, tracer] * termh[lon, ntopfl]
        for lev in range(ntopfl + 1, nlev - 1):
            for lon in range(nlon):
                for tracer in range(ntracer):
                    tracer_diffused[lon, lev, tracer] = (
                        qmx[lon, lev, tracer] + cch[lon, lev] * tracer_diffused[lon, lev - 1, tracer]
                    ) * termh[lon, lev]
        for lon in range(nlon):
            tmp1d = 1.0 / (1.0 + cch[lon, nlev - 1] * (1.0 - zeh[lon, nlev - 2]))
            for tracer in range(ntracer):
                tracer_diffused[lon, nlev - 1, tracer] = (
                    qmx[lon, nlev - 1, tracer]
                    + (0.0 if surface_flux_is_zero else dqbot[lon, tracer])
                    + cch[lon, nlev - 1] * tracer_diffused[lon, nlev - 2, tracer]
                ) * tmp1d
        for lev in range(nlev - 2, ntopfl - 1, -1):
            for lon in range(nlon):
                for tracer in range(ntracer):
                    tracer_diffused[lon, lev, tracer] = (
                        tracer_diffused[lon, lev, tracer] + zeh[lon, lev] * tracer_diffused[lon, lev + 1, tracer]
                    )

        negative_count = 0
        for lon in range(nlon):
            for lev in range(nlev):
                for tracer in range(ntracer):
                    if tracer_diffused[lon, lev, tracer] < 0.0:
                        negative_count += 1
                        tracer_diffused[lon, lev, tracer] = 0.0

        for lon in range(nlon):
            for tracer in range(ntracer):
                before_mass = 0.0
                if not surface_flux_is_zero:
                    before_mass = surface_flux_kg_m2_s[lon, tracer] * area_m2[lon] * dt_s
                after_mass = 0.0
                for lev in range(ntopfl, nlev):
                    before_mass += tracer_top[lon, lev, tracer] * dry_mass_top[lon, lev]
                    after_mass += tracer_diffused[lon, lev, tracer] * dry_mass_top[lon, lev]
                if abs(before_mass) > 0.0 and abs(after_mass) > 0.0:
                    ratio = before_mass / after_mass
                    for lev in range(ntopfl, nlev):
                        tracer_diffused[lon, lev, tracer] *= ratio

        zfq_scalar = np.zeros((nlon, nlev), dtype=np.float64)
        sphu_diffused = np.zeros((nlon, nlev), dtype=np.float64)
        for lon in range(nlon):
            zfq_scalar[lon, ntopfl] = shmx[lon, ntopfl] * termh[lon, ntopfl]
        for lev in range(ntopfl + 1, nlev - 1):
            for lon in range(nlon):
                zfq_scalar[lon, lev] = (shmx[lon, lev] + cch[lon, lev] * zfq_scalar[lon, lev - 1]) * termh[
                    lon, lev
                ]
        for lon in range(nlon):
            tmp1d = 1.0 / (1.0 + cch[lon, nlev - 1] * (1.0 - zeh[lon, nlev - 2]))
            zfq_scalar[lon, nlev - 1] = (
                shmx[lon, nlev - 1] + dshbot[lon] + cch[lon, nlev - 1] * zfq_scalar[lon, nlev - 2]
            ) * tmp1d
            sphu_diffused[lon, nlev - 1] = zfq_scalar[lon, nlev - 1]
        for lev in range(nlev - 2, ntopfl - 1, -1):
            for lon in range(nlon):
                sphu_diffused[lon, lev] = zfq_scalar[lon, lev] + zeh[lon, lev] * sphu_diffused[lon, lev + 1]
        for lon in range(nlon):
            for lev in range(nlev):
                if sphu_diffused[lon, lev] < 1.0e-12:
                    sphu_diffused[lon, lev] = 0.0

        return tracer_diffused, sphu_diffused, kvh, kvm, tpert, qpert, negative_count

else:

    def _tracer_working_mass_numba_kernel(tracer_conc: np.ndarray, dry_air_mass_top: np.ndarray) -> np.ndarray:
        raise RuntimeError("numba is not available")


    def _run_vdiff_latitude_numba_kernel(
        tracer_top: np.ndarray,
        u_top: np.ndarray,
        v_top: np.ndarray,
        temperature_top: np.ndarray,
        sphu_top: np.ndarray,
        pmid_top: np.ndarray,
        pint_top: np.ndarray,
        rpdel_top: np.ndarray,
        rpdeli_top: np.ndarray,
        zm_top: np.ndarray,
        thp_top: np.ndarray,
        dry_mass_top: np.ndarray,
        pblh_m: np.ndarray,
        hflux_w_m2: np.ndarray,
        water_flux_kg_m2_s: np.ndarray,
        surface_flux_kg_m2_s: np.ndarray,
        ustar_m_s: np.ndarray,
        area_m2: np.ndarray,
        dt_s: float,
        ntopfl: int,
        npbl: int,
        ml2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        raise RuntimeError("numba is not available")


    def _run_vdiffdr_fullgrid_zero_flux_numba_kernel(
        tracer_top: np.ndarray,
        u_top: np.ndarray,
        v_top: np.ndarray,
        temperature_top: np.ndarray,
        sphu_top: np.ndarray,
        pmid_hpa: np.ndarray,
        pint_hpa: np.ndarray,
        virtual_temperature_top: np.ndarray,
        bxheight_top: np.ndarray,
        dry_mass_top: np.ndarray,
        pblh_m: np.ndarray,
        hflux_w_m2: np.ndarray,
        water_flux_kg_m2_s: np.ndarray,
        ustar_m_s: np.ndarray,
        dt_s: float,
        npbl: int,
        *workspace: np.ndarray,
    ) -> int:
        raise RuntimeError("numba is not available")


def _pbldif_archived_pblh(
    *,
    th: np.ndarray,
    q: np.ndarray,
    z: np.ndarray,
    temperature: np.ndarray,
    pmid: np.ndarray,
    kvf: np.ndarray,
    surface_flux: np.ndarray,
    water_flux: np.ndarray,
    heat_flux: np.ndarray,
    pblh: np.ndarray,
    ustar: np.ndarray,
    npbl: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cgh = np.zeros_like(kvf)
    cgsh = np.zeros_like(kvf)
    cgs = np.zeros_like(kvf)
    cgq = np.zeros((kvf.shape[0], kvf.shape[1], surface_flux.shape[1]), dtype=np.float64)
    tpert = np.zeros(kvf.shape[0], dtype=np.float64)
    qpert = np.zeros(kvf.shape[0], dtype=np.float64)
    if not _PBLH_ARCHIVED:
        raise NotImplementedError("derived VDIFF PBL height is outside the configured GEOS-Chem path")
    _ = (th, q, z, temperature, pmid, water_flux, heat_flux, pblh, ustar, npbl)
    return cgh, cgq, cgsh, cgs, tpert, qpert


def _apply_pbldif_diffusivity(
    *,
    kvh: np.ndarray,
    kvm: np.ndarray,
    cgs: np.ndarray,
    cgh: np.ndarray,
    cgq: np.ndarray,
    cgsh: np.ndarray,
    tpert: np.ndarray,
    qpert: np.ndarray,
    th: np.ndarray,
    q: np.ndarray,
    z: np.ndarray,
    temperature: np.ndarray,
    pmid: np.ndarray,
    kvf: np.ndarray,
    surface_flux: np.ndarray,
    water_flux: np.ndarray,
    heat_flux: np.ndarray,
    pblh: np.ndarray,
    ustar: np.ndarray,
    npbl: int,
) -> None:
    nlon, nlev = th.shape
    rrho = RD_J_PER_KG_K * temperature[:, nlev - 1] / pmid[:, nlev - 1]
    khfs = heat_flux * rrho / CPAIR_J_PER_KG_K
    kshfs = water_flux * rrho
    kqfs = surface_flux * rrho[:, np.newaxis]
    thvsrf = th[:, nlev - 1] * (1.0 + 0.61 * q[:, nlev - 1])
    heatv = khfs + 0.61 * th[:, nlev - 1] * kshfs
    unstbl = heatv > 0.0
    sign_heat = np.where(heatv >= 0.0, 1.0e-10, -1.0e-10)
    obklen = -thvsrf * ustar**3 / (G0_M_PER_S2 * VON_KARMAN * (heatv + sign_heat))
    fak1 = ustar * pblh * VON_KARMAN
    phiminv = np.zeros(nlon, dtype=np.float64)
    phihinv = np.zeros(nlon, dtype=np.float64)
    wm = np.zeros(nlon, dtype=np.float64)
    fak2 = np.zeros(nlon, dtype=np.float64)
    fak3 = np.zeros(nlon, dtype=np.float64)
    phiminv[unstbl] = (1.0 - _BINM * pblh[unstbl] / obklen[unstbl]) ** _ONET
    phihinv[unstbl] = np.sqrt(1.0 - _BINH * pblh[unstbl] / obklen[unstbl])
    wm[unstbl] = ustar[unstbl] * phiminv[unstbl]
    fak2[unstbl] = wm[unstbl] * pblh[unstbl] * VON_KARMAN
    wstr = np.zeros(nlon, dtype=np.float64)
    wstr[unstbl] = (heatv[unstbl] * G0_M_PER_S2 * pblh[unstbl] / thvsrf[unstbl]) ** _ONET
    fak3[unstbl] = _FAKN * wstr[unstbl] / wm[unstbl]
    tpert[unstbl] = np.maximum(khfs[unstbl] * _FAK / wm[unstbl], 0.0)
    qpert[unstbl] = np.maximum(kshfs[unstbl] * _FAK / wm[unstbl], 0.0)
    stable_or_neutral = ~unstbl
    tpert[stable_or_neutral] = np.maximum(khfs[stable_or_neutral] * _FAK / ustar[stable_or_neutral], 0.0)
    qpert[stable_or_neutral] = np.maximum(kshfs[stable_or_neutral] * _FAK / ustar[stable_or_neutral], 0.0)

    pblk = np.zeros(nlon, dtype=np.float64)
    pr = np.zeros(nlon, dtype=np.float64)
    for lev in range(nlev - 1, nlev - npbl, -1):
        if lev <= 0:
            break
        zm = z[:, lev]
        zp = z[:, lev - 1]
        within = zm < pblh
        if not np.any(within):
            continue
        zmzp = 0.5 * (zm + zp)
        zh = np.zeros(nlon, dtype=np.float64)
        zl = np.zeros(nlon, dtype=np.float64)
        zzh = np.zeros(nlon, dtype=np.float64)
        zh[within] = zmzp[within] / pblh[within]
        zl[within] = zmzp[within] / obklen[within]
        zzh[within] = np.where(zh[within] <= 1.0, (1.0 - zh[within]) ** 2, 0.0)

        stable = within & ~unstbl
        if np.any(stable):
            pblk[stable] = np.where(
                zl[stable] <= 1.0,
                fak1[stable] * zh[stable] * zzh[stable] / (1.0 + _BETAS * zl[stable]),
                fak1[stable] * zh[stable] * zzh[stable] / (_BETAS + zl[stable]),
            )
            kvm[stable, lev] = np.maximum(pblk[stable], kvf[stable, lev])
            kvh[stable, lev] = kvm[stable, lev]

        unstable_surface = within & unstbl & (zh < _SFFRAC)
        if np.any(unstable_surface):
            term = (1.0 - _BETAM * zl[unstable_surface]) ** _ONET
            pblk[unstable_surface] = fak1[unstable_surface] * zh[unstable_surface] * zzh[unstable_surface] * term
            pr[unstable_surface] = term / np.sqrt(1.0 - _BETAH * zl[unstable_surface])

        unstable_outer = within & unstbl & (zh >= _SFFRAC)
        if np.any(unstable_outer):
            pblk[unstable_outer] = fak2[unstable_outer] * zh[unstable_outer] * zzh[unstable_outer]
            cgs[unstable_outer, lev] = fak3[unstable_outer] / (pblh[unstable_outer] * wm[unstable_outer])
            cgh[unstable_outer, lev] = khfs[unstable_outer] * cgs[unstable_outer, lev]
            pr[unstable_outer] = phiminv[unstable_outer] / phihinv[unstable_outer] + _CCON * fak3[unstable_outer] / _FAK
            cgsh[unstable_outer, lev] = kshfs[unstable_outer] * cgs[unstable_outer, lev]
            cgq[unstable_outer, lev, :] = kqfs[unstable_outer, :] * cgs[unstable_outer, lev, np.newaxis]

        unstable = within & unstbl
        if np.any(unstable):
            kvm[unstable, lev] = np.maximum(pblk[unstable], kvf[unstable, lev])
            kvh[unstable, lev] = np.maximum(pblk[unstable] / pr[unstable], kvf[unstable, lev])


def _countergradient_adjust_tracers(
    qp1: np.ndarray,
    rpdel: np.ndarray,
    pint: np.ndarray,
    temperature: np.ndarray,
    kvh: np.ndarray,
    cgq: np.ndarray,
    dt_s: float,
    ntopfl: int,
    npbl: int,
    qmincg: np.ndarray,
) -> np.ndarray:
    qmx = qp1.copy()
    nlev = qp1.shape[1]
    if npbl <= 1:
        return qmx
    potbar = _potbar(pint, temperature)
    ztodtgor = dt_s * G0_M_PER_S2 / RD_J_PER_KG_K
    start = max(ntopfl, nlev - npbl)
    for lev in range(start, nlev):
        tmp1 = ztodtgor * rpdel[:, lev]
        qmx[:, lev, :] = qp1[:, lev, :] + tmp1[:, np.newaxis] * (
            potbar[:, lev + 1, np.newaxis] * kvh[:, lev + 1, np.newaxis] * cgq[:, lev + 1, :]
            - potbar[:, lev, np.newaxis] * kvh[:, lev, np.newaxis] * cgq[:, lev, :]
        )
    adjust = np.any(qmx[:, start:, :] < qmincg[np.newaxis, np.newaxis, :], axis=1)
    for tracer_index in range(qp1.shape[2]):
        qmx[adjust[:, tracer_index], start:, tracer_index] = qp1[adjust[:, tracer_index], start:, tracer_index]
    return qmx


def _countergradient_adjust_scalar(
    value: np.ndarray,
    rpdel: np.ndarray,
    pint: np.ndarray,
    temperature: np.ndarray,
    kvh: np.ndarray,
    cg: np.ndarray,
    dt_s: float,
    ntopfl: int,
    npbl: int,
    *,
    minimum: float | None,
) -> np.ndarray:
    adjusted = value.copy()
    nlev = value.shape[1]
    if npbl <= 1:
        return adjusted
    potbar = _potbar(pint, temperature)
    ztodtgor = dt_s * G0_M_PER_S2 / RD_J_PER_KG_K
    start = max(ntopfl, nlev - npbl)
    for lev in range(start, nlev):
        tmp1 = ztodtgor * rpdel[:, lev]
        adjusted[:, lev] = value[:, lev] + tmp1 * (
            potbar[:, lev + 1] * kvh[:, lev + 1] * cg[:, lev + 1]
            - potbar[:, lev] * kvh[:, lev] * cg[:, lev]
        )
    if minimum is not None:
        mask = np.any(adjusted[:, start:] < minimum, axis=1)
        adjusted[mask, start:] = value[mask, start:]
    return adjusted


def _potbar(pint: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    nlev = temperature.shape[1]
    potbar = np.zeros((temperature.shape[0], nlev + 1), dtype=np.float64)
    for lev in range(1, nlev):
        potbar[:, lev] = pint[:, lev] / (0.5 * (temperature[:, lev] + temperature[:, lev - 1]))
    potbar[:, nlev] = pint[:, nlev] / temperature[:, nlev - 1]
    return potbar


def _diffusion_coefficients(
    kvh: np.ndarray,
    rpdel: np.ndarray,
    rpdeli: np.ndarray,
    pint: np.ndarray,
    temperature: np.ndarray,
    dt_s: float,
    ntopfl: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nlon, nlev = rpdel.shape
    cah = np.zeros((nlon, nlev), dtype=np.float64)
    cch = np.zeros_like(cah)
    zeh = np.zeros_like(cah)
    termh = np.zeros_like(cah)
    potbar = _potbar(pint, temperature)
    gorsq = (G0_M_PER_S2 / RD_J_PER_KG_K) ** 2
    for lev in range(ntopfl, nlev - 1):
        tmp2 = dt_s * gorsq * rpdeli[:, lev] * potbar[:, lev + 1] ** 2
        cah[:, lev] = kvh[:, lev + 1] * tmp2 * rpdel[:, lev]
        cch[:, lev + 1] = kvh[:, lev + 1] * tmp2 * rpdel[:, lev + 1]
    cah[:, nlev - 1] = 0.0
    termh[:, ntopfl] = 1.0 / (1.0 + cah[:, ntopfl])
    zeh[:, ntopfl] = cah[:, ntopfl] * termh[:, ntopfl]
    for lev in range(ntopfl + 1, nlev - 1):
        termh[:, lev] = 1.0 / (1.0 + cah[:, lev] + cch[:, lev] * (1.0 - zeh[:, lev - 1]))
        zeh[:, lev] = cah[:, lev] * termh[:, lev]
    return cch, zeh, termh


def _qvdiff(
    qm1: np.ndarray,
    qflx: np.ndarray,
    cc: np.ndarray,
    ze: np.ndarray,
    term: np.ndarray,
    ntopfl: int,
) -> np.ndarray:
    nlon, nlev, _ncnst = qm1.shape
    zfq = np.zeros_like(qm1)
    qp1 = np.zeros_like(qm1)
    zfq[:, ntopfl, :] = qm1[:, ntopfl, :] * term[:, ntopfl, np.newaxis]
    for lev in range(ntopfl + 1, nlev - 1):
        zfq[:, lev, :] = (qm1[:, lev, :] + cc[:, lev, np.newaxis] * zfq[:, lev - 1, :]) * term[
            :, lev, np.newaxis
        ]
    tmp1d = 1.0 / (1.0 + cc[:, nlev - 1] * (1.0 - ze[:, nlev - 2]))
    ze[:, nlev - 1] = 0.0
    zfq[:, nlev - 1, :] = (
        qm1[:, nlev - 1, :] + qflx + cc[:, nlev - 1, np.newaxis] * zfq[:, nlev - 2, :]
    ) * tmp1d[:, np.newaxis]
    qp1[:, nlev - 1, :] = zfq[:, nlev - 1, :]
    for lev in range(nlev - 2, ntopfl - 1, -1):
        qp1[:, lev, :] = zfq[:, lev, :] + ze[:, lev, np.newaxis] * qp1[:, lev + 1, :]
    return qp1


def _rescale_long_lived_mass(
    *,
    before: np.ndarray,
    after: np.ndarray,
    dry_mass_top: np.ndarray,
    surface_flux: np.ndarray,
    area_m2: np.ndarray,
    dt_s: float,
    ntopfl: int,
) -> np.ndarray:
    rescaled = after.copy()
    before_mass = np.sum(before[:, ntopfl:, :] * dry_mass_top[:, ntopfl:, np.newaxis], axis=1)
    before_mass = before_mass + surface_flux * area_m2[:, np.newaxis] * dt_s
    after_mass = np.sum(after[:, ntopfl:, :] * dry_mass_top[:, ntopfl:, np.newaxis], axis=1)
    ratio = np.ones_like(after_mass)
    safe = (np.abs(before_mass) > 0.0) & (np.abs(after_mass) > 0.0)
    ratio[safe] = before_mass[safe] / after_mass[safe]
    rescaled[:, ntopfl:, :] = rescaled[:, ntopfl:, :] * ratio[:, np.newaxis, :]
    return rescaled
