from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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


def _tracer_working_mass(tracer_conc: np.ndarray, dry_air_mass_top: np.ndarray) -> np.ndarray:
    """Return total tracer mass from VDIFF working layout."""

    return np.sum(tracer_conc * dry_air_mass_top[:, :, :, np.newaxis], axis=(0, 1, 2))


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
        column = _run_vdiff_latitude(
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
