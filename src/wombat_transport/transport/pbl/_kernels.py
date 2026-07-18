"""Low-level compiled kernels for VDIFF/PBL calculations."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from wombat_transport.constants import G0_M_PER_S2, RD_J_PER_KG_K
from wombat_transport.transport.numba_control import (
    configure_numba_threads,
    numba_available_and_enabled,
    numba_mode,
)
from wombat_transport.transport.pbl._reference import (
    CAPPA,
    CPAIR_J_PER_KG_K,
    VON_KARMAN,
    ZKMIN_M2_S,
    ZVIR,
    _BETAH,
    _BETAM,
    _BETAS,
    _BINH,
    _BINM,
    _CCON,
    _FAK,
    _FAKN,
    _ONET,
    _SFFRAC,
)

try:  # Optional acceleration path; NumPy remains the reference fallback.
    from numba import get_thread_id, njit, prange
except ImportError:  # pragma: no cover - exercised in environments without numba.
    get_thread_id = None
    njit = None
    prange = range

_NUMBA_AVAILABLE = njit is not None


def _tracer_working_mass_numba(tracer_conc: np.ndarray, dry_air_mass_top: np.ndarray) -> np.ndarray:
    if not _NUMBA_AVAILABLE:
        return np.sum(tracer_conc * dry_air_mass_top[:, :, :, np.newaxis], axis=(0, 1, 2))
    return _tracer_working_mass_numba_kernel(tracer_conc, dry_air_mass_top)


@dataclass(frozen=True)
class _VdiffFullGridWorkspace:
    nthreads: int
    tracer_out: np.ndarray
    sphu_out: np.ndarray
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
    qmx: np.ndarray
    adjust: np.ndarray
    tracer_diffused: np.ndarray
    tracer_ratio: np.ndarray
    tracer_after_mass: np.ndarray
    shmx: np.ndarray
    zfq_scalar: np.ndarray
    sphu_diffused: np.ndarray


_VDIFF_FULLGRID_WORKSPACES = threading.local()


def _get_vdiff_fullgrid_workspace(
    nthreads: int, nlev: int, nlat: int, nlon: int, ntracer: int
) -> _VdiffFullGridWorkspace:
    existing = getattr(_VDIFF_FULLGRID_WORKSPACES, "workspace", None)
    if (
        existing is not None
        and existing.nthreads == nthreads
        and existing.tracer_out.shape == (nlev, nlat, nlon, ntracer)
        and existing.sphu_out.shape == (nlev, nlat, nlon)
        and existing.pmid.shape == (nthreads, nlon, nlev)
        and existing.tracer_diffused.shape == (nthreads, nlon, nlev, ntracer)
        and existing.tracer_ratio.shape == (nthreads, nlon, ntracer)
        and existing.tracer_after_mass.shape == (nthreads, nlon, ntracer)
    ):
        return existing

    lev_shape = (nthreads, nlon, nlev)
    edge_shape = (nthreads, nlon, nlev + 1)
    scalar_shape = (nthreads, nlon)
    tracer_shape = (nthreads, nlon, ntracer)
    lev_tracer_shape = (nthreads, nlon, nlev, ntracer)
    workspace = _VdiffFullGridWorkspace(
        nthreads=nthreads,
        tracer_out=np.empty((nlev, nlat, nlon, ntracer), dtype=np.float64),
        sphu_out=np.empty((nlev, nlat, nlon), dtype=np.float64),
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
        tpert=np.empty(scalar_shape, dtype=np.float64),
        qpert=np.empty(scalar_shape, dtype=np.float64),
        tmp1=np.empty(scalar_shape, dtype=np.float64),
        dshbot=np.empty(scalar_shape, dtype=np.float64),
        rrho=np.empty(scalar_shape, dtype=np.float64),
        khfs=np.empty(scalar_shape, dtype=np.float64),
        kshfs=np.empty(scalar_shape, dtype=np.float64),
        thvsrf=np.empty(scalar_shape, dtype=np.float64),
        heatv=np.empty(scalar_shape, dtype=np.float64),
        obklen=np.empty(scalar_shape, dtype=np.float64),
        fak1=np.empty(scalar_shape, dtype=np.float64),
        phiminv=np.empty(scalar_shape, dtype=np.float64),
        phihinv=np.empty(scalar_shape, dtype=np.float64),
        wm=np.empty(scalar_shape, dtype=np.float64),
        fak2=np.empty(scalar_shape, dtype=np.float64),
        fak3=np.empty(scalar_shape, dtype=np.float64),
        pblk=np.empty(scalar_shape, dtype=np.float64),
        pr=np.empty(scalar_shape, dtype=np.float64),
        potbar=np.empty(edge_shape, dtype=np.float64),
        cah=np.empty(lev_shape, dtype=np.float64),
        cch=np.empty(lev_shape, dtype=np.float64),
        zeh=np.empty(lev_shape, dtype=np.float64),
        termh=np.empty(lev_shape, dtype=np.float64),
        qmx=np.empty(lev_tracer_shape, dtype=np.float64),
        adjust=np.empty(tracer_shape, dtype=np.bool_),
        tracer_diffused=np.empty(lev_tracer_shape, dtype=np.float64),
        tracer_ratio=np.empty(tracer_shape, dtype=np.float64),
        tracer_after_mass=np.empty(tracer_shape, dtype=np.float64),
        shmx=np.empty(lev_shape, dtype=np.float64),
        zfq_scalar=np.empty(lev_shape, dtype=np.float64),
        sphu_diffused=np.empty(lev_shape, dtype=np.float64),
    )
    _VDIFF_FULLGRID_WORKSPACES.workspace = workspace
    return workspace



def _numba_vdiff_mode() -> str:
    return numba_mode()


def _numba_vdiff_enabled() -> bool:
    return numba_available_and_enabled(available=_NUMBA_AVAILABLE)


def _numba_vdiff_thread_count() -> int:
    return configure_numba_threads(available=_NUMBA_AVAILABLE)


def _prepare_vdiff_met_plan_numba(
    *,
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
    area_m2: np.ndarray,
    dt_s: float,
    npbl: int,
    nthreads: int,
    plan_output: tuple[np.ndarray, ...],
    sphu_output_buffer: np.ndarray | None = None,
    diagnostic_plan_output: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Prepare meteorology-only VDIFF coefficients without tracer inputs."""
    if not _NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    nlev, nlat, nlon = temperature_top.shape
    workspace = _get_vdiff_fullgrid_workspace(nthreads, nlev, nlat, nlon, 0)
    empty_tracer = workspace.tracer_out
    empty_flux = empty_tracer[0]
    sphu_out = workspace.sphu_out
    if sphu_output_buffer is not None:
        sphu_out = np.asarray(sphu_output_buffer)
        if sphu_out.shape != sphu_top.shape or sphu_out.dtype != np.float64:
            raise ValueError("sphu_output_buffer must match sphu_top shape and float64 dtype")
        if not sphu_out.flags.c_contiguous or not sphu_out.flags.writeable:
            raise ValueError("sphu_output_buffer must be writable and C-contiguous")
        if np.shares_memory(sphu_out, sphu_top):
            raise ValueError("sphu_output_buffer must not overlap sphu_top")
    (
        plan_cch,
        plan_zeh,
        plan_termh,
        plan_cgs,
        plan_kvh,
        plan_potbar,
        plan_rpdel,
        plan_rrho,
        plan_tmp1,
    ) = plan_output
    expected_plan_shape = (nlev, nlat, nlon)
    if any(array.shape != expected_plan_shape for array in plan_output[:3]):
        raise ValueError(f"plan coefficient arrays must have shape {expected_plan_shape}")
    if any(array.shape != (nlev + 1, nlat, nlon) for array in plan_output[3:6]):
        raise ValueError("plan edge arrays have the wrong shape")
    if (
        plan_rpdel.shape != expected_plan_shape
        or plan_rrho.shape != (nlat, nlon)
        or plan_tmp1.shape != (nlat, nlon)
    ):
        raise ValueError("plan source arrays have the wrong shape")
    if diagnostic_plan_output is None:
        plan_kvm = np.empty((1, 1, 1), dtype=np.float64)
        plan_tpert = np.empty((1, 1), dtype=np.float64)
        plan_qpert = np.empty((1, 1), dtype=np.float64)
        capture_plan_diagnostics = False
    else:
        plan_kvm, plan_tpert, plan_qpert = diagnostic_plan_output
        if plan_kvm.shape != (nlev + 1, nlat, nlon):
            raise ValueError("diagnostic plan kvm array has the wrong shape")
        if plan_tpert.shape != (nlat, nlon) or plan_qpert.shape != (nlat, nlon):
            raise ValueError("diagnostic plan perturbation arrays have the wrong shape")
        capture_plan_diagnostics = True
    _run_vdiffdr_fullgrid_zero_flux_numba_kernel(
        empty_tracer,
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
        empty_flux,
        ustar_m_s,
        area_m2,
        dt_s,
        npbl,
        True,
        empty_tracer,
        sphu_out,
        pmid_hpa,
        False,
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
        workspace.qmx,
        workspace.adjust,
        workspace.tracer_diffused,
        workspace.tracer_ratio,
        workspace.tracer_after_mass,
        workspace.shmx,
        workspace.zfq_scalar,
        workspace.sphu_diffused,
        plan_cch,
        plan_zeh,
        plan_termh,
        plan_cgs,
        plan_kvh,
        plan_potbar,
        plan_rpdel,
        plan_rrho,
        plan_tmp1,
        True,
        True,
        plan_kvm,
        plan_tpert,
        plan_qpert,
        capture_plan_diagnostics,
    )
    return sphu_out


if njit is not None:

    @njit(cache=True, parallel=True, nogil=True)
    def _finalize_deferred_tpcore_poles_numba_kernel(
        tracer_mass: np.ndarray, pressure_mass_hpa: np.ndarray
    ) -> None:
        """Reproduce TPCORE's finalized pole copies before per-latitude VDIFF work."""
        nlev, nlat, nlon, ntracer = tracer_mass.shape
        for lev in prange(nlev):
            for lon in range(nlon):
                south_inv = 1.0 / pressure_mass_hpa[lev, 0, lon]
                north_inv = 1.0 / pressure_mass_hpa[lev, nlat - 1, lon]
                for tracer in range(ntracer):
                    south = tracer_mass[lev, 0, lon, tracer] * south_inv
                    north = tracer_mass[lev, nlat - 1, lon, tracer] * north_inv
                    if south < 0.0:
                        south = 1.0e-26
                    if north < 0.0:
                        north = 1.0e-26
                    tracer_mass[lev, 1, lon, tracer] = south
                    tracer_mass[lev, nlat - 2, lon, tracer] = north

    @njit(cache=True, nogil=True)
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


    @njit(inline="always", nogil=True)
    def _solve_vdiff_humidity_numba(
        nlev: int,
        ntopfl: int,
        lat: int,
        shmx: np.ndarray,
        termh: np.ndarray,
        cch: np.ndarray,
        zeh: np.ndarray,
        dshbot: np.ndarray,
        zfq_scalar: np.ndarray,
        sphu_diffused: np.ndarray,
        sphu_out: np.ndarray,
    ) -> None:
        nlon = shmx.shape[0]
        for lon in range(nlon):
            zfq_scalar[lon, ntopfl] = shmx[lon, ntopfl] * termh[lon, ntopfl]
        for lev in range(ntopfl + 1, nlev - 1):
            for lon in range(nlon):
                zfq_scalar[lon, lev] = (
                    shmx[lon, lev] + cch[lon, lev] * zfq_scalar[lon, lev - 1]
                ) * termh[lon, lev]
        for lon in range(nlon):
            tmp1d = 1.0 / (1.0 + cch[lon, nlev - 1] * (1.0 - zeh[lon, nlev - 2]))
            zfq_scalar[lon, nlev - 1] = (
                shmx[lon, nlev - 1]
                + dshbot[lon]
                + cch[lon, nlev - 1] * zfq_scalar[lon, nlev - 2]
            ) * tmp1d
            sphu_diffused[lon, nlev - 1] = zfq_scalar[lon, nlev - 1]
        for lev in range(nlev - 2, ntopfl - 1, -1):
            for lon in range(nlon):
                sphu_diffused[lon, lev] = (
                    zfq_scalar[lon, lev] + zeh[lon, lev] * sphu_diffused[lon, lev + 1]
                )
        for lon in range(nlon):
            for lev in range(nlev):
                value = sphu_diffused[lon, lev]
                if value < 1.0e-12:
                    value = 0.0
                sphu_out[lev, lat, lon] = value


    @njit(cache=True, parallel=True, nogil=True)
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
        surface_flux_kg_m2_s: np.ndarray,
        ustar_m_s: np.ndarray,
        area_m2: np.ndarray,
        dt_s: float,
        npbl: int,
        surface_flux_is_zero: bool,
        tracer_out: np.ndarray,
        sphu_out: np.ndarray,
        input_mass_pressure_hpa: np.ndarray,
        input_is_pressure_mass: bool,
        pmid_workspace: np.ndarray,
        pint_workspace: np.ndarray,
        rpdel_workspace: np.ndarray,
        rpdeli_workspace: np.ndarray,
        zm_workspace: np.ndarray,
        thp_workspace: np.ndarray,
        kvf_workspace: np.ndarray,
        kvh_workspace: np.ndarray,
        kvm_workspace: np.ndarray,
        cgsh_workspace: np.ndarray,
        cgs_workspace: np.ndarray,
        tpert_workspace: np.ndarray,
        qpert_workspace: np.ndarray,
        tmp1_workspace: np.ndarray,
        dshbot_workspace: np.ndarray,
        rrho_workspace: np.ndarray,
        khfs_workspace: np.ndarray,
        kshfs_workspace: np.ndarray,
        thvsrf_workspace: np.ndarray,
        heatv_workspace: np.ndarray,
        obklen_workspace: np.ndarray,
        fak1_workspace: np.ndarray,
        phiminv_workspace: np.ndarray,
        phihinv_workspace: np.ndarray,
        wm_workspace: np.ndarray,
        fak2_workspace: np.ndarray,
        fak3_workspace: np.ndarray,
        pblk_workspace: np.ndarray,
        pr_workspace: np.ndarray,
        potbar_workspace: np.ndarray,
        cah_workspace: np.ndarray,
        cch_workspace: np.ndarray,
        zeh_workspace: np.ndarray,
        termh_workspace: np.ndarray,
        qmx_workspace: np.ndarray,
        adjust_workspace: np.ndarray,
        tracer_diffused_workspace: np.ndarray,
        tracer_ratio_workspace: np.ndarray,
        tracer_after_mass_workspace: np.ndarray,
        shmx_workspace: np.ndarray,
        zfq_scalar_workspace: np.ndarray,
        sphu_diffused_workspace: np.ndarray,
        plan_cch: np.ndarray,
        plan_zeh: np.ndarray,
        plan_termh: np.ndarray,
        plan_cgs: np.ndarray,
        plan_kvh: np.ndarray,
        plan_potbar: np.ndarray,
        plan_rpdel: np.ndarray,
        plan_rrho: np.ndarray,
        plan_tmp1: np.ndarray,
        capture_plan: bool,
        plan_only: bool,
        plan_kvm: np.ndarray,
        plan_tpert: np.ndarray,
        plan_qpert: np.ndarray,
        capture_plan_diagnostics: bool,
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

        for lat in prange(nlat):
            thread_id = get_thread_id()
            source_lat = lat
            input_needs_conversion = input_is_pressure_mass and lat != 1 and lat != nlat - 2
            pmid = pmid_workspace[thread_id]
            pint = pint_workspace[thread_id]
            rpdel = rpdel_workspace[thread_id]
            rpdeli = rpdeli_workspace[thread_id]
            zm = zm_workspace[thread_id]
            thp = thp_workspace[thread_id]
            kvf = kvf_workspace[thread_id]
            kvh = kvh_workspace[thread_id]
            kvm = kvm_workspace[thread_id]
            cgsh = cgsh_workspace[thread_id]
            cgs = cgs_workspace[thread_id]
            tpert = tpert_workspace[thread_id]
            qpert = qpert_workspace[thread_id]
            tmp1 = tmp1_workspace[thread_id]
            dshbot = dshbot_workspace[thread_id]
            rrho = rrho_workspace[thread_id]
            khfs = khfs_workspace[thread_id]
            kshfs = kshfs_workspace[thread_id]
            thvsrf = thvsrf_workspace[thread_id]
            heatv = heatv_workspace[thread_id]
            obklen = obklen_workspace[thread_id]
            fak1 = fak1_workspace[thread_id]
            phiminv = phiminv_workspace[thread_id]
            phihinv = phihinv_workspace[thread_id]
            wm = wm_workspace[thread_id]
            fak2 = fak2_workspace[thread_id]
            fak3 = fak3_workspace[thread_id]
            pblk = pblk_workspace[thread_id]
            pr = pr_workspace[thread_id]
            potbar = potbar_workspace[thread_id]
            cah = cah_workspace[thread_id]
            cch = cch_workspace[thread_id]
            zeh = zeh_workspace[thread_id]
            termh = termh_workspace[thread_id]
            qmx = qmx_workspace[thread_id]
            adjust = adjust_workspace[thread_id]
            tracer_diffused = tracer_diffused_workspace[thread_id]
            tracer_ratio = tracer_ratio_workspace[thread_id]
            tracer_after_mass = tracer_after_mass_workspace[thread_id]
            shmx = shmx_workspace[thread_id]
            zfq_scalar = zfq_scalar_workspace[thread_id]
            sphu_diffused = sphu_diffused_workspace[thread_id]
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
                potbar[lon, 0] = 0.0
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
            if capture_plan:
                for lev in range(nlev):
                    for lon in range(nlon):
                        plan_cch[lev, lat, lon] = cch[lon, lev]
                        plan_zeh[lev, lat, lon] = zeh[lon, lev]
                        plan_termh[lev, lat, lon] = termh[lon, lev]
                        plan_rpdel[lev, lat, lon] = rpdel[lon, lev]
                for edge in range(nlev + 1):
                    for lon in range(nlon):
                        plan_cgs[edge, lat, lon] = cgs[lon, edge]
                        plan_kvh[edge, lat, lon] = kvh[lon, edge]
                        plan_potbar[edge, lat, lon] = potbar[lon, edge]
                for lon in range(nlon):
                    plan_rrho[lat, lon] = rrho[lon]
                    plan_tmp1[lat, lon] = tmp1[lon]
            if capture_plan_diagnostics:
                for edge in range(nlev + 1):
                    for lon in range(nlon):
                        plan_kvm[edge, lat, lon] = kvm[lon, edge]
                for lon in range(nlon):
                    plan_tpert[lat, lon] = tpert[lon]
                    plan_qpert[lat, lon] = qpert[lon]

            if capture_plan and plan_only:
                _solve_vdiff_humidity_numba(
                    nlev,
                    ntopfl,
                    lat,
                    shmx,
                    termh,
                    cch,
                    zeh,
                    dshbot,
                    zfq_scalar,
                    sphu_diffused,
                    sphu_out,
                )
                continue

            if not surface_flux_is_zero:
                for lon in range(nlon):
                    for tracer in range(ntracer):
                        adjust[lon, tracer] = False
                    for lev in range(nlev):
                        inv_input_mass = 1.0
                        if input_needs_conversion:
                            inv_input_mass = 1.0 / input_mass_pressure_hpa[lev, source_lat, lon]
                        for tracer in range(ntracer):
                            input_value = tracer_top[lev, source_lat, lon, tracer] * inv_input_mass
                            if input_needs_conversion and input_value < 0.0:
                                input_value = 1.0e-26
                            qmx[lon, lev, tracer] = input_value
                if npbl > 1:
                    for lev in range(start, nlev):
                        for lon in range(nlon):
                            scale = ztodtgor * rpdel[lon, lev]
                            term_next = potbar[lon, lev + 1] * kvh[lon, lev + 1]
                            term_now = potbar[lon, lev] * kvh[lon, lev]
                            inv_input_mass = 1.0
                            if input_needs_conversion:
                                inv_input_mass = 1.0 / input_mass_pressure_hpa[lev, source_lat, lon]
                            for tracer in range(ntracer):
                                input_value = tracer_top[lev, source_lat, lon, tracer] * inv_input_mass
                                if input_needs_conversion and input_value < 0.0:
                                    input_value = 1.0e-26
                                flux_rrho = surface_flux_kg_m2_s[lat, lon, tracer] * rrho[lon]
                                cgq_next = flux_rrho * cgs[lon, lev + 1]
                                cgq_now = flux_rrho * cgs[lon, lev]
                                qmx_value = input_value + scale * (
                                    term_next * cgq_next - term_now * cgq_now
                                )
                                qmx[lon, lev, tracer] = qmx_value
                                if qmx_value < 0.0:
                                    adjust[lon, tracer] = True
                    for lon in range(nlon):
                        for tracer in range(ntracer):
                            if adjust[lon, tracer]:
                                for lev in range(start, nlev):
                                    input_value = tracer_top[lev, source_lat, lon, tracer]
                                    if input_needs_conversion:
                                        input_value *= 1.0 / input_mass_pressure_hpa[lev, source_lat, lon]
                                        if input_value < 0.0:
                                            input_value = 1.0e-26
                                    qmx[lon, lev, tracer] = input_value

            for lon in range(nlon):
                dry_mass = dry_mass_top[ntopfl, lat, lon]
                inv_input_mass = 1.0
                if input_needs_conversion:
                    inv_input_mass = 1.0 / input_mass_pressure_hpa[ntopfl, source_lat, lon]
                for tracer in range(ntracer):
                    tracer_value = tracer_top[ntopfl, source_lat, lon, tracer] * inv_input_mass
                    if input_needs_conversion and tracer_value < 0.0:
                        tracer_value = 1.0e-26
                    tracer_ratio[lon, tracer] = tracer_value * dry_mass
                    tracer_diffused[lon, ntopfl, tracer] = (
                        (tracer_value if surface_flux_is_zero else qmx[lon, ntopfl, tracer]) * termh[lon, ntopfl]
                    )
            for lev in range(ntopfl + 1, nlev - 1):
                for lon in range(nlon):
                    dry_mass = dry_mass_top[lev, lat, lon]
                    cch_value = cch[lon, lev]
                    termh_value = termh[lon, lev]
                    inv_input_mass = 1.0
                    if input_needs_conversion:
                        inv_input_mass = 1.0 / input_mass_pressure_hpa[lev, source_lat, lon]
                    for tracer in range(ntracer):
                        tracer_value = tracer_top[lev, source_lat, lon, tracer] * inv_input_mass
                        if input_needs_conversion and tracer_value < 0.0:
                            tracer_value = 1.0e-26
                        tracer_ratio[lon, tracer] += tracer_value * dry_mass
                        source_value = tracer_value if surface_flux_is_zero else qmx[lon, lev, tracer]
                        tracer_diffused[lon, lev, tracer] = (
                            source_value
                            + cch_value * tracer_diffused[lon, lev - 1, tracer]
                        ) * termh_value
            for lon in range(nlon):
                tmp1d = 1.0 / (1.0 + cch[lon, nlev - 1] * (1.0 - zeh[lon, nlev - 2]))
                dry_mass = dry_mass_top[nlev - 1, lat, lon]
                cch_bottom = cch[lon, nlev - 1]
                inv_input_mass = 1.0
                if input_needs_conversion:
                    inv_input_mass = 1.0 / input_mass_pressure_hpa[nlev - 1, source_lat, lon]
                for tracer in range(ntracer):
                    tracer_value = tracer_top[nlev - 1, source_lat, lon, tracer] * inv_input_mass
                    if input_needs_conversion and tracer_value < 0.0:
                        tracer_value = 1.0e-26
                    tracer_ratio[lon, tracer] += tracer_value * dry_mass
                    source_value = tracer_value if surface_flux_is_zero else qmx[lon, nlev - 1, tracer]
                    tracer_diffused[lon, nlev - 1, tracer] = (
                        source_value
                        + (
                            0.0
                            if surface_flux_is_zero
                            else surface_flux_kg_m2_s[lat, lon, tracer] * tmp1[lon]
                        )
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
                    if not surface_flux_is_zero:
                        before_mass += surface_flux_kg_m2_s[lat, lon, tracer] * area_m2[lat, lon] * dt_s
                    after_mass = tracer_after_mass[lon, tracer]
                    if abs(before_mass) > 0.0 and abs(after_mass) > 0.0:
                        ratio = before_mass / after_mass
                    tracer_ratio[lon, tracer] = ratio
                for lev in range(ntopfl, nlev):
                    for tracer in range(ntracer):
                        tracer_out[lev, lat, lon, tracer] = tracer_diffused[lon, lev, tracer] * tracer_ratio[
                            lon, tracer
                        ]

            _solve_vdiff_humidity_numba(
                nlev,
                ntopfl,
                lat,
                shmx,
                termh,
                cch,
                zeh,
                dshbot,
                zfq_scalar,
                sphu_diffused,
                sphu_out,
            )

        return negative_count


else:

    def _finalize_deferred_tpcore_poles_numba_kernel(
        tracer_mass: np.ndarray, pressure_mass_hpa: np.ndarray
    ) -> None:
        raise RuntimeError("numba is not available")

    def _tracer_working_mass_numba_kernel(tracer_conc: np.ndarray, dry_air_mass_top: np.ndarray) -> np.ndarray:
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
        surface_flux_kg_m2_s: np.ndarray,
        ustar_m_s: np.ndarray,
        area_m2: np.ndarray,
        dt_s: float,
        npbl: int,
        surface_flux_is_zero: bool,
        tracer_out: np.ndarray,
        sphu_out: np.ndarray,
        input_mass_pressure_hpa: np.ndarray,
        input_is_pressure_mass: bool,
        *workspace: np.ndarray,
    ) -> int:
        raise RuntimeError("numba is not available")
