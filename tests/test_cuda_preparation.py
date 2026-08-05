from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wombat_transport.cuda import CudaRuntime, CudaUnavailableError
from wombat_transport.cuda.forcing import CudaForcingChunks
from wombat_transport.cuda.forcing import CudaForcingStep
from wombat_transport.cuda.preparation import CudaPlanPreparation
from wombat_transport.grid import TransportGrid
from wombat_transport.transport.pressure import dry_surface_pressure_hpa
from wombat_transport.transport.pressure import dry_air_mass_from_pressure
from wombat_transport.transport.pressure import dry_pressure_thickness_from_surface_hpa
from wombat_transport.transport.pressure import pressure_edges_from_surface_hpa
from wombat_transport.transport.pressure import wet_surface_pressure_hpa
from wombat_transport.transport.driver import _hydrostatic_box_height_m
from wombat_transport.transport.driver import _virtual_temperature_k
from wombat_transport.transport.pbl._plan import prepare_vdiff_met_plan
from wombat_transport.transport.tpcore._plan import prepare_tpcore_met_plan
from wombat_transport.transport.tpcore._reference import build_tpcore_static_terms
from wombat_transport.transport.forcing import TransportForcingChunkSelection


def _runtime_or_skip() -> CudaRuntime:
    try:
        return CudaRuntime()
    except CudaUnavailableError as exc:
        pytest.skip(str(exc))


@pytest.mark.cuda
def test_cuda_forcing_chunks_upload_once_and_select_views():
    runtime = _runtime_or_skip()
    a1 = SimpleNamespace(
        **{
            name: np.full((2, 5, 4), index, dtype=np.float64)
            for index, name in enumerate(("pblh", "hflux", "eflux", "ustar"))
        }
    )
    a3 = SimpleNamespace(
        **{
            name: np.full((2, 3, 5, 4), index, dtype=np.float64)
            for index, name in enumerate(
                ("u", "v", "dtrain", "dqrcu", "reevapcn")
            )
        },
        cmfmc=np.ones((2, 4, 5, 4), dtype=np.float64),
    )
    i3 = SimpleNamespace(
        surface_pressure=np.ones((3, 5, 4), dtype=np.float64),
        qv=np.ones((3, 3, 5, 4), dtype=np.float64),
        temperature=np.ones((3, 3, 5, 4), dtype=np.float64),
    )
    selection = TransportForcingChunkSelection(
        a1_block=a1,
        a3_block=a3,
        i3_block=i3,
        a1_offset=1,
        a3_offset=1,
        i3_start_offset=1,
        i3_end_offset=2,
        i3_restart_offset=1,
        start_fraction=0.2,
        end_fraction=0.3,
        midpoint_fraction=0.25,
    )
    chunks = CudaForcingChunks(runtime, dtype=np.float64)

    first = chunks.select(selection)
    after_first = runtime.transfer_stats
    second = chunks.select(selection)

    assert runtime.transfer_stats == after_first
    assert first.u_m_s.data.ptr == second.u_m_s.data.ptr
    np.testing.assert_array_equal(
        runtime.to_host(first.cmfmc_kg_m2_s),
        np.ones((3, 5, 4), dtype=np.float64),
    )


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_cuda_resident_tpcore_preparation_matches_cpu_strict(dtype):
    runtime = _runtime_or_skip()
    nlev, nlat, nlon = 3, 5, 4
    lat = np.linspace(-90.0, 90.0, nlat)
    lon = np.linspace(-180.0, 90.0, nlon)
    area = np.repeat(
        np.linspace(1.0, 3.0, nlat)[:, None],
        nlon,
        axis=1,
    )
    grid = TransportGrid(
        lat_deg=lat,
        lon_deg=lon,
        lev=np.arange(1.0, nlev + 1),
        area_m2=area,
        hyai_hpa=np.array([0.0, 0.0, 0.0, 0.01]),
        hybi=np.array([1.0, 0.7, 0.3, 0.0]),
        template_path=Path("unused.nc4"),
    )
    static = build_tpcore_static_terms(
        area_m2=area,
        hyai_hpa=grid.hyai_hpa,
        hybi=grid.hybi,
        lat_deg=lat,
    )
    horizontal = np.arange(nlat * nlon, dtype=np.float64).reshape(nlat, nlon)
    center = np.arange(nlev * nlat * nlon, dtype=np.float64).reshape(
        nlev,
        nlat,
        nlon,
    )
    ps0 = 99_000.0 + 3.0 * horizontal
    ps1 = 99_300.0 + 2.0 * horizontal
    q0 = 0.001 + center * 1.0e-7
    q1 = 0.0012 + center * 2.0e-7
    t0 = 260.0 + center * 0.01
    t1 = 261.0 + center * 0.02
    u = 4.0 + center * 0.001
    v = -2.0 + center * 0.0005
    dtrain = center * 1.0e-9
    dqrcu = center * 2.0e-10
    reevapcn = center * 3.0e-10
    cmfmc = center * 4.0e-5
    start_fraction = 0.2
    end_fraction = 0.25
    midpoint_fraction = 0.225
    dry0 = dry_surface_pressure_hpa(
        ps0[None],
        q0[None],
        grid.hyai_hpa,
        grid.hybi,
        area_m2=area,
    )[0]
    dry1 = dry_surface_pressure_hpa(
        ps1[None],
        q1[None],
        grid.hyai_hpa,
        grid.hybi,
        area_m2=area,
    )[0]
    wet0 = wet_surface_pressure_hpa(ps0[None], area_m2=area)[0]
    wet1 = wet_surface_pressure_hpa(ps1[None], area_m2=area)[0]
    p1 = dry0 + (dry1 - dry0) * start_fraction
    p2 = dry0 + (dry1 - dry0) * end_fraction
    expected = prepare_tpcore_met_plan(
        p1_hpa=p1,
        p2_hpa=p2,
        u_m_s=u,
        v_m_s=v,
        area_m2=area,
        hyai_hpa=grid.hyai_hpa,
        hybi=grid.hybi,
        lat_deg=lat,
        dt_s=600.0,
        static_terms=static,
    )
    forcing = CudaForcingStep(
        pblh_m=runtime.to_device(np.full((nlat, nlon), 1000.0)),
        hflux_w_m2=runtime.to_device(np.ones((nlat, nlon))),
        eflux_w_m2=runtime.to_device(np.ones((nlat, nlon))),
        ustar_m_s=runtime.to_device(np.ones((nlat, nlon))),
        u_m_s=runtime.to_device(u),
        v_m_s=runtime.to_device(v),
        dtrain_kg_m2_s=runtime.to_device(dtrain),
        dqrcu_kg_kg_s=runtime.to_device(dqrcu),
        reevapcn_kg_kg_s=runtime.to_device(reevapcn),
        cmfmc_kg_m2_s=runtime.to_device(cmfmc),
        surface_pressure_start_pa=runtime.to_device(ps0),
        surface_pressure_end_pa=runtime.to_device(ps1),
        qv_start=runtime.to_device(q0),
        qv_end=runtime.to_device(q1),
        temperature_start_k=runtime.to_device(t0),
        temperature_end_k=runtime.to_device(t1),
        start_fraction=start_fraction,
        end_fraction=end_fraction,
        midpoint_fraction=midpoint_fraction,
    )
    builder = CudaPlanPreparation(
        runtime,
        dtype=dtype,
        grid=grid,
        tpcore_static_terms=static,
        initial_dry_surface_pressure_hpa=p1,
    )
    actual = builder.prepare_tpcore_step(forcing, dt_s=600.0)
    met = builder.meteorology

    np.testing.assert_allclose(
        runtime.to_host(met.wet_surface_pressure_start_hpa),
        wet0 + (wet1 - wet0) * start_fraction,
        rtol=2e-15,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        runtime.to_host(met.dry_surface_pressure_hpa),
        p2,
        rtol=2e-15,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        runtime.to_host(met.specific_humidity_kg_kg),
        q0 + (q1 - q0) * midpoint_fraction,
        rtol=0.0,
        atol=2e-19,
    )
    expected_arrays = {
        "delp1": expected.setup.delp1_hpa,
        "delp2": expected.setup.delp2_hpa,
        "pu": expected.setup.pu_hpa,
        "xmass": expected.setup.xmass_hpa,
        "ymass": expected.setup.ymass_hpa,
        "vertical_mass_flux": expected.setup.vertical_mass_flux_hpa,
        "normalized_vertical_courant": expected.normalized_vertical_courant,
        "cx": expected.setup.cx,
        "cy": expected.setup.cy,
        "ua": expected.ua,
        "va": expected.va,
        "jn": expected.jn,
        "js": expected.js,
    }
    for name, expected_array in expected_arrays.items():
        if np.issubdtype(expected_array.dtype, np.floating):
            expected_array = expected_array.astype(dtype)
        np.testing.assert_allclose(
            runtime.to_host(getattr(actual, name)),
            expected_array,
            rtol=3e-14 if dtype is np.float64 else 3e-6,
            atol=3e-14 if dtype is np.float64 else 3e-7,
            err_msg=name,
        )

    q_mid = q0 + (q1 - q0) * midpoint_fraction
    temperature_mid = t0 + (t1 - t0) * midpoint_fraction
    wet_end = wet0 + (wet1 - wet0) * end_fraction
    pedge = pressure_edges_from_surface_hpa(
        wet_end[None],
        grid.hyai_hpa,
        grid.hybi,
    )[0]
    pmid = 0.5 * (pedge[:-1] + pedge[1:])
    virtual_temperature = _virtual_temperature_k(temperature_mid, q_mid)
    bxheight = _hydrostatic_box_height_m(pedge, virtual_temperature)
    next_delp = dry_pressure_thickness_from_surface_hpa(
        p2[None],
        grid.hyai_hpa,
        grid.hybi,
    )
    next_mass = dry_air_mass_from_pressure(next_delp, area)
    expected_vdiff = prepare_vdiff_met_plan(
        u_top=u[::-1],
        v_top=v[::-1],
        temperature_top=temperature_mid[::-1],
        sphu_top=q_mid[::-1],
        pmid_hpa=pmid[::-1],
        pint_hpa=pedge[::-1],
        virtual_temperature_top=virtual_temperature[::-1],
        bxheight_top=bxheight[::-1],
        dry_mass_top=next_mass[0, ::-1],
        pblh_m=np.full((nlat, nlon), 1000.0),
        hflux_w_m2=np.full((nlat, nlon), 1.0),
        water_flux_kg_m2_s=np.full((nlat, nlon), 1.0 / 2.5104e6),
        ustar_m_s=np.full((nlat, nlon), 1.0),
        area_m2=area,
        dt_s=600.0,
        workers=1,
    )
    vdiff, convection = builder.prepare_vdiff_and_convection(
        forcing,
        dt_s=600.0,
    )
    vdiff_arrays = {
        "cch": expected_vdiff.cch,
        "zeh": expected_vdiff.zeh,
        "termh": expected_vdiff.termh,
        "cgs": expected_vdiff.cgs,
        "kvh": expected_vdiff.kvh,
        "potbar": expected_vdiff.potbar,
        "rpdel": expected_vdiff.rpdel,
        "rrho": expected_vdiff.rrho,
        "tmp1": expected_vdiff.tmp1,
        "dry_mass": expected_vdiff.dry_mass,
        "specific_humidity_after": expected_vdiff.specific_humidity_after,
    }
    for name, expected_array in vdiff_arrays.items():
        actual_array = runtime.to_host(getattr(vdiff, name))
        if name in {"cch"}:
            actual_array = actual_array[1:]
            expected_array = expected_array[1:]
        elif name in {"zeh", "termh"}:
            actual_array = actual_array[:-1]
            expected_array = expected_array[:-1]
        expected_array = expected_array.astype(dtype)
        np.testing.assert_allclose(
            actual_array,
            expected_array,
            rtol=3e-13 if dtype is np.float64 else 3e-6,
            atol=3e-13 if dtype is np.float64 else 3e-7,
            err_msg=name,
        )
    np.testing.assert_array_equal(
        runtime.to_host(vdiff.start_level),
        np.array([expected_vdiff.start_level], dtype=np.int32),
    )
    convection_expected = {
        "cmfmc": cmfmc[::-1],
        "dtrain": dtrain[::-1],
        "dqrcu": dqrcu[::-1],
        "reevapcn": reevapcn[::-1],
        "delp_hpa": next_delp[0, ::-1],
        "delp_dry": next_delp[0, ::-1],
        "bmass": next_delp[0, ::-1] * (100.0 / 9.80665),
    }
    for name, expected_array in convection_expected.items():
        expected_array = expected_array.astype(dtype)
        np.testing.assert_allclose(
            runtime.to_host(getattr(convection, name)),
            expected_array,
            rtol=3e-15 if dtype is np.float64 else 3e-6,
            atol=3e-15 if dtype is np.float64 else 3e-7,
            err_msg=name,
        )
