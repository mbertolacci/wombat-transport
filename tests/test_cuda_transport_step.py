from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import pytest

from wombat_transport.cuda import CudaRuntime, CudaUnavailableError
from wombat_transport.cuda.transport_step import CudaTransportStepExecutor
from wombat_transport.cuda.transport_step import CudaTransportStepPlans
from wombat_transport.transport.pbl._plan import VdiffPlan
from wombat_transport.transport.tpcore import run_tpcore_one_step_with_setup
from wombat_transport.transport.tpcore._plan import prepare_tpcore_met_plan


CASE_DIR = Path("tests/fixtures/tpcore_snapshot_v2")


def _cuda_runtime_or_skip() -> CudaRuntime:
    try:
        return CudaRuntime()
    except CudaUnavailableError as exc:
        pytest.skip(str(exc))


def _to_blocks(values: np.ndarray, width: int) -> np.ndarray:
    nlev, nlat, nlon, ntracer = values.shape
    blocks = np.zeros(
        ((ntracer + width - 1) // width, nlev, nlat, nlon, width),
        dtype=values.dtype,
    )
    for tracer in range(ntracer):
        block, lane = divmod(tracer, width)
        blocks[block, ..., lane] = values[..., tracer]
    return blocks


def _from_blocks(values: np.ndarray, tracer_count: int) -> np.ndarray:
    output = np.empty((*values.shape[1:-1], tracer_count), dtype=values.dtype)
    for tracer in range(tracer_count):
        block, lane = divmod(tracer, values.shape[-1])
        output[..., tracer] = values[block, ..., lane]
    return output


def _identity_vdiff_plan(
    shape: tuple[int, int, int],
    area_m2: np.ndarray,
) -> VdiffPlan:
    nlev, nlat, nlon = shape
    center = np.zeros(shape, dtype=np.float64)
    edge = np.zeros((nlev + 1, nlat, nlon), dtype=np.float64)
    horizontal = np.ones((nlat, nlon), dtype=np.float64)
    return VdiffPlan(
        cch=center,
        zeh=center,
        termh=np.ones_like(center),
        cgs=edge,
        kvh=edge,
        potbar=edge,
        rpdel=center,
        rrho=horizontal,
        tmp1=horizontal,
        dry_mass=np.ones_like(center),
        area_m2=np.ascontiguousarray(area_m2),
        dt_s=600.0,
        start_level=0,
        specific_humidity_after=center,
    )


@pytest.mark.cuda
def test_cuda_transport_step_composes_resident_operators(monkeypatch):
    monkeypatch.setenv("WOMBAT_NUMBA", "1")
    monkeypatch.setenv("WOMBAT_NUMBA_THREADS", "1")
    with netCDF4.Dataset(CASE_DIR / "tpcore_input.nc") as dataset:
        values = {
            name: np.ascontiguousarray(variable[:], dtype=np.float64)
            for name, variable in dataset.variables.items()
            if name != "tracer_name"
        }
        dt_s = float(dataset.dt_s)

    host_tpcore = prepare_tpcore_met_plan(
        p1_hpa=values["p1_hpa"],
        p2_hpa=values["p2_hpa"],
        u_m_s=values["u_m_s"],
        v_m_s=values["v_m_s"],
        area_m2=values["area_m2"],
        hyai_hpa=values["hyai"],
        hybi=values["hybi"],
        lat_deg=values["lat"],
        dt_s=dt_s,
    )
    expected = run_tpcore_one_step_with_setup(
        tracer_conc=values["tracer_conc"],
        setup=host_tpcore.setup,
        area_m2=values["area_m2"],
    ).tracer_conc_after
    nlev, nlat, nlon, tracer_count = values["tracer_conc"].shape
    host_vdiff = _identity_vdiff_plan(
        (nlev, nlat, nlon),
        values["area_m2"],
    )

    runtime = _cuda_runtime_or_skip()
    executor = CudaTransportStepExecutor(runtime, dtype=np.float64)
    width = 2
    plans = CudaTransportStepPlans(
        tpcore=executor.tpcore.upload_plan(host_tpcore),
        vdiff=executor.vdiff.upload_plan(host_vdiff),
        convection=executor.convection.upload_plan(
            cmfmc_kg_m2_s=np.zeros((nlev, nlat, nlon)),
            dtrain_kg_m2_s=np.zeros((nlev, nlat, nlon)),
            dqrcu_kg_kg_s=np.zeros((nlev, nlat, nlon)),
            reevapcn_kg_kg_s=np.zeros((nlev, nlat, nlon)),
            delp_dry_hpa=np.ones((nlev, nlat, nlon)),
            delp_hpa=np.ones((nlev, nlat, nlon)),
            area_m2=values["area_m2"],
            dt_s=dt_s,
            reconstruct_conv_precip_flux=False,
        ),
        surface_flux_blocks=runtime.zeros(
            (
                (tracer_count + width - 1) // width,
                nlat,
                nlon,
                width,
            ),
            dtype=np.float64,
        ),
        has_surface_flux=False,
    )
    tracer_blocks = runtime.to_device(_to_blocks(values["tracer_conc"], width))
    runtime.reset_transfer_stats()

    result = executor.apply(
        tracer_blocks,
        plans,
        tracer_count=tracer_count,
        capture_vdiff_handoff=True,
    )

    assert runtime.transfer_stats.host_to_device_count == 0
    assert runtime.transfer_stats.device_to_host_count == 0
    assert runtime.shares_memory(result.tracer_blocks, tracer_blocks)
    assert result.tpcore_tracer_blocks is not None
    actual_tpcore = _from_blocks(
        runtime.to_host(result.tpcore_tracer_blocks),
        tracer_count,
    )
    actual_vdiff = _from_blocks(
        runtime.to_host(result.vdiff_tracer_blocks),
        tracer_count,
    )
    actual_final = _from_blocks(
        runtime.to_host(result.tracer_blocks),
        tracer_count,
    )
    np.testing.assert_allclose(
        actual_tpcore,
        expected,
        rtol=2.0e-13,
        atol=2.0e-18,
    )
    np.testing.assert_array_equal(actual_vdiff, actual_tpcore)
    np.testing.assert_array_equal(actual_final, actual_tpcore)
