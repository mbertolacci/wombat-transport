from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import pytest

from wombat_transport.cuda import CudaRuntime, CudaUnavailableError
from wombat_transport.transport.convection import run_cloud_convection_one_step
from wombat_transport.transport.convection._cuda import CudaConvectionExecutor


CASE_DIR = Path("tests/fixtures/convection_real_sampled_v2")


def _cuda_runtime_or_skip() -> CudaRuntime:
    try:
        return CudaRuntime()
    except CudaUnavailableError as exc:
        pytest.skip(str(exc))


def _load_case(monkeypatch):
    monkeypatch.setenv("WOMBAT_NUMBA", "1")
    monkeypatch.setenv("WOMBAT_NUMBA_THREADS", "1")
    with netCDF4.Dataset(CASE_DIR / "convection_input.nc") as dataset:
        values = {
            name: np.ascontiguousarray(variable[:], dtype=np.float64)
            for name, variable in dataset.variables.items()
            if name
            not in {
                "lat",
                "lon",
                "tracer_name",
                "source_lat_index",
                "source_lon_index",
            }
        }
        dt_s = float(dataset.dt_s)
        reconstruct = bool(dataset.reconstruct_conv_precip_flux)
    expected = run_cloud_convection_one_step(
        tracer_conc=values["tracer_conc"],
        cmfmc_kg_m2_s=values["cmfmc_kg_m2_s"],
        dtrain_kg_m2_s=values["dtrain_kg_m2_s"],
        dqrcu_kg_kg_s=values["dqrcu_kg_kg_s"],
        reevapcn_kg_kg_s=values["reevapcn_kg_kg_s"],
        delp_dry_hpa=values["delp_dry_hpa"],
        delp_hpa=values["delp_hpa"],
        area_m2=values["area_m2"],
        dt_s=dt_s,
        reconstruct_conv_precip_flux=reconstruct,
        diagnostics=True,
    )
    with netCDF4.Dataset(CASE_DIR / "convection_output.nc") as dataset:
        geos = np.asarray(
            dataset.variables["tracer_conc_after"][:],
            dtype=np.float64,
        )
    return values, dt_s, reconstruct, expected, geos


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


def _upload_plan(executor, values, dt_s, reconstruct):
    return executor.upload_plan(
        cmfmc_kg_m2_s=values["cmfmc_kg_m2_s"],
        dtrain_kg_m2_s=values["dtrain_kg_m2_s"],
        dqrcu_kg_kg_s=values["dqrcu_kg_kg_s"],
        reevapcn_kg_kg_s=values["reevapcn_kg_kg_s"],
        delp_dry_hpa=values["delp_dry_hpa"],
        delp_hpa=values["delp_hpa"],
        area_m2=values["area_m2"],
        dt_s=dt_s,
        reconstruct_conv_precip_flux=reconstruct,
    )


@pytest.mark.cuda
def test_cuda_convection_float64_matches_cpu_and_geos(monkeypatch):
    runtime = _cuda_runtime_or_skip()
    values, dt_s, reconstruct, expected, geos = _load_case(monkeypatch)
    tracer_count = values["tracer_conc"].shape[-1]
    executor = CudaConvectionExecutor(runtime, dtype=np.float64)
    plan = _upload_plan(executor, values, dt_s, reconstruct)
    blocks = runtime.to_device(_to_blocks(values["tracer_conc"], 8))
    runtime.reset_transfer_stats()

    result = executor.apply_blocks(
        blocks,
        plan,
        tracer_count=tracer_count,
        diagnostics=True,
    )

    assert runtime.transfer_stats.host_to_device_count == 0
    assert runtime.transfer_stats.device_to_host_count == 0
    actual = _from_blocks(runtime.to_host(result.tracer_blocks), tracer_count)
    actual_diag = _from_blocks(
        runtime.to_host(result.diag14_mass_flux),
        tracer_count,
    )
    np.testing.assert_allclose(actual, expected.tracer_conc, rtol=2.0e-15, atol=2.0e-20)
    np.testing.assert_allclose(
        actual_diag,
        expected.diag14_mass_flux,
        rtol=3.0e-15,
        atol=5.0e-10,
    )
    np.testing.assert_allclose(actual, geos, rtol=2.0e-15, atol=3.0e-19)


@pytest.mark.cuda
def test_cuda_convection_float32_has_bounded_drift(monkeypatch):
    runtime = _cuda_runtime_or_skip()
    values, dt_s, reconstruct, expected, _ = _load_case(monkeypatch)
    tracer_count = values["tracer_conc"].shape[-1]
    executor = CudaConvectionExecutor(runtime, dtype=np.float32)
    plan = _upload_plan(executor, values, dt_s, reconstruct)
    blocks = runtime.to_device(
        _to_blocks(values["tracer_conc"], 8),
        dtype=np.float32,
    )

    result = executor.apply_blocks(blocks, plan, tracer_count=tracer_count)
    actual = _from_blocks(
        runtime.to_host(result.tracer_blocks),
        tracer_count,
    ).astype(np.float64)

    np.testing.assert_allclose(
        actual,
        expected.tracer_conc,
        rtol=3.0e-6,
        atol=5.0e-11,
    )
    assert np.all(np.isfinite(actual))
    assert np.count_nonzero(actual < 0.0) == expected.negative_count_after
