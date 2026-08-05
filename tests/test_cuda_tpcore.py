from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import pytest

from wombat_transport.cuda import CudaRuntime, CudaUnavailableError
from wombat_transport.transport.tpcore import run_tpcore_one_step_with_setup
from wombat_transport.transport.tpcore._cuda import (
    CudaTpcoreExecutor,
    _zonal_warps_per_block,
)
from wombat_transport.transport.tpcore._plan import prepare_tpcore_met_plan


TPCORE_CASES = (
    Path("tests/fixtures/tpcore_snapshot_v2"),
    Path("tests/fixtures/tpcore_x_fxppm_low_courant_v2"),
    Path("tests/fixtures/tpcore_x_large_courant_polar_v2"),
)


@pytest.mark.parametrize(
    ("dtype", "tracer_count", "expected"),
    (
        (np.dtype(np.float64), 128, 4),
        (np.dtype(np.float32), 24, 4),
        (np.dtype(np.float32), 32, 8),
        (np.dtype(np.float32), 512, 8),
    ),
)
def test_cuda_tpcore_selects_zonal_launch_geometry(
    dtype,
    tracer_count,
    expected,
):
    assert _zonal_warps_per_block(dtype, tracer_count) == expected


def _cuda_runtime_or_skip() -> CudaRuntime:
    try:
        return CudaRuntime()
    except CudaUnavailableError as exc:
        pytest.skip(str(exc))


def _load_case(path: Path, monkeypatch):
    monkeypatch.setenv("WOMBAT_NUMBA", "1")
    monkeypatch.setenv("WOMBAT_NUMBA_THREADS", "1")
    with netCDF4.Dataset(path / "tpcore_input.nc") as dataset:
        values = {
            name: np.ascontiguousarray(variable[:], dtype=np.float64)
            for name, variable in dataset.variables.items()
            if name not in {"tracer_name"}
        }
        dt_s = float(dataset.dt_s)
    plan = prepare_tpcore_met_plan(
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
        setup=plan.setup,
        area_m2=values["area_m2"],
    )
    with netCDF4.Dataset(path / "tpcore_output.nc") as dataset:
        geos = np.asarray(
            dataset.variables["tracer_conc_after"][:],
            dtype=np.float64,
        )
    return values, plan, expected, geos


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


@pytest.mark.cuda
@pytest.mark.parametrize("case_dir", TPCORE_CASES, ids=lambda path: path.name)
def test_cuda_tpcore_float64_matches_cpu_and_geos(case_dir, monkeypatch):
    runtime = _cuda_runtime_or_skip()
    values, host_plan, expected, geos = _load_case(case_dir, monkeypatch)
    tracer_count = values["tracer_conc"].shape[-1]
    executor = CudaTpcoreExecutor(runtime, dtype=np.float64)
    plan = executor.upload_plan(host_plan)
    tracer = runtime.to_device(_to_blocks(values["tracer_conc"], 2))
    runtime.reset_transfer_stats()

    result = executor.apply_blocks(
        tracer,
        plan,
        tracer_count=tracer_count,
        capture_handoffs=True,
    )

    assert runtime.transfer_stats.host_to_device_count == 0
    assert runtime.transfer_stats.device_to_host_count == 0
    actual = _from_blocks(runtime.to_host(result.tracer_blocks), tracer_count)
    np.testing.assert_allclose(actual, expected.tracer_conc_after, rtol=2.0e-13, atol=2.0e-18)
    np.testing.assert_allclose(actual, geos, rtol=3.0e-12, atol=1.0e-11)
    assert result.q_after_horizontal is not None
    assert result.dq_after_horizontal is not None
    assert result.dq_after_vertical is not None


@pytest.mark.cuda
def test_cuda_tpcore_float32_has_bounded_drift(monkeypatch):
    runtime = _cuda_runtime_or_skip()
    values, host_plan, expected, _ = _load_case(TPCORE_CASES[0], monkeypatch)
    tracer_count = values["tracer_conc"].shape[-1]
    executor = CudaTpcoreExecutor(runtime, dtype=np.float32)
    plan = executor.upload_plan(host_plan)
    tracer = runtime.to_device(
        _to_blocks(values["tracer_conc"], 2),
        dtype=np.float32,
    )

    result = executor.apply_blocks(tracer, plan, tracer_count=tracer_count)
    actual = _from_blocks(
        runtime.to_host(result.tracer_blocks),
        tracer_count,
    ).astype(np.float64)

    np.testing.assert_allclose(
        actual,
        expected.tracer_conc_after,
        rtol=2.0e-5,
        atol=2.0e-9,
    )
    assert np.all(np.isfinite(actual))
    assert np.count_nonzero(actual < 0.0) == 0
