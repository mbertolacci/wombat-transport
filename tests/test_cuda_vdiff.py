from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import pytest

from wombat_transport.cuda import CudaRuntime, CudaUnavailableError
from wombat_transport.transport.pbl import LATVAP_J_PER_KG
from wombat_transport.transport.pbl import run_vdiffdr_one_step
from wombat_transport.transport.pbl._cuda import CudaVdiffExecutor
from wombat_transport.transport.pbl._plan import prepare_vdiff_met_plan

VDIFF_CASES = (
    Path("tests/fixtures/vdiff_snapshot_v2/vdiff_input.nc"),
    Path("tests/fixtures/vdiff_nonzero_surface_flux_v2/vdiff_input.nc"),
    Path("tests/fixtures/vdiff_negative_clipping_v2/vdiff_input.nc"),
)


def _cuda_runtime_or_skip() -> CudaRuntime:
    try:
        return CudaRuntime()
    except CudaUnavailableError as exc:
        pytest.skip(str(exc))


def _load_vdiff_case(
    path: Path,
    monkeypatch,
    *,
    lane_factors: np.ndarray | None = None,
):
    monkeypatch.setenv("WOMBAT_NUMBA", "1")
    monkeypatch.setenv("WOMBAT_NUMBA_THREADS", "1")
    with netCDF4.Dataset(path) as dataset:
        values = {
            name: np.asarray(variable[:], dtype=np.float64)
            for name, variable in dataset.variables.items()
            if name not in {"lat", "lon"}
        }
        dt_s = float(dataset.dt_s)
    if lane_factors is not None:
        factors = np.asarray(lane_factors, dtype=np.float64)
        if factors.ndim != 1:
            raise ValueError("lane_factors must be one-dimensional")
        values["tracer_conc"] = np.ascontiguousarray(
            values["tracer_conc"][..., :1] * factors
        )
        values["surface_flux_kg_m2_s"] = np.ascontiguousarray(
            values["surface_flux_kg_m2_s"][..., :1] * factors
        )

    operator_kwargs = {
        "tracer_conc": values["tracer_conc"],
        "u_m_s": values["u_m_s"],
        "v_m_s": values["v_m_s"],
        "temperature_k": values["temperature_k"],
        "specific_humidity_kg_kg": values["specific_humidity_kg_kg"],
        "pmid_hpa": values["pmid_hpa"],
        "pedge_hpa": values["pedge_hpa"],
        "virtual_temperature_k": values["virtual_temperature_k"],
        "bxheight_m": values["bxheight_m"],
        "dry_air_mass_kg": values["dry_air_mass_kg"],
        "pbl_top_m": values["pbl_top_m"],
        "hflux_w_m2": values["hflux_w_m2"],
        "eflux_w_m2": values["eflux_w_m2"],
        "ustar_m_s": values["ustar_m_s"],
        "area_m2": values["area_m2"],
        "surface_flux_kg_m2_s": values["surface_flux_kg_m2_s"],
        "dt_s": dt_s,
    }
    expected = run_vdiffdr_one_step(**operator_kwargs, diagnostics=False)
    plan = prepare_vdiff_met_plan(
        u_top=values["u_m_s"],
        v_top=values["v_m_s"],
        temperature_top=values["temperature_k"],
        sphu_top=values["specific_humidity_kg_kg"],
        pmid_hpa=values["pmid_hpa"],
        pint_hpa=values["pedge_hpa"],
        virtual_temperature_top=values["virtual_temperature_k"],
        bxheight_top=values["bxheight_m"],
        dry_mass_top=values["dry_air_mass_kg"],
        pblh_m=values["pbl_top_m"],
        hflux_w_m2=values["hflux_w_m2"],
        water_flux_kg_m2_s=values["eflux_w_m2"] / LATVAP_J_PER_KG,
        ustar_m_s=values["ustar_m_s"],
        area_m2=values["area_m2"],
        dt_s=dt_s,
        workers=1,
    )
    return values, plan, expected


@pytest.mark.cuda
@pytest.mark.parametrize("case_path", VDIFF_CASES, ids=lambda path: path.parent.name)
def test_cuda_vdiff_float64_matches_cpu_operator(case_path, monkeypatch):
    runtime = _cuda_runtime_or_skip()
    values, host_plan, expected = _load_vdiff_case(case_path, monkeypatch)
    executor = CudaVdiffExecutor(runtime, dtype=np.float64)
    device_plan = executor.upload_plan(host_plan)
    device_tracer = runtime.to_device(values["tracer_conc"])
    device_flux = runtime.to_device(values["surface_flux_kg_m2_s"])
    has_flux = bool(np.any(values["surface_flux_kg_m2_s"] != 0.0))
    runtime.reset_transfer_stats()

    result = executor.apply(
        device_tracer,
        device_plan,
        device_flux,
        has_flux=has_flux,
    )

    assert runtime.transfer_stats.host_to_device_count == 0
    assert runtime.transfer_stats.device_to_host_count == 0
    actual = runtime.to_host(result.tracer_conc)
    actual_humidity = runtime.to_host(result.specific_humidity_kg_kg)
    negative_count = int(runtime.to_host(result.negative_count_before_clip))

    np.testing.assert_allclose(
        actual,
        expected.tracer_conc,
        rtol=5.0e-15,
        atol=5.0e-20,
    )
    np.testing.assert_array_equal(
        actual_humidity,
        expected.specific_humidity_kg_kg,
    )
    assert negative_count == expected.negative_count_before_clip
    assert runtime.transfer_stats.host_to_device_count == 0
    assert runtime.transfer_stats.device_to_host_count == 3

    expected_mass = np.sum(
        expected.tracer_conc * values["dry_air_mass_kg"][..., None],
        axis=(0, 1, 2),
    )
    actual_mass = np.sum(
        actual * values["dry_air_mass_kg"][..., None],
        axis=(0, 1, 2),
    )
    np.testing.assert_allclose(actual_mass, expected_mass, rtol=5.0e-15)


@pytest.mark.cuda
@pytest.mark.parametrize("case_path", VDIFF_CASES, ids=lambda path: path.parent.name)
def test_cuda_vdiff_float32_has_bounded_drift(case_path, monkeypatch):
    runtime = _cuda_runtime_or_skip()
    values, host_plan, expected = _load_vdiff_case(case_path, monkeypatch)
    executor = CudaVdiffExecutor(runtime, dtype=np.float32)
    device_plan = executor.upload_plan(host_plan)
    device_tracer = runtime.to_device(values["tracer_conc"], dtype=np.float32)
    device_flux = runtime.to_device(
        values["surface_flux_kg_m2_s"],
        dtype=np.float32,
    )

    result = executor.apply(
        device_tracer,
        device_plan,
        device_flux,
        has_flux=bool(np.any(values["surface_flux_kg_m2_s"] != 0.0)),
    )
    actual = runtime.to_host(result.tracer_conc).astype(np.float64)
    negative_count = int(runtime.to_host(result.negative_count_before_clip))

    np.testing.assert_allclose(
        actual,
        expected.tracer_conc,
        rtol=2.0e-6,
        atol=5.0e-11,
    )
    assert negative_count == expected.negative_count_before_clip
    assert np.all(np.isfinite(actual))

    expected_mass = np.sum(
        expected.tracer_conc * values["dry_air_mass_kg"][..., None],
        axis=(0, 1, 2),
    )
    actual_mass = np.sum(
        actual * values["dry_air_mass_kg"][..., None],
        axis=(0, 1, 2),
    )
    np.testing.assert_allclose(actual_mass, expected_mass, rtol=2.0e-6)


@pytest.mark.cuda
def test_cuda_vdiff_handles_multiple_tracer_blocks_and_padded_lanes(monkeypatch):
    runtime = _cuda_runtime_or_skip()
    factors = np.array(
        [0.25, 0.5, 1.0, 2.0, 4.0, 0.0, 1.5, 3.0, 0.75, 0.0],
        dtype=np.float64,
    )
    block_width = 8
    block_count = 2
    values, host_plan, expected = _load_vdiff_case(
        VDIFF_CASES[1],
        monkeypatch,
        lane_factors=factors,
    )
    nlev, nlat, nlon, _ = values["tracer_conc"].shape
    tracer_blocks = np.zeros(
        (block_count, nlev, nlat, nlon, block_width),
        dtype=np.float64,
    )
    flux_blocks = np.zeros(
        (block_count, nlat, nlon, block_width),
        dtype=np.float64,
    )
    for tracer in range(factors.size):
        block, lane = divmod(tracer, block_width)
        tracer_blocks[block, ..., lane] = values["tracer_conc"][..., tracer]
        flux_blocks[block, ..., lane] = values["surface_flux_kg_m2_s"][..., tracer]
    executor = CudaVdiffExecutor(runtime, dtype=np.float64)
    device_plan = executor.upload_plan(host_plan)
    device_tracer = runtime.to_device(tracer_blocks)
    device_flux = runtime.to_device(flux_blocks)
    reusable_output = runtime.empty(device_tracer.shape, dtype=np.float64)
    runtime.reset_transfer_stats()

    first = executor.apply_blocks(
        device_tracer,
        device_plan,
        device_flux,
        has_flux=True,
        tracer_count=factors.size,
        output=reusable_output,
    )
    runtime.synchronize()
    second = executor.apply_blocks(
        device_tracer,
        device_plan,
        device_flux,
        has_flux=True,
        tracer_count=factors.size,
        output=reusable_output,
    )
    runtime.synchronize()

    assert first.tracer_conc is reusable_output
    assert second.tracer_conc is reusable_output
    assert runtime.transfer_stats.host_to_device_count == 0
    assert runtime.transfer_stats.device_to_host_count == 0
    actual_blocks = runtime.to_host(second.tracer_conc)
    actual = np.concatenate(
        (
            actual_blocks[0],
            actual_blocks[1, ..., : factors.size - block_width],
        ),
        axis=-1,
    )
    np.testing.assert_allclose(
        actual,
        expected.tracer_conc,
        rtol=5.0e-15,
        atol=5.0e-20,
    )
    np.testing.assert_array_equal(actual[..., 5], 0.0)
    np.testing.assert_array_equal(actual[..., 9], 0.0)
    np.testing.assert_array_equal(
        actual_blocks[1, ..., factors.size - block_width :],
        0.0,
    )


@pytest.mark.cuda
def test_cuda_vdiff_rejects_overlapping_output(monkeypatch):
    runtime = _cuda_runtime_or_skip()
    values, host_plan, _ = _load_vdiff_case(VDIFF_CASES[0], monkeypatch)
    executor = CudaVdiffExecutor(runtime, dtype=np.float64)
    device_plan = executor.upload_plan(host_plan)
    device_tracer = runtime.to_device(values["tracer_conc"])
    device_flux = runtime.to_device(values["surface_flux_kg_m2_s"])

    with pytest.raises(ValueError, match="must not overlap"):
        executor.apply(
            device_tracer,
            device_plan,
            device_flux,
            has_flux=False,
            output=device_tracer,
        )
