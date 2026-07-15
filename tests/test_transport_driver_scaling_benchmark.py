from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest


def _load_benchmark_module():
    path = Path(__file__).parents[1] / "tools" / "benchmark_transport_driver_scaling.py"
    spec = importlib.util.spec_from_file_location("benchmark_transport_driver_scaling", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark_module()


def test_transport_driver_tracer_state_bytes_scale_linearly_with_tracer_count():
    grid_shape = (47, 91, 144)

    one = benchmark._tracer_state_bytes(1, grid_shape)
    many = benchmark._tracer_state_bytes(512, grid_shape)

    assert one == 47 * 91 * 144 * 8
    assert many == one * 512


def test_transport_driver_estimated_peak_bytes_include_working_set_multiplier_and_fixed_overhead():
    estimate = benchmark._estimate_peak_bytes(
        4,
        (2, 3, 5),
        multiplier=3.0,
        fixed_overhead_bytes=1024,
    )

    assert estimate == 4 * 2 * 3 * 5 * 8 * 3 + 1024


def test_transport_driver_count_is_allowed_skips_when_estimate_exceeds_limit():
    allowed, reason = benchmark._count_is_allowed(estimated_peak_bytes=2048, memory_limit_bytes=1024)

    assert not allowed
    assert "exceeds memory limit" in reason


def test_transport_driver_memory_limit_bytes_parses_gib_values():
    assert benchmark._memory_limit_bytes("2.5") == int(2.5 * 1024**3)

    with pytest.raises(ValueError):
        benchmark._memory_limit_bytes("0")


def test_transport_driver_write_csv_records_stage_columns_and_skipped_rows():
    row = benchmark.BenchmarkRow(
        tracer_count=512,
        status="skipped",
        repeat=1,
        best_wall_s=None,
        mean_wall_s=None,
        best_setup_s=None,
        best_tpcore_s=None,
        best_vdiff_s=None,
        best_convection_s=None,
        best_overhead_s=None,
        mean_setup_s=None,
        mean_tpcore_s=None,
        mean_vdiff_s=None,
        mean_convection_s=None,
        mean_overhead_s=None,
        seconds_per_tracer=None,
        tracers_per_second=None,
        gridcell_tracers_per_second=None,
        tracer_state_mib=2406.0,
        estimated_peak_mib=58280.0,
        memory_limit_mib=16000.0,
        peak_rss_mib=128.0,
        checksum=None,
        reason="estimated peak exceeds memory limit",
    )
    handle = io.StringIO()

    benchmark._write_csv([row], handle)

    assert "best_tpcore_s" in handle.getvalue()
    assert "best_vdiff_s" in handle.getvalue()
    assert "best_convection_s" in handle.getvalue()
    assert "512,skipped,1" in handle.getvalue()


def test_transport_driver_mean_run_averages_stage_times():
    runs = [
        benchmark.TimedRun(10.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.1),
        benchmark.TimedRun(20.0, 3.0, 4.0, 5.0, 6.0, 2.0, 0.3),
    ]

    mean = benchmark._mean_run(runs)

    assert mean.total_s == 15.0
    assert mean.setup_s == 2.0
    assert mean.tpcore_s == 3.0
    assert mean.vdiff_s == 4.0
    assert mean.convection_s == 5.0
    assert mean.overhead_s == 1.0
    assert mean.checksum == pytest.approx(0.2)


def test_transport_driver_synthetic_builder_uses_canonical_tracer_field():
    inputs = benchmark._build_synthetic_driver_inputs(
        Path("tests/fixtures/io_readers_v1/run.yml"),
        3,
        dt_s=600.0,
    )

    assert inputs.grid.shape == (47, 91, 144)
    assert inputs.tracer_field.data.shape == (1, 47, 91, 144, 3)
    assert inputs.forcing.u_m_s.shape == (1, 47, 91, 144)
    assert inputs.forcing.convective_mass_flux_kg_m2_s.shape == (1, 47, 91, 144)
    assert inputs.forcing.convective_precip_mm_day.shape == (1, 91, 144)
