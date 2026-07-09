from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest


def _load_benchmark_module():
    path = Path(__file__).parents[1] / "tools" / "benchmark_tpcore_scaling.py"
    spec = importlib.util.spec_from_file_location("benchmark_tpcore_scaling", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark_module()


def test_tpcore_numba_defaults_to_all(monkeypatch):
    from wombat_transport.transport import tpcore

    monkeypatch.delenv("WOMBAT_TPCORE_NUMBA", raising=False)

    assert tpcore._numba_tpcore_mode() == "all"
    assert tpcore._numba_tpcore_x_enabled()
    assert tpcore._numba_tpcore_y_enabled()
    assert tpcore._numba_tpcore_z_enabled()
    assert tpcore._numba_tpcore_prepass_enabled()


def test_tracer_state_bytes_scale_linearly_with_tracer_count():
    grid_shape = (47, 91, 144)

    one = benchmark._tracer_state_bytes(1, grid_shape)
    many = benchmark._tracer_state_bytes(512, grid_shape)

    assert one == 47 * 91 * 144 * 8
    assert many == one * 512


def test_estimated_peak_bytes_include_working_set_multiplier_and_fixed_overhead():
    grid_shape = (2, 3, 5)

    estimate = benchmark._estimate_peak_bytes(
        4,
        grid_shape,
        multiplier=3.0,
        fixed_overhead_bytes=1024,
    )

    assert estimate == 4 * 2 * 3 * 5 * 8 * 3 + 1024


def test_count_is_allowed_skips_when_estimate_exceeds_limit():
    allowed, reason = benchmark._count_is_allowed(estimated_peak_bytes=2048, memory_limit_bytes=1024)

    assert not allowed
    assert "exceeds memory limit" in reason


def test_count_is_allowed_runs_when_auto_limit_is_unavailable():
    allowed, reason = benchmark._count_is_allowed(estimated_peak_bytes=2048, memory_limit_bytes=None)

    assert allowed
    assert "memory limit unavailable" in reason


def test_memory_limit_bytes_parses_gib_values():
    assert benchmark._memory_limit_bytes("2.5") == int(2.5 * 1024**3)

    with pytest.raises(ValueError):
        benchmark._memory_limit_bytes("0")


def test_write_csv_records_skipped_rows():
    row = benchmark.BenchmarkRow(
        tracer_count=512,
        status="skipped",
        repeat=1,
        best_wall_s=None,
        mean_wall_s=None,
        seconds_per_tracer=None,
        tracers_per_second=None,
        gridcell_tracers_per_second=None,
        tracer_state_mib=2406.0,
        estimated_peak_mib=29384.0,
        memory_limit_mib=16000.0,
        peak_rss_mib=128.0,
        checksum=None,
        reason="estimated peak exceeds memory limit",
    )
    handle = io.StringIO()

    benchmark._write_csv([row], handle)

    assert "tracer_count,status,repeat" in handle.getvalue()
    assert "512,skipped,1" in handle.getvalue()
    assert "estimated peak exceeds memory limit" in handle.getvalue()
