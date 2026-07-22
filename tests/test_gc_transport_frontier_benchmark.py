from __future__ import annotations

import importlib.util
import netCDF4
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


frontier = _load_tool("benchmark_gc_transport_frontier", "tools/benchmark_gc_transport_frontier.py")
pjc_generator = _load_tool(
    "generate_pjc_frontier_source",
    "tools/gc_harness/generate_pjc_frontier_source.py",
)
vdiff_generator = _load_tool(
    "generate_vdiff_harness_source_for_frontier_test",
    "tools/gc_harness/generate_vdiff_harness_source.py",
)


def test_generate_gc_specs_covers_balanced_process_openmp_factorizations():
    specs = frontier.generate_specs(
        cpus=(0, 2, 4, 6),
        core_counts=(1, 4),
        tracer_counts=(10,),
        grid_shape=(47, 91, 144),
    )

    assert {
        (spec.total_cores, spec.processes, spec.threads_per_process)
        for spec in specs
    } == {(1, 1, 1), (4, 1, 4), (4, 2, 2), (4, 4, 1)}
    two_process = next(spec for spec in specs if spec.processes == 2)
    assert two_process.rank_tracers == (5, 5)
    assert two_process.rank_cpus == ((0, 2), (4, 6))
    assert all(spec.executor == "gc-openmp" and spec.block_width == 0 for spec in specs)


def test_gc_fixture_contract_uses_existing_chain_handoffs(tmp_path):
    assert frontier._fixture_paths(tmp_path) == (
        tmp_path / "transport_chain_input.nc",
        tmp_path / "vdiff_input.nc",
        tmp_path / "convection_input.nc",
    )


def test_gc_fixture_grid_shape_uses_lev_lat_lon_order(tmp_path):
    path = tmp_path / "fixture.nc"
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("lon", 144)
        dataset.createDimension("lat", 91)
        dataset.createDimension("lev", 47)

    assert frontier._read_fixture_grid_shape(path) == (47, 91, 144)


def test_pjc_frontier_generator_initializes_north_pole_cosine(tmp_path):
    source = tmp_path / "pjc.F90"
    output = tmp_path / "pjc.frontier.F90"
    source.write_text("before\n" + pjc_generator.OLD + "after\n", encoding="utf-8")

    pjc_generator.generate(source, output)

    generated = output.read_text(encoding="utf-8")
    assert "COSE_FV(J2_GL+1) = 0.e+0_fp" in generated
    assert generated.count("SINE_FV(J2_GL+1)") == 1


def test_vdiff_frontier_generation_exposes_driver_without_trace_copy(tmp_path):
    source = tmp_path / "vdiff.F90"
    output = tmp_path / "vdiff.frontier.F90"
    source.write_text("  PUBLIC :: Max_PblHt_For_Vdiff\n", encoding="utf-8")

    vdiff_generator.generate(source, output, with_trace=False)

    generated = output.read_text(encoding="utf-8")
    assert "PUBLIC :: VDIFFDR" in generated
    assert "Vdiff_Trace" not in generated
