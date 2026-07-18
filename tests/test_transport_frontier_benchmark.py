from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = REPO_ROOT / "tools/benchmark_transport_frontier.py"
    spec = importlib.util.spec_from_file_location("benchmark_transport_frontier", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


frontier = _load_module()


def test_parse_cpu_list_preserves_explicit_order():
    assert frontier.parse_cpu_list("8,10,12-14,2") == (8, 10, 12, 13, 14, 2)


def test_balanced_split_assigns_every_tracer():
    assert frontier.balanced_split(10, 3) == (4, 3, 3)


def test_generate_specs_covers_balanced_factorizations_and_useful_blocks():
    specs = frontier.generate_specs(
        cpus=(0, 2, 4, 6),
        core_counts=(1, 4),
        tracer_counts=(32,),
        executors=("spatial", "blocks"),
        block_widths=(8, 16),
        grid_shape=(47, 46, 72),
    )

    assert len(specs) == 7
    assert {
        (spec.total_cores, spec.processes, spec.threads_per_process, spec.executor, spec.block_width)
        for spec in specs
    } == {
        (1, 1, 1, "spatial", 0),
        (4, 1, 4, "spatial", 0),
        (4, 1, 4, "blocks", 8),
        (4, 1, 4, "blocks", 16),
        (4, 2, 2, "spatial", 0),
        (4, 2, 2, "blocks", 8),
        (4, 4, 1, "spatial", 0),
    }
    two_process = next(spec for spec in specs if spec.processes == 2 and spec.executor == "spatial")
    assert two_process.rank_tracers == (16, 16)
    assert two_process.rank_cpus == ((0, 2), (4, 6))


def test_taskset_binder_uses_rank_cpu_list():
    assert frontier.binder_command(
        "taskset", (2, 6, 10), memory_policy="bind"
    ) == ["taskset", "--cpu-list", "2,6,10"]


def test_dry_run_matrix_is_valid_csv(capsys):
    specs = frontier.generate_specs(
        cpus=(0, 2),
        core_counts=(2,),
        tracer_counts=(24,),
        executors=("spatial",),
        block_widths=(8,),
        grid_shape=(47, 46, 72),
    )

    frontier._print_specs(specs)

    rows = list(csv.DictReader(capsys.readouterr().out.splitlines()))
    assert len(rows) == 2
    assert rows[0]["rank_cpus"] in {"0,2", "0;2"}


def test_report_selects_median_winner_and_writes_plots(tmp_path):
    cases = tmp_path / "cases"
    for config_id, wall, executor in (
        ("slow", 2.0, "spatial"),
        ("fast", 1.0, "blocks"),
    ):
        case = cases / config_id
        case.mkdir(parents=True)
        config = {
            "config_id": config_id,
            "total_tracers": 16,
            "total_cores": 4,
            "processes": 1,
            "threads_per_process": 4,
            "executor": executor,
            "block_width": 8 if executor == "blocks" else 0,
            "rank_tracers": [16],
            "rank_cpus": [[0, 1, 2, 3]],
            "estimated_peak_bytes": 1024,
            "binder": "taskset",
        }
        result = {
            "status": "completed",
            "reason": "",
            "best_effective_s": wall,
            "median_effective_s": wall,
            "mean_effective_s": wall,
            "workers": [{"threading_layer": "tbb"}],
            "iterations": [
                {
                    "iteration": 0,
                    "effective_wall_s": wall,
                    "rank_spread_percent": 0.0,
                    "ranks": [
                        {
                            "rank": 0,
                            "wall_s": wall,
                            "checksum": 1.0,
                            "peak_rss_mib": 100.0,
                        }
                    ],
                }
            ],
        }
        (case / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (case / "result.json").write_text(json.dumps(result), encoding="utf-8")

    frontier._write_reports(tmp_path)

    with (tmp_path / "summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    winner = next(row for row in rows if row["best_for_tracers_and_cores"] == "True")
    assert winner["config_id"] == "fast"
    assert float(winner["ensemble_steps_per_s"]) == 1.0
    assert float(winner["aggregate_tracer_steps_per_s"]) == 16.0
    assert (tmp_path / "ensemble_steps_per_s.svg").is_file()
    assert (tmp_path / "aggregate_tracer_steps_per_s.svg").is_file()
    assert "1p×4t/blocks-8" in (tmp_path / "winners.md").read_text(encoding="utf-8")
