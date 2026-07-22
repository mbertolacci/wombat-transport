from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, TextIO
from xml.sax.saxutils import escape


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_RUN_CONFIG = Path(
    "validation_runs/cases/realistic_restart_noemis_2x25/wombat/main/run.yml"
)
DEFAULT_BLOCK_WIDTHS = (8, 16)
STATE_MULTIPLIER = 24.0
PROCESS_OVERHEAD_BYTES = 768 * 1024**2
AUTO_MEMORY_FRACTION = 0.55


@dataclass(frozen=True)
class ConfigSpec:
    config_id: str
    total_tracers: int
    total_cores: int
    processes: int
    threads_per_process: int
    executor: str
    block_width: int
    rank_tracers: tuple[int, ...]
    rank_cpus: tuple[tuple[int, ...], ...]
    estimated_peak_bytes: int


@dataclass
class WorkerProcess:
    rank: int
    process: subprocess.Popen[str]
    stderr_handle: TextIO


SUMMARY_FIELDS = (
    "config_id",
    "status",
    "total_tracers",
    "total_cores",
    "tracers_per_core",
    "processes",
    "threads_per_process",
    "min_tracers_per_process",
    "max_tracers_per_process",
    "executor",
    "block_width",
    "binder",
    "repeat",
    "best_effective_s",
    "median_effective_s",
    "mean_effective_s",
    "ensemble_steps_per_s",
    "aggregate_tracer_steps_per_s",
    "tracer_steps_per_s_per_core",
    "median_rank_spread_percent",
    "total_peak_rss_gib",
    "estimated_peak_gib",
    "best_for_tracers_and_cores",
    "best_for_tracer_count",
    "best_for_tracer_and_executor",
    "rank_cpus",
    "threading_layers",
    "reason",
)

ITERATION_FIELDS = (
    "config_id",
    "iteration",
    "rank",
    "rank_tracers",
    "rank_cpus",
    "rank_wall_s",
    "effective_wall_s",
    "rank_spread_percent",
    "checksum",
    "peak_rss_mib",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the process/thread/executor frontier of synthetic transport."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run a frontier sweep and write reports")
    _add_run_arguments(run_parser)
    report_parser = subparsers.add_parser("report", help="rebuild CSV and SVG reports")
    report_parser.add_argument("results_dir", type=Path)
    worker_parser = subparsers.add_parser("_worker", help="internal bound-worker protocol")
    worker_parser.add_argument("--spec", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "run":
        return _run_frontier(args)
    if args.command == "report":
        _write_reports(args.results_dir.resolve())
        return 0
    return _worker_main(args.spec)


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-config", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument("--cpus", required=True, help="ordered Linux CPU list, e.g. 0,2,4,6 or 0-7")
    parser.add_argument("--core-counts", type=_positive_int, nargs="+", required=True)
    parser.add_argument("--tracer-counts", type=_positive_int, nargs="+", required=True)
    parser.add_argument("--block-widths", type=_positive_int, nargs="+", default=list(DEFAULT_BLOCK_WIDTHS))
    parser.add_argument("--executors", choices=("spatial", "blocks"), nargs="+", default=("spatial", "blocks"))
    parser.add_argument("--binder", choices=("auto", "taskset", "numactl", "none"), default="auto")
    parser.add_argument("--allow-smt", action="store_true", help="suppress warnings for selected SMT siblings")
    parser.add_argument(
        "--memory-policy",
        choices=("bind", "local", "interleave"),
        default="bind",
        help="numactl memory policy; ignored by other binders",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--warmup", type=_nonnegative_int, default=2)
    parser.add_argument("--repeat", type=_positive_int, default=7)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--order", choices=("generated", "shuffled"), default="shuffled")
    parser.add_argument("--max-memory-gb", default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def _run_frontier(args: argparse.Namespace) -> int:
    cpus = parse_cpu_list(args.cpus)
    available = _available_cpus()
    unavailable = [cpu for cpu in cpus if cpu not in available]
    if unavailable and args.binder != "none":
        raise SystemExit(f"selected CPUs are outside this process affinity mask: {unavailable}")
    if max(args.core_counts) > len(cpus):
        raise SystemExit("a requested core count exceeds the number of selected CPUs")
    sibling_groups = _selected_smt_groups(cpus)
    if sibling_groups and not args.allow_smt:
        formatted = "; ".join(_format_cpu_tuple(group) for group in sibling_groups)
        print(
            f"Warning: selected CPUs include SMT siblings ({formatted}); "
            "core counts therefore include logical hardware threads. Pass --allow-smt to acknowledge.",
            file=sys.stderr,
        )

    python = args.python.absolute()
    grid_shape = _read_grid_shape(args.run_config.resolve(), python)
    specs = generate_specs(
        cpus=cpus,
        core_counts=tuple(sorted(set(args.core_counts))),
        tracer_counts=tuple(sorted(set(args.tracer_counts))),
        executors=tuple(dict.fromkeys(args.executors)),
        block_widths=tuple(sorted(set(args.block_widths))),
        grid_shape=grid_shape,
    )
    if args.order == "shuffled":
        random.Random(args.seed).shuffle(specs)

    if args.dry_run:
        _print_specs(specs)
        return 0

    root = args.output_dir.resolve()
    if root.exists() and not args.resume:
        raise SystemExit(f"output directory already exists; pass --resume to continue: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        manifest = _system_manifest(args, cpus, grid_shape, specs)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    memory_limit = _memory_limit_bytes(args.max_memory_gb)
    binder = resolve_binder(
        args.binder,
        probe_cpus=cpus[:1],
        memory_policy=args.memory_policy,
    )
    for index, spec in enumerate(specs, start=1):
        case_dir = root / "cases" / spec.config_id
        if (case_dir / ".complete").is_file():
            print(f"[{index}/{len(specs)}] {spec.config_id}: already complete", flush=True)
            continue
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_json(case_dir / "config.json", {**_spec_json(spec), "binder": binder})
        if memory_limit is not None and spec.estimated_peak_bytes > memory_limit:
            _write_json(
                case_dir / "result.json",
                {
                    "status": "skipped",
                    "reason": (
                        f"estimated peak {spec.estimated_peak_bytes / 1024**3:.2f} GiB exceeds "
                        f"limit {memory_limit / 1024**3:.2f} GiB"
                    ),
                    "iterations": [],
                    "workers": [],
                },
            )
            (case_dir / ".complete").touch()
            print(f"[{index}/{len(specs)}] {spec.config_id}: skipped for memory", flush=True)
            continue
        print(f"[{index}/{len(specs)}] {spec.config_id}", flush=True)
        try:
            result = _run_config(
                spec,
                case_dir=case_dir,
                run_config=args.run_config.resolve(),
                python=python,
                binder=binder,
                memory_policy=args.memory_policy,
                warmup=args.warmup,
                repeat=args.repeat,
                cache_dir=root / "numba-cache",
            )
        except Exception as exc:
            _write_json(
                case_dir / "result.json",
                {"status": "failed", "reason": str(exc), "iterations": [], "workers": []},
            )
            _write_reports(root)
            raise
        _write_json(case_dir / "result.json", result)
        (case_dir / ".complete").touch()
        print(
            f"  median {result['median_effective_s']:.6f} s, "
            f"{spec.total_tracers / result['median_effective_s']:.1f} tracer-steps/s",
            flush=True,
        )

    _write_reports(root)
    return 0


def generate_specs(
    *,
    cpus: tuple[int, ...],
    core_counts: tuple[int, ...],
    tracer_counts: tuple[int, ...],
    executors: tuple[str, ...],
    block_widths: tuple[int, ...],
    grid_shape: tuple[int, int, int],
) -> list[ConfigSpec]:
    specs: list[ConfigSpec] = []
    for total_tracers in tracer_counts:
        for total_cores in core_counts:
            selected = cpus[:total_cores]
            for processes in _divisors(total_cores):
                if processes > total_tracers:
                    continue
                threads = total_cores // processes
                rank_tracers = balanced_split(total_tracers, processes)
                rank_cpus = tuple(
                    tuple(selected[rank * threads : (rank + 1) * threads])
                    for rank in range(processes)
                )
                candidates: list[tuple[str, int]] = []
                if threads == 1 or "spatial" in executors:
                    candidates.append(("spatial", 0))
                if threads > 1 and "blocks" in executors:
                    for width in block_widths:
                        if any(math.ceil(count / width) >= 2 for count in rank_tracers):
                            candidates.append(("blocks", width))
                for executor, width in candidates:
                    config_id = (
                        f"n{total_tracers:05d}-c{total_cores:03d}-p{processes:03d}-t{threads:03d}-"
                        f"{executor}-w{width:03d}"
                    )
                    estimated = sum(
                        int(STATE_MULTIPLIER * _state_bytes(count, grid_shape) + PROCESS_OVERHEAD_BYTES)
                        for count in rank_tracers
                    )
                    specs.append(
                        ConfigSpec(
                            config_id=config_id,
                            total_tracers=total_tracers,
                            total_cores=total_cores,
                            processes=processes,
                            threads_per_process=threads,
                            executor=executor,
                            block_width=width,
                            rank_tracers=rank_tracers,
                            rank_cpus=rank_cpus,
                            estimated_peak_bytes=estimated,
                        )
                    )
    return specs


def balanced_split(total: int, parts: int) -> tuple[int, ...]:
    quotient, remainder = divmod(total, parts)
    return tuple(quotient + (rank < remainder) for rank in range(parts))


def parse_cpu_list(value: str) -> tuple[int, ...]:
    cpus: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            raise argparse.ArgumentTypeError("CPU list contains an empty item")
        if "-" in token:
            fields = token.split("-", 1)
            try:
                start, stop = (int(field) for field in fields)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"invalid CPU range {token!r}") from exc
            if start < 0 or stop < start:
                raise argparse.ArgumentTypeError(f"invalid CPU range {token!r}")
            cpus.extend(range(start, stop + 1))
        else:
            try:
                cpu = int(token)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"invalid CPU {token!r}") from exc
            if cpu < 0:
                raise argparse.ArgumentTypeError("CPU identifiers must be non-negative")
            cpus.append(cpu)
    if not cpus or len(cpus) != len(set(cpus)):
        raise argparse.ArgumentTypeError("CPU list must be non-empty and contain no duplicates")
    return tuple(cpus)


def resolve_binder(
    requested: str,
    *,
    probe_cpus: tuple[int, ...] = (),
    memory_policy: str = "bind",
) -> str:
    if requested == "auto":
        if shutil.which("numactl") is not None:
            if not probe_cpus:
                return "numactl"
            probe = subprocess.run(
                [*binder_command("numactl", probe_cpus, memory_policy=memory_policy), "true"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if probe.returncode == 0:
                return "numactl"
            print("Warning: numactl binding probe failed; falling back to taskset.", file=sys.stderr)
        if shutil.which("taskset") is not None:
            return "taskset"
        raise SystemExit("neither numactl nor taskset is available; use --binder none explicitly")
    if requested != "none" and shutil.which(requested) is None:
        raise SystemExit(f"requested binder is unavailable: {requested}")
    return requested


def binder_command(
    binder: str,
    cpus: tuple[int, ...],
    *,
    memory_policy: str,
) -> list[str]:
    cpu_text = ",".join(str(cpu) for cpu in cpus)
    if binder == "none":
        return []
    if binder == "taskset":
        return ["taskset", "--cpu-list", cpu_text]
    discovered_nodes = {_cpu_numa_node(cpu) for cpu in cpus}
    if None in discovered_nodes:
        raise RuntimeError("could not determine NUMA node for every selected CPU")
    nodes = sorted(int(node) for node in discovered_nodes if node is not None)
    node_text = ",".join(str(node) for node in nodes)
    command = ["numactl", f"--physcpubind={cpu_text}"]
    if memory_policy == "local":
        command.append("--localalloc")
    elif memory_policy == "interleave":
        command.append(f"--interleave={node_text}")
    else:
        command.append(f"--membind={node_text}")
    return command


def _run_config(
    spec: ConfigSpec,
    *,
    case_dir: Path,
    run_config: Path,
    python: Path,
    binder: str,
    memory_policy: str,
    warmup: int,
    repeat: int,
    cache_dir: Path,
) -> dict[str, Any]:
    workers: list[WorkerProcess] = []
    ready_rows: list[dict[str, Any]] = []
    try:
        for rank in range(spec.processes):
            worker_spec = {
                "rank": rank,
                "tracer_count": spec.rank_tracers[rank],
                "tracer_offset": sum(spec.rank_tracers[:rank]),
                "block_width": (
                    spec.rank_tracers[rank] if spec.executor == "spatial" else spec.block_width
                ),
                "executor": spec.executor,
                "threads": spec.threads_per_process,
                "warmup": warmup,
                "run_config": str(run_config),
            }
            spec_path = case_dir / f"worker_{rank:03d}.json"
            _write_json(spec_path, worker_spec)
            stderr_handle = (case_dir / f"rank_{rank:03d}.stderr.log").open("w", encoding="utf-8")
            command = [
                *binder_command(binder, spec.rank_cpus[rank], memory_policy=memory_policy),
                str(python),
                str(SCRIPT_PATH),
                "_worker",
                "--spec",
                str(spec_path),
            ]
            environment = os.environ.copy()
            environment.update(
                PYTHONPATH=str(SCRIPT_PATH.parents[1] / "src"),
                WOMBAT_NUMBA="1",
                WOMBAT_NUMBA_THREADS=str(spec.threads_per_process),
                NUMBA_NUM_THREADS=str(spec.threads_per_process),
                NUMBA_CACHE_DIR=str(cache_dir),
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
            )
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_handle,
                text=True,
                bufsize=1,
                env=environment,
            )
            workers.append(WorkerProcess(rank, process, stderr_handle))

        for worker in workers:
            row = _read_worker_row(worker)
            if row.get("event") != "ready":
                raise RuntimeError(f"rank {worker.rank} did not become ready: {row}")
            ready_rows.append(row)

        iterations: list[dict[str, Any]] = []
        for iteration in range(repeat):
            target_ns = time.monotonic_ns() + 100_000_000
            command = json.dumps({"command": "run", "target_ns": target_ns}) + "\n"
            for worker in workers:
                assert worker.process.stdin is not None
                worker.process.stdin.write(command)
                worker.process.stdin.flush()
            rank_rows = []
            for worker in workers:
                row = _read_worker_row(worker)
                if row.get("event") != "done":
                    raise RuntimeError(f"rank {worker.rank} failed iteration {iteration}: {row}")
                rank_rows.append(row)
            effective = (max(int(row["completed_ns"]) for row in rank_rows) - target_ns) / 1.0e9
            walls = [float(row["wall_s"]) for row in rank_rows]
            spread = 100.0 * (max(walls) - min(walls)) / max(walls)
            iterations.append(
                {
                    "iteration": iteration,
                    "effective_wall_s": effective,
                    "rank_spread_percent": spread,
                    "ranks": rank_rows,
                }
            )

        for worker in workers:
            assert worker.process.stdin is not None
            worker.process.stdin.write(json.dumps({"command": "stop"}) + "\n")
            worker.process.stdin.flush()
        for worker in workers:
            worker.process.wait(timeout=30)
            if worker.process.returncode:
                raise RuntimeError(f"rank {worker.rank} exited with {worker.process.returncode}")

        effective_values = [float(row["effective_wall_s"]) for row in iterations]
        return {
            "status": "completed",
            "reason": "",
            "iterations": iterations,
            "workers": ready_rows,
            "best_effective_s": min(effective_values),
            "median_effective_s": _median(effective_values),
            "mean_effective_s": sum(effective_values) / len(effective_values),
        }
    finally:
        for worker in workers:
            if worker.process.poll() is None:
                worker.process.terminate()
                try:
                    worker.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.process.kill()
            worker.stderr_handle.close()


def _read_worker_row(worker: WorkerProcess) -> dict[str, Any]:
    assert worker.process.stdout is not None
    line = worker.process.stdout.readline()
    if not line:
        code = worker.process.poll()
        raise RuntimeError(f"rank {worker.rank} closed its output unexpectedly (exit {code})")
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"rank {worker.rank} returned invalid JSON: {line.rstrip()}") from exc


def _worker_main(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    os.environ["WOMBAT_NUMBA_THREADS"] = str(spec["threads"])

    import numpy as np
    from numba import threading_layer

    tools_dir = str(SCRIPT_PATH.parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from benchmark_transport_driver_scaling import _build_synthetic_driver_inputs
    from wombat_transport.transport._executor import TransportExecutor
    from wombat_transport.transport.driver import (
        build_tpcore_static_terms,
        run_transport_step_with_executor,
    )

    inputs = _build_synthetic_driver_inputs(
        Path(spec["run_config"]), int(spec["tracer_count"]), dt_s=600.0
    )
    if int(spec["tracer_offset"]):
        inputs.tracer_field.block_data[...] += int(spec["tracer_offset"]) * 1.0e-7
    state = inputs.tracer_field.reblock(int(spec["block_width"]))
    executor = TransportExecutor.create(state)
    static_terms = build_tpcore_static_terms(
        area_m2=inputs.grid.area_m2,
        hyai_hpa=inputs.grid.hyai_hpa,
        hybi=inputs.grid.hybi,
        lat_deg=inputs.grid.lat_deg,
    )
    dry_air_mass = None

    def logical_checksum() -> float:
        total = sum(float(np.sum(state.tracer(index))) for index in range(state.tracer_count))
        cells = math.prod(state.shape[:-1]) * state.tracer_count
        return total / cells

    def run_step() -> tuple[float, int, float]:
        nonlocal state, dry_air_mass
        start = time.perf_counter()
        result = run_transport_step_with_executor(
            state,
            inputs.forcing,
            inputs.grid,
            executor,
            dt_s=inputs.dt_s,
            dry_air_mass_kg=dry_air_mass,
            tpcore_static_terms=static_terms,
            validate_tpcore_branches=False,
            execution=spec["executor"],
        )
        wall = time.perf_counter() - start
        state = result.state
        dry_air_mass = result.dry_air_mass_kg
        completed_ns = time.monotonic_ns()
        return wall, completed_ns, logical_checksum()

    for _ in range(int(spec["warmup"])):
        run_step()
    print(
        json.dumps(
            {
                "event": "ready",
                "rank": int(spec["rank"]),
                "threading_layer": threading_layer(),
                "peak_rss_mib": _peak_rss_mib(),
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        command = json.loads(line)
        if command["command"] == "stop":
            return 0
        target_ns = int(command["target_ns"])
        delay = (target_ns - time.monotonic_ns()) / 1.0e9
        if delay > 0.0:
            time.sleep(delay)
        wall, completed_ns, checksum = run_step()
        print(
            json.dumps(
                {
                    "event": "done",
                    "rank": int(spec["rank"]),
                    "wall_s": wall,
                    "completed_ns": completed_ns,
                    "checksum": checksum,
                    "peak_rss_mib": _peak_rss_mib(),
                }
            ),
            flush=True,
        )
    return 0


def _write_reports(root: Path) -> None:
    rows, iterations = _read_results(root)
    completed = [row for row in rows if row["status"] == "completed"]
    core_winners: dict[tuple[int, int], dict[str, Any]] = {}
    winners: dict[int, dict[str, Any]] = {}
    strategy_winners: dict[tuple[int, str], dict[str, Any]] = {}
    for row in completed:
        tracers = int(row["total_tracers"])
        core_key = (tracers, int(row["total_cores"]))
        current = core_winners.get(core_key)
        if current is None or float(row["median_effective_s"]) < float(current["median_effective_s"]):
            core_winners[core_key] = row
        current = winners.get(tracers)
        if current is None or float(row["median_effective_s"]) < float(current["median_effective_s"]):
            winners[tracers] = row
        strategy_key = (tracers, str(row["executor"]))
        current = strategy_winners.get(strategy_key)
        if current is None or float(row["median_effective_s"]) < float(current["median_effective_s"]):
            strategy_winners[strategy_key] = row
    for row in rows:
        tracers = int(row["total_tracers"])
        core_key = (tracers, int(row["total_cores"]))
        row["best_for_tracers_and_cores"] = row is core_winners.get(core_key)
        row["best_for_tracer_count"] = row is winners.get(tracers)
        row["best_for_tracer_and_executor"] = row is strategy_winners.get(
            (tracers, str(row["executor"]))
        )

    _write_csv(root / "summary.csv", SUMMARY_FIELDS, rows)
    _write_csv(root / "iterations.csv", ITERATION_FIELDS, iterations)
    _write_winners(root / "winners.md", winners.values())
    strategy_rows = sorted(
        strategy_winners.values(),
        key=lambda row: (int(row["total_tracers"]), str(row["executor"])),
    )
    if strategy_rows:
        _write_frontier_svg(root / "transport_frontier.svg", strategy_rows)
    (root / "ensemble_steps_per_s.svg").unlink(missing_ok=True)
    (root / "seconds_per_step.svg").unlink(missing_ok=True)
    (root / "aggregate_tracer_steps_per_s.svg").unlink(missing_ok=True)
    print(f"Wrote {len(rows)} configurations to {root / 'summary.csv'}")


def _read_results(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    for config_path in sorted((root / "cases").glob("*/config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        result_path = config_path.with_name("result.json")
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        status = result.get("status", "invalid")
        total = int(config["total_tracers"])
        cores = int(config["total_cores"])
        effective = float(result.get("median_effective_s", 0.0))
        iterations = result.get("iterations", [])
        spreads = [float(item["rank_spread_percent"]) for item in iterations]
        peak_rss = 0.0
        for item in iterations:
            peak_rss = max(peak_rss, sum(float(rank["peak_rss_mib"]) for rank in item["ranks"]))
            for rank in item["ranks"]:
                iteration_rows.append(
                    {
                        "config_id": config["config_id"],
                        "iteration": item["iteration"],
                        "rank": rank["rank"],
                        "rank_tracers": config["rank_tracers"][int(rank["rank"])],
                        "rank_cpus": _format_cpu_tuple(config["rank_cpus"][int(rank["rank"])]),
                        "rank_wall_s": rank["wall_s"],
                        "effective_wall_s": item["effective_wall_s"],
                        "rank_spread_percent": item["rank_spread_percent"],
                        "checksum": rank["checksum"],
                        "peak_rss_mib": rank["peak_rss_mib"],
                    }
                )
        workers = result.get("workers", [])
        row = {
            "config_id": config["config_id"],
            "status": status,
            "total_tracers": total,
            "total_cores": cores,
            "tracers_per_core": total / cores,
            "processes": config["processes"],
            "threads_per_process": config["threads_per_process"],
            "min_tracers_per_process": min(config["rank_tracers"]),
            "max_tracers_per_process": max(config["rank_tracers"]),
            "executor": config["executor"],
            "block_width": config["block_width"],
            "binder": config.get("binder", ""),
            "repeat": len(iterations),
            "best_effective_s": result.get("best_effective_s", ""),
            "median_effective_s": result.get("median_effective_s", ""),
            "mean_effective_s": result.get("mean_effective_s", ""),
            "ensemble_steps_per_s": 1.0 / effective if effective else "",
            "aggregate_tracer_steps_per_s": total / effective if effective else "",
            "tracer_steps_per_s_per_core": total / effective / cores if effective else "",
            "median_rank_spread_percent": _median(spreads) if spreads else "",
            "total_peak_rss_gib": peak_rss / 1024.0 if peak_rss else "",
            "estimated_peak_gib": int(config["estimated_peak_bytes"]) / 1024.0**3,
            "best_for_tracers_and_cores": False,
            "best_for_tracer_count": False,
            "best_for_tracer_and_executor": False,
            "rank_cpus": ";".join(_format_cpu_tuple(value) for value in config["rank_cpus"]),
            "threading_layers": ";".join(str(worker.get("threading_layer", "")) for worker in workers),
            "reason": result.get("reason", ""),
        }
        rows.append(row)
    return rows, iteration_rows


def _write_frontier_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 1200, 1050
    left, right = 120, 120
    top_plot_y, bottom_plot_y = 80, 570
    plot_height = 390
    plot_width = width - left - right
    tracer_values = sorted({int(row["total_tracers"]) for row in rows})
    x_indices = {tracers: index for index, tracers in enumerate(tracer_values)}
    series = {
        executor: sorted(
            (row for row in rows if row["executor"] == executor),
            key=lambda row: int(row["total_tracers"]),
        )
        for executor in ("spatial", "blocks")
    }
    throughput_max = max(float(row["aggregate_tracer_steps_per_s"]) for row in rows) * 1.18
    seconds_max = max(float(row["median_effective_s"]) for row in rows) * 1.18

    def x_pos(index: int) -> float:
        if len(tracer_values) == 1:
            return left + plot_width / 2.0
        return left + index * plot_width / (len(tracer_values) - 1)

    def y_pos(value: float, plot_y: float, maximum: float) -> float:
        return plot_y + plot_height - (value / maximum) * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="22">Optimal transport frontier by execution strategy</text>',
    ]

    panels = (
        (top_plot_y, throughput_max, "aggregate_tracer_steps_per_s", "Aggregate transport throughput", "tracer-steps / s"),
        (bottom_plot_y, seconds_max, "median_effective_s", "Fastest transport step time", "seconds / step"),
    )
    for plot_y, maximum, _value_field, title, y_label in panels:
        lines.append(f'<text x="{width / 2}" y="{plot_y - 22}" text-anchor="middle" font-family="sans-serif" font-size="19">{escape(title)}</text>')
        lines.append(f'<line x1="{left}" y1="{plot_y}" x2="{left}" y2="{plot_y + plot_height}" stroke="#111827"/>')
        lines.append(f'<line x1="{left}" y1="{plot_y + plot_height}" x2="{left + plot_width}" y2="{plot_y + plot_height}" stroke="#111827"/>')
        for tick in range(6):
            value = maximum * tick / 5.0
            y = y_pos(value, plot_y, maximum)
            lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb"/>')
            lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.1f}</text>')
        lines.append(f'<text x="20" y="{plot_y + plot_height / 2}" text-anchor="middle" transform="rotate(-90 20 {plot_y + plot_height / 2})" font-family="sans-serif" font-size="14">{escape(y_label)}</text>')

    for index, tracers in enumerate(tracer_values):
        x = x_pos(index)
        for plot_y in (top_plot_y, bottom_plot_y):
            lines.append(f'<line x1="{x:.2f}" y1="{plot_y}" x2="{x:.2f}" y2="{plot_y + plot_height}" stroke="#f3f4f6"/>')
        lines.append(f'<text x="{x:.2f}" y="{bottom_plot_y + plot_height + 24}" text-anchor="middle" font-family="sans-serif" font-size="12">{tracers}</text>')
    lines.append(f'<text x="{left + plot_width / 2}" y="{height - 25}" text-anchor="middle" font-family="sans-serif" font-size="14">tracers in ensemble</text>')

    styles = {
        "spatial": ("#2563eb", -13),
        "blocks": ("#dc2626", 21),
    }
    for plot_y, maximum, value_field, _, _ in panels:
        for executor in ("spatial", "blocks"):
            group = series[executor]
            if not group:
                continue
            color, label_offset = styles[executor]
            points = " ".join(
                f'{x_pos(x_indices[int(row["total_tracers"])]):.2f},'
                f'{y_pos(float(row[value_field]), plot_y, maximum):.2f}'
                for row in group
            )
            lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
            for row in group:
                x = x_pos(x_indices[int(row["total_tracers"])])
                y = y_pos(float(row[value_field]), plot_y, maximum)
                label = _winner_label(row)
                lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}"><title>{escape(label)}</title></circle>')
                if plot_y == top_plot_y:
                    label_y = max(plot_y + 10, y + label_offset)
                    annotation = _plot_label(row)
                    lines.append(f'<text x="{x:.2f}" y="{label_y:.2f}" text-anchor="middle" transform="rotate(-32 {x:.2f} {label_y:.2f})" font-family="sans-serif" font-size="10" fill="{color}">{escape(annotation)}</text>')
    legend_y = top_plot_y + 20
    for index, executor in enumerate(("spatial", "blocks")):
        color, _ = styles[executor]
        legend_x = left + 18 + index * 110
        lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x + 30}" y="{legend_y + 4}" font-family="sans-serif" font-size="12">{executor}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_winners(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: int(row["total_tracers"]))
    lines = [
        "| Tracers | Cores | Winner | Seconds/step | Tracer-step/s | RSS GiB |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            f"| {row['total_tracers']} | {row['total_cores']} | {_winner_label(row)} | "
            f"{float(row['median_effective_s']):.6f} | "
            f"{float(row['aggregate_tracer_steps_per_s']):.1f} | "
            f"{float(row['total_peak_rss_gib'] or 0.0):.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _winner_label(row: dict[str, Any]) -> str:
    base = f"{row['processes']}p×{row['threads_per_process']}t"
    if row["executor"] == "blocks":
        return f"{base}/blocks-{row['block_width']}"
    return f"{base}/{row['executor']}"


def _plot_label(row: dict[str, Any]) -> str:
    base = f"{row['processes']}p×{row['threads_per_process']}t"
    if row["executor"] == "spatial":
        return base
    if row["executor"] == "blocks":
        return f"{base} b{row['block_width']}"
    return f"{base} {row['executor']}"


def _system_manifest(
    args: argparse.Namespace,
    cpus: tuple[int, ...],
    grid_shape: tuple[int, int, int],
    specs: list[ConfigSpec],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": sys.argv,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "cpu_model": _cpu_model(),
        "selected_cpus": cpus,
        "cpu_topology": {str(cpu): _cpu_topology(cpu) for cpu in cpus},
        "cpu_numa_nodes": {str(cpu): _cpu_numa_node(cpu) for cpu in cpus},
        "selected_smt_groups": _selected_smt_groups(cpus),
        "grid_shape": grid_shape,
        "run_config": str(args.run_config.resolve()),
        "git": _git_state(),
        "packages": _python_packages(args.python.absolute()),
        "order": args.order,
        "seed": args.seed,
        "configuration_count": len(specs),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }


def _read_grid_shape(run_config: Path, python: Path) -> tuple[int, int, int]:
    code = (
        "from wombat_transport.run_config import load_run_config; "
        "from wombat_transport.grid import load_transport_grid; "
        "import sys; c=load_run_config(sys.argv[1]); "
        "print(','.join(map(str, load_transport_grid(c.grid_template).shape)))"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SCRIPT_PATH.parents[1] / "src")
    completed = subprocess.run(
        [str(python), "-c", code, str(run_config)],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    if completed.returncode:
        detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "unknown error"
        raise RuntimeError(f"could not load the grid from {run_config}: {detail}")
    values = tuple(int(value) for value in completed.stdout.strip().split(","))
    if len(values) != 3:
        raise RuntimeError(f"unexpected grid shape: {completed.stdout!r}")
    return values


def _spec_json(spec: ConfigSpec) -> dict[str, Any]:
    payload = asdict(spec)
    payload["rank_tracers"] = list(spec.rank_tracers)
    payload["rank_cpus"] = [list(value) for value in spec.rank_cpus]
    return payload


def _print_specs(specs: list[ConfigSpec]) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(
        (
            "config_id",
            "total_tracers",
            "total_cores",
            "processes",
            "threads",
            "executor",
            "width",
            "rank_tracers",
            "rank_cpus",
            "estimated_gib",
        )
    )
    for spec in specs:
        writer.writerow(
            (
                spec.config_id,
                spec.total_tracers,
                spec.total_cores,
                spec.processes,
                spec.threads_per_process,
                spec.executor,
                spec.block_width,
                "/".join(map(str, spec.rank_tracers)),
                ";".join(_format_cpu_tuple(value) for value in spec.rank_cpus),
                f"{spec.estimated_peak_bytes / 1024**3:.3f}",
            )
        )


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format_cpu_tuple(cpus: Iterable[int]) -> str:
    return ",".join(str(cpu) for cpu in cpus)


def _divisors(value: int) -> tuple[int, ...]:
    return tuple(candidate for candidate in range(1, value + 1) if value % candidate == 0)


def _state_bytes(tracers: int, grid_shape: tuple[int, int, int]) -> int:
    return int(tracers) * math.prod(grid_shape) * 8


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _memory_limit_bytes(value: str) -> int | None:
    if value == "none":
        return None
    if value == "auto":
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size * AUTO_MEMORY_FRACTION)
    parsed = float(value)
    if parsed <= 0.0:
        raise SystemExit("--max-memory-gb must be positive, auto, or none")
    return int(parsed * 1024**3)


def _available_cpus() -> set[int]:
    if hasattr(os, "sched_getaffinity"):
        return set(os.sched_getaffinity(0))
    return set(range(os.cpu_count() or 1))


def _cpu_numa_node(cpu: int) -> int | None:
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}")
    nodes = sorted(path.glob("node[0-9]*"))
    if not nodes:
        return 0 if path.exists() else None
    return int(nodes[0].name[4:])


def _cpu_topology(cpu: int) -> dict[str, Any]:
    root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")

    def read_int(name: str) -> int | None:
        try:
            return int((root / name).read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    try:
        siblings = parse_cpu_list((root / "thread_siblings_list").read_text(encoding="utf-8").strip())
    except (OSError, argparse.ArgumentTypeError):
        siblings = ()
    return {
        "package": read_int("physical_package_id"),
        "core": read_int("core_id"),
        "thread_siblings": siblings,
    }


def _selected_smt_groups(cpus: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    selected = set(cpus)
    groups: set[tuple[int, ...]] = set()
    for cpu in cpus:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        try:
            siblings = tuple(item for item in parse_cpu_list(path.read_text(encoding="utf-8").strip()) if item in selected)
        except (OSError, argparse.ArgumentTypeError):
            continue
        if len(siblings) > 1:
            groups.add(tuple(sorted(siblings)))
    return tuple(sorted(groups))


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def _git_state() -> dict[str, Any]:
    root = SCRIPT_PATH.parents[1]
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else "",
        "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def _python_packages(python: Path) -> dict[str, str]:
    code = "import numba, numpy; print(numpy.__version__); print(numba.__version__)"
    completed = subprocess.run(
        [str(python), "-c", code],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        return {"numpy": "", "numba": ""}
    lines = completed.stdout.splitlines()
    return {
        "numpy": lines[0] if lines else "",
        "numba": lines[1] if len(lines) > 1 else "",
    }


def _peak_rss_mib() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss / 1024.0**2 if sys.platform == "darwin" else rss / 1024.0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
