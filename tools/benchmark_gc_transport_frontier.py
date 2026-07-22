#!/usr/bin/env python3
"""Benchmark the GEOS-Chem transport process/OpenMP frontier.

The coordinator intentionally mirrors ``benchmark_transport_frontier.py``:
each configuration uses the same nested CPU scopes, balanced process shards,
synchronized iterations, metrics, and report format.  The bound worker is a
GEOS-Chem-backed executable that keeps the complete TPCORE -> VDIFF ->
convection chain in memory; fixture I/O and operator initialization happen
before warmup and measurement.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time
from typing import Any, TextIO


SCRIPT_PATH = Path(__file__).resolve()
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

import benchmark_transport_frontier as common  # noqa: E402


DEFAULT_RUN_CONFIG = Path(
    "validation_runs/cases/realistic_restart_noemis_2x25/wombat/main/run.yml"
)
DEFAULT_EXECUTABLE = Path("tools/gc_harness/build/gc_transport_frontier_harness")
DEFAULT_FIXTURE_DIR = Path("oracle_data/base_initial_transport_chain_v3")
STATE_MULTIPLIER = 24.0
PROCESS_OVERHEAD_BYTES = 768 * 1024**2


class WorkerProcess:
    def __init__(
        self,
        rank: int,
        process: subprocess.Popen[str],
        stderr_handle: TextIO,
    ) -> None:
        self.rank = rank
        self.process = process
        self.stderr_handle = stderr_handle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the GEOS-Chem PJC/TPCORE/VDIFF/convection process/OpenMP frontier."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run a frontier sweep and write reports")
    _add_run_arguments(run_parser)
    report_parser = subparsers.add_parser("report", help="rebuild CSV and SVG reports")
    report_parser.add_argument("results_dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "report":
        common._write_reports(args.results_dir.resolve())
        return 0
    return _run_frontier(args)


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-config", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--cpus", required=True, help="ordered Linux CPU list, e.g. 0,2,4,6 or 0-7")
    parser.add_argument("--core-counts", type=common._positive_int, nargs="+", required=True)
    parser.add_argument("--tracer-counts", type=common._positive_int, nargs="+", required=True)
    parser.add_argument("--binder", choices=("auto", "taskset", "numactl", "none"), default="auto")
    parser.add_argument("--allow-smt", action="store_true")
    parser.add_argument(
        "--memory-policy",
        choices=("bind", "local", "interleave"),
        default="bind",
        help="numactl memory policy; ignored by other binders",
    )
    parser.add_argument("--warmup", type=common._nonnegative_int, default=2)
    parser.add_argument("--repeat", type=common._positive_int, default=7)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--order", choices=("generated", "shuffled"), default="shuffled")
    parser.add_argument("--max-memory-gb", default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def generate_specs(
    *,
    cpus: tuple[int, ...],
    core_counts: tuple[int, ...],
    tracer_counts: tuple[int, ...],
    grid_shape: tuple[int, int, int],
) -> list[common.ConfigSpec]:
    specs: list[common.ConfigSpec] = []
    for total_tracers in tracer_counts:
        for total_cores in core_counts:
            selected = cpus[:total_cores]
            for processes in common._divisors(total_cores):
                if processes > total_tracers:
                    continue
                threads = total_cores // processes
                rank_tracers = common.balanced_split(total_tracers, processes)
                rank_cpus = tuple(
                    tuple(selected[rank * threads : (rank + 1) * threads])
                    for rank in range(processes)
                )
                estimated = sum(
                    int(
                        STATE_MULTIPLIER * common._state_bytes(count, grid_shape)
                        + PROCESS_OVERHEAD_BYTES
                    )
                    for count in rank_tracers
                )
                config_id = (
                    f"n{total_tracers:05d}-c{total_cores:03d}-"
                    f"p{processes:03d}-t{threads:03d}-gc-openmp"
                )
                specs.append(
                    common.ConfigSpec(
                        config_id=config_id,
                        total_tracers=total_tracers,
                        total_cores=total_cores,
                        processes=processes,
                        threads_per_process=threads,
                        executor="gc-openmp",
                        block_width=0,
                        rank_tracers=rank_tracers,
                        rank_cpus=rank_cpus,
                        estimated_peak_bytes=estimated,
                    )
                )
    return specs


def _run_frontier(args: argparse.Namespace) -> int:
    cpus = common.parse_cpu_list(args.cpus)
    available = common._available_cpus()
    unavailable = [cpu for cpu in cpus if cpu not in available]
    if unavailable and args.binder != "none":
        raise SystemExit(f"selected CPUs are outside this process affinity mask: {unavailable}")
    if max(args.core_counts) > len(cpus):
        raise SystemExit("a requested core count exceeds the number of selected CPUs")
    sibling_groups = common._selected_smt_groups(cpus)
    if sibling_groups and not args.allow_smt:
        formatted = "; ".join(common._format_cpu_tuple(group) for group in sibling_groups)
        print(
            f"Warning: selected CPUs include SMT siblings ({formatted}); pass --allow-smt to acknowledge.",
            file=sys.stderr,
        )

    executable = args.executable.resolve()
    fixture_dir = args.fixture_dir.resolve()
    fixture_paths = _fixture_paths(fixture_dir)
    if not args.dry_run:
        if not executable.is_file():
            raise SystemExit(f"GEOS-Chem benchmark executable does not exist: {executable}")
        if not os.access(executable, os.X_OK):
            raise SystemExit(f"GEOS-Chem benchmark executable is not executable: {executable}")
        missing = [path for path in fixture_paths if not path.is_file()]
        if missing:
            raise SystemExit("missing GC chain fixture inputs: " + ", ".join(map(str, missing)))

    os.environ.setdefault(
        "NUMBA_CACHE_DIR",
        str(Path(tempfile.gettempdir()) / "wombat-transport-numba-cache"),
    )
    grid_shape = common._read_grid_shape(args.run_config.resolve(), Path(sys.executable))
    if not args.dry_run:
        fixture_shape = _read_fixture_grid_shape(fixture_paths[0])
        if fixture_shape != grid_shape:
            raise SystemExit(
                f"fixture grid {fixture_shape} does not match run-config grid {grid_shape}"
            )
    specs = generate_specs(
        cpus=cpus,
        core_counts=tuple(sorted(set(args.core_counts))),
        tracer_counts=tuple(sorted(set(args.tracer_counts))),
        grid_shape=grid_shape,
    )
    if args.order == "shuffled":
        random.Random(args.seed).shuffle(specs)
    if args.dry_run:
        common._print_specs(specs)
        return 0

    root = args.output_dir.resolve()
    if root.exists() and not args.resume:
        raise SystemExit(f"output directory already exists; pass --resume to continue: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        common._write_json(
            manifest_path,
            _system_manifest(args, cpus, grid_shape, specs, executable, fixture_paths),
        )

    memory_limit = common._memory_limit_bytes(args.max_memory_gb)
    binder = common.resolve_binder(
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
        common._write_json(
            case_dir / "config.json",
            {**_spec_json(spec), "binder": binder},
        )
        if memory_limit is not None and spec.estimated_peak_bytes > memory_limit:
            result = {
                "status": "skipped",
                "reason": (
                    f"estimated peak {spec.estimated_peak_bytes / 1024**3:.2f} GiB exceeds "
                    f"limit {memory_limit / 1024**3:.2f} GiB"
                ),
                "iterations": [],
                "workers": [],
            }
            common._write_json(case_dir / "result.json", result)
            (case_dir / ".complete").touch()
            print(f"[{index}/{len(specs)}] {spec.config_id}: skipped for memory", flush=True)
            continue
        print(f"[{index}/{len(specs)}] {spec.config_id}", flush=True)
        try:
            result = _run_config(
                spec,
                case_dir=case_dir,
                executable=executable,
                fixture_paths=fixture_paths,
                binder=binder,
                memory_policy=args.memory_policy,
                warmup=args.warmup,
                repeat=args.repeat,
            )
        except Exception as exc:
            result = {"status": "failed", "reason": str(exc), "iterations": [], "workers": []}
        common._write_json(case_dir / "result.json", result)
        (case_dir / ".complete").touch()
        if result["status"] == "completed":
            print(
                f"  median {result['median_effective_s']:.6f} s/step, "
                f"{spec.total_tracers / result['median_effective_s']:.1f} tracer-steps/s",
                flush=True,
            )
        else:
            print(f"  failed: {result['reason']}", flush=True)
        common._write_reports(root)
    common._write_reports(root)
    return 0


def _run_config(
    spec: common.ConfigSpec,
    *,
    case_dir: Path,
    executable: Path,
    fixture_paths: tuple[Path, ...],
    binder: str,
    memory_policy: str,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    workers: list[WorkerProcess] = []
    ready_rows: list[dict[str, Any]] = []
    try:
        for rank in range(spec.processes):
            stderr_handle = (case_dir / f"rank_{rank:03d}.stderr.log").open("w", encoding="utf-8")
            command = [
                *common.binder_command(binder, spec.rank_cpus[rank], memory_policy=memory_policy),
                str(executable),
                *(str(path) for path in fixture_paths),
                str(spec.rank_tracers[rank]),
                str(warmup),
            ]
            environment = os.environ.copy()
            environment.update(
                OMP_NUM_THREADS=str(spec.threads_per_process),
                OMP_DYNAMIC="FALSE",
                OMP_PROC_BIND="TRUE",
                OMP_WAIT_POLICY="ACTIVE",
                OMP_STACKSIZE=environment.get("OMP_STACKSIZE", "256M"),
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
            fields = _read_worker_fields(worker)
            if fields[0] != "READY" or len(fields) != 2:
                raise RuntimeError(f"rank {worker.rank} did not become ready: {' '.join(fields)}")
            actual_threads = int(fields[1])
            if actual_threads != spec.threads_per_process:
                raise RuntimeError(
                    f"rank {worker.rank} reported {actual_threads} OpenMP threads, "
                    f"expected {spec.threads_per_process}"
                )
            ready_rows.append(
                {
                    "rank": worker.rank,
                    "threading_layer": "openmp",
                    "openmp_threads": actual_threads,
                    "peak_rss_mib": _process_peak_rss_mib(worker.process.pid),
                }
            )

        iterations: list[dict[str, Any]] = []
        for iteration in range(repeat):
            target_ns = time.monotonic_ns() + 100_000_000
            for worker in workers:
                assert worker.process.stdin is not None
                worker.process.stdin.write(f"RUN {target_ns}\n")
                worker.process.stdin.flush()
            rank_rows: list[dict[str, Any]] = []
            for worker in workers:
                fields = _read_worker_fields(worker)
                if fields[0] != "DONE" or len(fields) != 4:
                    raise RuntimeError(
                        f"rank {worker.rank} returned an invalid iteration row: {' '.join(fields)}"
                    )
                started_ns = int(fields[1])
                completed_ns = int(fields[2])
                rank_rows.append(
                    {
                        "rank": worker.rank,
                        "wall_s": (completed_ns - started_ns) / 1.0e9,
                        "completed_ns": completed_ns,
                        "checksum": float(fields[3]),
                        "peak_rss_mib": _process_peak_rss_mib(worker.process.pid),
                    }
                )
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
            worker.process.stdin.write("STOP 0\n")
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
            "median_effective_s": common._median(effective_values),
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


def _read_worker_fields(worker: WorkerProcess) -> list[str]:
    assert worker.process.stdout is not None
    line = worker.process.stdout.readline()
    if not line:
        code = worker.process.poll()
        raise RuntimeError(f"rank {worker.rank} closed its output unexpectedly (exit {code})")
    return line.split()


def _fixture_paths(fixture_dir: Path) -> tuple[Path, ...]:
    return (
        fixture_dir / "transport_chain_input.nc",
        fixture_dir / "vdiff_input.nc",
        fixture_dir / "convection_input.nc",
    )


def _read_fixture_grid_shape(path: Path) -> tuple[int, int, int]:
    import netCDF4

    with netCDF4.Dataset(path) as dataset:
        return tuple(len(dataset.dimensions[name]) for name in ("lev", "lat", "lon"))


def _spec_json(spec: common.ConfigSpec) -> dict[str, Any]:
    payload = asdict(spec)
    payload["rank_tracers"] = list(spec.rank_tracers)
    payload["rank_cpus"] = [list(value) for value in spec.rank_cpus]
    return payload


def _process_peak_rss_mib(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def _system_manifest(
    args: argparse.Namespace,
    cpus: tuple[int, ...],
    grid_shape: tuple[int, int, int],
    specs: list[common.ConfigSpec],
    executable: Path,
    fixture_paths: tuple[Path, ...],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "engine": "geos-chem-classic-harness",
        "measurement": "in-memory PJC -> TPCORE -> VDIFF -> convection",
        "pjc_mass_flux_handoff": "direct PJC output",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": sys.argv,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": common._cpu_model(),
        "selected_cpus": cpus,
        "cpu_topology": {str(cpu): common._cpu_topology(cpu) for cpu in cpus},
        "cpu_numa_nodes": {str(cpu): common._cpu_numa_node(cpu) for cpu in cpus},
        "selected_smt_groups": common._selected_smt_groups(cpus),
        "grid_shape": grid_shape,
        "run_config": str(args.run_config.resolve()),
        "git": common._git_state(),
        "executable": str(executable),
        "executable_sha256": _sha256(executable),
        "fixture_inputs": {str(path): _sha256(path) for path in fixture_paths},
        "openmp": {
            "dynamic": False,
            "proc_bind": True,
            "wait_policy": "ACTIVE",
            "stacksize": os.environ.get("OMP_STACKSIZE", "256M"),
        },
        "order": args.order,
        "seed": args.seed,
        "configuration_count": len(specs),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
