from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


@dataclass(frozen=True)
class Case:
    grid: str
    tracers: int
    steps: int
    directory: str
    family: str
    geos_wall: dict[int, float]
    geos_rate: dict[int, float]


@dataclass(frozen=True)
class Result:
    grid: str
    tracers: int
    threads: int
    executor: str
    block_width: int
    wall_seconds: float


CASES = (
    Case(
        "2x2.5",
        1,
        288,
        "realistic_restart_noemis_2x25",
        "readme_scaling",
        {1: 66.10, 2: 49.25, 4: 39.06, 8: 34.44},
        {1: 4.4, 2: 5.8, 4: 7.4, 8: 8.4},
    ),
    Case(
        "2x2.5",
        24,
        144,
        "residual_24tracer_emissions_1day_2x25",
        "readme_scaling",
        {1: 200.46, 2: 124.01, 4: 83.28, 8: 65.83},
        {1: 17.2, 2: 27.9, 4: 41.5, 8: 52.5},
    ),
    Case(
        "2x2.5",
        100,
        144,
        "residual_100tracer_emissions_1day_2x25",
        "readme_scaling_100",
        {1: 710.41, 2: 436.12, 4: 286.84, 8: 219.09},
        {1: 20.3, 2: 33.0, 4: 50.2, 8: 65.7},
    ),
    Case(
        "4x5",
        1,
        288,
        "realistic_restart_noemis_4x5",
        "readme_scaling",
        {1: 16.51, 2: 13.61, 4: 11.49, 8: 10.63},
        {1: 17.4, 2: 21.2, 4: 25.1, 8: 27.1},
    ),
    Case(
        "4x5",
        24,
        144,
        "residual_24tracer_emissions_1day_4x5",
        "readme_scaling",
        {1: 49.38, 2: 30.46, 4: 20.30, 8: 15.80},
        {1: 70.0, 2: 113.4, 4: 170.3, 8: 218.7},
    ),
    Case(
        "4x5",
        100,
        144,
        "residual_100tracer_emissions_1day_4x5",
        "readme_scaling_100",
        {1: 176.16, 2: 105.78, 4: 69.75, 8: 54.37},
        {1: 81.7, 2: 136.1, 4: 206.5, 8: 264.9},
    ),
)

GENERATED_NAMES = {
    "OutputDir",
    "Restarts",
    "timing.txt",
    "validation_timing.json",
    "wombat.log",
    "wombat_run_metadata.json",
}


def _template(materialized_root: Path, case: Case) -> Path:
    return materialized_root / f"{case.family}_t1" / case.directory / "main" / "wombat"


def _copy_template(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"materialized Wombat run is unavailable: {source}")
    destination.mkdir(parents=True)
    for item in source.iterdir():
        if item.name in GENERATED_NAMES:
            continue
        target = destination / item.name
        if item.is_symlink():
            target.symlink_to(item.resolve(), target_is_directory=item.resolve().is_dir())
        elif item.is_dir():
            shutil.copytree(item, target, symlinks=True)
        else:
            shutil.copy2(item, target)
    (destination / "OutputDir").mkdir()
    (destination / "Restarts").mkdir()


def _run(
    *,
    repo: Path,
    python: Path,
    source: Path,
    run_root: Path,
    case: Case,
    threads: int,
    executor: str,
    block_width: int,
) -> Result:
    run_dir = run_root / case.directory / f"t{threads}-{executor}-w{block_width}"
    _copy_template(source, run_dir)
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(repo / "src"),
        WOMBAT_NUMBA="1",
        WOMBAT_NUMBA_THREADS=str(threads),
        WOMBAT_TRANSPORT_EXECUTOR=executor,
        WOMBAT_TRANSPORT_BLOCK_WIDTH=str(block_width),
        NUMBA_NUM_THREADS=str(threads),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    command = [str(python), "-m", "wombat_transport.run", "run.yml"]
    log_path = run_dir / "wombat.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=run_dir,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(f"benchmark failed with exit code {completed.returncode}: {log_path}")
    print(
        f"{case.grid:6} {case.tracers:3} tracers t{threads} "
        f"{executor:7} w{block_width}: {wall:.2f} s",
        flush=True,
    )
    result = Result(case.grid, case.tracers, threads, executor, block_width, wall)
    shutil.rmtree(run_dir)
    return result


def _best_results(results: list[Result]) -> list[Result]:
    best: dict[tuple[str, int, int], Result] = {}
    for result in results:
        key = (result.grid, result.tracers, result.threads)
        if key not in best or result.wall_seconds < best[key].wall_seconds:
            best[key] = result
    return [best[key] for key in sorted(best, key=lambda item: (item[0] != "2x2.5", item[1], item[2]))]


def _markdown(results: list[Result]) -> str:
    cases = {(case.grid, case.tracers): case for case in CASES}
    lines = [
        "| Grid | Tracers | Threads | Executor | Width | GEOS-Chem wall s | Wombat wall s "
        "| GEOS-Chem tracer-steps/s | Wombat tracer-steps/s | Wombat speedup |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in _best_results(results):
        case = cases[(result.grid, result.tracers)]
        tracer_steps = case.steps * case.tracers
        geos_wall = case.geos_wall.get(result.threads)
        geos_wall_text = f"{geos_wall:.2f}" if geos_wall is not None else "--"
        geos_rate = case.geos_rate.get(result.threads)
        geos_rate_text = f"{geos_rate:.1f}" if geos_rate is not None else "--"
        speedup_text = (
            f"{geos_wall / result.wall_seconds:.2f}x" if geos_wall is not None else "--"
        )
        lines.append(
            f"| {result.grid} | {result.tracers} | {result.threads} | {result.executor} "
            f"| {result.block_width} "
            f"| {geos_wall_text} | {result.wall_seconds:.2f} "
            f"| {geos_rate_text} | {tracer_steps / result.wall_seconds:.1f} "
            f"| {speedup_text} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Refresh the Wombat side of the end-to-end benchmarks documented in the README."
    )
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--grids", nargs="+", choices=("2x2.5", "4x5"))
    parser.add_argument("--tracers", nargs="+", type=int, choices=(1, 24, 100))
    parser.add_argument("--executors", nargs="+", choices=("spatial", "blocks"), default=None)
    parser.add_argument("--block-widths", nargs="+", type=int, default=[8, 16, 25])
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="do not run one unmeasured case per selected executor to populate Numba caches",
    )
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.work_root.exists():
        parser.error(f"work root already exists: {args.work_root}")
    if any(width < 1 for width in args.block_widths) or any(
        threads < 1 for threads in args.threads
    ):
        parser.error("thread counts and block width must be positive")
    repo = Path(__file__).resolve().parents[1]
    executors = args.executors or ["spatial", "blocks"]
    results: list[Result] = []
    try:
        selected_cases = [
            case
            for case in CASES
            if (args.grids is None or case.grid in args.grids)
            and (args.tracers is None or case.tracers in args.tracers)
        ]
        if not args.no_warmup and "spatial" in executors:
            warmup = next(case for case in CASES if case.grid == "4x5" and case.tracers == 1)
            _run(
                repo=repo,
                python=args.python.absolute(),
                source=_template(args.materialized_root.resolve(), warmup),
                run_root=args.work_root.resolve() / "warmup",
                case=warmup,
                threads=max(args.threads),
                executor="spatial",
                block_width=1,
            )
        if not args.no_warmup and "blocks" in executors:
            warmup = next(case for case in CASES if case.grid == "4x5" and case.tracers == 24)
            _run(
                repo=repo,
                python=args.python.absolute(),
                source=_template(args.materialized_root.resolve(), warmup),
                run_root=args.work_root.resolve() / "warmup",
                case=warmup,
                threads=max(args.threads),
                executor="blocks",
                block_width=args.block_widths[0],
            )
        for case in selected_cases:
            source = _template(args.materialized_root.resolve(), case)
            for threads in args.threads:
                for executor in executors:
                    if executor == "blocks" and (case.tracers == 1 or threads == 1):
                        continue
                    widths = [case.tracers] if executor == "spatial" else args.block_widths
                    for width in widths:
                        results.append(
                            _run(
                                repo=repo,
                                python=args.python.absolute(),
                                source=source,
                                run_root=args.work_root.resolve(),
                                case=case,
                                threads=threads,
                                executor=executor,
                                block_width=width,
                            )
                        )
    except (OSError, RuntimeError) as exc:
        print(f"documented benchmark failed: {exc}", file=sys.stderr)
        return 2

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "results": [asdict(result) for result in results],
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\n" + _markdown(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
