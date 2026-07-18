from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from _perf_support import parse_perf_stat_summary, profile_environment, run_perf_bundle


DEFAULT_RUN_CONFIG = Path("validation_runs/cases/realistic_restart_noemis_2x25/wombat/main/run.yml")
DEFAULT_OUTPUT_DIR = Path("/tmp/wombat-transport-operator-profile")
DEFAULT_DT_S = 600.0
OPERATORS = ("vdiff", "convection")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports: list[Path] = []
    for operator in args.operators:
        operator_dir = args.output_dir / operator
        operator_dir.mkdir(parents=True, exist_ok=True)
        benchmark_csv = operator_dir / f"{operator}_{args.tracers}_benchmark.csv"
        benchmark = _run_benchmark(operator, args, benchmark_csv)
        perf_outputs: dict[str, Path] = {}
        perf_summary: dict[str, str] = {}
        if not args.skip_perf:
            perf_outputs = _run_perf(operator, args, operator_dir)
            perf_summary = parse_perf_stat_summary(perf_outputs.get("perf_stat"))
        report_path = operator_dir / f"{operator}_{args.tracers}_profile.md"
        _write_report(
            report_path,
            operator=operator,
            args=args,
            benchmark=benchmark,
            perf_summary=perf_summary,
            perf_outputs=perf_outputs,
        )
        reports.append(report_path)

    for report_path in reports:
        print(report_path)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile VDIFF and convection single-thread Numba workloads.")
    parser.add_argument("--run-config", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument("--operators", nargs="+", choices=OPERATORS, default=list(OPERATORS))
    parser.add_argument("--tracers", type=int, default=96)
    parser.add_argument("--repeat", type=int, default=5, help="Benchmark repeats for the report wall-time row.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--perf-repeat", type=int, default=16)
    parser.add_argument("--perf-warmup", type=int, default=2)
    parser.add_argument("--perf-delay-ms", type=int, default=2000)
    parser.add_argument("--dt-s", type=float, default=DEFAULT_DT_S)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-perf", action="store_true")
    args = parser.parse_args(argv)
    if args.tracers <= 0:
        parser.error("--tracers must be positive")
    if args.repeat <= 0 or args.warmup < 0 or args.perf_repeat <= 0 or args.perf_warmup < 0:
        parser.error("repeat counts must be positive and warmup must be non-negative")
    if args.dt_s <= 0.0:
        parser.error("--dt-s must be positive")
    return args


def _benchmark_script(operator: str) -> Path:
    if operator == "vdiff":
        return Path(__file__).with_name("benchmark_vdiff_scaling.py")
    if operator == "convection":
        return Path(__file__).with_name("benchmark_convection_scaling.py")
    raise ValueError(f"unsupported operator {operator!r}")


def _run_benchmark(operator: str, args: argparse.Namespace, output_csv: Path) -> dict[str, str]:
    cmd = [
        sys.executable,
        str(_benchmark_script(operator)),
        "--run-config",
        str(args.run_config),
        "--counts",
        str(args.tracers),
        "--repeat",
        str(args.repeat),
        "--warmup",
        str(args.warmup),
        "--dt-s",
        str(args.dt_s),
        "--output",
        str(output_csv),
    ]
    subprocess.run(cmd, check=True, env=profile_environment())
    with output_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"expected one benchmark row in {output_csv}, found {len(rows)}")
    return rows[0]


def _run_perf(operator: str, args: argparse.Namespace, output_dir: Path) -> dict[str, Path]:
    common = [
        sys.executable,
        str(_benchmark_script(operator)),
        "--run-config",
        str(args.run_config),
        "--counts",
        str(args.tracers),
        "--repeat",
        str(args.perf_repeat),
        "--warmup",
        str(args.perf_warmup),
        "--dt-s",
        str(args.dt_s),
    ]
    return run_perf_bundle(
        command=common,
        output_dir=output_dir,
        stem=f"{operator}_{args.tracers}",
        delay_ms=args.perf_delay_ms,
        env=profile_environment(),
    )


def _write_report(
    path: Path,
    *,
    operator: str,
    args: argparse.Namespace,
    benchmark: dict[str, str],
    perf_summary: dict[str, str],
    perf_outputs: dict[str, Path],
) -> None:
    lines = [
        f"# {operator.upper()} Numba profile ({args.tracers} tracers)",
        "",
        "## Benchmark",
        "",
        f"- status: `{benchmark.get('status', '')}`",
        f"- best wall: `{benchmark.get('best_wall_s', '')} s`",
        f"- mean wall: `{benchmark.get('mean_wall_s', '')} s`",
        f"- checksum: `{benchmark.get('checksum', '')}`",
    ]
    if operator == "convection":
        lines.extend(
            [
                f"- active columns: `{benchmark.get('active_columns', '')}`",
                f"- total columns: `{benchmark.get('total_columns', '')}`",
            ]
        )

    if perf_summary:
        lines.extend(
            [
                "",
                "## Perf Stat",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
            ]
        )
        for key in (
            "task_clock_ms",
            "ipc",
            "backend_bound_pct",
            "frontend_bound_pct",
            "bad_speculation_pct",
            "retiring_pct",
            "branch_miss_pct",
            "l1d_miss_pct",
            "llc_miss_pct",
            "page_faults",
        ):
            lines.append(f"| `{key}` | {perf_summary.get(key, '')} |")

    if perf_outputs:
        lines.extend(["", "## Perf Artifacts", ""])
        for name, artifact_path in perf_outputs.items():
            lines.append(f"- `{name}`: `{artifact_path}`")
    else:
        lines.extend(["", "## Perf Artifacts", "", "- perf was skipped or not available."])

    lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
