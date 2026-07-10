from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_RUN_CONFIG = Path("base_wombat/run.yml")
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
            perf_summary = _parse_perf_stat_summary(perf_outputs.get("perf_stat"))
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
    subprocess.run(cmd, check=True, env=_profile_env())
    with output_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"expected one benchmark row in {output_csv}, found {len(rows)}")
    return rows[0]


def _run_perf(operator: str, args: argparse.Namespace, output_dir: Path) -> dict[str, Path]:
    perf = shutil.which("perf")
    if perf is None:
        return {}

    env = _profile_env()
    env["NUMBA_ENABLE_PROFILING"] = "1"
    env["NUMBA_DEBUGINFO"] = "1"
    benchmark_script = str(_benchmark_script(operator))
    common = [
        sys.executable,
        benchmark_script,
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

    stat_csv = output_dir / f"{operator}_{args.tracers}_perf_stat_benchmark.csv"
    stat_txt = output_dir / f"{operator}_{args.tracers}_perf_stat.txt"
    stat_cmd = [
        perf,
        "stat",
        "-d",
        "--delay",
        str(args.perf_delay_ms),
        "--",
        *common,
        "--output",
        str(stat_csv),
    ]
    stat = subprocess.run(stat_cmd, env=env, text=True, capture_output=True, check=False)
    stat_txt.write_text(stat.stdout + stat.stderr)

    outputs = {"perf_stat": stat_txt, "perf_stat_benchmark": stat_csv}
    if stat.returncode != 0:
        return outputs

    perf_data = output_dir / f"{operator}_{args.tracers}.perf.data"
    record_csv = output_dir / f"{operator}_{args.tracers}_perf_record_benchmark.csv"
    record_cmd = [
        perf,
        "record",
        "--delay",
        str(args.perf_delay_ms),
        "-F",
        "999",
        "--call-graph",
        "dwarf",
        "-o",
        str(perf_data),
        "--",
        *common,
        "--output",
        str(record_csv),
    ]
    record = subprocess.run(record_cmd, env=env, text=True, capture_output=True, check=False)
    (output_dir / f"{operator}_{args.tracers}_perf_record.log").write_text(record.stdout + record.stderr)
    outputs["perf_data"] = perf_data
    outputs["perf_record_benchmark"] = record_csv
    if record.returncode != 0:
        return outputs

    dso_report = output_dir / f"{operator}_{args.tracers}_perf_dso_report.txt"
    symbol_report = output_dir / f"{operator}_{args.tracers}_perf_symbol_report.txt"
    _write_perf_report(perf, perf_data, dso_report, ["--sort", "dso"])
    _write_perf_report(perf, perf_data, symbol_report, ["--sort", "symbol,dso", "--percent-limit", "0.75"])
    outputs["perf_dso_report"] = dso_report
    outputs["perf_symbol_report"] = symbol_report
    return outputs


def _write_perf_report(perf: str, perf_data: Path, output_path: Path, extra_args: list[str]) -> None:
    cmd = [perf, "report", "--stdio", "--no-children", "-i", str(perf_data), *extra_args]
    report = subprocess.run(cmd, text=True, capture_output=True, check=False)
    output_path.write_text(report.stdout + report.stderr)


def _parse_perf_stat_summary(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    text = path.read_text()
    patterns = {
        "task_clock_ms": r"^\s*([0-9.]+)\s+msec task-clock",
        "ipc": r"cpu_core/instructions/.*#\s*([0-9.]+)\s+insn per cycle",
        "branch_miss_pct": r"cpu_core/branch-misses/.*#\s*([0-9.]+)%\s+of all branches",
        "backend_bound_pct": r"#\s*([0-9.]+)\s*%\s+tma_backend_bound",
        "frontend_bound_pct": r"#\s*([0-9.]+)\s*%\s+tma_frontend_bound",
        "bad_speculation_pct": r"#\s*([0-9.]+)\s*%\s+tma_bad_speculation",
        "retiring_pct": r"#\s*([0-9.]+)\s*%\s+tma_retiring",
        "l1d_miss_pct": r"L1-dcache-load-misses.*#\s*([0-9.]+)%\s+of all L1-dcache accesses",
        "llc_miss_pct": r"LLC-load-misses\s+#\s*([0-9.]+)%\s+of all LL-cache accesses",
        "page_faults": r"^\s*([0-9]+)\s+page-faults",
    }
    summary: dict[str, str] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            summary[name] = match.group(1)
    return summary


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


def _profile_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("NUMBA_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/wombat-pycache")
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/wombat-numba-cache")
    return env


if __name__ == "__main__":
    raise SystemExit(main())
