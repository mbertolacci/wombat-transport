from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path


def profile_environment(*, numba_thread_vars: tuple[str, ...] = ()) -> dict[str, str]:
    env = os.environ.copy()
    if "NUMBA_NUM_THREADS" not in env:
        env["NUMBA_NUM_THREADS"] = next(
            (env[name] for name in numba_thread_vars if name in env),
            "1",
        )
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/wombat-pycache")
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/wombat-numba-cache")
    return env


def run_perf_bundle(
    *,
    command: list[str],
    output_dir: Path,
    stem: str,
    delay_ms: int,
    env: dict[str, str],
) -> dict[str, Path]:
    perf = shutil.which("perf")
    if perf is None:
        return {}

    perf_env = env.copy()
    perf_env["NUMBA_ENABLE_PROFILING"] = "1"
    perf_env["NUMBA_DEBUGINFO"] = "1"

    stat_csv = output_dir / f"{stem}_perf_stat_benchmark.csv"
    stat_txt = output_dir / f"{stem}_perf_stat.txt"
    stat = subprocess.run(
        [perf, "stat", "-d", "--delay", str(delay_ms), "--", *command, "--output", str(stat_csv)],
        env=perf_env,
        text=True,
        capture_output=True,
        check=False,
    )
    stat_txt.write_text(stat.stdout + stat.stderr, encoding="utf-8")

    outputs = {"perf_stat": stat_txt, "perf_stat_benchmark": stat_csv}
    if stat.returncode != 0:
        return outputs

    perf_data = output_dir / f"{stem}.perf.data"
    record_csv = output_dir / f"{stem}_perf_record_benchmark.csv"
    record = subprocess.run(
        [
            perf,
            "record",
            "--delay",
            str(delay_ms),
            "-F",
            "999",
            "--call-graph",
            "dwarf",
            "-o",
            str(perf_data),
            "--",
            *command,
            "--output",
            str(record_csv),
        ],
        env=perf_env,
        text=True,
        capture_output=True,
        check=False,
    )
    (output_dir / f"{stem}_perf_record.log").write_text(record.stdout + record.stderr, encoding="utf-8")
    outputs["perf_data"] = perf_data
    outputs["perf_record_benchmark"] = record_csv
    if record.returncode != 0:
        return outputs

    dso_report = output_dir / f"{stem}_perf_dso_report.txt"
    symbol_report = output_dir / f"{stem}_perf_symbol_report.txt"
    _write_perf_report(perf, perf_data, dso_report, ["--sort", "dso"])
    _write_perf_report(perf, perf_data, symbol_report, ["--sort", "symbol,dso", "--percent-limit", "0.75"])
    outputs["perf_dso_report"] = dso_report
    outputs["perf_symbol_report"] = symbol_report
    return outputs


def parse_perf_stat_summary(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
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


def _write_perf_report(perf: str, perf_data: Path, output_path: Path, extra_args: list[str]) -> None:
    report = subprocess.run(
        [perf, "report", "--stdio", "--no-children", "-i", str(perf_data), *extra_args],
        text=True,
        capture_output=True,
        check=False,
    )
    output_path.write_text(report.stdout + report.stderr, encoding="utf-8")
