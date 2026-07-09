from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit

from wombat_transport.transport.tpcore import _numba as nb
from wombat_transport.transport.tpcore._core import _set_cross_terms, _set_jn_js, setup_tpcore_terms


DEFAULT_RUN_CONFIG = Path("base_wombat/run.yml")
DEFAULT_OUTPUT_DIR = Path("/tmp/wombat-tpcore-numba-profile")
DEFAULT_DT_S = 600.0
STAGE_PERF_STAGES = (
    "ytp_horizontal_mass_flux",
    "xtp_horizontal_mass_flux",
    "fzppm_vertical",
)


@njit(cache=False)
def _init_level_kernel(q: np.ndarray, dq1: np.ndarray, delp1: np.ndarray, level: int) -> None:
    nlat = q.shape[1]
    nlon = q.shape[2]
    ntracer = q.shape[3]
    for j in range(nlat):
        for i in range(nlon):
            mass = delp1[level, j, i]
            for tracer in range(ntracer):
                dq1[level, j, i, tracer] = q[level, j, i, tracer] * mass


@njit(cache=False)
def _update_q_level_kernel(q: np.ndarray, adx: np.ndarray, ady: np.ndarray, level: int) -> None:
    nlat = q.shape[1]
    nlon = q.shape[2]
    ntracer = q.shape[3]
    for j in range(nlat):
        for i in range(nlon):
            for tracer in range(ntracer):
                q[level, j, i, tracer] += adx[j, i, tracer] + ady[j, i, tracer]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.stage_worker is not None:
        return _run_stage_worker(args)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    bench = _load_benchmark_module()
    inputs = bench._build_synthetic_tpcore_inputs(args.run_config, args.tracers, dt_s=args.dt_s)
    setup = _make_setup(inputs)

    _warm_numba(inputs, setup)

    benchmark_csv = output_dir / f"tpcore_{args.tracers}_benchmark.csv"
    benchmark = _run_benchmark_script(args, benchmark_csv)
    staged_rows = _measure_staged(inputs, setup, repeat=args.stage_repeat)
    codegen_rows = _inspect_codegen()

    perf_outputs: dict[str, Path] = {}
    if not args.skip_perf:
        perf_outputs = _run_perf(args, output_dir)
    stage_perf_rows: list[dict[str, str]] = []
    if args.stage_perf:
        stage_perf_rows = _run_stage_perf(args, output_dir)

    report_path = output_dir / f"tpcore_{args.tracers}_profile.md"
    _write_report(
        report_path,
        args=args,
        benchmark=benchmark,
        staged_rows=staged_rows,
        codegen_rows=codegen_rows,
        perf_outputs=perf_outputs,
        stage_perf_rows=stage_perf_rows,
    )
    print(report_path)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the single-thread Numba TPCORE path.")
    parser.add_argument("--run-config", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument("--tracers", type=int, default=96)
    parser.add_argument("--repeat", type=int, default=5, help="Timed full TPCORE benchmark repeats.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--stage-repeat", type=int, default=5)
    parser.add_argument("--perf-repeat", type=int, default=8)
    parser.add_argument("--perf-delay-ms", type=int, default=5000)
    parser.add_argument("--dt-s", type=float, default=DEFAULT_DT_S)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-perf", action="store_true", help="Skip perf stat/record/report.")
    parser.add_argument(
        "--stage-perf",
        action="store_true",
        help="Collect perf stat counters for isolated TPCORE suboperators.",
    )
    parser.add_argument(
        "--stage-perf-stages",
        nargs="+",
        choices=STAGE_PERF_STAGES,
        default=list(STAGE_PERF_STAGES),
        help="Suboperators to profile when --stage-perf is set.",
    )
    parser.add_argument("--stage-perf-iterations", type=int, default=20)
    parser.add_argument("--stage-worker", choices=STAGE_PERF_STAGES, help=argparse.SUPPRESS)
    parser.add_argument("--stage-worker-iterations", type=int, default=20, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.tracers <= 0:
        parser.error("--tracers must be positive")
    if (
        args.repeat <= 0
        or args.warmup < 0
        or args.stage_repeat <= 0
        or args.perf_repeat <= 0
        or args.stage_perf_iterations <= 0
        or args.stage_worker_iterations <= 0
    ):
        parser.error("repeat counts must be positive and warmup must be non-negative")
    if args.dt_s <= 0.0:
        parser.error("--dt-s must be positive")
    return args


def _load_benchmark_module() -> Any:
    path = Path(__file__).with_name("benchmark_tpcore_scaling.py").resolve()
    spec = importlib.util.spec_from_file_location("benchmark_tpcore_scaling", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_setup(inputs: Any) -> Any:
    return setup_tpcore_terms(
        p1_hpa=inputs.p1_hpa,
        p2_hpa=inputs.p2_hpa,
        u_m_s=inputs.u_m_s,
        v_m_s=inputs.v_m_s,
        area_m2=inputs.area_m2,
        hyai_hpa=inputs.hyai_hpa,
        hybi=inputs.hybi,
        lat_deg=inputs.lat_deg,
        dt_s=inputs.dt_s,
    )


def _warm_numba(inputs: Any, setup: Any) -> None:
    nb._advect_tracers_fused_numba(tracer_conc=inputs.tracer_conc, setup=setup, area_m2=inputs.area_m2, fill=True)
    _run_staged_once(inputs, setup)


def _run_benchmark_script(args: argparse.Namespace, output_csv: Path) -> dict[str, str]:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("benchmark_tpcore_scaling.py")),
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


def _measure_staged(inputs: Any, setup: Any, *, repeat: int) -> list[dict[str, float]]:
    rows = []
    for _ in range(repeat):
        rows.append(_run_staged_once(inputs, setup))
    return rows


def _run_staged_once(inputs: Any, setup: Any) -> dict[str, float]:
    times: dict[str, float] = {}
    t0 = time.perf_counter()
    nlev, nlat, nlon, ntracer = inputs.tracer_conc.shape
    q = np.ascontiguousarray(inputs.tracer_conc).copy()
    dq1 = np.empty_like(q)
    prepass_workspace = nb._make_tpcore_prepass_numba_workspace(nlat, nlon, ntracer)
    x_workspace = nb._make_xtp_numba_workspace(nlat, nlon, ntracer)
    y_workspace = nb._make_ytp_numba_workspace(nlat, nlon, ntracer)
    ua, va = _set_cross_terms(setup.cx, setup.cy)
    jn, js = _set_jn_js(setup.cx)
    qqu, qqv, adx, ady = prepass_workspace
    dcx, fx, al_x, ar_x, a6_x, dc_x, qa_x = x_workspace
    dcy, al_y, ar_y, a6_y = y_workspace
    _add_time(times, "python_copy_workspace_cross_terms", t0)

    for level in range(nlev):
        t = time.perf_counter()
        nb._average_const_poles_batch_numba_kernel(q[level], setup.delp1_hpa[level], inputs.area_m2[:, 0])
        _init_level_kernel(q, dq1, setup.delp1_hpa, level)
        _add_time(times, "poles_plus_dq_init", t)

        t = time.perf_counter()
        nb._calc_advec_cross_terms_batch_numba_kernel(
            q[level], ua[level], va[level], int(jn[level]), int(js[level]), qqu, qqv
        )
        _add_time(times, "calc_cross_terms", t)

        t = time.perf_counter()
        nb._xadv_dao2_batch_numba_kernel(qqv, ua[level], int(jn[level]), int(js[level]), adx)
        _add_time(times, "xadv_dao2", t)

        t = time.perf_counter()
        nb._yadv_dao2_batch_numba_kernel(qqu, va[level], ady)
        _add_time(times, "yadv_dao2", t)

        t = time.perf_counter()
        _update_q_level_kernel(q, adx, ady, level)
        _add_time(times, "q_prepass_update", t)

        t = time.perf_counter()
        nb._xtp_batch_numba_kernel(
            dq1[level],
            qqv,
            setup.pu_hpa[level],
            setup.cx[level],
            setup.xmass_hpa[level],
            int(jn[level]),
            int(js[level]),
            dcx,
            fx,
            al_x,
            ar_x,
            a6_x,
            dc_x,
            qa_x,
        )
        _add_time(times, "xtp_horizontal_mass_flux", t)

        t = time.perf_counter()
        nb._ytp_batch_numba_kernel(
            dq1[level],
            qqu,
            qqv,
            setup.cy[level],
            setup.ymass_hpa[level],
            setup.geofac,
            setup.geofac_pc,
            dcy,
            al_y,
            ar_y,
            a6_y,
        )
        _add_time(times, "ytp_horizontal_mass_flux", t)

    t = time.perf_counter()
    nb._fzppm_batch_numba_kernel(setup.delp1_hpa, setup.vertical_mass_flux_hpa, dq1, q)
    _add_time(times, "fzppm_vertical", t)

    t = time.perf_counter()
    nb._qckxyz_batch_numba_kernel(dq1)
    _add_time(times, "qckxyz_fill", t)

    t = time.perf_counter()
    nb._finalize_tpcore_output_numba_kernel(dq1, setup.delp2_hpa)
    _add_time(times, "finalize_output", t)
    times["checksum"] = float(np.mean(dq1[0, 0, 0, :]))
    return times


def _run_stage_worker(args: argparse.Namespace) -> int:
    bench = _load_benchmark_module()
    inputs = bench._build_synthetic_tpcore_inputs(args.run_config, args.tracers, dt_s=args.dt_s)
    setup = _make_setup(inputs)
    _warm_numba(inputs, setup)
    state = _prepare_stage_state(inputs, setup)
    print("READY", flush=True)
    sys.stdin.readline()
    start = time.perf_counter()
    checksum = 0.0
    for _ in range(args.stage_worker_iterations):
        checksum = _run_stage_perf_kernel(args.stage_worker, state)
    elapsed = time.perf_counter() - start
    print(f"stage,{args.stage_worker}")
    print(f"iterations,{args.stage_worker_iterations}")
    print(f"elapsed_s,{elapsed:.9f}")
    print(f"checksum,{checksum:.16g}")
    return 0


def _prepare_stage_state(inputs: Any, setup: Any) -> dict[str, Any]:
    nlev, nlat, nlon, ntracer = inputs.tracer_conc.shape
    q = np.ascontiguousarray(inputs.tracer_conc).copy()
    dq1 = np.empty_like(q)
    qqu_levels = np.empty_like(q)
    qqv_levels = np.empty_like(q)
    prepass_workspace = nb._make_tpcore_prepass_numba_workspace(nlat, nlon, ntracer)
    x_workspace = nb._make_xtp_numba_workspace(nlat, nlon, ntracer)
    y_workspace = nb._make_ytp_numba_workspace(nlat, nlon, ntracer)
    ua, va = _set_cross_terms(setup.cx, setup.cy)
    jn, js = _set_jn_js(setup.cx)
    qqu, qqv, adx, ady = prepass_workspace
    dcx, fx, al_x, ar_x, a6_x, dc_x, qa_x = x_workspace
    dcy, al_y, ar_y, a6_y = y_workspace

    for level in range(nlev):
        nb._average_const_poles_batch_numba_kernel(q[level], setup.delp1_hpa[level], inputs.area_m2[:, 0])
        _init_level_kernel(q, dq1, setup.delp1_hpa, level)
        nb._calc_advec_cross_terms_batch_numba_kernel(
            q[level], ua[level], va[level], int(jn[level]), int(js[level]), qqu, qqv
        )
        qqu_levels[level] = qqu
        qqv_levels[level] = qqv
        nb._xadv_dao2_batch_numba_kernel(qqv, ua[level], int(jn[level]), int(js[level]), adx)
        nb._yadv_dao2_batch_numba_kernel(qqu, va[level], ady)
        _update_q_level_kernel(q, adx, ady, level)
        nb._xtp_batch_numba_kernel(
            dq1[level],
            qqv,
            setup.pu_hpa[level],
            setup.cx[level],
            setup.xmass_hpa[level],
            int(jn[level]),
            int(js[level]),
            dcx,
            fx,
            al_x,
            ar_x,
            a6_x,
            dc_x,
            qa_x,
        )
        nb._ytp_batch_numba_kernel(
            dq1[level],
            qqu,
            qqv,
            setup.cy[level],
            setup.ymass_hpa[level],
            setup.geofac,
            setup.geofac_pc,
            dcy,
            al_y,
            ar_y,
            a6_y,
        )

    return {
        "setup": setup,
        "q": q,
        "dq1": dq1,
        "qqu_levels": qqu_levels,
        "qqv_levels": qqv_levels,
        "x_workspace": x_workspace,
        "y_workspace": y_workspace,
        "jn": jn,
        "js": js,
    }


def _run_stage_perf_kernel(stage: str, state: dict[str, Any]) -> float:
    setup = state["setup"]
    dq1 = state["dq1"]
    nlev = dq1.shape[0]
    if stage == "fzppm_vertical":
        nb._fzppm_batch_numba_kernel(setup.delp1_hpa, setup.vertical_mass_flux_hpa, dq1, state["q"])
        return float(np.mean(dq1[0, 0, 0, :]))

    if stage == "xtp_horizontal_mass_flux":
        dcx, fx, al_x, ar_x, a6_x, dc_x, qa_x = state["x_workspace"]
        for level in range(nlev):
            nb._xtp_batch_numba_kernel(
                dq1[level],
                state["qqv_levels"][level],
                setup.pu_hpa[level],
                setup.cx[level],
                setup.xmass_hpa[level],
                int(state["jn"][level]),
                int(state["js"][level]),
                dcx,
                fx,
                al_x,
                ar_x,
                a6_x,
                dc_x,
                qa_x,
            )
        return float(np.mean(dq1[0, 0, 0, :]))

    if stage == "ytp_horizontal_mass_flux":
        dcy, al_y, ar_y, a6_y = state["y_workspace"]
        for level in range(nlev):
            nb._ytp_batch_numba_kernel(
                dq1[level],
                state["qqu_levels"][level],
                state["qqv_levels"][level],
                setup.cy[level],
                setup.ymass_hpa[level],
                setup.geofac,
                setup.geofac_pc,
                dcy,
                al_y,
                ar_y,
                a6_y,
            )
        return float(np.mean(dq1[0, 0, 0, :]))

    raise ValueError(f"unsupported stage {stage!r}")


def _add_time(times: dict[str, float], name: str, start: float) -> None:
    times[name] = times.get(name, 0.0) + (time.perf_counter() - start)


def _inspect_codegen() -> list[dict[str, Any]]:
    kernels = [
        ("fused", nb._advect_tracers_fused_numba_kernel),
        ("average_poles", nb._average_const_poles_batch_numba_kernel),
        ("calc_cross_terms", nb._calc_advec_cross_terms_batch_numba_kernel),
        ("xadv_dao2", nb._xadv_dao2_batch_numba_kernel),
        ("yadv_dao2", nb._yadv_dao2_batch_numba_kernel),
        ("xtp", nb._xtp_batch_numba_kernel),
        ("ytp", nb._ytp_batch_numba_kernel),
        ("fzppm", nb._fzppm_batch_numba_kernel),
        ("qckxyz", nb._qckxyz_batch_numba_kernel),
        ("finalize", nb._finalize_tpcore_output_numba_kernel),
    ]
    rows: list[dict[str, Any]] = []
    for name, fn in kernels:
        sig = fn.signatures[0]
        asm = fn.inspect_asm(sig)
        llvm = fn.inspect_llvm(sig)
        rows.append(
            {
                "kernel": name,
                "signatures": len(fn.signatures),
                "asm_lines": asm.count("\n"),
                "llvm_lines": llvm.count("\n"),
                "packed_fp_instr": len(re.findall(r"\bv(?:add|sub|mul|div|min|max|fmadd|fnmadd)[a-z0-9_]*p[ds]\b", asm)),
                "scalar_fp_instr": len(re.findall(r"\bv(?:add|sub|mul|div|min|max|fmadd|fnmadd)[a-z0-9_]*s[ds]\b", asm)),
                "branch_instr": len(re.findall(r"\n\s*j[a-z]+\s", asm)),
                "call_instr": len(re.findall(r"\n\s*callq?\s", asm)),
                "llvm_vector_body": llvm.count("vector.body"),
                "llvm_vector_double": len(re.findall(r"<[0-9]+ x double>", llvm)),
                "llvm_nrt_alloc": llvm.count("NRT_MemInfo_alloc"),
            }
        )
    return rows


def _run_perf(args: argparse.Namespace, output_dir: Path) -> dict[str, Path]:
    perf = shutil.which("perf")
    if perf is None:
        return {}
    env = _profile_env()
    env["NUMBA_ENABLE_PROFILING"] = "1"
    env["NUMBA_DEBUGINFO"] = "1"

    benchmark_script = str(Path(__file__).with_name("benchmark_tpcore_scaling.py"))
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
        "1",
        "--dt-s",
        str(args.dt_s),
    ]

    stat_csv = output_dir / f"tpcore_{args.tracers}_perf_stat_benchmark.csv"
    stat_txt = output_dir / f"tpcore_{args.tracers}_perf_stat.txt"
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

    perf_data = output_dir / f"tpcore_{args.tracers}.perf.data"
    record_csv = output_dir / f"tpcore_{args.tracers}_perf_record_benchmark.csv"
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
    (output_dir / f"tpcore_{args.tracers}_perf_record.log").write_text(record.stdout + record.stderr)
    outputs["perf_data"] = perf_data
    outputs["perf_record_benchmark"] = record_csv
    if record.returncode != 0:
        return outputs

    dso_report = output_dir / f"tpcore_{args.tracers}_perf_dso_report.txt"
    symbol_report = output_dir / f"tpcore_{args.tracers}_perf_symbol_report.txt"
    _write_perf_report(perf, perf_data, dso_report, ["--sort", "dso"])
    _write_perf_report(perf, perf_data, symbol_report, ["--sort", "symbol,dso", "--percent-limit", "0.75"])
    outputs["perf_dso_report"] = dso_report
    outputs["perf_symbol_report"] = symbol_report
    return outputs


def _run_stage_perf(args: argparse.Namespace, output_dir: Path) -> list[dict[str, str]]:
    perf = shutil.which("perf")
    if perf is None:
        return []
    rows: list[dict[str, str]] = []
    for stage in args.stage_perf_stages:
        stat_path = output_dir / f"tpcore_{args.tracers}_{stage}_perf_stat.txt"
        worker_log_path = output_dir / f"tpcore_{args.tracers}_{stage}_worker.log"
        cmd = [
            sys.executable,
            str(Path(__file__)),
            "--run-config",
            str(args.run_config),
            "--tracers",
            str(args.tracers),
            "--dt-s",
            str(args.dt_s),
            "--stage-worker",
            stage,
            "--stage-worker-iterations",
            str(args.stage_perf_iterations),
        ]
        worker = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_profile_env(),
        )
        try:
            ready = worker.stdout.readline() if worker.stdout is not None else ""
            if ready.strip() != "READY":
                stdout, stderr = worker.communicate(timeout=10)
                worker_log_path.write_text(ready + stdout + stderr)
                rows.append({"stage": stage, "status": "failed", "perf_stat": str(stat_path)})
                continue

            perf_cmd = [perf, "stat", "-d", "-p", str(worker.pid), "-o", str(stat_path)]
            perf_proc = subprocess.Popen(perf_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            time.sleep(0.2)
            if worker.stdin is not None:
                worker.stdin.write("\n")
                worker.stdin.flush()
            stdout, stderr = worker.communicate(timeout=120)
            if perf_proc.poll() is None:
                perf_proc.send_signal(signal.SIGINT)
            perf_stdout, perf_stderr = perf_proc.communicate(timeout=30)
            worker_log_path.write_text(stdout + stderr + perf_stdout + perf_stderr)
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.communicate()

        summary = _parse_perf_stat_summary(stat_path)
        summary.update(
            {
                "stage": stage,
                "status": "completed" if stat_path.exists() else "missing",
                "perf_stat": str(stat_path),
                "worker_log": str(worker_log_path),
            }
        )
        rows.append(summary)
    return rows


def _parse_perf_stat_summary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text()
    patterns = {
        "task_clock_ms": r"^\s*([0-9.]+)\s+msec task-clock",
        "ipc": r"cpu_core/instructions/.*#\s*([0-9.]+)\s+insn per cycle",
        "branch_miss_pct": r"cpu_core/branch-misses/.*#\s*([0-9.]+)%\s+of all branches",
        "backend_bound_pct": r"#\s*([0-9.]+)\s*%\s+tma_backend_bound",
        "l1d_miss_pct": r"L1-dcache-load-misses.*#\s*([0-9.]+)%\s+of all L1-dcache accesses",
        "llc_miss_pct": r"LLC-load-misses\s+#\s*([0-9.]+)%\s+of all LL-cache accesses",
    }
    summary: dict[str, str] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            summary[name] = match.group(1)
    return summary


def _write_perf_report(perf: str, perf_data: Path, output_path: Path, extra_args: list[str]) -> None:
    cmd = [perf, "report", "--stdio", "--no-children", "-i", str(perf_data), *extra_args]
    report = subprocess.run(cmd, text=True, capture_output=True, check=False)
    output_path.write_text(report.stdout + report.stderr)


def _profile_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("NUMBA_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/wombat-pycache")
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/wombat-numba-cache")
    return env


def _write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    benchmark: dict[str, str],
    staged_rows: list[dict[str, float]],
    codegen_rows: list[dict[str, Any]],
    perf_outputs: dict[str, Path],
    stage_perf_rows: list[dict[str, str]],
) -> None:
    stage_means = _stage_means(staged_rows)
    stage_total = sum(value for key, value in stage_means.items() if key != "checksum")
    lines = [
        f"# TPCORE Numba profile ({args.tracers} tracers)",
        "",
        "## Benchmark",
        "",
        f"- best wall: `{benchmark['best_wall_s']} s`",
        f"- mean wall: `{benchmark['mean_wall_s']} s`",
        f"- checksum: `{benchmark['checksum']}`",
        "",
        "## Staged Timing",
        "",
        "| Stage | Mean s | Percent |",
        "| --- | ---: | ---: |",
    ]
    for name, value in sorted(
        ((key, value) for key, value in stage_means.items() if key != "checksum"),
        key=lambda item: item[1],
        reverse=True,
    ):
        lines.append(f"| `{name}` | {value:.9f} | {value / stage_total * 100.0:.2f}% |")

    lines.extend(
        [
            "",
            "## Codegen Summary",
            "",
            "| Kernel | Vector bodies | Vector doubles | NRT alloc refs | Branch instr | Calls |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in codegen_rows:
        lines.append(
            f"| `{row['kernel']}` | {row['llvm_vector_body']} | {row['llvm_vector_double']} | "
            f"{row['llvm_nrt_alloc']} | {row['branch_instr']} | {row['call_instr']} |"
        )

    if perf_outputs:
        lines.extend(["", "## Perf Artifacts", ""])
        for name, artifact_path in perf_outputs.items():
            lines.append(f"- `{name}`: `{artifact_path}`")
    else:
        lines.extend(["", "## Perf Artifacts", "", "- perf was skipped or not available."])

    if stage_perf_rows:
        lines.extend(
            [
                "",
                "## Stage Perf Stat",
                "",
                "| Stage | Task ms | IPC | Backend % | Branch miss % | L1D miss % | LLC miss % |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in stage_perf_rows:
            lines.append(
                f"| `{row['stage']}` | {row.get('task_clock_ms', '')} | {row.get('ipc', '')} | "
                f"{row.get('backend_bound_pct', '')} | {row.get('branch_miss_pct', '')} | "
                f"{row.get('l1d_miss_pct', '')} | {row.get('llc_miss_pct', '')} |"
            )
        lines.extend(["", "Stage perf artifacts:"])
        for row in stage_perf_rows:
            lines.append(f"- `{row['stage']}`: `{row.get('perf_stat', '')}`")

    lines.append("")
    path.write_text("\n".join(lines))


def _stage_means(rows: list[dict[str, float]]) -> dict[str, float]:
    names = rows[0].keys()
    return {name: sum(row[name] for row in rows) / float(len(rows)) for name in names}


if __name__ == "__main__":
    raise SystemExit(main())
