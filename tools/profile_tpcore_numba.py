from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from _perf_support import parse_perf_stat_summary, profile_environment, run_perf_bundle
from wombat_transport.transport.numba_control import configure_numba_threads
from wombat_transport.transport.tpcore import _numba as nb
from wombat_transport.transport.tpcore._native import setup_tpcore_terms


DEFAULT_RUN_CONFIG = Path("validation_runs/cases/realistic_restart_noemis_2x25/wombat/main/run.yml")
DEFAULT_OUTPUT_DIR = Path("/tmp/wombat-tpcore-numba-profile")
DEFAULT_DT_S = 600.0
STAGE_PERF_STAGES = (
    "python_copy_workspace_cross_terms",
    "poles_plus_dq_init",
    "calc_cross_terms",
    "xadv_dao2",
    "yadv_dao2",
    "ytp_horizontal_mass_flux",
    "xtp_horizontal_mass_flux",
    "fzppm_vertical",
    "qckxyz_fill_finalize",
)


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
    path_census: Path | None = None
    if args.path_census:
        path_census = output_dir / f"tpcore_{args.tracers}_path_census.json"
        path_census.write_text(
            json.dumps(_count_paths(setup, args.tracers), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    report_path = output_dir / f"tpcore_{args.tracers}_profile.md"
    _write_report(
        report_path,
        args=args,
        benchmark=benchmark,
        staged_rows=staged_rows,
        codegen_rows=codegen_rows,
        perf_outputs=perf_outputs,
        stage_perf_rows=stage_perf_rows,
        path_census=path_census,
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
        "--path-census",
        action="store_true",
        help="Write geometry-dependent hot-kernel path counts alongside the profile.",
    )
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
    parser.add_argument("--stage-worker-seconds", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--stage-worker-direct", action="store_true", help=argparse.SUPPRESS)
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
        or args.stage_worker_seconds < 0.0
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
    nthreads = configure_numba_threads(available=True)
    workspace = nb._get_tpcore_numba_workspace(nlev, nlat, nlon, ntracer, nthreads)
    nb._set_cross_terms_numba_kernel(setup.cx, setup.cy, workspace.ua, workspace.va)
    nb._set_jn_js_numba_kernel(setup.cx, workspace.jn, workspace.js)
    qqu = workspace.qqu
    qqv = workspace.qqv
    ua = workspace.ua
    va = workspace.va
    jn = workspace.jn
    js = workspace.js
    x_workspace = workspace.x_workspace
    y_workspace = workspace.y_workspace
    dcx, fx, al_x, ar_x, a6_x, dc_x, qa_x = x_workspace
    dcy, al_y, ar_y, a6_y, south_flux_y, north_flux_y, south_dao2_y, north_dao2_y = y_workspace
    z_workspace = workspace.z_workspace
    _add_time(times, "python_copy_workspace_cross_terms", t0)

    for level in range(nlev):
        t = time.perf_counter()
        nb._average_const_poles_batch_numba_kernel(q[level], setup.delp1_hpa[level], inputs.area_m2[:, 0])
        nb._init_dq_mass_numba_kernel(q[level], dq1[level], setup.delp1_hpa[level])
        _add_time(times, "poles_plus_dq_init", t)

        t = time.perf_counter()
        nb._calc_advec_cross_terms_batch_numba_kernel(
            q[level], ua[level], va[level], int(jn[level]), int(js[level]), qqu, qqv
        )
        _add_time(times, "calc_cross_terms", t)

        t = time.perf_counter()
        nb._xadv_dao2_apply_batch_numba_kernel(q[level], qqv, ua[level], int(jn[level]), int(js[level]))
        _add_time(times, "xadv_dao2", t)

        t = time.perf_counter()
        nb._yadv_dao2_apply_batch_numba_kernel(q[level], qqu, va[level], south_dao2_y, north_dao2_y)
        _add_time(times, "yadv_dao2", t)

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
            south_flux_y,
            north_flux_y,
        )
        _add_time(times, "ytp_horizontal_mass_flux", t)

    t = time.perf_counter()
    nb._fzppm_batch_numba_kernel(setup.delp1_hpa, setup.vertical_mass_flux_hpa, dq1, q, *z_workspace)
    _add_time(times, "fzppm_vertical", t)

    t = time.perf_counter()
    if nb._qckxyz_needs_fill_numba_kernel(dq1):
        nb._qckxyz_batch_numba_kernel(dq1)
    nb._finalize_tpcore_output_numba_kernel(dq1, setup.delp2_hpa)
    _add_time(times, "qckxyz_fill_finalize", t)
    times["checksum"] = float(np.mean(dq1[0, 0, 0, :]))
    return times


def _run_stage_worker(args: argparse.Namespace) -> int:
    bench = _load_benchmark_module()
    inputs = bench._build_synthetic_tpcore_inputs(args.run_config, args.tracers, dt_s=args.dt_s)
    setup = _make_setup(inputs)
    _warm_numba(inputs, setup)
    state = _prepare_stage_state(inputs, setup)
    checksum = _run_stage_perf_kernel(args.stage_worker, state)
    if not args.stage_worker_direct:
        print("READY", flush=True)
        sys.stdin.readline()
    start = time.perf_counter()
    iterations = 0
    while iterations < args.stage_worker_iterations or (
        args.stage_worker_seconds > 0.0 and time.perf_counter() - start < args.stage_worker_seconds
    ):
        checksum = _run_stage_perf_kernel(args.stage_worker, state)
        iterations += 1
    elapsed = time.perf_counter() - start
    print(f"stage,{args.stage_worker}")
    print(f"iterations,{iterations}")
    print(f"elapsed_s,{elapsed:.9f}")
    print(f"checksum,{checksum:.16g}")
    return 0


def _prepare_stage_state(inputs: Any, setup: Any) -> dict[str, Any]:
    nlev, nlat, nlon, ntracer = inputs.tracer_conc.shape
    q = np.ascontiguousarray(inputs.tracer_conc).copy()
    dq1 = np.empty_like(q)
    q_for_x = np.empty_like(q)
    q_for_y = np.empty_like(q)
    qqu_levels = np.empty_like(q)
    qqv_levels = np.empty_like(q)
    nthreads = configure_numba_threads(available=True)
    workspace = nb._get_tpcore_numba_workspace(nlev, nlat, nlon, ntracer, nthreads)
    nb._set_cross_terms_numba_kernel(setup.cx, setup.cy, workspace.ua, workspace.va)
    nb._set_jn_js_numba_kernel(setup.cx, workspace.jn, workspace.js)
    qqu = workspace.qqu
    qqv = workspace.qqv
    ua = workspace.ua
    va = workspace.va
    jn = workspace.jn
    js = workspace.js
    x_workspace = workspace.x_workspace
    y_workspace = workspace.y_workspace
    dcx, fx, al_x, ar_x, a6_x, dc_x, qa_x = x_workspace
    dcy, al_y, ar_y, a6_y, south_flux_y, north_flux_y, south_dao2_y, north_dao2_y = y_workspace

    for level in range(nlev):
        nb._average_const_poles_batch_numba_kernel(q[level], setup.delp1_hpa[level], inputs.area_m2[:, 0])
        nb._init_dq_mass_numba_kernel(q[level], dq1[level], setup.delp1_hpa[level])
        nb._calc_advec_cross_terms_batch_numba_kernel(
            q[level], ua[level], va[level], int(jn[level]), int(js[level]), qqu, qqv
        )
        qqu_levels[level] = qqu
        qqv_levels[level] = qqv
        q_for_x[level] = q[level]
        q_for_y[level] = q[level]
        nb._xadv_dao2_apply_batch_numba_kernel(q[level], qqv, ua[level], int(jn[level]), int(js[level]))
        nb._yadv_dao2_apply_batch_numba_kernel(q[level], qqu, va[level], south_dao2_y, north_dao2_y)
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
            south_flux_y,
            north_flux_y,
        )

    return {
        "inputs": inputs,
        "setup": setup,
        "q": q,
        "q_for_x": q_for_x,
        "q_for_y": q_for_y,
        "dq1": dq1,
        "prepass_workspace": (qqu, qqv),
        "qqu_levels": qqu_levels,
        "qqv_levels": qqv_levels,
        "x_workspace": x_workspace,
        "y_workspace": y_workspace,
        "z_workspace": workspace.z_workspace,
        "ua": ua,
        "va": va,
        "jn": jn,
        "js": js,
    }


def _run_stage_perf_kernel(stage: str, state: dict[str, Any]) -> float:
    inputs = state["inputs"]
    setup = state["setup"]
    q = state["q"]
    dq1 = state["dq1"]
    nlev = dq1.shape[0]
    nlat = dq1.shape[1]
    nlon = dq1.shape[2]
    ntracer = dq1.shape[3]
    if stage == "python_copy_workspace_cross_terms":
        q_work = np.ascontiguousarray(inputs.tracer_conc).copy()
        dq1_work = np.empty_like(q_work)
        nthreads = configure_numba_threads(available=True)
        workspace = nb._get_tpcore_numba_workspace(nlev, nlat, nlon, ntracer, nthreads)
        nb._set_cross_terms_numba_kernel(setup.cx, setup.cy, workspace.ua, workspace.va)
        nb._set_jn_js_numba_kernel(setup.cx, workspace.jn, workspace.js)
        return float(
            np.mean(q_work[0, 0, 0, :])
            + 0.0 * dq1_work.size
            + 0.0 * workspace.qqu.size
            + 0.0 * workspace.x_workspace[0].size
            + 0.0 * workspace.y_workspace[0].size
            + 0.0 * workspace.ua[0, 0, 0]
            + 0.0 * workspace.va[0, 0, 0]
            + 0.0 * workspace.jn[0]
            + 0.0 * workspace.js[0]
        )

    if stage == "poles_plus_dq_init":
        for level in range(nlev):
            nb._average_const_poles_batch_numba_kernel(q[level], setup.delp1_hpa[level], inputs.area_m2[:, 0])
            nb._init_dq_mass_numba_kernel(q[level], dq1[level], setup.delp1_hpa[level])
        return float(np.mean(dq1[0, 0, 0, :]))

    if stage == "calc_cross_terms":
        qqu, qqv = state["prepass_workspace"]
        for level in range(nlev):
            nb._calc_advec_cross_terms_batch_numba_kernel(
                q[level],
                state["ua"][level],
                state["va"][level],
                int(state["jn"][level]),
                int(state["js"][level]),
                qqu,
                qqv,
            )
        return float(np.mean(qqu[0, 0, :]) + np.mean(qqv[0, 0, :]))

    if stage == "xadv_dao2":
        for level in range(nlev):
            nb._xadv_dao2_apply_batch_numba_kernel(
                state["q_for_x"][level],
                state["qqv_levels"][level],
                state["ua"][level],
                int(state["jn"][level]),
                int(state["js"][level]),
            )
        return float(np.mean(state["q_for_x"][0, 0, 0, :]))

    if stage == "yadv_dao2":
        for level in range(nlev):
            _, _, _, _, _, _, south_dao2_y, north_dao2_y = state["y_workspace"]
            nb._yadv_dao2_apply_batch_numba_kernel(
                state["q_for_y"][level],
                state["qqu_levels"][level],
                state["va"][level],
                south_dao2_y,
                north_dao2_y,
            )
        return float(np.mean(state["q_for_y"][0, 0, 0, :]))

    if stage == "fzppm_vertical":
        nb._fzppm_batch_numba_kernel(setup.delp1_hpa, setup.vertical_mass_flux_hpa, dq1, state["q"], *state["z_workspace"])
        return float(np.mean(dq1[0, 0, 0, :]))

    if stage == "qckxyz_fill_finalize":
        if nb._qckxyz_needs_fill_numba_kernel(dq1):
            nb._qckxyz_batch_numba_kernel(dq1)
        nb._finalize_tpcore_output_numba_kernel(dq1, setup.delp2_hpa)
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
        dcy, al_y, ar_y, a6_y, south_flux_y, north_flux_y, _, _ = state["y_workspace"]
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
                south_flux_y,
                north_flux_y,
            )
        return float(np.mean(dq1[0, 0, 0, :]))

    raise ValueError(f"unsupported stage {stage!r}")


def _add_time(times: dict[str, float], name: str, start: float) -> None:
    times[name] = times.get(name, 0.0) + (time.perf_counter() - start)


def _inspect_codegen() -> list[dict[str, Any]]:
    kernels = [
        ("fused", nb._advect_tracers_fused_numba_kernel),
        ("set_cross_terms", nb._set_cross_terms_numba_kernel),
        ("set_jn_js", nb._set_jn_js_numba_kernel),
        ("average_poles", nb._average_const_poles_batch_numba_kernel),
        ("calc_cross_terms", nb._calc_advec_cross_terms_batch_numba_kernel),
        ("xadv_dao2", nb._xadv_dao2_batch_numba_kernel),
        ("yadv_dao2", nb._yadv_dao2_batch_numba_kernel),
        ("xadv_dao2_apply", nb._xadv_dao2_apply_batch_numba_kernel),
        ("yadv_dao2_apply", nb._yadv_dao2_apply_batch_numba_kernel),
        ("xtp", nb._xtp_batch_numba_kernel),
        ("ytp", nb._ytp_batch_numba_kernel),
        ("fzppm", nb._fzppm_batch_numba_kernel),
        ("qckxyz_needs_fill", nb._qckxyz_needs_fill_numba_kernel),
        ("qckxyz", nb._qckxyz_batch_numba_kernel),
        ("finalize", nb._finalize_tpcore_output_numba_kernel),
    ]
    rows: list[dict[str, Any]] = []
    for name, fn in kernels:
        if not fn.signatures:
            continue
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


def _count_paths(setup: Any, ntracer: int) -> dict[str, Any]:
    nlev, nlat, nlon = setup.cx.shape
    jn = np.empty(nlev, dtype=np.int64)
    js = np.empty_like(jn)
    nb._set_jn_js_numba_kernel(setup.cx, jn, js)
    j1p = 2
    j2p = nlat - 3
    jvan = max(1, nlat // 18)
    x_rows = {"edge": 0, "near_pole": 0, "ppm": 0, "large_courant": 0}
    x_large_cells = {"positive": 0, "negative": 0, "fractional": 0}
    x_flux_sign = {"positive": 0, "nonpositive": 0}
    for level in range(nlev):
        for j in range(j1p, j2p + 1):
            values = setup.cx[level, j]
            if j > int(js[level]) and j < int(jn[level]):
                if j == j1p or j == j2p:
                    x_rows["edge"] += 1
                elif j <= j1p + jvan or j >= j2p - jvan:
                    x_rows["near_pole"] += 1
                else:
                    x_rows["ppm"] += 1
            else:
                x_rows["large_courant"] += 1
                x_large_cells["positive"] += int(np.count_nonzero(values > 1.0))
                x_large_cells["negative"] += int(np.count_nonzero(values < -1.0))
                x_large_cells["fractional"] += int(np.count_nonzero((values >= -1.0) & (values <= 1.0)))
            x_flux_sign["positive"] += int(np.count_nonzero(values > 0.0))
            x_flux_sign["nonpositive"] += int(np.count_nonzero(values <= 0.0))

    y_values = setup.cy[:, j1p : j2p + 2, :]
    z_values = setup.vertical_mass_flux_hpa[:, np.r_[0, np.arange(2, nlat - 2), nlat - 1], :]
    y_positive = int(np.count_nonzero(y_values > 0.0))
    y_total = int(y_values.size)
    z_positive = int(np.count_nonzero(z_values[:-1] > 0.0))
    z_total = int(z_values[:-1].size)
    x_total_rows = sum(x_rows.values())
    return {
        "shape": {"levels": nlev, "latitudes": nlat, "longitudes": nlon, "tracers": ntracer},
        "xtp": {
            "rows": x_rows,
            "row_percent": {name: value / x_total_rows * 100.0 for name, value in x_rows.items()},
            "large_courant_cells": x_large_cells,
            "flux_sign_cells": x_flux_sign,
            "ppm_limiter_evaluations": x_rows["ppm"] * nlon * ntracer,
        },
        "ytp": {
            "positive_flux_cells": y_positive,
            "nonpositive_flux_cells": y_total - y_positive,
            "limiter_evaluations": nlev * nlon * (nlat - 2) * ntracer,
        },
        "fzppm": {
            "processed_latitude_rows": nlat - 2,
            "skipped_latitude_rows": 2,
            "positive_flux_interfaces": z_positive,
            "nonpositive_flux_interfaces": z_total - z_positive,
            "edge_limiter_evaluations": (nlat - 2) * nlon * 4 * ntracer,
            "interior_limiter_evaluations": (nlat - 2) * nlon * max(nlev - 4, 0) * ntracer,
        },
    }


def _run_perf(args: argparse.Namespace, output_dir: Path) -> dict[str, Path]:
    common = [
        sys.executable,
        str(Path(__file__).with_name("benchmark_tpcore_scaling.py")),
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
    return run_perf_bundle(
        command=common,
        output_dir=output_dir,
        stem=f"tpcore_{args.tracers}",
        delay_ms=args.perf_delay_ms,
        env=_profile_env(),
    )


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

        summary = parse_perf_stat_summary(stat_path)
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


def _profile_env() -> dict[str, str]:
    return profile_environment(numba_thread_vars=("WOMBAT_NUMBA_THREADS",))


def _write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    benchmark: dict[str, str],
    staged_rows: list[dict[str, float]],
    codegen_rows: list[dict[str, Any]],
    perf_outputs: dict[str, Path],
    stage_perf_rows: list[dict[str, str]],
    path_census: Path | None,
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

    if path_census is not None:
        lines.extend(["", "## Path Census", "", f"- `{path_census}`"])

    lines.append("")
    path.write_text("\n".join(lines))


def _stage_means(rows: list[dict[str, float]]) -> dict[str, float]:
    names = rows[0].keys()
    return {name: sum(row[name] for row in rows) / float(len(rows)) for name in names}


if __name__ == "__main__":
    raise SystemExit(main())
