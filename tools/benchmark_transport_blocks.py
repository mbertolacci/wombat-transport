from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from numba import set_num_threads

from _scaling_support import positive_int
from benchmark_tpcore_scaling import _build_synthetic_tpcore_inputs
from benchmark_vdiff_scaling import _build_synthetic_vdiff_inputs
from benchmark_convection_scaling import _build_synthetic_convection_inputs
from wombat_transport.transport._numba_transport import apply_numba_transport
from wombat_transport.transport._numba_transport import make_numba_transport_workspace
from wombat_transport.transport.convection import G0_100, run_cloud_convection_one_step
from wombat_transport.transport.pbl import LATVAP_J_PER_KG, run_vdiffdr_one_step
from wombat_transport.transport.pbl._numba_transport import prepare_vdiff_zero_flux_block_plan
from wombat_transport.transport.tpcore import run_tpcore_one_step_with_setup, setup_tpcore_terms
from wombat_transport.transport.tpcore._numba_transport import load_tracer_block_workspace
from wombat_transport.transport.tpcore._numba_transport import prepare_tpcore_block_plan


FIELDS = (
    "tracer_count",
    "mode",
    "lane_width",
    "workers",
    "best_apply_s",
    "mean_apply_s",
    "plan_s",
    "best_total_s",
    "speedup_vs_fused",
    "array_equal",
    "max_abs_error",
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    os.environ["WOMBAT_NUMBA_THREADS"] = str(args.workers)
    set_num_threads(args.workers)
    rows = []
    for ntracer in args.counts:
        tpcore = _build_synthetic_tpcore_inputs(args.run_config, ntracer, dt_s=args.dt_s)
        vdiff = _build_synthetic_vdiff_inputs(
            args.run_config,
            ntracer,
            dt_s=args.dt_s,
            surface_flux_kg_m2_s=args.surface_flux_kg_m2_s,
        )
        convection = _build_synthetic_convection_inputs(args.run_config, ntracer, dt_s=args.dt_s)
        setup = setup_tpcore_terms(
            p1_hpa=tpcore.p1_hpa,
            p2_hpa=tpcore.p2_hpa,
            u_m_s=tpcore.u_m_s,
            v_m_s=tpcore.v_m_s,
            area_m2=tpcore.area_m2,
            hyai_hpa=tpcore.hyai_hpa,
            hybi=tpcore.hybi,
            lat_deg=tpcore.lat_deg,
            dt_s=tpcore.dt_s,
        )

        def fused_chain() -> np.ndarray:
            after_tpcore = run_tpcore_one_step_with_setup(
                tracer_conc=tpcore.tracer_conc,
                setup=setup,
                area_m2=tpcore.area_m2,
                validate_branches=False,
                reuse_output=True,
            ).tracer_conc_after
            after_vdiff = run_vdiffdr_one_step(
                tracer_conc=after_tpcore,
                u_m_s=vdiff.u_m_s,
                v_m_s=vdiff.v_m_s,
                temperature_k=vdiff.temperature_k,
                specific_humidity_kg_kg=vdiff.specific_humidity_kg_kg,
                pmid_hpa=vdiff.pmid_hpa,
                pedge_hpa=vdiff.pedge_hpa,
                virtual_temperature_k=vdiff.virtual_temperature_k,
                bxheight_m=vdiff.bxheight_m,
                dry_air_mass_kg=vdiff.dry_air_mass_kg,
                pbl_top_m=vdiff.pbl_top_m,
                hflux_w_m2=vdiff.hflux_w_m2,
                eflux_w_m2=vdiff.eflux_w_m2,
                ustar_m_s=vdiff.ustar_m_s,
                area_m2=vdiff.area_m2,
                dt_s=vdiff.dt_s,
                surface_flux_kg_m2_s=vdiff.surface_flux_kg_m2_s,
                diagnostics=False,
                reuse_output=True,
            ).tracer_conc
            if not args.include_convection:
                return after_vdiff
            return run_cloud_convection_one_step(
                tracer_conc=after_vdiff,
                cmfmc_kg_m2_s=convection.cmfmc_kg_m2_s,
                dtrain_kg_m2_s=convection.dtrain_kg_m2_s,
                dqrcu_kg_kg_s=convection.dqrcu_kg_kg_s,
                reevapcn_kg_kg_s=convection.reevapcn_kg_kg_s,
                delp_dry_hpa=convection.delp_dry_hpa,
                delp_hpa=convection.delp_hpa,
                area_m2=convection.area_m2,
                bxheight_m=convection.bxheight_m,
                pficu_kg_m2_s=convection.pficu_kg_m2_s,
                pflcu_kg_m2_s=convection.pflcu_kg_m2_s,
                temperature_k=convection.temperature_k,
                precccon_mm_day=convection.precccon_mm_day,
                dt_s=convection.dt_s,
                reconstruct_conv_precip_flux=convection.reconstruct_conv_precip_flux,
                diagnostics=False,
                reuse_output=True,
            ).tracer_conc

        fused_times, reference = _time(fused_chain, args.warmup, args.repeat)
        fused_best = min(fused_times)
        rows.append(_row(ntracer, "fused", 0, args.workers, fused_times, 0.0, fused_best, reference, reference))

        for lane_width in args.lanes:
            transport_workspace = make_numba_transport_workspace(
                tpcore.tracer_conc.shape, lane_width, args.workers
            )
            workspace = transport_workspace.tpcore
            def prepare_plans():
                tpcore_plan = prepare_tpcore_block_plan(setup=setup, area_m2=tpcore.area_m2)
                vdiff_plan = prepare_vdiff_zero_flux_block_plan(
                    u_top=vdiff.u_m_s,
                    v_top=vdiff.v_m_s,
                    temperature_top=vdiff.temperature_k,
                    sphu_top=vdiff.specific_humidity_kg_kg,
                    pmid_hpa=vdiff.pmid_hpa,
                    pint_hpa=vdiff.pedge_hpa,
                    virtual_temperature_top=vdiff.virtual_temperature_k,
                    bxheight_top=vdiff.bxheight_m,
                    dry_mass_top=vdiff.dry_air_mass_kg,
                    pblh_m=vdiff.pbl_top_m,
                    hflux_w_m2=vdiff.hflux_w_m2,
                    water_flux_kg_m2_s=vdiff.eflux_w_m2 / LATVAP_J_PER_KG,
                    ustar_m_s=vdiff.ustar_m_s,
                    area_m2=vdiff.area_m2,
                    dt_s=vdiff.dt_s,
                    workers=args.workers,
                    workspace=transport_workspace.vdiff_plan,
                )
                return tpcore_plan, vdiff_plan

            for _ in range(args.warmup):
                tpcore_plan, vdiff_plan = prepare_plans()
            plan_times = []
            for _ in range(args.repeat):
                plan_start = time.perf_counter()
                tpcore_plan, vdiff_plan = prepare_plans()
                plan_times.append(time.perf_counter() - plan_start)
            plan_s = min(plan_times)

            def load() -> None:
                load_tracer_block_workspace(tpcore.tracer_conc, workspace)

            if args.include_convection:
                def numba_transport(execution: str) -> np.ndarray:
                    apply_numba_transport(
                        tpcore_plan=tpcore_plan,
                        vdiff_plan=vdiff_plan,
                        workspace=transport_workspace,
                        surface_flux_kg_m2_s=vdiff.surface_flux_kg_m2_s,
                        cmfmc=convection.cmfmc_kg_m2_s,
                        dtrain=convection.dtrain_kg_m2_s,
                        delp_hpa=convection.delp_hpa,
                        delp_dry=convection.delp_dry_hpa,
                        bmass=convection.delp_dry_hpa * G0_100,
                        dqrcu=convection.dqrcu_kg_kg_s,
                        reevapcn=convection.reevapcn_kg_kg_s,
                        reconstruct_conv_precip_flux=convection.reconstruct_conv_precip_flux,
                        internal_steps=max(int(convection.dt_s) // 300, 1),
                        internal_dt_s=(
                            convection.dt_s / max(int(convection.dt_s) // 300, 1)
                        ),
                        execution=execution,
                    )
                    return workspace.blocks[0].q

                for execution in ("serial", "spatial", "blocks"):
                    transport_times, _ = _time_preloaded(
                        load,
                        lambda execution=execution: numba_transport(execution),
                        args.warmup,
                        args.repeat,
                    )
                    transport_actual = _unpack_q(workspace)
                    rows.append(
                        _row(
                            ntracer,
                            f"{execution}-numba-transport",
                            lane_width,
                            args.workers,
                            transport_times,
                            plan_s,
                            fused_best,
                            transport_actual,
                            reference,
                        )
                    )

    output = args.output.open("w", newline="", encoding="utf-8") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            output.close()
    return 0


def _time(call, warmup: int, repeat: int) -> tuple[list[float], np.ndarray]:
    result = None
    for _ in range(warmup):
        result = call()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = call()
        times.append(time.perf_counter() - start)
    return times, result


def _time_preloaded(load, call, warmup: int, repeat: int) -> tuple[list[float], np.ndarray]:
    result = None
    for _ in range(warmup):
        load()
        result = call()
    times = []
    for _ in range(repeat):
        load()
        start = time.perf_counter()
        result = call()
        times.append(time.perf_counter() - start)
    return times, result


def _unpack_q(workspace) -> np.ndarray:
    nlev, nlat, nlon, ntracer = workspace.tracer_shape
    output = np.empty((nlev, nlat, nlon, ntracer), dtype=np.float64)
    for index, block in enumerate(workspace.blocks):
        start = index * workspace.lane_width
        stop = min(start + workspace.lane_width, ntracer)
        output[:, :, :, start:stop] = block.q[:, :, :, : stop - start]
    return output


def _row(ntracer, mode, lane, workers, times, plan_s, fused_best, actual, reference):
    best = min(times)
    total = best + plan_s
    return {
        "tracer_count": ntracer,
        "mode": mode,
        "lane_width": lane,
        "workers": workers,
        "best_apply_s": f"{best:.9f}",
        "mean_apply_s": f"{np.mean(times):.9f}",
        "plan_s": f"{plan_s:.9f}",
        "best_total_s": f"{total:.9f}",
        "speedup_vs_fused": f"{fused_best / total:.6f}",
        "array_equal": str(bool(np.array_equal(actual, reference))).lower(),
        "max_abs_error": f"{np.max(np.abs(actual - reference)):.16g}",
    }


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Benchmark persistent TPCORE-to-VDIFF tracer blocks.")
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--counts", type=positive_int, nargs="+", default=[64, 96, 192])
    parser.add_argument("--lanes", type=positive_int, nargs="+", default=[8, 16])
    parser.add_argument("--workers", type=positive_int, default=8)
    parser.add_argument("--repeat", type=positive_int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dt-s", type=float, default=600.0)
    parser.add_argument("--surface-flux-kg-m2-s", type=float, default=0.0)
    parser.add_argument("--include-convection", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
