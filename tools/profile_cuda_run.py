"""Profile one ordinary resident CUDA run with host and CUDA-event regions."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import wombat_transport.runner as runner_module
import wombat_transport.cuda.history as cuda_history_module
from wombat_transport.cuda.executor import CudaRunExecutor
from wombat_transport.cuda.forcing import CudaForcingChunks
from wombat_transport.cuda.history import CudaHistoryAverageMaterializer
from wombat_transport.cuda.preparation import CudaPlanPreparation
from wombat_transport.cuda.runtime import CudaRuntime
from wombat_transport.emissions import EmissionsOperator
from wombat_transport.obsoperator.manager import ObsOperatorManager
from wombat_transport.obsoperator.sampling_cuda import CudaObsSampler
from wombat_transport.obsoperator.writer import _ObsOperatorNetCDFWriter
from wombat_transport.output import (
    HistoryOutputManager,
    _AverageCollection,
    _StreamingSpeciesConcFile,
)
from wombat_transport.run_config import RunConfig, load_run_config
from wombat_transport.transport.convection._cuda import (
    CudaConvectionExecutor,
)
from wombat_transport.transport.forcing import TransportForcingProvider
from wombat_transport.transport.pbl._cuda import CudaVdiffExecutor
from wombat_transport.transport.tpcore._cuda import CudaTpcoreExecutor


class RunProfiler:
    """Collect nested host timings and asynchronous CUDA event pairs."""

    def __init__(self, cupy: Any, *, nvtx: bool) -> None:
        self.cupy = cupy
        self.nvtx = nvtx
        self.capturing = False
        self.host_seconds: defaultdict[str, float] = defaultdict(float)
        self.host_counts: defaultdict[str, int] = defaultdict(int)
        self.transfer_bytes: defaultdict[str, int] = defaultdict(int)
        self.transfer_records: list[dict[str, Any]] = []
        self.host_stack: list[str] = []
        self.device_events: defaultdict[str, list[tuple[Any, Any]]] = (
            defaultdict(list)
        )
        self.kernel_attributes: dict[str, dict[str, Any]] = {}

    def range(self, label: str) -> Any:
        if not self.nvtx or not self.capturing:
            return nullcontext()
        from cupyx.profiler import time_range

        return time_range(label)

    def timed_method(
        self,
        cls: type[Any],
        name: str,
        label: str,
        *,
        device: bool = False,
        byte_arg: int | None = None,
        transfer_kind: str | None = None,
    ) -> None:
        original = getattr(cls, name)

        def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if not self.capturing:
                return original(instance, *args, **kwargs)
            parent = self.host_stack[-1] if self.host_stack else None
            start_wall = perf_counter()
            start_event = None
            end_event = None
            if device:
                start_event = self.cupy.cuda.Event()
                end_event = self.cupy.cuda.Event()
                start_event.record()
            self.host_stack.append(label)
            try:
                with self.range(label):
                    result = original(instance, *args, **kwargs)
            finally:
                self.host_stack.pop()
            elapsed_ms = (perf_counter() - start_wall) * 1000.0
            if end_event is not None:
                end_event.record()
                self.device_events[label].append((start_event, end_event))
            self.host_seconds[label] += elapsed_ms / 1000.0
            self.host_counts[label] += 1
            if byte_arg is not None and len(args) > byte_arg:
                byte_count = int(args[byte_arg].nbytes)
                self.transfer_bytes[label] += byte_count
                self.transfer_records.append(
                    {
                        "kind": transfer_kind,
                        "bytes": byte_count,
                        "host_ms": elapsed_ms,
                        "parent": parent,
                    }
                )
            return result

        setattr(cls, name, wrapped)

    def timed_runner_function(self, name: str, label: str) -> None:
        self.timed_function(runner_module, name, label)

    def timed_function(
        self,
        owner: Any,
        name: str,
        label: str,
        *,
        device: bool = False,
    ) -> None:
        original = getattr(owner, name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if not self.capturing:
                return original(*args, **kwargs)
            start = perf_counter()
            start_event = None
            end_event = None
            if device:
                start_event = self.cupy.cuda.Event()
                end_event = self.cupy.cuda.Event()
                start_event.record()
            self.host_stack.append(label)
            try:
                with self.range(label):
                    result = original(*args, **kwargs)
            finally:
                self.host_stack.pop()
            if end_event is not None:
                end_event.record()
                self.device_events[label].append((start_event, end_event))
            self.host_seconds[label] += perf_counter() - start
            self.host_counts[label] += 1
            return result

        setattr(owner, name, wrapped)

    def instrument_kernel_attributes(
        self,
        cls: type[Any],
        attributes: tuple[tuple[str, str], ...],
    ) -> None:
        original = cls.__init__
        profiler = self

        class TimedKernel:
            def __init__(self, kernel: Any, label: str) -> None:
                self.kernel = kernel
                self.label = label
                profiler.kernel_attributes[label] = {
                    key: _json_scalar(value)
                    for key, value in kernel.attributes.items()
                }

            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                if not profiler.capturing:
                    return self.kernel(*args, **kwargs)
                start = profiler.cupy.cuda.Event()
                end = profiler.cupy.cuda.Event()
                start.record()
                with profiler.range(self.label):
                    result = self.kernel(*args, **kwargs)
                end.record()
                profiler.device_events[self.label].append((start, end))
                return result

            @property
            def attributes(self) -> Any:
                return self.kernel.attributes

        def wrapped(instance: Any, *args: Any, **kwargs: Any) -> None:
            original(instance, *args, **kwargs)
            for attribute, label in attributes:
                kernel = getattr(instance, attribute)
                setattr(instance, attribute, TimedKernel(kernel, label))

        cls.__init__ = wrapped

    def report_regions(self) -> dict[str, Any]:
        host = [
            {
                "label": label,
                "calls": self.host_counts[label],
                "total_ms": self.host_seconds[label] * 1000.0,
                "bytes": self.transfer_bytes.get(label, 0),
            }
            for label in sorted(self.host_seconds)
        ]
        device = []
        for label in sorted(self.device_events):
            events = self.device_events[label]
            total_ms = sum(
                self.cupy.cuda.get_elapsed_time(start, end)
                for start, end in events
            )
            device.append(
                {
                    "label": label,
                    "calls": len(events),
                    "total_ms": total_ms,
                }
            )
        return {
            "host": host,
            "device": device,
            "kernel_attributes": self.kernel_attributes,
            "transfers": self.transfer_records,
        }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    os.environ["WOMBAT_BACKEND"] = "cuda"
    os.environ["WOMBAT_CUDA_DTYPE"] = args.dtype
    os.environ["WOMBAT_TRANSPORT_BLOCK_WIDTH"] = str(args.block_width)
    runtime = CudaRuntime(args.device)
    cupy = runtime.array_module
    profiler = RunProfiler(cupy, nvtx=args.nvtx)
    _install_instrumentation(profiler)
    source_config = load_run_config(args.config)
    temporary_root = None
    if args.run_dir is None:
        temporary_root = TemporaryDirectory(prefix="wombat-cuda-profile-run-")
        profile_root = Path(temporary_root.name)
    else:
        profile_root = args.run_dir.resolve()
        profile_root.mkdir(parents=True, exist_ok=True)

    if args.warmup_steps:
        warm_config = _redirect_config(
            source_config,
            profile_root / "warmup",
            name_suffix="profile_warmup",
        )
        warm = runner_module.run_tracer_simulation(
            warm_config,
            max_steps=args.warmup_steps,
        )
        del warm
        gc.collect()
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()

    config = _redirect_config(
        source_config,
        profile_root / "timed",
        name_suffix="profile_timed",
    )
    profiler.capturing = True
    whole_start = cupy.cuda.Event()
    whole_end = cupy.cuda.Event()
    whole_start.record()
    start_wall = perf_counter()
    with profiler.range("run.total"):
        result = runner_module.run_tracer_simulation(
            config,
            max_steps=args.steps,
        )
    whole_end.record()
    whole_end.synchronize()
    wall_seconds = perf_counter() - start_wall
    profiler.capturing = False

    memory_pool = cupy.get_default_memory_pool()
    pinned_pool = cupy.get_default_pinned_memory_pool()
    artifacts = _artifact_summary(config.root)
    report = {
        "metadata": {
            "config": str(args.config.resolve()),
            "dtype": args.dtype,
            "block_width": args.block_width,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "profile_run_root": str(config.root),
            "profile_run_root_temporary": args.run_dir is None,
            "device_id": args.device,
            "device": runtime.device_info.name,
            "compute_capability": runtime.device_info.compute_capability,
            "cupy": cupy.__version__,
            "cuda_driver_version": int(
                cupy.cuda.runtime.driverGetVersion()
            ),
            "cuda_runtime_version": int(
                cupy.cuda.runtime.runtimeGetVersion()
            ),
            "nvtx": args.nvtx,
            "process_cpu_affinity": (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
            "nested_device_regions_overlap_parent_regions": True,
        },
        "summary": {
            "wall_seconds": wall_seconds,
            "device_span_ms": cupy.cuda.get_elapsed_time(
                whole_start,
                whole_end,
            ),
            "final_state_bytes": int(result.state.block_data.nbytes),
            "device_pool_used_bytes": int(memory_pool.used_bytes()),
            "device_pool_total_bytes": int(memory_pool.total_bytes()),
            "pinned_pool_free_blocks": int(pinned_pool.n_free_blocks()),
            "output_artifact_count": artifacts["count"],
            "output_artifact_bytes": artifacts["bytes"],
        },
        "regions": profiler.report_regions(),
        "output_artifacts": artifacts["files"],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{text}\n", encoding="utf-8")
        print(args.output)
    if temporary_root is not None:
        temporary_root.cleanup()
    return 0


def _install_instrumentation(profiler: RunProfiler) -> None:
    for cls, name, label in (
        (CudaRunExecutor, "__init__", "cuda.initialize"),
        (
            TransportForcingProvider,
            "chunks_for_step",
            "host.forcing_select",
        ),
        (
            EmissionsOperator,
            "evaluate_surface_flux",
            "host.emissions",
        ),
        (HistoryOutputManager, "prepare_step", "history.prepare"),
        (HistoryOutputManager, "complete_step", "history.complete"),
        (HistoryOutputManager, "close", "history.close"),
        (_AverageCollection, "_append_average", "history.append_average"),
        (
            _StreamingSpeciesConcFile,
            "append_average",
            "history.netcdf_append",
        ),
        (
            _StreamingSpeciesConcFile,
            "_write_average",
            "history.netcdf_write_fields",
        ),
        (
            _StreamingSpeciesConcFile,
            "_write_preaveraged",
            "history.netcdf_write_fields",
        ),
        (ObsOperatorManager, "sample", "obsoperator.manager_sample"),
        (ObsOperatorManager, "_initialize_for_date", "obsoperator.plan_update"),
        (ObsOperatorManager, "close", "obsoperator.close"),
        (
            _ObsOperatorNetCDFWriter,
            "write_completed",
            "obsoperator.stage_completed",
        ),
        (_ObsOperatorNetCDFWriter, "flush", "obsoperator.netcdf_flush"),
    ):
        profiler.timed_method(cls, name, label)

    for cls, name, label, byte_arg, transfer_kind in (
        (CudaRuntime, "to_device", "cuda.h2d_allocate", 0, "h2d_allocate"),
        (CudaRuntime, "copy_to_device", "cuda.h2d_refresh", 1, "h2d_refresh"),
        (CudaRuntime, "to_host", "cuda.d2h", 0, "d2h"),
        (CudaForcingChunks, "select", "cuda.forcing_select", None, None),
        (
            CudaPlanPreparation,
            "prepare_tpcore_step",
            "cuda.prepare_tpcore",
            None,
            None,
        ),
        (
            CudaPlanPreparation,
            "prepare_vdiff_and_convection",
            "cuda.prepare_vdiff_convection",
            None,
            None,
        ),
        (
            CudaTpcoreExecutor,
            "apply_blocks",
            "cuda.transport_tpcore",
            None,
            None,
        ),
        (
            CudaVdiffExecutor,
            "apply_blocks",
            "cuda.transport_vdiff",
            None,
            None,
        ),
        (
            CudaConvectionExecutor,
            "apply_blocks",
            "cuda.transport_convection",
            None,
            None,
        ),
        (
            CudaObsSampler,
            "sample",
            "cuda.obsoperator_sample",
            None,
            None,
        ),
        (
            CudaObsSampler,
            "sync_to_host",
            "cuda.obsoperator_sync",
            None,
            None,
        ),
        (
            CudaHistoryAverageMaterializer,
            "materialize",
            "cuda.history_materialize",
            None,
            None,
        ),
    ):
        profiler.timed_method(
            cls,
            name,
            label,
            device=True,
            byte_arg=byte_arg,
            transfer_kind=transfer_kind,
        )

    profiler.instrument_kernel_attributes(
        CudaTpcoreExecutor,
        (
            ("_horizontal_poles", "kernel.tpcore_horizontal_poles"),
            (
                "_horizontal_initialize",
                "kernel.tpcore_horizontal_initialize",
            ),
            ("_horizontal", "kernel.tpcore_horizontal"),
            ("_vertical", "kernel.tpcore_vertical"),
        ),
    )
    profiler.instrument_kernel_attributes(
        CudaVdiffExecutor,
        (("_kernel", "kernel.vdiff"),),
    )
    profiler.instrument_kernel_attributes(
        CudaConvectionExecutor,
        (("_kernel", "kernel.convection"),),
    )
    profiler.instrument_kernel_attributes(
        CudaPlanPreparation,
        tuple(
            (f"_{name}", f"kernel.prepare_{name}")
            for name in (
                "prepare_surface_endpoints",
                "average_surface_poles",
                "interpolate_meteorology",
                "tpcore_pressure_delta_terms",
                "tpcore_pressure_delta_sum",
                "tpcore_apply_pressure_fix",
                "tpcore_average_pressure",
                "tpcore_mass_flux",
                "tpcore_divergence_interior",
                "tpcore_divergence_poles",
                "tpcore_sum_vertical",
                "tpcore_pressure_rows",
                "tpcore_meridional_correction",
                "tpcore_zonal_correction",
                "tpcore_apply_correction",
                "tpcore_copy_pressure",
                "tpcore_pressure_terms",
                "tpcore_vertical_flux",
                "tpcore_cross_terms",
                "tpcore_initialize_jn_js",
                "cast_plan",
                "compute_vdiff_start",
                "prepare_vdiff",
                "prepare_convection",
            )
        ),
    )
    profiler.instrument_kernel_attributes(
        CudaObsSampler,
        (("_kernel", "kernel.obsoperator"),),
    )
    profiler.timed_function(
        cuda_history_module,
        "accumulate_history_sums",
        "cuda.history_accumulate",
        device=True,
    )

    for name, label in (
        ("initialize_tracers", "host.initialize_tracers"),
        ("load_transport_grid", "host.load_grid"),
        ("build_tpcore_static_terms", "host.build_static"),
        ("_load_emissions_operator", "host.load_emissions"),
        ("_load_simulation_forcing", "host.initial_forcing"),
    ):
        profiler.timed_runner_function(name, label)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile one ordinary Wombat CUDA run with nested host and "
            "CUDA-event regions."
        )
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--steps", type=_positive_int, default=18)
    parser.add_argument("--warmup-steps", type=_nonnegative_int, default=1)
    parser.add_argument("--block-width", type=_positive_int, default=32)
    parser.add_argument("--device", type=_nonnegative_int, default=0)
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help=(
            "Keep redirected HISTORY, ObsOperator, restart, and metadata "
            "artifacts under this directory; otherwise use a temporary root."
        ),
    )
    return parser.parse_args(argv)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _redirect_config(
    config: RunConfig,
    root: Path,
    *,
    name_suffix: str,
) -> RunConfig:
    """Redirect all profiler-owned writes while preserving input locations."""

    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    outputs = deepcopy(config.outputs)
    if outputs:
        outputs["expid"] = str(root / "history" / "GEOSChem")
        obsoperator = outputs.get("obsoperator")
        if isinstance(obsoperator, dict):
            input_file = obsoperator.get("input_file")
            if input_file is not None:
                input_path = Path(str(input_file))
                if not input_path.is_absolute():
                    input_path = config.root / input_path
                obsoperator["input_file"] = str(input_path.resolve())
            for key in ("output_file", "restart_file"):
                value = obsoperator.get(key)
                if value is not None:
                    obsoperator[key] = str(
                        root / "obsoperator" / Path(str(value)).name
                    )
    emissions = config.emissions
    if isinstance(emissions, str):
        emissions_path = Path(emissions)
        if not emissions_path.is_absolute():
            emissions = str((config.root / emissions_path).resolve())
    return replace(
        config,
        name=f"{config.name}_{name_suffix}",
        root=root,
        source_run_dir=root,
        output_dir=root / "OutputDir",
        emissions=emissions,
        outputs=outputs,
    )


def _artifact_summary(root: Path) -> dict[str, Any]:
    files = []
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        size = path.stat().st_size
        total_bytes += size
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": size,
            }
        )
    return {"count": len(files), "bytes": total_bytes, "files": files}


if __name__ == "__main__":
    raise SystemExit(main())
