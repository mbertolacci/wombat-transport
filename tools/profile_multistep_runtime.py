from __future__ import annotations

import argparse
import cProfile
import csv
import json
import pstats
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

from yaml12 import read_yaml, write_yaml

from wombat_transport import emissions as emissions_mod
from wombat_transport import output as output_mod
from wombat_transport import runner as runner_mod
from wombat_transport.run_config import load_run_config
from wombat_transport.transport import driver as driver_mod


DEFAULT_COUNTS = (1, 24, 96)
DEFAULT_START = "2014-09-01 00:00"
DEFAULT_END = "2014-09-03 00:00"
DEFAULT_SOURCE_CONFIG = Path("validation_runs/cases/residual_24tracer_emissions_2month/wombat/main/run.yml")


@dataclass
class CallTimer:
    calls: int = 0
    total_s: float = 0.0
    child_s: float = 0.0

    @property
    def exclusive_s(self) -> float:
        return self.total_s - self.child_s


@dataclass
class ProfileRow:
    tracer_count: int
    stage: str
    calls: int
    total_s: float
    exclusive_s: float
    percent_total: float


@dataclass
class ProfileResult:
    tracer_count: int
    total_s: float
    transport_steps: int
    emissions_steps: int
    total_emitted_mass_kg: float
    timers: dict[str, CallTimer] = field(default_factory=dict)

    def rows(self) -> list[ProfileRow]:
        rows = []
        for stage, timer in sorted(self.timers.items()):
            rows.append(
                ProfileRow(
                    tracer_count=self.tracer_count,
                    stage=stage,
                    calls=timer.calls,
                    total_s=timer.total_s,
                    exclusive_s=timer.exclusive_s,
                    percent_total=timer.exclusive_s / self.total_s * 100.0 if self.total_s else 0.0,
                )
            )
        return rows


class RuntimeProfiler:
    def __init__(self) -> None:
        self.timers: dict[str, CallTimer] = {}
        self._local = threading.local()
        self._lock = threading.Lock()

    def _stack(self) -> list[tuple[str, float, float]]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    def wrap(self, stage: str, function: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            stack = self._stack()
            stack.append((stage, time.perf_counter(), 0.0))
            try:
                return function(*args, **kwargs)
            finally:
                finished_stage, start, child_s = stack.pop()
                elapsed = time.perf_counter() - start
                with self._lock:
                    timer = self.timers.setdefault(finished_stage, CallTimer())
                    timer.calls += 1
                    timer.total_s += elapsed
                    timer.child_s += child_s
                if stack:
                    parent_stage, parent_start, parent_child = stack.pop()
                    stack.append((parent_stage, parent_start, parent_child + elapsed))

        return wrapper


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    source_config = args.source_config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.warmup_steps:
        warmup_run_dir = _prepare_run_dir(
            source_config,
            output_dir / "warmup" / "wombat",
            tracer_count=1,
            start=args.start,
            end=args.end,
            outputs_enabled=False,
        )
        _run_profiled(warmup_run_dir / "run.yml", tracer_count=1, max_steps=args.warmup_steps, cprofile_path=None)
        shutil.rmtree(warmup_run_dir, ignore_errors=True)

    results: list[ProfileResult] = []
    for count in args.counts:
        run_dir = _prepare_run_dir(
            source_config,
            output_dir / f"tracers_{count:03d}" / "wombat",
            tracer_count=count,
            start=args.start,
            end=args.end,
            outputs_enabled=args.outputs,
            field_offset=args.field_offset,
            species_conc_frequency=args.species_conc_frequency,
            species_conc_duration=args.species_conc_duration,
            output_compression=args.output_compression,
            output_compression_level=args.output_compression_level,
            output_shuffle=args.output_shuffle,
            output_rank4_chunking=tuple(args.output_rank4_chunking) if args.output_rank4_chunking else None,
            output_writer=args.output_writer,
        )
        cprofile_path = output_dir / f"cprofile_{count:03d}.prof" if args.cprofile else None
        result = _run_profiled(run_dir / "run.yml", tracer_count=count, max_steps=None, cprofile_path=cprofile_path)
        results.append(result)

    _write_stage_csv(output_dir / "stage_times.csv", results)
    _write_summary_json(output_dir / "summary.json", results)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile real multi-step Wombat runs by runner stage.")
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("validation_runs/work/profile_multistep_runtime"))
    parser.add_argument("--counts", type=int, nargs="+", default=list(DEFAULT_COUNTS))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--outputs", action="store_true", help="Keep configured HISTORY-like outputs enabled.")
    parser.add_argument(
        "--species-conc-frequency",
        default=None,
        help="Override SpeciesConc collection frequency, e.g. '00000000 010000' for hourly samples.",
    )
    parser.add_argument(
        "--species-conc-duration",
        default=None,
        help="Override SpeciesConc collection duration, e.g. '00000001 000000' for daily files.",
    )
    parser.add_argument(
        "--output-compression",
        choices=("on", "off"),
        default=None,
        help="Override output compression enabled state.",
    )
    parser.add_argument(
        "--output-compression-level",
        type=int,
        default=None,
        help="Override output compression level.",
    )
    parser.add_argument(
        "--output-shuffle",
        choices=("on", "off"),
        default=None,
        help="Override output compression shuffle filter.",
    )
    parser.add_argument(
        "--output-rank4-chunking",
        type=int,
        nargs=4,
        default=None,
        metavar=("TIME", "LEV", "LAT", "LON"),
        help="Override rank-4 output chunks, e.g. 1 47 91 144.",
    )
    parser.add_argument(
        "--output-writer",
        choices=("sync", "threaded"),
        default=None,
        help="Override outputs.writer.",
    )
    parser.add_argument("--cprofile", action="store_true", help="Write one cProfile .prof per tracer count.")
    parser.add_argument(
        "--field-offset",
        type=int,
        default=0,
        help="Offset into source emissions fields when assigning synthetic species. Useful for one-tracer hourly cases.",
    )
    args = parser.parse_args(argv)
    if any(count <= 0 for count in args.counts):
        parser.error("--counts values must be positive")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")
    return args


def _prepare_run_dir(
    source_config_path: Path,
    run_dir: Path,
    *,
    tracer_count: int,
    start: str,
    end: str,
    outputs_enabled: bool,
    field_offset: int = 0,
    species_conc_frequency: str | None = None,
    species_conc_duration: str | None = None,
    output_compression: str | None = None,
    output_compression_level: int | None = None,
    output_shuffle: str | None = None,
    output_rank4_chunking: tuple[int, int, int, int] | None = None,
    output_writer: str | None = None,
) -> Path:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    source_root = source_config_path.parent
    run_config = read_yaml(source_config_path) or {}
    source_species = read_yaml(source_root / str(run_config["species_database"])) or {}
    emissions_ref = run_config["emissions"]
    if not isinstance(emissions_ref, str):
        raise TypeError("source config must reference an emissions YAML file")
    source_emissions = read_yaml(source_root / emissions_ref) or {}

    species = _build_species(source_species, tracer_count)
    emissions = _build_emissions(source_emissions, tuple(species.keys()), field_offset=field_offset)

    write_yaml(species, run_dir / "species_database.yml")
    write_yaml(emissions, run_dir / "emissions.yml")

    run_config["name"] = f"profile_multistep_{tracer_count:03d}_tracers"
    run_config["source_run_dir"] = "."
    run_config["species_database"] = "./species_database.yml"
    run_config["initial_restart"] = None
    run_config["grid_template"] = "../../../../../base/Restarts/GEOSChem.Restart.20140901_0000z.nc4"
    run_config["output_dir"] = "./OutputDir"
    run_config["simulation"]["start"] = start
    run_config["simulation"]["end"] = end
    run_config["meteorology"]["root"] = "../../../../../ExtData/GEOS_2x2.5/MERRA2"
    run_config["emissions"] = "emissions.yml"
    run_config["logging"] = {"level": "warning"}
    run_config["diagnostics"] = {}
    run_config["comparison"] = {}
    run_config["validation"] = {}
    if not outputs_enabled:
        run_config["outputs"] = {}
    elif (
        species_conc_frequency
        or species_conc_duration
        or output_compression
        or output_compression_level is not None
        or output_shuffle
        or output_rank4_chunking is not None
        or output_writer
    ):
        outputs = run_config.get("outputs", {})
        if output_writer:
            outputs["writer"] = output_writer
            run_config["outputs"] = outputs
        if output_compression or output_compression_level is not None or output_shuffle:
            compression = dict(outputs.get("compression", {}))
            if output_compression:
                compression["enabled"] = output_compression == "on"
            if output_compression_level is not None:
                compression["level"] = output_compression_level
            if output_shuffle:
                compression["shuffle"] = output_shuffle == "on"
            outputs["compression"] = compression
            run_config["outputs"] = outputs
        if output_rank4_chunking is not None:
            chunking = dict(outputs.get("chunking", {}))
            chunking["rank4"] = list(output_rank4_chunking)
            outputs["chunking"] = chunking
            run_config["outputs"] = outputs
        for name, collection in run_config.get("outputs", {}).get("collections", {}).items():
            if str(name).startswith("SpeciesConc"):
                if species_conc_frequency:
                    collection["frequency"] = species_conc_frequency
                if species_conc_duration:
                    collection["duration"] = species_conc_duration

    write_yaml(run_config, run_dir / "run.yml")
    return run_dir


def _build_species(source_species: dict[str, object], tracer_count: int) -> dict[str, object]:
    source_names = list(source_species.keys())
    if not source_names:
        raise ValueError("source species database is empty")
    species: dict[str, object] = {}
    for index in range(tracer_count):
        source_name = source_names[index % len(source_names)]
        item = dict(source_species[source_name])  # type: ignore[arg-type]
        item["FullName"] = f"profile_tracer_{index + 1:03d}"
        species[f"p{index + 1:03d}"] = item
    return species


def _build_emissions(
    source_emissions: dict[str, object],
    species_names: tuple[str, ...],
    *,
    field_offset: int = 0,
) -> dict[str, object]:
    fields = list(source_emissions.get("fields", ()))  # type: ignore[union-attr]
    if not fields:
        raise ValueError("source emissions config has no fields")
    configured: list[dict[str, object]] = []
    for index, species_name in enumerate(species_names):
        template = dict(fields[(index + field_offset) % len(fields)])
        template["name"] = f"profile_{index + 1:03d}_{template['name']}"
        template["species"] = species_name
        configured.append(template)
    return {
        "unit_conversion": source_emissions.get("unit_conversion", "none"),
        "missing_species": source_emissions.get("missing_species", "zero"),
        "scales": source_emissions.get("scales", {}),
        "fields": configured,
    }


def _run_profiled(
    run_config_path: Path,
    *,
    tracer_count: int,
    max_steps: int | None,
    cprofile_path: Path | None,
) -> ProfileResult:
    config = load_run_config(run_config_path)
    profiler = RuntimeProfiler()

    def run() -> ProfileResult:
        with _patched_runtime(profiler):
            start = time.perf_counter()
            result = runner_mod.run_tracer_simulation(config, max_steps=max_steps)
            total_s = time.perf_counter() - start
        return ProfileResult(
            tracer_count=tracer_count,
            total_s=total_s,
            transport_steps=result.transport_steps,
            emissions_steps=result.emissions_steps,
            total_emitted_mass_kg=result.total_emitted_mass,
            timers=profiler.timers,
        )

    if cprofile_path is None:
        return run()

    cprofile_path.parent.mkdir(parents=True, exist_ok=True)
    profile = cProfile.Profile()
    result = profile.runcall(run)
    profile.dump_stats(str(cprofile_path))
    with (cprofile_path.with_suffix(".txt")).open("w", encoding="utf-8") as handle:
        stats = pstats.Stats(profile, stream=handle).strip_dirs().sort_stats("cumtime")
        stats.print_stats(80)
    return result


@contextmanager
def _patched_runtime(profiler: RuntimeProfiler) -> Iterator[None]:
    originals: list[tuple[object, str, Callable]] = [
        (runner_mod, "_load_simulation_forcing", runner_mod._load_simulation_forcing),
        (emissions_mod.EmissionsOperator, "evaluate", emissions_mod.EmissionsOperator.evaluate),
        (emissions_mod.EmissionsOperator, "evaluate_surface_flux", emissions_mod.EmissionsOperator.evaluate_surface_flux),
        (emissions_mod.EmissionsOperator, "_read_configured_array", emissions_mod.EmissionsOperator._read_configured_array),
        (runner_mod, "emitted_mass_by_tracer_for_step", runner_mod.emitted_mass_by_tracer_for_step),
        (driver_mod, "run_transport_one_step", driver_mod.run_transport_one_step),
        (driver_mod, "setup_tpcore_terms", driver_mod.setup_tpcore_terms),
        (driver_mod, "run_tpcore_one_step_with_setup", driver_mod.run_tpcore_one_step_with_setup),
        (driver_mod, "run_vdiffdr_one_step", driver_mod.run_vdiffdr_one_step),
        (driver_mod, "run_cloud_convection_one_step", driver_mod.run_cloud_convection_one_step),
        (output_mod.HistoryOutputManager, "record_step", output_mod.HistoryOutputManager.record_step),
        (output_mod.HistoryOutputManager, "close", output_mod.HistoryOutputManager.close),
        (output_mod, "write_species_conc_collection", output_mod.write_species_conc_collection),
        (output_mod, "write_restart_collection", output_mod.write_restart_collection),
        (output_mod._StreamingSpeciesConcFile, "_open", output_mod._StreamingSpeciesConcFile._open),
        (output_mod._StreamingSpeciesConcFile, "append_average", output_mod._StreamingSpeciesConcFile.append_average),
        (output_mod._StreamingSpeciesConcFile, "close", output_mod._StreamingSpeciesConcFile.close),
    ]
    stage_names = {
        "_load_simulation_forcing": "met_forcing",
        "evaluate": "emissions_evaluate",
        "evaluate_surface_flux": "emissions_evaluate",
        "_read_configured_array": "emissions_read_array",
        "emitted_mass_by_tracer_for_step": "emissions_mass_sum",
        "run_transport_one_step": "transport_total",
        "setup_tpcore_terms": "transport_setup_tpcore",
        "run_tpcore_one_step_with_setup": "transport_tpcore",
        "run_vdiffdr_one_step": "transport_vdiff",
        "run_cloud_convection_one_step": "transport_convection",
        "record_step": "output_record_step",
        "close": "output_close",
        "write_species_conc_collection": "output_write_species_conc",
        "write_restart_collection": "output_write_restart",
        "_open": "output_open_species_conc",
        "append_average": "output_append_species_conc",
    }
    try:
        for obj, name, original in originals:
            if obj is output_mod._StreamingSpeciesConcFile and name == "close":
                stage = "output_close_species_conc"
            else:
                stage = stage_names[name]
            setattr(obj, name, profiler.wrap(stage, original))
        yield
    finally:
        for obj, name, original in originals:
            setattr(obj, name, original)


def _write_stage_csv(path: Path, results: list[ProfileResult]) -> None:
    fields = ["tracer_count", "stage", "calls", "total_s", "exclusive_s", "percent_total"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "tracer_count": result.tracer_count,
                    "stage": "run_total",
                    "calls": 1,
                    "total_s": f"{result.total_s:.9f}",
                    "exclusive_s": f"{result.total_s:.9f}",
                    "percent_total": "100.000000",
                }
            )
            for row in result.rows():
                writer.writerow(
                    {
                        "tracer_count": row.tracer_count,
                        "stage": row.stage,
                        "calls": row.calls,
                        "total_s": f"{row.total_s:.9f}",
                        "exclusive_s": f"{row.exclusive_s:.9f}",
                        "percent_total": f"{row.percent_total:.6f}",
                    }
                )


def _write_summary_json(path: Path, results: list[ProfileResult]) -> None:
    payload = []
    for result in results:
        payload.append(
            {
                "tracer_count": result.tracer_count,
                "total_s": result.total_s,
                "transport_steps": result.transport_steps,
                "emissions_steps": result.emissions_steps,
                "total_emitted_mass_kg": result.total_emitted_mass_kg,
                "timers": {name: asdict(timer) | {"exclusive_s": timer.exclusive_s} for name, timer in result.timers.items()},
            }
        )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
