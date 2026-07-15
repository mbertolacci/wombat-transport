from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable


RUN_FIELDS = (
    "repetition",
    "total_tracers",
    "processes",
    "threads_per_process",
    "tracers_per_process",
    "writer",
    "status",
    "transport_steps",
    "emissions_steps",
    "effective_wall_s",
    "effective_s_per_step",
    "aggregate_tracer_steps_per_s",
    "threaded_speedup_same_repeat",
    "rank_spread_percent",
    "slowest_rank",
    "critical_met_s",
    "critical_emissions_s",
    "critical_transport_s",
    "critical_obsoperator_s",
    "critical_output_foreground_s",
    "critical_output_worker_s",
    "total_peak_rss_gib",
    "total_output_gib",
    "case_dir",
    "reason",
)

MODE_FIELDS = (
    "total_tracers",
    "processes",
    "threads_per_process",
    "tracers_per_process",
    "writer",
    "completed_repetitions",
    "mean_effective_wall_s",
    "stdev_effective_wall_s",
    "mean_effective_s_per_step",
    "aggregate_tracer_steps_per_s",
    "threaded_speedup",
    "mean_rank_spread_percent",
    "mean_critical_met_s",
    "mean_critical_emissions_s",
    "mean_critical_transport_s",
    "mean_critical_obsoperator_s",
    "mean_critical_output_foreground_s",
    "mean_critical_output_worker_s",
    "mean_total_peak_rss_gib",
    "mean_total_output_gib",
)


@dataclass(frozen=True)
class CaseSpec:
    repetition: int
    total_tracers: int
    processes: int
    threads_per_process: int
    tracers_per_process: int
    writer: str
    case_dir: Path


@dataclass(frozen=True)
class RunRow:
    spec: CaseSpec
    status: str
    transport_steps: int | None = None
    emissions_steps: int | None = None
    effective_wall_s: float | None = None
    effective_s_per_step: float | None = None
    aggregate_tracer_steps_per_s: float | None = None
    threaded_speedup_same_repeat: float | None = None
    rank_spread_percent: float | None = None
    slowest_rank: int | None = None
    critical_met_s: float | None = None
    critical_emissions_s: float | None = None
    critical_transport_s: float | None = None
    critical_obsoperator_s: float | None = None
    critical_output_foreground_s: float | None = None
    critical_output_worker_s: float | None = None
    total_peak_rss_gib: float | None = None
    total_output_gib: float | None = None
    reason: str = ""

    def as_csv_row(self, root: Path) -> dict[str, str]:
        spec = self.spec
        return {
            "repetition": str(spec.repetition),
            "total_tracers": str(spec.total_tracers),
            "processes": str(spec.processes),
            "threads_per_process": str(spec.threads_per_process),
            "tracers_per_process": str(spec.tracers_per_process),
            "writer": spec.writer,
            "status": self.status,
            "transport_steps": _format_int(self.transport_steps),
            "emissions_steps": _format_int(self.emissions_steps),
            "effective_wall_s": _format_float(self.effective_wall_s),
            "effective_s_per_step": _format_float(self.effective_s_per_step),
            "aggregate_tracer_steps_per_s": _format_float(self.aggregate_tracer_steps_per_s),
            "threaded_speedup_same_repeat": _format_float(self.threaded_speedup_same_repeat),
            "rank_spread_percent": _format_float(self.rank_spread_percent),
            "slowest_rank": _format_int(self.slowest_rank),
            "critical_met_s": _format_float(self.critical_met_s),
            "critical_emissions_s": _format_float(self.critical_emissions_s),
            "critical_transport_s": _format_float(self.critical_transport_s),
            "critical_obsoperator_s": _format_float(self.critical_obsoperator_s),
            "critical_output_foreground_s": _format_float(self.critical_output_foreground_s),
            "critical_output_worker_s": _format_float(self.critical_output_worker_s),
            "total_peak_rss_gib": _format_float(self.total_peak_rss_gib),
            "total_output_gib": _format_float(self.total_output_gib),
            "case_dir": str(spec.case_dir.relative_to(root)),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ModeRow:
    total_tracers: int
    processes: int
    threads_per_process: int
    tracers_per_process: int
    writer: str
    completed_repetitions: int
    mean_effective_wall_s: float
    stdev_effective_wall_s: float
    mean_effective_s_per_step: float
    aggregate_tracer_steps_per_s: float
    threaded_speedup: float | None
    mean_rank_spread_percent: float
    mean_critical_met_s: float
    mean_critical_emissions_s: float
    mean_critical_transport_s: float
    mean_critical_obsoperator_s: float
    mean_critical_output_foreground_s: float
    mean_critical_output_worker_s: float
    mean_total_peak_rss_gib: float
    mean_total_output_gib: float

    def as_csv_row(self) -> dict[str, str]:
        return {
            field: str(getattr(self, field)) if field in {"writer", "completed_repetitions"} else _format_float(getattr(self, field))
            for field in MODE_FIELDS
        } | {
            "total_tracers": str(self.total_tracers),
            "processes": str(self.processes),
            "threads_per_process": str(self.threads_per_process),
            "tracers_per_process": str(self.tracers_per_process),
        }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.results_dir.resolve()
    specs = _read_manifest(root)
    runs = [_summarize_case(root, spec) for spec in specs]
    runs = _annotate_run_speedups(runs)
    modes = _aggregate_modes(runs)

    runs_output = args.runs_output or root / "summary_runs.csv"
    modes_output = args.modes_output or root / "summary_modes.csv"
    _write_runs(runs, root, runs_output)
    _write_modes(modes, modes_output)
    _print_modes(modes, runs_output, modes_output)

    incomplete = sum(row.status != "completed" for row in runs)
    if incomplete:
        print(f"Warning: {incomplete} of {len(runs)} cases are incomplete; they remain in the run CSV.", file=sys.stderr)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize sync/threaded production-like multistep I/O benchmarks.")
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--runs-output", type=Path)
    parser.add_argument("--modes-output", type=Path)
    return parser.parse_args(argv)


def _read_manifest(root: Path) -> list[CaseSpec]:
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        raise SystemExit(f"Manifest not found: {manifest}")
    specs: list[CaseSpec] = []
    with manifest.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            try:
                specs.append(
                    CaseSpec(
                        repetition=int(record["repetition"]),
                        total_tracers=int(record["total_tracers"]),
                        processes=int(record["processes"]),
                        threads_per_process=int(record["threads_per_process"]),
                        tracers_per_process=int(record["tracers_per_process"]),
                        writer=record["writer"],
                        case_dir=root / record["case_dir"],
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(f"Invalid record in {manifest}: {record}") from exc
    if not specs:
        raise SystemExit(f"Manifest is empty: {manifest}")
    return sorted(specs, key=lambda spec: (spec.total_tracers, spec.repetition, spec.writer))


def _summarize_case(root: Path, spec: CaseSpec) -> RunRow:
    if not (spec.case_dir / ".complete").is_file():
        return RunRow(spec=spec, status="incomplete", reason="completion marker is missing")

    rank_results: list[dict[str, Any]] = []
    for rank in range(spec.processes):
        path = spec.case_dir / f"rank_{rank}" / "summary.json"
        if not path.is_file():
            return RunRow(spec=spec, status="incomplete", reason=f"missing rank_{rank}/summary.json")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return RunRow(spec=spec, status="invalid", reason=f"cannot read rank {rank}: {exc}")
        if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
            return RunRow(spec=spec, status="invalid", reason=f"rank {rank} summary must contain one result")
        result = payload[0]
        if int(result.get("tracer_count", -1)) != spec.tracers_per_process:
            return RunRow(spec=spec, status="invalid", reason=f"rank {rank} tracer count does not match manifest")
        rank_results.append(result)

    try:
        walls = [_finite_float(result["total_s"], "total_s") for result in rank_results]
        step_counts = [int(result["transport_steps"]) for result in rank_results]
        emissions_counts = [int(result["emissions_steps"]) for result in rank_results]
        if len(set(step_counts)) != 1 or step_counts[0] <= 0:
            raise ValueError(f"inconsistent transport step counts: {step_counts}")
        if len(set(emissions_counts)) != 1:
            raise ValueError(f"inconsistent emissions step counts: {emissions_counts}")
        slowest_rank = max(range(spec.processes), key=walls.__getitem__)
        critical = rank_results[slowest_rank]
        wall = walls[slowest_rank]
        steps = step_counts[0]
        return RunRow(
            spec=spec,
            status="completed",
            transport_steps=steps,
            emissions_steps=emissions_counts[0],
            effective_wall_s=wall,
            effective_s_per_step=wall / steps,
            aggregate_tracer_steps_per_s=spec.total_tracers * steps / wall,
            rank_spread_percent=100.0 * (max(walls) - min(walls)) / max(walls),
            slowest_rank=slowest_rank,
            critical_met_s=_timer_value(critical, "met_forcing", "total_s"),
            critical_emissions_s=_emissions_time(critical),
            critical_transport_s=_timer_value(critical, "transport_total", "total_s"),
            critical_obsoperator_s=_obsoperator_time(critical),
            critical_output_foreground_s=_output_foreground_time(critical),
            critical_output_worker_s=_output_worker_time(critical),
            total_peak_rss_gib=sum(_finite_float(result["peak_rss_mib"], "peak_rss_mib") for result in rank_results)
            / 1024.0,
            total_output_gib=sum(int(result.get("output_bytes", 0)) for result in rank_results) / 1024.0**3,
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return RunRow(spec=spec, status="invalid", reason=str(exc))


def _timer_value(result: dict[str, Any], name: str, field: str) -> float:
    timer = result.get("timers", {}).get(name)
    if timer is None:
        return 0.0
    return _finite_float(timer[field], f"{name}.{field}")


def _emissions_time(result: dict[str, Any]) -> float:
    names = ("emissions_evaluate", "emissions_read_array", "emissions_mass_sum")
    return sum(_timer_value(result, name, "exclusive_s") for name in names)


def _output_foreground_time(result: dict[str, Any]) -> float:
    return _timer_value(result, "output_record_step", "total_s") + _timer_value(result, "output_close", "total_s")


def _obsoperator_time(result: dict[str, Any]) -> float:
    return _timer_value(result, "obsoperator_sample", "total_s") + _timer_value(result, "obsoperator_close", "total_s")


def _output_worker_time(result: dict[str, Any]) -> float:
    names = ("output_open_species_conc", "output_append_species_conc", "output_close_species_conc")
    return sum(_timer_value(result, name, "total_s") for name in names)


def _finite_float(value: Any, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} is not finite")
    return parsed


def _annotate_run_speedups(rows: list[RunRow]) -> list[RunRow]:
    sync_walls = {
        (row.spec.repetition, row.spec.total_tracers): row.effective_wall_s
        for row in rows
        if row.status == "completed" and row.spec.writer == "sync"
    }
    annotated: list[RunRow] = []
    for row in rows:
        if row.status != "completed" or row.effective_wall_s is None:
            annotated.append(row)
            continue
        sync_wall = sync_walls.get((row.spec.repetition, row.spec.total_tracers))
        speedup = sync_wall / row.effective_wall_s if sync_wall is not None else None
        annotated.append(replace(row, threaded_speedup_same_repeat=speedup))
    return annotated


def _aggregate_modes(rows: list[RunRow]) -> list[ModeRow]:
    groups: dict[tuple[int, int, int, int, str], list[RunRow]] = {}
    for row in rows:
        if row.status != "completed":
            continue
        spec = row.spec
        key = (spec.total_tracers, spec.processes, spec.threads_per_process, spec.tracers_per_process, spec.writer)
        groups.setdefault(key, []).append(row)

    mean_walls = {key: statistics.mean(_values(group, "effective_wall_s")) for key, group in groups.items()}
    modes: list[ModeRow] = []
    for key, group in sorted(groups.items()):
        total, processes, threads, batch, writer = key
        walls = _values(group, "effective_wall_s")
        steps = {row.transport_steps for row in group}
        if len(steps) != 1 or None in steps:
            raise ValueError(f"inconsistent transport steps for {key}: {steps}")
        transport_steps = int(next(iter(steps)))
        sync_key = (total, processes, threads, batch, "sync")
        speedup = mean_walls[sync_key] / statistics.mean(walls) if sync_key in mean_walls else None
        modes.append(
            ModeRow(
                total_tracers=total,
                processes=processes,
                threads_per_process=threads,
                tracers_per_process=batch,
                writer=writer,
                completed_repetitions=len(group),
                mean_effective_wall_s=statistics.mean(walls),
                stdev_effective_wall_s=statistics.stdev(walls) if len(walls) > 1 else 0.0,
                mean_effective_s_per_step=statistics.mean(walls) / transport_steps,
                aggregate_tracer_steps_per_s=total * transport_steps / statistics.mean(walls),
                threaded_speedup=speedup,
                mean_rank_spread_percent=statistics.mean(_values(group, "rank_spread_percent")),
                mean_critical_met_s=statistics.mean(_values(group, "critical_met_s")),
                mean_critical_emissions_s=statistics.mean(_values(group, "critical_emissions_s")),
                mean_critical_transport_s=statistics.mean(_values(group, "critical_transport_s")),
                mean_critical_obsoperator_s=statistics.mean(_values(group, "critical_obsoperator_s")),
                mean_critical_output_foreground_s=statistics.mean(_values(group, "critical_output_foreground_s")),
                mean_critical_output_worker_s=statistics.mean(_values(group, "critical_output_worker_s")),
                mean_total_peak_rss_gib=statistics.mean(_values(group, "total_peak_rss_gib")),
                mean_total_output_gib=statistics.mean(_values(group, "total_output_gib")),
            )
        )
    return modes


def _values(rows: Iterable[RunRow], field: str) -> list[float]:
    values = [getattr(row, field) for row in rows]
    if any(value is None for value in values):
        raise ValueError(f"missing {field} in completed rows")
    return [float(value) for value in values]


def _write_runs(rows: list[RunRow], root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row(root))


def _write_modes(rows: list[ModeRow], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MODE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())


def _print_modes(rows: list[ModeRow], runs_output: Path, modes_output: Path) -> None:
    if not rows:
        print(f"No completed cases. Wrote {runs_output} and {modes_output}")
        return
    print("| Tracers | Writer | Repeats | Wall s | s/step | Tracer-steps/s | vs sync | RSS GiB | Output GiB |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row.total_tracers} | {row.writer} | {row.completed_repetitions} | "
            f"{row.mean_effective_wall_s:.2f} | {row.mean_effective_s_per_step:.4f} | "
            f"{row.aggregate_tracer_steps_per_s:.1f} | {_format_ratio(row.threaded_speedup)} | "
            f"{row.mean_total_peak_rss_gib:.1f} | {row.mean_total_output_gib:.2f} |"
        )
    print()
    print("Critical-rank cumulative stage times:")
    print("| Tracers | Writer | Met s | Emissions s | Transport s | ObsOperator s | HISTORY foreground s | HISTORY worker s |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row.total_tracers} | {row.writer} | {row.mean_critical_met_s:.2f} | "
            f"{row.mean_critical_emissions_s:.2f} | {row.mean_critical_transport_s:.2f} | "
            f"{row.mean_critical_obsoperator_s:.2f} | {row.mean_critical_output_foreground_s:.2f} | "
            f"{row.mean_critical_output_worker_s:.2f} |"
        )
    print(f"\nWrote run-level results to {runs_output}")
    print(f"Wrote mode averages to {modes_output}")


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.8f}"


def _format_int(value: int | None) -> str:
    return "" if value is None else str(value)


def _format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}x"


if __name__ == "__main__":
    raise SystemExit(main())
