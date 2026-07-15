from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable


OUTPUT_FIELDS = (
    "total_tracers",
    "total_cores",
    "processes",
    "threads_per_process",
    "tracers_per_process",
    "status",
    "effective_best_s",
    "effective_mean_s",
    "aggregate_tracers_per_s",
    "tracers_per_s_per_core",
    "throughput_vs_1p_same_cores",
    "rank_spread_percent",
    "slowest_rank",
    "best_setup_s",
    "best_tpcore_s",
    "best_vdiff_s",
    "best_convection_s",
    "best_overhead_s",
    "total_peak_rss_gib",
    "best_for_core_budget",
    "best_for_total_tracers",
    "pareto_for_total_tracers",
    "case_dir",
    "reason",
)


@dataclass(frozen=True)
class CaseSpec:
    total_tracers: int
    total_cores: int
    processes: int
    threads_per_process: int
    tracers_per_process: int
    case_dir: Path


@dataclass(frozen=True)
class SummaryRow:
    spec: CaseSpec
    status: str
    effective_best_s: float | None = None
    effective_mean_s: float | None = None
    aggregate_tracers_per_s: float | None = None
    tracers_per_s_per_core: float | None = None
    throughput_vs_1p_same_cores: float | None = None
    rank_spread_percent: float | None = None
    slowest_rank: int | None = None
    best_setup_s: float | None = None
    best_tpcore_s: float | None = None
    best_vdiff_s: float | None = None
    best_convection_s: float | None = None
    best_overhead_s: float | None = None
    total_peak_rss_gib: float | None = None
    best_for_core_budget: bool = False
    best_for_total_tracers: bool = False
    pareto_for_total_tracers: bool = False
    reason: str = ""

    def as_csv_row(self, root: Path) -> dict[str, str]:
        spec = self.spec
        return {
            "total_tracers": str(spec.total_tracers),
            "total_cores": str(spec.total_cores),
            "processes": str(spec.processes),
            "threads_per_process": str(spec.threads_per_process),
            "tracers_per_process": str(spec.tracers_per_process),
            "status": self.status,
            "effective_best_s": _format_float(self.effective_best_s),
            "effective_mean_s": _format_float(self.effective_mean_s),
            "aggregate_tracers_per_s": _format_float(self.aggregate_tracers_per_s),
            "tracers_per_s_per_core": _format_float(self.tracers_per_s_per_core),
            "throughput_vs_1p_same_cores": _format_float(self.throughput_vs_1p_same_cores),
            "rank_spread_percent": _format_float(self.rank_spread_percent),
            "slowest_rank": "" if self.slowest_rank is None else str(self.slowest_rank),
            "best_setup_s": _format_float(self.best_setup_s),
            "best_tpcore_s": _format_float(self.best_tpcore_s),
            "best_vdiff_s": _format_float(self.best_vdiff_s),
            "best_convection_s": _format_float(self.best_convection_s),
            "best_overhead_s": _format_float(self.best_overhead_s),
            "total_peak_rss_gib": _format_float(self.total_peak_rss_gib),
            "best_for_core_budget": _format_bool(self.best_for_core_budget),
            "best_for_total_tracers": _format_bool(self.best_for_total_tracers),
            "pareto_for_total_tracers": _format_bool(self.pareto_for_total_tracers),
            "case_dir": str(self.spec.case_dir.relative_to(root)),
            "reason": self.reason,
        }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.results_dir.resolve()
    specs = _read_manifest(root)
    rows = [_summarize_case(root, spec) for spec in specs]
    rows = _annotate(rows)

    output = args.output or root / "summary.csv"
    _write_csv(rows, root, output)
    _print_summary(rows, output)

    incomplete = sum(row.status != "completed" for row in rows)
    if incomplete:
        print(f"Warning: {incomplete} of {len(rows)} cases are incomplete; they remain in the CSV.", file=sys.stderr)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate the fixed-total, single-socket transport scaling experiment."
    )
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path, help="Output CSV path (default: RESULTS_DIR/summary.csv).")
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
                        total_tracers=int(record["total_tracers"]),
                        total_cores=int(record["total_cores"]),
                        processes=int(record["processes"]),
                        threads_per_process=int(record["threads_per_process"]),
                        tracers_per_process=int(record["tracers_per_process"]),
                        case_dir=root / record["case_dir"],
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(f"Invalid record in {manifest}: {record}") from exc
    if not specs:
        raise SystemExit(f"Manifest is empty: {manifest}")
    return sorted(specs, key=lambda spec: (spec.total_tracers, spec.total_cores, spec.processes))


def _summarize_case(root: Path, spec: CaseSpec) -> SummaryRow:
    relative_case = spec.case_dir.relative_to(root)
    if not (spec.case_dir / ".complete").is_file():
        return SummaryRow(spec=spec, status="incomplete", reason="completion marker is missing")

    rank_rows: list[dict[str, str]] = []
    for rank in range(spec.processes):
        rank_path = spec.case_dir / f"rank_{rank}.csv"
        if not rank_path.is_file():
            return SummaryRow(spec=spec, status="incomplete", reason=f"missing {relative_case}/rank_{rank}.csv")
        records = _read_rank_csv(rank_path)
        if len(records) != 1:
            return SummaryRow(
                spec=spec,
                status="invalid",
                reason=f"expected one record in {relative_case}/rank_{rank}.csv, found {len(records)}",
            )
        record = records[0]
        if record.get("status") != "completed":
            reason = record.get("reason") or f"rank {rank} status is {record.get('status', 'missing')}"
            return SummaryRow(spec=spec, status="failed", reason=reason)
        try:
            tracer_count = int(record["tracer_count"])
        except (KeyError, ValueError) as exc:
            return SummaryRow(spec=spec, status="invalid", reason=f"invalid tracer count for rank {rank}: {exc}")
        if tracer_count != spec.tracers_per_process:
            return SummaryRow(
                spec=spec,
                status="invalid",
                reason=f"rank {rank} has {tracer_count} tracers, expected {spec.tracers_per_process}",
            )
        rank_rows.append(record)

    try:
        best_wall = [_required_float(record, "best_wall_s") for record in rank_rows]
        mean_wall = [_required_float(record, "mean_wall_s") for record in rank_rows]
        peak_rss_mib = [_required_float(record, "peak_rss_mib") for record in rank_rows]
        slowest_rank = max(range(spec.processes), key=best_wall.__getitem__)
        slowest = rank_rows[slowest_rank]
        effective_best = best_wall[slowest_rank]
        effective_mean = max(mean_wall)
        aggregate_rate = spec.total_tracers / effective_best
        rank_spread = 100.0 * (max(best_wall) - min(best_wall)) / max(best_wall)
        return SummaryRow(
            spec=spec,
            status="completed",
            effective_best_s=effective_best,
            effective_mean_s=effective_mean,
            aggregate_tracers_per_s=aggregate_rate,
            tracers_per_s_per_core=aggregate_rate / spec.total_cores,
            rank_spread_percent=rank_spread,
            slowest_rank=slowest_rank,
            best_setup_s=_required_float(slowest, "best_setup_s"),
            best_tpcore_s=_required_float(slowest, "best_tpcore_s"),
            best_vdiff_s=_required_float(slowest, "best_vdiff_s"),
            best_convection_s=_required_float(slowest, "best_convection_s"),
            best_overhead_s=_required_float(slowest, "best_overhead_s"),
            total_peak_rss_gib=sum(peak_rss_mib) / 1024.0,
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return SummaryRow(spec=spec, status="invalid", reason=f"invalid rank timing: {exc}")


def _read_rank_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _required_float(record: dict[str, str], field: str) -> float:
    value = float(record[field])
    if not math.isfinite(value):
        raise ValueError(f"{field} is not finite")
    return value


def _annotate(rows: list[SummaryRow]) -> list[SummaryRow]:
    completed = [row for row in rows if row.status == "completed"]
    one_process = {
        (row.spec.total_tracers, row.spec.total_cores): row
        for row in completed
        if row.spec.processes == 1
    }
    best_by_budget = _best_rows(completed, key=lambda row: (row.spec.total_tracers, row.spec.total_cores))
    best_by_total = _best_rows(completed, key=lambda row: row.spec.total_tracers)

    annotated: list[SummaryRow] = []
    for row in rows:
        if row.status != "completed":
            annotated.append(row)
            continue
        baseline = one_process.get((row.spec.total_tracers, row.spec.total_cores))
        baseline_rate = baseline.aggregate_tracers_per_s if baseline is not None else None
        relative_rate = None
        if baseline_rate is not None and row.aggregate_tracers_per_s is not None:
            relative_rate = row.aggregate_tracers_per_s / baseline_rate
        annotated.append(
            replace(
                row,
                throughput_vs_1p_same_cores=relative_rate,
                best_for_core_budget=row is best_by_budget[(row.spec.total_tracers, row.spec.total_cores)],
                best_for_total_tracers=row is best_by_total[row.spec.total_tracers],
                pareto_for_total_tracers=_is_pareto(row, completed),
            )
        )
    return annotated


def _best_rows(rows: Iterable[SummaryRow], *, key) -> dict[object, SummaryRow]:
    best: dict[object, SummaryRow] = {}
    for row in rows:
        group = key(row)
        current = best.get(group)
        if current is None or _rate(row) > _rate(current):
            best[group] = row
    return best


def _is_pareto(candidate: SummaryRow, completed: list[SummaryRow]) -> bool:
    candidate_rate = _rate(candidate)
    for other in completed:
        if other.spec.total_tracers != candidate.spec.total_tracers or other is candidate:
            continue
        other_rate = _rate(other)
        no_more_cores = other.spec.total_cores <= candidate.spec.total_cores
        no_less_throughput = other_rate >= candidate_rate
        strictly_better = other.spec.total_cores < candidate.spec.total_cores or other_rate > candidate_rate
        if no_more_cores and no_less_throughput and strictly_better:
            return False
    return True


def _rate(row: SummaryRow) -> float:
    if row.aggregate_tracers_per_s is None:
        return float("-inf")
    return row.aggregate_tracers_per_s


def _write_csv(rows: list[SummaryRow], root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row(root))


def _print_summary(rows: list[SummaryRow], output: Path) -> None:
    winners = [row for row in rows if row.best_for_core_budget]
    if not winners:
        print(f"No completed cases. Wrote {output}")
        return

    print("| Tracers | Cores | Processes | Threads/process | Tracers/process | Effective s | Tracers/s | vs 1p |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in winners:
        spec = row.spec
        print(
            f"| {spec.total_tracers} | {spec.total_cores} | {spec.processes} | "
            f"{spec.threads_per_process} | {spec.tracers_per_process} | "
            f"{row.effective_best_s:.4f} | {row.aggregate_tracers_per_s:.1f} | "
            f"{_format_ratio(row.throughput_vs_1p_same_cores)} |"
        )

    print()
    print("Best measured configuration for each total tracer count:")
    for row in rows:
        if row.best_for_total_tracers:
            spec = row.spec
            print(
                f"- {spec.total_tracers} tracers: {row.aggregate_tracers_per_s:.1f} tracers/s "
                f"with {spec.processes} x {spec.threads_per_process} threads "
                f"({spec.total_cores} cores, {spec.tracers_per_process} tracers/process)"
            )
    print(f"\nWrote {len(rows)} rows to {output}")


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.8f}"


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}x"


if __name__ == "__main__":
    raise SystemExit(main())
