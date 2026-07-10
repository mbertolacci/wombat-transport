#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import netCDF4
import numpy as np


FIELD_MAP = {
    "tracer_conc": ("tracer", 1.0, "level_tracer"),
    "surface_flux_kg_m2_s": ("surface_flux", 1.0, "column_tracer"),
    "delp_dry_hpa": ("delp_dry_hpa", 1.0, "level"),
    "ad_kg": ("dry_air_mass_kg", 1.0, "level"),
    "bxheight_m": ("bxheight", 1.0, "level"),
    "pbl_top_m": ("pbl_top_m", 1.0, "column"),
    "sphu_g_kg": ("sphu", 1.0e-3, "level"),
    "temperature_k": ("temperature", 1.0, "level"),
    "cmfmc_kg_m2_s": ("cmfmc", 1.0, "level"),
    "dtrain_kg_m2_s": ("dtrain", 1.0, "level"),
    "dqrcu_kg_kg_s": ("dqrcu", 1.0, "level"),
    "reevapcn_kg_kg_s": ("reevapcn", 1.0, "level"),
    "pficu_kg_m2_s": ("pficu", 1.0, "level"),
}


@dataclass
class Metrics:
    field: str
    count: int = 0
    max_abs: float = 0.0
    mean_abs_sum: float = 0.0
    mean_bias_sum: float = 0.0
    max_location: tuple[str, int, int, int, int] | None = None

    def add(self, diff: float, location: tuple[str, int, int, int, int]) -> None:
        abs_diff = abs(diff)
        self.count += 1
        self.mean_abs_sum += abs_diff
        self.mean_bias_sum += diff
        if abs_diff >= self.max_abs:
            self.max_abs = abs_diff
            self.max_location = location

    @property
    def mean_abs(self) -> float:
        return self.mean_abs_sum / self.count if self.count else float("nan")

    @property
    def mean_bias(self) -> float:
        return self.mean_bias_sum / self.count if self.count else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare a GC main-loop CSV trace to a Wombat main-loop NetCDF trace.")
    parser.add_argument("wombat_trace", type=Path)
    parser.add_argument("gc_trace_csv", type=Path)
    parser.add_argument("--field", action="append", choices=sorted(FIELD_MAP), default=None)
    parser.add_argument("--air-mw-g-mol", type=float, default=28.965, help="Dry-air molecular weight for GC kg/kg to mol/mol conversion.")
    parser.add_argument("--tracer-mw-g-mol", type=float, default=44.01, help="Tracer molecular weight for GC kg/kg to mol/mol conversion.")
    parser.add_argument("--by-boundary", action="store_true", help="Report one metrics row per field and boundary.")
    args = parser.parse_args()

    fields = tuple(args.field or FIELD_MAP)
    tracer_unit_scale = args.air_mw_g_mol / args.tracer_mw_g_mol
    metrics = _metrics_container(fields, by_boundary=args.by_boundary)
    with netCDF4.Dataset(args.wombat_trace) as wombat:
        record_lookup = _wombat_record_lookup(wombat)
        column_lookup = _wombat_column_lookup(wombat)
        wombat_values = _load_wombat_values(wombat, fields)
        occurrences: dict[str, int] = defaultdict(int)
        last_call_index = None
        with args.gc_trace_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                call_index = int(row["call_index"])
                boundary = row["boundary"]
                if call_index != last_call_index:
                    occurrences[boundary] += 1
                    last_call_index = call_index
                occurrence = occurrences[boundary] - 1
                record = record_lookup.get((boundary, occurrence))
                if record is None:
                    continue
                column = column_lookup.get((int(row["j"]) - 1, int(row["i"]) - 1))
                if column is None:
                    continue
                lev = int(row["l"]) - 1
                tracer = int(row["tracer"]) - 1
                for field in fields:
                    gc_name, scale, kind = FIELD_MAP[field]
                    if field == "tracer_conc":
                        scale *= tracer_unit_scale
                    value = float(row[field])
                    if value < -1.0e35:
                        continue
                    actual = _wombat_value(wombat_values, gc_name, kind, record, lev, column, tracer)
                    if not np.isfinite(actual):
                        continue
                    diff = actual - value * scale
                    _metric_for(metrics, field, boundary, by_boundary=args.by_boundary).add(
                        float(diff), (boundary, occurrence, lev + 1, column, tracer + 1)
                    )

    print("field,count,max_abs,mean_abs,mean_bias,max_location")
    items = metrics.values() if not args.by_boundary else [item for key, item in sorted(metrics.items())]
    for item in items:
        print(
            f"{item.field},{item.count},{item.max_abs:.12e},{item.mean_abs:.12e},"
            f"{item.mean_bias:.12e},{item.max_location}"
        )
    return 0


def _metrics_container(fields: tuple[str, ...], *, by_boundary: bool):
    if not by_boundary:
        return {field: Metrics(field) for field in fields}
    return {}


def _metric_for(metrics, field: str, boundary: str, *, by_boundary: bool) -> Metrics:
    if not by_boundary:
        return metrics[field]
    key = (field, boundary)
    if key not in metrics:
        metrics[key] = Metrics(f"{field}:{boundary}")
    return metrics[key]


def _wombat_record_lookup(dataset: netCDF4.Dataset) -> dict[tuple[str, int], int]:
    boundaries = [str(value).strip("\x00 ") for value in netCDF4.chartostring(dataset.variables["boundary"][:])]
    counts: dict[str, int] = defaultdict(int)
    lookup: dict[tuple[str, int], int] = {}
    for index, boundary in enumerate(boundaries):
        occurrence = counts[boundary]
        lookup[(boundary, occurrence)] = index
        counts[boundary] += 1
    return lookup


def _wombat_column_lookup(dataset: netCDF4.Dataset) -> dict[tuple[int, int], int]:
    lat_indices = np.asarray(dataset.variables["lat_index"][:], dtype=np.int64)
    lon_indices = np.asarray(dataset.variables["lon_index"][:], dtype=np.int64)
    return {(int(lat), int(lon)): index for index, (lat, lon) in enumerate(zip(lat_indices, lon_indices))}


def _load_wombat_values(dataset: netCDF4.Dataset, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    values = {}
    for field in fields:
        variable_name = FIELD_MAP[field][0]
        if variable_name in values or variable_name not in dataset.variables:
            continue
        values[variable_name] = np.asarray(dataset.variables[variable_name][:])
    return values


def _wombat_value(
    values: dict[str, np.ndarray],
    variable_name: str,
    kind: str,
    record: int,
    lev: int,
    column: int,
    tracer: int,
) -> float:
    if variable_name not in values:
        return float("nan")
    variable = values[variable_name]
    if kind == "level_tracer":
        return float(variable[record, lev, column, tracer])
    if kind == "column_tracer":
        return float(variable[record, column, tracer])
    if kind == "level":
        return float(variable[record, lev, column])
    if kind == "column":
        return float(variable[record, column])
    raise AssertionError(f"unhandled comparison kind {kind}")


if __name__ == "__main__":
    raise SystemExit(main())
