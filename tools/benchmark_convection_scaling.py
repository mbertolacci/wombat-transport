from __future__ import annotations

import argparse
import csv
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import netCDF4
import numpy as np

from wombat_transport.transport.convection import run_cloud_convection_one_step


DEFAULT_INPUT = Path("tests/fixtures/convection_real_sampled_v1/convection_input.nc")


@dataclass(frozen=True)
class ConvectionInputs:
    tracer_conc: np.ndarray
    cmfmc_kg_m2_s: np.ndarray
    dtrain_kg_m2_s: np.ndarray
    dqrcu_kg_kg_s: np.ndarray
    reevapcn_kg_kg_s: np.ndarray
    delp_dry_hpa: np.ndarray
    delp_hpa: np.ndarray
    area_m2: np.ndarray
    bxheight_m: np.ndarray
    pficu_kg_m2_s: np.ndarray
    pflcu_kg_m2_s: np.ndarray
    temperature_k: np.ndarray
    precccon_mm_day: np.ndarray
    dt_s: float
    reconstruct_conv_precip_flux: bool


@dataclass(frozen=True)
class BenchmarkRow:
    version: str
    fixture: str
    repeat: int
    best_wall_s: float
    mean_wall_s: float
    tracer_count: int
    grid_shape: tuple[int, int, int]
    active_columns: int
    total_columns: int
    checksum: float
    peak_rss_mib: float

    def as_csv_row(self) -> dict[str, str]:
        return {
            "version": self.version,
            "fixture": self.fixture,
            "repeat": str(self.repeat),
            "best_wall_s": f"{self.best_wall_s:.8f}",
            "mean_wall_s": f"{self.mean_wall_s:.8f}",
            "tracer_count": str(self.tracer_count),
            "grid_shape": "x".join(str(value) for value in self.grid_shape),
            "active_columns": str(self.active_columns),
            "total_columns": str(self.total_columns),
            "checksum": f"{self.checksum:.16g}",
            "peak_rss_mib": f"{self.peak_rss_mib:.3f}",
        }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    inputs = _read_inputs(args.input)
    row = _benchmark_inputs(inputs, fixture=args.input, repeat=args.repeat, version=args.version)
    if args.output is None:
        _write_csv([row], sys.stdout)
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        _write_csv([row], handle)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Wombat cloud convection on a NetCDF fixture.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--repeat", type=_positive_int, default=7)
    parser.add_argument("--version", default="current")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _read_inputs(path: Path) -> ConvectionInputs:
    with netCDF4.Dataset(path) as dataset:
        return ConvectionInputs(
            tracer_conc=np.asarray(dataset.variables["tracer_conc"][:], dtype=np.float64),
            cmfmc_kg_m2_s=np.asarray(dataset.variables["cmfmc_kg_m2_s"][:], dtype=np.float64),
            dtrain_kg_m2_s=np.asarray(dataset.variables["dtrain_kg_m2_s"][:], dtype=np.float64),
            dqrcu_kg_kg_s=np.asarray(dataset.variables["dqrcu_kg_kg_s"][:], dtype=np.float64),
            reevapcn_kg_kg_s=np.asarray(dataset.variables["reevapcn_kg_kg_s"][:], dtype=np.float64),
            delp_dry_hpa=np.asarray(dataset.variables["delp_dry_hpa"][:], dtype=np.float64),
            delp_hpa=np.asarray(dataset.variables["delp_hpa"][:], dtype=np.float64),
            area_m2=np.asarray(dataset.variables["area_m2"][:], dtype=np.float64),
            bxheight_m=np.asarray(dataset.variables["bxheight_m"][:], dtype=np.float64),
            pficu_kg_m2_s=np.asarray(dataset.variables["pficu_kg_m2_s"][:], dtype=np.float64),
            pflcu_kg_m2_s=np.asarray(dataset.variables["pflcu_kg_m2_s"][:], dtype=np.float64),
            temperature_k=np.asarray(dataset.variables["temperature_k"][:], dtype=np.float64),
            precccon_mm_day=np.asarray(dataset.variables["precccon_mm_day"][:], dtype=np.float64),
            dt_s=float(dataset.dt_s),
            reconstruct_conv_precip_flux=bool(getattr(dataset, "reconstruct_conv_precip_flux", 0)),
        )


def _benchmark_inputs(inputs: ConvectionInputs, *, fixture: Path, repeat: int, version: str) -> BenchmarkRow:
    active = (np.max(np.abs(inputs.cmfmc_kg_m2_s), axis=0) > 1.0e-14) | (
        np.max(np.abs(inputs.dtrain_kg_m2_s), axis=0) > 1.0e-14
    )
    elapsed: list[float] = []
    checksum = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        result = run_cloud_convection_one_step(
            tracer_conc=inputs.tracer_conc,
            cmfmc_kg_m2_s=inputs.cmfmc_kg_m2_s,
            dtrain_kg_m2_s=inputs.dtrain_kg_m2_s,
            dqrcu_kg_kg_s=inputs.dqrcu_kg_kg_s,
            reevapcn_kg_kg_s=inputs.reevapcn_kg_kg_s,
            delp_dry_hpa=inputs.delp_dry_hpa,
            delp_hpa=inputs.delp_hpa,
            area_m2=inputs.area_m2,
            bxheight_m=inputs.bxheight_m,
            pficu_kg_m2_s=inputs.pficu_kg_m2_s,
            pflcu_kg_m2_s=inputs.pflcu_kg_m2_s,
            temperature_k=inputs.temperature_k,
            precccon_mm_day=inputs.precccon_mm_day,
            dt_s=inputs.dt_s,
            reconstruct_conv_precip_flux=inputs.reconstruct_conv_precip_flux,
        )
        elapsed.append(time.perf_counter() - start)
        checksum = float(np.mean(result.tracer_conc))
    return BenchmarkRow(
        version=version,
        fixture=str(fixture),
        repeat=repeat,
        best_wall_s=min(elapsed),
        mean_wall_s=sum(elapsed) / len(elapsed),
        tracer_count=int(inputs.tracer_conc.shape[0]),
        grid_shape=tuple(int(value) for value in inputs.tracer_conc.shape[1:]),
        active_columns=int(active.sum()),
        total_columns=int(active.size),
        checksum=checksum,
        peak_rss_mib=float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0,
    )


def _write_csv(rows: list[BenchmarkRow], handle: TextIO) -> None:
    fieldnames = list(rows[0].as_csv_row()) if rows else list(BenchmarkRow.__dataclass_fields__)
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row.as_csv_row())


if __name__ == "__main__":
    raise SystemExit(main())
