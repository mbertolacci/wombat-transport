from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import netCDF4
import numpy as np
import yaml


class ValidationRunError(RuntimeError):
    pass


@dataclass(frozen=True)
class MetricRow:
    case: str
    mode: str
    stage: str
    comparison: str
    kind: str
    file: str
    variable: str
    max_abs_error: float
    mean_abs_error: float
    n_values: int
    tolerance_mode: str


def load_case_manifest(case_dir: str | Path) -> dict[str, Any]:
    path = Path(case_dir) / "case.yml"
    with path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    if int(manifest.get("schema_version", 0)) != 1:
        raise ValidationRunError(f"{path} must declare schema_version: 1")
    if not manifest.get("name"):
        raise ValidationRunError(f"{path} is missing name")
    if not isinstance(manifest.get("stages"), list) or not manifest["stages"]:
        raise ValidationRunError(f"{path} must define at least one stage")
    return manifest


def compare_case(
    case_dir: str | Path,
    *,
    mode: str = "quick",
    work_dir: str | Path = "validation_runs/work",
) -> tuple[list[MetricRow], Path]:
    case_path = Path(case_dir).resolve()
    manifest = load_case_manifest(case_path)
    case_name = str(manifest["name"])
    repo = Path(__file__).resolve().parents[1]
    case_work = Path(work_dir)
    if not case_work.is_absolute():
        case_work = repo / case_work
    case_work = case_work / case_name

    requested = _comparison_ids_for_mode(manifest, mode)
    rows: list[MetricRow] = []
    for stage in manifest["stages"]:
        stage_id = str(stage.get("id", ""))
        if not stage_id:
            raise ValidationRunError(f"case {case_name} has a stage without id")
        context = {"repo": repo, "case": case_path, "work": case_work, "stage": stage_id}
        geoschem_dir = _engine_work_dir(stage, "geoschem", context)
        wombat_dir = _engine_work_dir(stage, "wombat", context)
        for comparison in stage.get("comparisons", ()):
            comparison_id = str(comparison.get("id", ""))
            if comparison_id not in requested:
                continue
            rows.extend(
                _compare_collection(
                    case_name=case_name,
                    mode=mode,
                    stage_id=stage_id,
                    comparison=comparison,
                    geoschem_dir=geoschem_dir,
                    wombat_dir=wombat_dir,
                )
            )

    if not rows:
        raise ValidationRunError(f"case {case_name} mode {mode!r} selected no comparison rows")

    output_dir = case_work / "compare" / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics_csv(output_dir / "metrics.csv", rows)
    _write_summary_json(output_dir / "summary.json", manifest, mode, rows)
    return rows, output_dir


def _comparison_ids_for_mode(manifest: dict[str, Any], mode: str) -> set[str]:
    modes = manifest.get("modes", {})
    if mode not in modes:
        raise ValidationRunError(f"case {manifest.get('name')} has no mode {mode!r}")
    comparisons = modes[mode].get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValidationRunError(f"case {manifest.get('name')} mode {mode!r} has no comparisons")
    return {str(item) for item in comparisons}


def _engine_work_dir(stage: dict[str, Any], engine: str, context: dict[str, Path | str]) -> Path:
    engines = stage.get("engines", {})
    if engine not in engines:
        raise ValidationRunError(f"stage {stage.get('id')} is missing engine {engine}")
    raw = engines[engine].get("work_dir")
    if raw is None:
        raise ValidationRunError(f"stage {stage.get('id')} engine {engine} is missing work_dir")
    return _resolve_template(raw, context)


def _compare_collection(
    *,
    case_name: str,
    mode: str,
    stage_id: str,
    comparison: dict[str, Any],
    geoschem_dir: Path,
    wombat_dir: Path,
) -> list[MetricRow]:
    comparison_id = str(comparison.get("id", ""))
    kind = str(comparison.get("kind", ""))
    geoschem_files = _glob_required(geoschem_dir, str(comparison.get("geoschem_glob", "")))
    wombat_files = _glob_required(wombat_dir, str(comparison.get("wombat_glob", "")))
    pairs = _pair_by_name(geoschem_files, wombat_files, comparison_id)
    fields = tuple(str(item) for item in comparison.get("fields", ()))
    if not fields:
        raise ValidationRunError(f"comparison {comparison_id} has no fields")
    tolerance_mode = str(comparison.get("tolerance_mode", "report"))

    rows: list[MetricRow] = []
    for name, geoschem_path, wombat_path in pairs:
        with netCDF4.Dataset(geoschem_path) as geoschem, netCDF4.Dataset(wombat_path) as wombat:
            variables = _variables_for_fields(geoschem, wombat, fields)
            for variable_name in variables:
                reference = np.asarray(geoschem.variables[variable_name][:], dtype=np.float64)
                candidate = np.asarray(wombat.variables[variable_name][:], dtype=np.float64)
                if candidate.shape != reference.shape:
                    raise ValidationRunError(
                        f"{comparison_id}:{name}:{variable_name} shape mismatch "
                        f"wombat={candidate.shape} geoschem={reference.shape}"
                    )
                abs_error = np.abs(candidate - reference)
                rows.append(
                    MetricRow(
                        case=case_name,
                        mode=mode,
                        stage=stage_id,
                        comparison=comparison_id,
                        kind=kind,
                        file=name,
                        variable=variable_name,
                        max_abs_error=float(np.max(abs_error)) if abs_error.size else 0.0,
                        mean_abs_error=float(np.mean(abs_error)) if abs_error.size else 0.0,
                        n_values=int(abs_error.size),
                        tolerance_mode=tolerance_mode,
                    )
                )
    return rows


def _glob_required(root: Path, pattern: str) -> list[Path]:
    if not pattern:
        raise ValidationRunError("comparison glob must not be empty")
    files = sorted(root.glob(pattern))
    if not files:
        raise ValidationRunError(f"no files matched {root / pattern}")
    return files


def _pair_by_name(geoschem_files: list[Path], wombat_files: list[Path], comparison_id: str) -> list[tuple[str, Path, Path]]:
    geoschem = {path.name: path for path in geoschem_files}
    wombat = {path.name: path for path in wombat_files}
    missing_wombat = sorted(set(geoschem) - set(wombat))
    missing_geoschem = sorted(set(wombat) - set(geoschem))
    if missing_wombat or missing_geoschem:
        raise ValidationRunError(
            f"comparison {comparison_id} file mismatch; "
            f"missing_wombat={missing_wombat} missing_geoschem={missing_geoschem}"
        )
    return [(name, geoschem[name], wombat[name]) for name in sorted(geoschem)]


def _variables_for_fields(
    geoschem: netCDF4.Dataset,
    wombat: netCDF4.Dataset,
    fields: tuple[str, ...],
) -> tuple[str, ...]:
    names: list[str] = []
    for field in fields:
        if field == "SpeciesConcVV_?ALL?":
            names.extend(_matching_variables(geoschem, wombat, "SpeciesConcVV_"))
        elif field == "SpeciesRst_?ALL?":
            names.extend(_matching_variables(geoschem, wombat, "SpeciesRst_"))
        else:
            if field not in geoschem.variables:
                raise ValidationRunError(f"GEOS-Chem output is missing variable {field}")
            if field not in wombat.variables:
                raise ValidationRunError(f"Wombat output is missing variable {field}")
            names.append(field)
    unique = tuple(dict.fromkeys(names))
    if not unique:
        raise ValidationRunError(f"fields {fields} matched no variables")
    return unique


def _matching_variables(geoschem: netCDF4.Dataset, wombat: netCDF4.Dataset, prefix: str) -> tuple[str, ...]:
    geoschem_names = {name for name in geoschem.variables if name.startswith(prefix)}
    wombat_names = {name for name in wombat.variables if name.startswith(prefix)}
    missing_wombat = sorted(geoschem_names - wombat_names)
    missing_geoschem = sorted(wombat_names - geoschem_names)
    if missing_wombat or missing_geoschem:
        raise ValidationRunError(
            f"variable mismatch for {prefix}; missing_wombat={missing_wombat} missing_geoschem={missing_geoschem}"
        )
    return tuple(sorted(geoschem_names))


def _resolve_template(value: str, context: dict[str, Path | str]) -> Path:
    resolved = str(value).format(**{key: str(item) for key, item in context.items()})
    return Path(resolved).resolve()


def _write_metrics_csv(path: Path, rows: list[MetricRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MetricRow.__dataclass_fields__.keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _write_summary_json(path: Path, manifest: dict[str, Any], mode: str, rows: list[MetricRow]) -> None:
    payload = {
        "schema_version": 1,
        "case": manifest["name"],
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(rows),
        "max_abs_error": max(row.max_abs_error for row in rows),
        "mean_abs_error_max": max(row.mean_abs_error for row in rows),
        "comparisons": sorted({row.comparison for row in rows}),
        "stages": sorted({row.stage for row in rows}),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare existing GEOS-Chem and Wombat validation-run outputs.")
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--mode", default="quick")
    parser.add_argument("--work-dir", type=Path, default=Path("validation_runs/work"))
    args = parser.parse_args(argv)

    try:
        rows, output_dir = compare_case(args.case_dir, mode=args.mode, work_dir=args.work_dir)
    except ValidationRunError as exc:
        print(f"validation comparison failed: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {len(rows)} metric rows to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
