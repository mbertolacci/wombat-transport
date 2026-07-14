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
from yaml12 import read_yaml


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
    reference_engine: str = "geoschem"
    candidate_engine: str = "wombat"


def load_case_manifest(case_dir: str | Path) -> dict[str, Any]:
    path = Path(case_dir) / "case.yml"
    manifest = read_yaml(path) or {}
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
    reference_work_dir: str | Path | None = None,
    candidate_work_dir: str | Path | None = None,
    reference_engine: str = "geoschem",
    candidate_engine: str = "wombat",
    output_dir: str | Path | None = None,
) -> tuple[list[MetricRow], Path]:
    case_path = Path(case_dir).resolve()
    manifest = load_case_manifest(case_path)
    case_name = str(manifest["name"])
    repo = Path(__file__).resolve().parents[1]
    reference_root = _resolve_work_root(repo, reference_work_dir or work_dir)
    candidate_root = _resolve_work_root(repo, candidate_work_dir or work_dir)
    reference_case_work = reference_root / case_name
    candidate_case_work = candidate_root / case_name

    requested = _comparison_ids_for_mode(manifest, mode)
    rows: list[MetricRow] = []
    for stage in manifest["stages"]:
        stage_id = str(stage.get("id", ""))
        if not stage_id:
            raise ValidationRunError(f"case {case_name} has a stage without id")
        reference_context = {"repo": repo, "case": case_path, "work": reference_case_work, "stage": stage_id}
        candidate_context = {"repo": repo, "case": case_path, "work": candidate_case_work, "stage": stage_id}
        reference_dir = _engine_work_dir(stage, reference_engine, reference_context)
        candidate_dir = _engine_work_dir(stage, candidate_engine, candidate_context)
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
                    geoschem_dir=reference_dir,
                    wombat_dir=candidate_dir,
                    reference_engine=reference_engine,
                    candidate_engine=candidate_engine,
                )
            )

    if not rows:
        raise ValidationRunError(f"case {case_name} mode {mode!r} selected no comparison rows")

    if output_dir is None:
        resolved_output_dir = candidate_case_work / "compare" / mode
    else:
        resolved_output_dir = Path(output_dir)
        if not resolved_output_dir.is_absolute():
            resolved_output_dir = repo / resolved_output_dir
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics_csv(resolved_output_dir / "metrics.csv", rows)
    _write_summary_json(resolved_output_dir / "summary.json", manifest, mode, rows)
    return rows, resolved_output_dir


def _resolve_work_root(repo: Path, value: str | Path) -> Path:
    root = Path(value)
    return root.resolve() if root.is_absolute() else (repo / root).resolve()


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
    reference_engine: str = "geoschem",
    candidate_engine: str = "wombat",
) -> list[MetricRow]:
    comparison_id = str(comparison.get("id", ""))
    kind = str(comparison.get("kind", ""))
    geoschem_files = _glob_required(geoschem_dir, str(comparison.get("geoschem_glob", "")))
    wombat_files = _glob_required(wombat_dir, str(comparison.get("wombat_glob", "")))
    tolerance_mode = str(comparison.get("tolerance_mode", "report"))
    if kind == "obsoperator":
        return _compare_obsoperator_files(
            case_name=case_name,
            mode=mode,
            stage_id=stage_id,
            comparison_id=comparison_id,
            reference_files=geoschem_files,
            candidate_files=wombat_files,
            tolerance_mode=tolerance_mode,
            reference_engine=reference_engine,
            candidate_engine=candidate_engine,
        )

    pairs = _pair_by_name(geoschem_files, wombat_files, comparison_id)
    fields = tuple(str(item) for item in comparison.get("fields", ()))
    if not fields:
        raise ValidationRunError(f"comparison {comparison_id} has no fields")

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
                        reference_engine=reference_engine,
                        candidate_engine=candidate_engine,
                    )
                )
    return rows


def _compare_obsoperator_files(
    *,
    case_name: str,
    mode: str,
    stage_id: str,
    comparison_id: str,
    reference_files: list[Path],
    candidate_files: list[Path],
    tolerance_mode: str,
    reference_engine: str,
    candidate_engine: str,
) -> list[MetricRow]:
    reference_by_name = {path.name: path for path in reference_files}
    candidate_by_name = {path.name: path for path in candidate_files}
    rows: list[MetricRow] = []
    for filename in sorted(set(reference_by_name) | set(candidate_by_name)):
        reference_path = reference_by_name.get(filename)
        candidate_path = candidate_by_name.get(filename)
        if reference_path is None or candidate_path is None:
            rows.append(
                _metric_row(
                    case_name=case_name,
                    mode=mode,
                    stage_id=stage_id,
                    comparison_id=comparison_id,
                    filename=filename,
                    variable="missing_file_in_reference" if reference_path is None else "missing_file_in_candidate",
                    errors=np.ones(1, dtype=np.float64),
                    tolerance_mode=tolerance_mode,
                    reference_engine=reference_engine,
                    candidate_engine=candidate_engine,
                )
            )
            continue
        with netCDF4.Dataset(reference_path) as reference_dataset, netCDF4.Dataset(candidate_path) as candidate_dataset:
            reference = _obsoperator_sample_map(reference_dataset)
            candidate = _obsoperator_sample_map(candidate_dataset)
        reference_keys = set(reference)
        candidate_keys = set(candidate)
        shared = sorted(reference_keys & candidate_keys)
        missing = reference_keys - candidate_keys
        extra = candidate_keys - reference_keys
        reference_samples = np.asarray([reference[key] for key in shared], dtype=np.float64)
        candidate_samples = np.asarray([candidate[key] for key in shared], dtype=np.float64)
        rows.extend(
            (
                _metric_row(
                    case_name=case_name,
                    mode=mode,
                    stage_id=stage_id,
                    comparison_id=comparison_id,
                    filename=filename,
                    variable="missing_samples_in_candidate",
                    errors=_structural_errors(len(missing), len(reference_keys)),
                    tolerance_mode=tolerance_mode,
                    reference_engine=reference_engine,
                    candidate_engine=candidate_engine,
                ),
                _metric_row(
                    case_name=case_name,
                    mode=mode,
                    stage_id=stage_id,
                    comparison_id=comparison_id,
                    filename=filename,
                    variable="extra_samples_in_candidate",
                    errors=_structural_errors(len(extra), len(candidate_keys)),
                    tolerance_mode=tolerance_mode,
                    reference_engine=reference_engine,
                    candidate_engine=candidate_engine,
                ),
                _metric_row(
                    case_name=case_name,
                    mode=mode,
                    stage_id=stage_id,
                    comparison_id=comparison_id,
                    filename=filename,
                    variable="sample",
                    errors=np.abs(candidate_samples - reference_samples),
                    tolerance_mode=tolerance_mode,
                    reference_engine=reference_engine,
                    candidate_engine=candidate_engine,
                ),
                _metric_row(
                    case_name=case_name,
                    mode=mode,
                    stage_id=stage_id,
                    comparison_id=comparison_id,
                    filename=filename,
                    variable="sample_tolerance_failures",
                    errors=np.logical_not(
                        np.isclose(reference_samples, candidate_samples, rtol=1.0e-6, atol=1.0e-12)
                    ).astype(np.float64),
                    tolerance_mode=tolerance_mode,
                    reference_engine=reference_engine,
                    candidate_engine=candidate_engine,
                ),
            )
        )
    return rows


def _metric_row(
    *,
    case_name: str,
    mode: str,
    stage_id: str,
    comparison_id: str,
    filename: str,
    variable: str,
    errors: np.ndarray,
    tolerance_mode: str,
    reference_engine: str,
    candidate_engine: str,
) -> MetricRow:
    return MetricRow(
        case=case_name,
        mode=mode,
        stage=stage_id,
        comparison=comparison_id,
        kind="obsoperator",
        file=filename,
        variable=variable,
        max_abs_error=float(np.max(errors)) if errors.size else 0.0,
        mean_abs_error=float(np.mean(errors)) if errors.size else 0.0,
        n_values=int(errors.size),
        tolerance_mode=tolerance_mode,
        reference_engine=reference_engine,
        candidate_engine=candidate_engine,
    )


def _structural_errors(mismatch_count: int, total_count: int) -> np.ndarray:
    errors = np.zeros(total_count, dtype=np.float64)
    if mismatch_count:
        errors[:mismatch_count] = 1.0
    return errors


def _obsoperator_sample_map(dataset: netCDF4.Dataset) -> dict[tuple[str, str], float]:
    required = {"id", "field", "id_index", "field_index", "sample"}
    missing = sorted(required - set(dataset.variables))
    if missing:
        raise ValidationRunError(f"ObsOperator output is missing variables {missing}")
    ids = _decode_nul_padded_strings(dataset.variables["id"], "entries")
    fields = _decode_nul_padded_strings(dataset.variables["field"], "fields")
    id_index = np.asarray(dataset.variables["id_index"][:], dtype=np.int64)
    field_index = np.asarray(dataset.variables["field_index"][:], dtype=np.int64)
    samples = np.asarray(dataset.variables["sample"][:], dtype=np.float64)
    if id_index.shape != field_index.shape or id_index.shape != samples.shape:
        raise ValidationRunError("ObsOperator index and sample arrays have different shapes")
    if np.any(id_index < 1) or np.any(id_index > len(ids)):
        raise ValidationRunError("ObsOperator id_index is out of bounds")
    if np.any(field_index < 1) or np.any(field_index > len(fields)):
        raise ValidationRunError("ObsOperator field_index is out of bounds")
    result: dict[tuple[str, str], float] = {}
    for id_value, field_value, sample in zip(id_index, field_index, samples, strict=True):
        key = (ids[int(id_value) - 1], fields[int(field_value) - 1])
        if key in result:
            raise ValidationRunError(f"ObsOperator output contains duplicate sample key {key!r}")
        result[key] = float(sample)
    return result


def _decode_nul_padded_strings(variable: netCDF4.Variable, row_dimension: str) -> tuple[str, ...]:
    values = np.asarray(variable[:], dtype="S1")
    dimensions = tuple(variable.dimensions)
    if values.ndim != 2 or row_dimension not in dimensions:
        raise ValidationRunError(
            f"ObsOperator variable {variable.name} has unexpected dimensions {dimensions}"
        )
    row_axis = dimensions.index(row_dimension)
    rows = values if row_axis == 0 else values.T
    decoded: list[str] = []
    for row in rows:
        encoded = row.tobytes()
        nul = encoded.find(b"\x00")
        if nul >= 0:
            encoded = encoded[:nul]
        decoded.append(encoded.decode("utf-8"))
    return tuple(decoded)


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
        "reference_engines": sorted({row.reference_engine for row in rows}),
        "candidate_engines": sorted({row.candidate_engine for row in rows}),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare existing GEOS-Chem and Wombat validation-run outputs.")
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--mode", default="quick")
    parser.add_argument("--work-dir", type=Path, default=Path("validation_runs/work"))
    parser.add_argument("--reference-work-dir", type=Path)
    parser.add_argument("--candidate-work-dir", type=Path)
    parser.add_argument("--reference-engine", choices=("geoschem", "wombat"), default="geoschem")
    parser.add_argument("--candidate-engine", choices=("geoschem", "wombat"), default="wombat")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    try:
        rows, output_dir = compare_case(
            args.case_dir,
            mode=args.mode,
            work_dir=args.work_dir,
            reference_work_dir=args.reference_work_dir,
            candidate_work_dir=args.candidate_work_dir,
            reference_engine=args.reference_engine,
            candidate_engine=args.candidate_engine,
            output_dir=args.output_dir,
        )
    except ValidationRunError as exc:
        print(f"validation comparison failed: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {len(rows)} metric rows to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
