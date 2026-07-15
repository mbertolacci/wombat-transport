from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

from yaml12 import read_yaml


class ValidationMatrixError(RuntimeError):
    pass


def _resolve_root(repo: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (repo / value).resolve()


def _thread_root(prefix: Path, threads: int) -> Path:
    return prefix.with_name(f"{prefix.name}_t{threads}")


def _case_dirs(repo: Path, requested: list[Path]) -> list[Path]:
    if requested:
        return [_resolve_root(repo, item) for item in requested]
    return sorted((repo / "validation_runs" / "cases").iterdir())


def _load_manifest(case_dir: Path) -> dict[str, Any]:
    manifest = read_yaml(case_dir / "case.yml") or {}
    if int(manifest.get("schema_version", 0)) != 1 or not manifest.get("name"):
        raise ValidationMatrixError(f"invalid validation manifest: {case_dir / 'case.yml'}")
    return manifest


def _stage_dir(stage: dict[str, Any], engine: str, case_work: Path) -> Path:
    raw = str(stage["engines"][engine]["work_dir"])
    return Path(raw.format(work=case_work)).resolve()


def _materialize_stage(
    *,
    repo: Path,
    case_dir: Path,
    case_name: str,
    stage: dict[str, Any],
    thread_work_root: Path,
    compressed_inputs: Path,
    uncompressed_inputs: Path,
    binary_work_root: Path,
    geoschem_executable: Path | None,
) -> dict[str, Path]:
    case_work = thread_work_root / case_name
    result: dict[str, Path] = {}
    for engine in ("geoschem", "wombat"):
        destination = _stage_dir(stage, engine, case_work)
        if destination.exists():
            raise ValidationMatrixError(f"refusing to overwrite existing run directory: {destination}")
        template = case_dir / str(stage["engines"][engine]["template_dir"])
        shutil.copytree(template, destination)
        obs_target = uncompressed_inputs if engine == "geoschem" else compressed_inputs
        (destination / "ObsOperator").symlink_to(obs_target, target_is_directory=True)
        if engine == "geoschem":
            binary = geoschem_executable or (
                binary_work_root / case_name / str(stage["id"]) / "geoschem" / "gcclassic"
            )
            binary = binary.resolve()
            if not binary.is_file():
                raise ValidationMatrixError(f"compiled GEOS-Chem executable is unavailable: {binary}")
            (destination / "gcclassic").symlink_to(binary)
        result[engine] = destination
    return result


def _dates_for_stage(stage: dict[str, Any]) -> list[datetime]:
    start = datetime.strptime(str(stage["start"]), "%Y-%m-%d %H:%M")
    end = datetime.strptime(str(stage["end"]), "%Y-%m-%d %H:%M")
    current = datetime(start.year, start.month, start.day)
    dates: list[datetime] = []
    while current < end:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def _prepare_uncompressed_inputs(stages: list[dict[str, Any]], source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for day in sorted({item for stage in stages for item in _dates_for_stage(stage)}):
        stem = f"obsoperator-{day:%Y%m%d}.yml"
        compressed = source / f"{stem}.gz"
        output = destination / stem
        if output.is_file():
            continue
        if not compressed.is_file():
            raise ValidationMatrixError(f"missing ObsOperator input: {compressed}")
        temporary = output.with_suffix(output.suffix + ".tmp")
        with gzip.open(compressed, "rb") as input_handle, temporary.open("wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle)
        temporary.replace(output)


def _link_obsoperator_restart(stage: dict[str, Any], stage_dirs: dict[str, dict[str, Path]]) -> None:
    dependency = stage.get("depends_on")
    if not dependency:
        return
    start = datetime.strptime(str(stage["start"]), "%Y-%m-%d %H:%M")
    filename = f"Wombat.ObsOperator.Restart.{start:%Y%m%d_%H%M%S}.nc4"
    source = stage_dirs[str(dependency)]["wombat"] / "Restarts" / filename
    if not source.is_file():
        raise ValidationMatrixError(f"required ObsOperator restart is unavailable: {source}")
    destination_dir = stage_dirs[str(stage["id"])]["wombat"] / "Restarts"
    destination_dir.mkdir(parents=True, exist_ok=True)
    (destination_dir / filename).symlink_to(source.resolve())


def _run_stage(
    *,
    repo: Path,
    case_name: str,
    stage_id: str,
    engine: str,
    run_dir: Path,
    threads: int,
    commit: str,
    geoschem_commit: str,
) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo / "src")
    if engine == "geoschem":
        environment.update(
            OMP_NUM_THREADS=str(threads),
            OMP_STACKSIZE=environment.get("OMP_STACKSIZE", "1000M"),
            OMP_PROC_BIND="close",
            OMP_PLACES="cores",
        )
        command = ["/usr/bin/time", "-p", "-o", "timing.txt", "./run_geoschem.sh"]
        log_path = run_dir / "run_wrapper.log"
    else:
        environment.update(
            WOMBAT_NUMBA="1",
            WOMBAT_NUMBA_THREADS=str(threads),
            NUMBA_NUM_THREADS=str(threads),
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        command = [
            "/usr/bin/time",
            "-p",
            "-o",
            "timing.txt",
            str(repo / ".venv" / "bin" / "python"),
            "-m",
            "wombat_transport.run",
            "run.yml",
        ]
        log_path = run_dir / "wombat.log"

    print(f"starting {case_name}/{stage_id}/{engine}/t{threads}: {run_dir}", flush=True)
    started = datetime.now(timezone.utc)
    start_clock = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=run_dir,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    ended = datetime.now(timezone.utc)
    metadata = {
        "schema_version": 1,
        "case": case_name,
        "stage": stage_id,
        "engine": engine,
        "threads": threads,
        "git_commit": commit,
        "geoschem_commit": geoschem_commit,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "elapsed_seconds": time.perf_counter() - start_clock,
        "return_code": completed.returncode,
        "command": command,
    }
    (run_dir / "validation_timing.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise ValidationMatrixError(
            f"{case_name}/{stage_id}/{engine}/t{threads} failed with exit code {completed.returncode}; "
            f"see {log_path}"
        )
    print(
        f"completed {case_name}/{stage_id}/{engine}/t{threads} "
        f"in {metadata['elapsed_seconds']:.2f} s",
        flush=True,
    )


def run_matrix(
    *,
    repo: Path,
    cases: list[Path],
    threads: list[int],
    work_prefix: Path,
    compressed_inputs: Path,
    uncompressed_inputs: Path,
    binary_work_root: Path,
    geoschem_executable: Path | None,
    prepare_only: bool,
) -> None:
    manifests = [(case, _load_manifest(case)) for case in cases]
    all_stages = [stage for _, manifest in manifests for stage in manifest["stages"]]
    _prepare_uncompressed_inputs(all_stages, compressed_inputs, uncompressed_inputs)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    geoschem_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo / "GCClassic" / "src" / "GEOS-Chem", text=True
    ).strip()

    for thread_count in threads:
        if thread_count < 1:
            raise ValidationMatrixError("thread counts must be positive")
        thread_work_root = _thread_root(work_prefix, thread_count)
        for case_dir, manifest in manifests:
            case_name = str(manifest["name"])
            stage_dirs: dict[str, dict[str, Path]] = {}
            for stage in manifest["stages"]:
                stage_id = str(stage["id"])
                stage_dirs[stage_id] = _materialize_stage(
                    repo=repo,
                    case_dir=case_dir,
                    case_name=case_name,
                    stage=stage,
                    thread_work_root=thread_work_root,
                    compressed_inputs=compressed_inputs,
                    uncompressed_inputs=uncompressed_inputs,
                    binary_work_root=binary_work_root,
                    geoschem_executable=geoschem_executable,
                )
            if prepare_only:
                continue
            for stage in manifest["stages"]:
                stage_id = str(stage["id"])
                _run_stage(
                    repo=repo,
                    case_name=case_name,
                    stage_id=stage_id,
                    engine="geoschem",
                    run_dir=stage_dirs[stage_id]["geoschem"],
                    threads=thread_count,
                    commit=commit,
                    geoschem_commit=geoschem_commit,
                )
                _link_obsoperator_restart(stage, stage_dirs)
                _run_stage(
                    repo=repo,
                    case_name=case_name,
                    stage_id=stage_id,
                    engine="wombat",
                    run_dir=stage_dirs[stage_id]["wombat"],
                    threads=thread_count,
                    commit=commit,
                    geoschem_commit=geoschem_commit,
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GEOS-Chem/Wombat validation cases at multiple thread counts.")
    parser.add_argument("case_dirs", nargs="*", type=Path)
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2])
    parser.add_argument(
        "--work-prefix",
        type=Path,
        default=Path("validation_runs/work/obsoperator"),
    )
    parser.add_argument("--obsoperator-dir", type=Path, default=Path("external_data/obsoperator"))
    parser.add_argument(
        "--uncompressed-obsoperator-dir",
        type=Path,
        default=Path("validation_runs/work/obsoperator-yaml"),
    )
    parser.add_argument("--binary-work-dir", type=Path, default=Path("validation_runs/work"))
    parser.add_argument(
        "--geoschem-executable",
        type=Path,
        help="use one prebuilt gcclassic executable for every case and stage",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    try:
        run_matrix(
            repo=repo,
            cases=_case_dirs(repo, args.case_dirs),
            threads=args.threads,
            work_prefix=_resolve_root(repo, args.work_prefix),
            compressed_inputs=_resolve_root(repo, args.obsoperator_dir),
            uncompressed_inputs=_resolve_root(repo, args.uncompressed_obsoperator_dir),
            binary_work_root=_resolve_root(repo, args.binary_work_dir),
            geoschem_executable=(
                _resolve_root(repo, args.geoschem_executable) if args.geoschem_executable else None
            ),
            prepare_only=args.prepare_only,
        )
    except (OSError, KeyError, TypeError, ValidationMatrixError, subprocess.SubprocessError) as exc:
        print(f"validation matrix failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
