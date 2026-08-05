from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from wombat_transport.output import OutputCompressionConfig, parse_output_storage


@dataclass(frozen=True)
class ObsOperatorConfig:
    activate: bool = False
    verbose: bool = False
    input_file: str | None = None
    output_file: str | None = None
    restart_file: str | None = None
    restart_missing: str = "warn"
    compression: OutputCompressionConfig = field(
        default_factory=OutputCompressionConfig
    )

def parse_obsoperator_config(outputs: dict[str, Any]) -> ObsOperatorConfig:
    raw = outputs.get("obsoperator", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("outputs.obsoperator must be a mapping")
    activate = bool(raw.get("activate", False))
    verbose = bool(raw.get("verbose", False))
    input_file = _optional_config_string(raw, "input_file")
    output_file = _optional_config_string(raw, "output_file")
    restart_file = _optional_config_string(raw, "restart_file")
    restart_missing = str(raw.get("restart_missing", "warn"))
    compression = parse_output_storage(
        raw,
        label="outputs.obsoperator",
    ).compression
    if restart_missing not in {"warn", "error", "ignore"}:
        raise ValueError("outputs.obsoperator.restart_missing must be 'warn', 'error', or 'ignore'")
    if "input_mode" in raw or "writer" in raw:
        raise ValueError("outputs.obsoperator async input_mode/writer options are no longer supported")
    if activate and input_file is None:
        raise KeyError("outputs.obsoperator.input_file is required when ObsOperator is active")
    if activate and output_file is None:
        raise KeyError("outputs.obsoperator.output_file is required when ObsOperator is active")
    if activate and restart_file is None:
        raise KeyError("outputs.obsoperator.restart_file is required when ObsOperator is active")
    return ObsOperatorConfig(
        activate=activate,
        verbose=verbose,
        input_file=input_file,
        output_file=output_file,
        restart_file=restart_file,
        restart_missing=restart_missing,
        compression=compression,
    )


def expand_obsoperator_template(template: str, timestamp: datetime) -> str:
    return (
        str(template)
        .replace("YYYY", f"{timestamp.year:04d}")
        .replace("MM", f"{timestamp.month:02d}")
        .replace("DD", f"{timestamp.day:02d}")
        .replace("hh", f"{timestamp.hour:02d}")
        .replace("mm", f"{timestamp.minute:02d}")
        .replace("ss", f"{timestamp.second:02d}")
    )

def _resolve_template_path(root: Path, template: str | None, timestamp: datetime) -> Path:
    if template is None:
        raise ValueError("ObsOperator path template is missing")
    path = Path(expand_obsoperator_template(template, timestamp))
    if not path.is_absolute():
        path = root / path
    return path.resolve()

def _optional_config_string(raw: dict[str, Any], key: str) -> str | None:
    if key not in raw or raw[key] is None:
        return None
    value = str(raw[key]).strip()
    return value or None
