from __future__ import annotations

import logging
from os import PathLike
from typing import Any, TextIO

import yaml

try:
    from yaml12 import parse_yaml as _parse_yaml12
except ImportError:  # pragma: no cover - exercised in incomplete installations.
    _parse_yaml12 = None


logger = logging.getLogger(__name__)
_yaml12_fallback_warned = False


def load_yaml(source: str | TextIO) -> Any:
    """Load established Wombat configuration YAML with PyYAML semantics."""

    return yaml.safe_load(source)


def dump_yaml(data: Any, stream: TextIO | None = None, **kwargs: Any) -> str | None:
    """Write YAML with the project's established PyYAML representation."""

    return yaml.safe_dump(data, stream, **kwargs)


def load_yaml12(source: str | TextIO, *, source_name: str | PathLike[str] | None = None) -> Any:
    """Load YAML 1.2, falling back visibly if py-yaml12 is unavailable."""

    text = source.read() if hasattr(source, "read") else source
    if _parse_yaml12 is not None:
        return _parse_yaml12(text)

    global _yaml12_fallback_warned
    if not _yaml12_fallback_warned:
        location = f" for {source_name}" if source_name is not None else ""
        logger.warning(
            "MAJOR PERFORMANCE WARNING: py-yaml12 is unavailable; using the "
            "slower PyYAML fallback%s. Install py-yaml12 for normal ObsOperator performance.",
            location,
        )
        _yaml12_fallback_warned = True
    return yaml.safe_load(text)
