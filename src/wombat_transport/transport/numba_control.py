from __future__ import annotations

import os


FALSEY_NUMBA_VALUES = frozenset({"0", "false", "no", "off", "none"})

try:  # Optional acceleration dependency.
    from numba import set_num_threads
except ImportError:  # pragma: no cover - exercised in environments without numba.
    set_num_threads = None


def numba_mode(operator_env: str) -> str:
    """Return the effective Numba mode for one transport operator.

    Operator-specific environment variables override the global
    ``WOMBAT_NUMBA`` switch. With neither set, optional Numba acceleration is
    enabled when the dependency is importable.
    """

    return os.environ.get(operator_env, os.environ.get("WOMBAT_NUMBA", "1")).lower()


def numba_enabled(operator_env: str, *, available: bool) -> bool:
    if not available:
        return False
    return numba_mode(operator_env) not in FALSEY_NUMBA_VALUES


def numba_thread_count(operator_env: str) -> int:
    """Return configured Numba worker count for one transport operator."""

    value = os.environ.get(f"{operator_env}_THREADS", os.environ.get("WOMBAT_NUMBA_THREADS", "1"))
    try:
        count = int(value)
    except ValueError as exc:
        raise ValueError(f"{operator_env}_THREADS/WOMBAT_NUMBA_THREADS must be a positive integer") from exc
    if count < 1:
        raise ValueError(f"{operator_env}_THREADS/WOMBAT_NUMBA_THREADS must be a positive integer")
    return count


def apply_numba_thread_count(operator_env: str, *, available: bool) -> int:
    """Set and return the configured Numba worker count if Numba is available."""

    count = numba_thread_count(operator_env)
    if available and set_num_threads is not None:
        set_num_threads(count)
    return count
