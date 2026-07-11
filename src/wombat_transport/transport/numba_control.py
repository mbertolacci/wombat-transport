from __future__ import annotations

import os


FALSEY_NUMBA_VALUES = frozenset({"0", "false", "no", "off", "none"})


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
