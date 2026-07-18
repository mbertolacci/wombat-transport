from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar


FALSEY_NUMBA_VALUES = frozenset({"0", "false", "no", "off", "none"})
NUMBA_ENV = "WOMBAT_NUMBA"
NUMBA_THREADS_ENV = "WOMBAT_NUMBA_THREADS"

try:  # Optional acceleration dependency.
    from numba import set_num_threads
    from numba.core.compiler import CompilerBase, DefaultPassBuilder
except ImportError:  # pragma: no cover - exercised in environments without numba.
    set_num_threads = None
    CompilerBase = None
    DefaultPassBuilder = None


if CompilerBase is not None:

    class NoAliasCompiler(CompilerBase):
        """Compile an internal kernel whose array arguments never overlap."""

        def define_pipelines(self):
            self.state.flags.noalias = True
            return [DefaultPassBuilder.define_nopython_pipeline(self.state)]

else:  # pragma: no cover - exercised in environments without numba.
    NoAliasCompiler = None


_numba_warning_emitted = False
_numba_threads_configured = False
_configured_numba_thread_count: int | None = None
_numba_configuration_lock = threading.Lock()
_transport_numba_execution_lock = threading.Lock()
_P = ParamSpec("_P")
_R = TypeVar("_R")


def synchronized_transport_numba(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Serialize transport kernels while allowing unrelated Python threads to run."""

    @wraps(function)
    def synchronized(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with _transport_numba_execution_lock:
            return function(*args, **kwargs)

    return synchronized


def numba_mode() -> str:
    """Return the repository-wide Numba mode."""

    return os.environ.get(NUMBA_ENV, "1").lower()


def numba_enabled(*, available: bool) -> bool:
    """Return whether the repository-wide Numba backend is enabled."""

    return available and numba_mode() not in FALSEY_NUMBA_VALUES


def numba_available_and_enabled(*, available: bool | None = None) -> bool:
    """Return whether the repository-wide Numba backend is enabled and available."""

    dependency_available = set_num_threads is not None
    if available is not None:
        dependency_available = dependency_available and available
    return numba_enabled(available=dependency_available)


def numba_thread_count() -> int:
    """Return the repository-wide Numba worker count."""

    value = os.environ.get(NUMBA_THREADS_ENV, "1")
    try:
        count = int(value)
    except ValueError as exc:
        raise ValueError(f"{NUMBA_THREADS_ENV} must be a positive integer") from exc
    if count < 1:
        raise ValueError(f"{NUMBA_THREADS_ENV} must be a positive integer")
    return count


def configure_numba_threads(*, available: bool) -> int:
    """Configure Numba's repository-wide worker count once per process."""

    global _configured_numba_thread_count, _numba_threads_configured
    count = numba_thread_count()
    if not available or set_num_threads is None:
        return count
    with _numba_configuration_lock:
        if not _numba_threads_configured:
            set_num_threads(count)
            _configured_numba_thread_count = count
            _numba_threads_configured = True
        assert _configured_numba_thread_count is not None
        return _configured_numba_thread_count


def warn_if_numba_disabled(logger: logging.Logger) -> None:
    """Emit one prominent warning when the repository-wide Numba backend is disabled."""

    global _numba_warning_emitted
    if _numba_warning_emitted:
        return

    if set_num_threads is None:
        reason = "Numba is unavailable"
    elif not numba_available_and_enabled():
        reason = f"Numba is disabled by {NUMBA_ENV}"
    else:
        return

    logger.warning(
        "MAJOR PERFORMANCE WARNING: %s; accelerated Wombat paths are disabled.",
        reason,
    )
    _numba_warning_emitted = True
