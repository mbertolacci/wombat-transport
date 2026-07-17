"""Tracer-independent preparation for the compiled TPCORE operator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wombat_transport.transport.tpcore import _kernels
from wombat_transport.transport.tpcore.types import TpcoreSetup


@dataclass(frozen=True)
class TpcorePlan:
    """TPCORE setup and derived values shared by every tracer block."""

    setup: TpcoreSetup
    area_1d_m2: np.ndarray
    ua: np.ndarray
    va: np.ndarray
    jn: np.ndarray
    js: np.ndarray


def prepare_tpcore_plan(*, setup: TpcoreSetup, area_m2: np.ndarray) -> TpcorePlan:
    """Prepare tracer-independent TPCORE values for one transport step."""

    if not _kernels._NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    if area_m2.shape != setup.delp1_hpa.shape[1:]:
        raise ValueError("area_m2 shape does not match the TPCORE setup")
    ua = np.empty_like(setup.cx)
    va = np.empty_like(setup.cy)
    jn = np.empty(setup.cx.shape[0], dtype=np.int64)
    js = np.empty(setup.cx.shape[0], dtype=np.int64)
    _kernels._set_cross_terms_numba_kernel(setup.cx, setup.cy, ua, va)
    _kernels._set_jn_js_numba_kernel(setup.cx, jn, js)
    return TpcorePlan(
        setup=setup,
        area_1d_m2=np.ascontiguousarray(area_m2[:, 0], dtype=np.float64),
        ua=ua,
        va=va,
        jn=jn,
        js=js,
    )
