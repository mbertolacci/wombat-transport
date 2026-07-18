from __future__ import annotations

from wombat_transport.transport.tpcore._operator import (
    _numba_tpcore_enabled as _numba_tpcore_enabled,
    _numba_tpcore_mode as _numba_tpcore_mode,
    run_tpcore_one_step,
    run_tpcore_one_step_with_setup,
)
from wombat_transport.transport.tpcore._reference import _average_const_poles_batch as _average_const_poles_batch
from wombat_transport.transport.tpcore._reference import _calc_advec_cross_terms as _calc_advec_cross_terms
from wombat_transport.transport.tpcore._reference import (
    analyze_tpcore_branches,
    build_tpcore_static_terms,
    setup_tpcore_terms,
    trace_tpcore_one_step,
    trace_tpcore_one_step_with_setup,
    validate_tpcore_branch_support,
)
from wombat_transport.transport.tpcore.types import (
    TpcoreBranchReport,
    TpcoreSetup,
    TpcoreState,
    TpcoreStaticTerms,
    TpcoreTrace,
)

__all__ = [
    "TpcoreBranchReport",
    "TpcoreSetup",
    "TpcoreState",
    "TpcoreStaticTerms",
    "TpcoreTrace",
    "analyze_tpcore_branches",
    "build_tpcore_static_terms",
    "run_tpcore_one_step",
    "run_tpcore_one_step_with_setup",
    "setup_tpcore_terms",
    "trace_tpcore_one_step",
    "trace_tpcore_one_step_with_setup",
    "validate_tpcore_branch_support",
]
