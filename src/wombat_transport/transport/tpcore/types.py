from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wombat_transport.transport.pjc import PjcHorizontalGeometry


@dataclass(frozen=True)
class TpcoreState:
    """One-step TPCORE state arrays in canonical transport order.

    Arrays exposed by this dataclass use the project orientation
    ``(lev_top, lat, lon)`` for 3-D fields and ``(lev_top, lat, lon, tracer)``
    for tracers.
    """

    tracer_conc_after: np.ndarray
    xmass_hpa: np.ndarray
    ymass_hpa: np.ndarray
    surface_pressure_hpa: np.ndarray
    delp1_hpa: np.ndarray
    delpm_hpa: np.ndarray
    delp2_hpa: np.ndarray
    vertical_mass_flux_hpa: np.ndarray


@dataclass(frozen=True)
class TpcoreDeferredState:
    """Borrowed TPCORE handoff with tracers still stored as pressure-weighted mass.

    ``tracer_mass_after_hpa`` is workspace-owned and remains valid only until
    the next TPCORE call with the same shape and thread configuration. Production
    transport consumes it synchronously in VDIFF before that can occur.
    """

    tracer_mass_after_hpa: np.ndarray
    xmass_hpa: np.ndarray
    ymass_hpa: np.ndarray
    surface_pressure_hpa: np.ndarray
    delp1_hpa: np.ndarray
    delpm_hpa: np.ndarray
    delp2_hpa: np.ndarray
    vertical_mass_flux_hpa: np.ndarray


@dataclass(frozen=True)
class TpcoreTrace:
    """Optional one-step tracer checkpoints in canonical transport order."""

    q_after_pole_average: np.ndarray
    dq_after_init: np.ndarray
    q_after_cross_terms: np.ndarray
    dq_after_xtp: np.ndarray
    dq_after_ytp: np.ndarray
    dq_after_fzppm: np.ndarray
    dq_after_fill: np.ndarray
    tracer_conc_after: np.ndarray


@dataclass(frozen=True)
class TpcoreSetup:
    xmass_hpa: np.ndarray
    ymass_hpa: np.ndarray
    surface_pressure_hpa: np.ndarray
    delp1_hpa: np.ndarray
    delpm_hpa: np.ndarray
    delp2_hpa: np.ndarray
    pu_hpa: np.ndarray
    vertical_mass_flux_hpa: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    geofac: np.ndarray
    geofac_pc: float


@dataclass(frozen=True)
class TpcoreGridIdentity:
    """Immutable source values used to build cached TPCORE grid terms."""

    area_m2: np.ndarray
    hyai_hpa: np.ndarray
    hybi: np.ndarray
    lat_deg: np.ndarray


@dataclass(frozen=True)
class TpcoreStaticTerms:
    grid_identity: TpcoreGridIdentity
    pjc_geometry: PjcHorizontalGeometry
    ak_top_hpa: np.ndarray
    dap_top_hpa: np.ndarray
    dbk_top: np.ndarray
    dap_geos_hpa: np.ndarray
    dbk_geos: np.ndarray


@dataclass(frozen=True)
class TpcoreBranchReport:
    shape: tuple[int, int, int]
    max_abs_cx: float
    max_abs_cy: float
    has_large_cx: bool
    has_large_cy: bool
    needs_fxppm: bool
    x_ffsl_active: bool
    x_ffsl_endpoint_active: bool
    x_near_pole_vanleer_active: bool
    jn: tuple[int, ...]
    js: tuple[int, ...]
    unsupported_reasons: tuple[str, ...]

    @property
    def is_supported(self) -> bool:
        return not self.unsupported_reasons
