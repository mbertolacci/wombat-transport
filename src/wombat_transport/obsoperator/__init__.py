"""GEOS-Chem-compatible observation operator support."""

from wombat_transport.obsoperator.config import (
    ObsOperatorConfig,
    expand_obsoperator_template,
    parse_obsoperator_config,
)
from wombat_transport.obsoperator.manager import ObsOperatorManager

__all__ = [
    "ObsOperatorConfig",
    "ObsOperatorManager",
    "expand_obsoperator_template",
    "parse_obsoperator_config",
]
