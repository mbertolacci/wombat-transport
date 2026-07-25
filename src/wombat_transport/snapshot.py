from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from wombat_transport.fields import TracerField
from wombat_transport.transport.forcing import TransportForcing


@dataclass(frozen=True)
class CompletedStepSnapshot:
    timestamp: datetime
    state: TracerField
    delp_dry_hpa: np.ndarray
    forcing: TransportForcing
