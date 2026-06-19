from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TracerField:
    """Stacked tracer data plus enough metadata for direct comparison."""

    names: tuple[str, ...]
    data: np.ndarray
    units: tuple[str, ...]
    coords: dict[str, np.ndarray]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape
