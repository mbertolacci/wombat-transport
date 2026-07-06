from __future__ import annotations

import numpy as np

def scalar_mass_by_tracer(field_data: np.ndarray, dry_air_mass_kg: np.ndarray) -> np.ndarray:
    return np.sum(np.asarray(field_data) * np.asarray(dry_air_mass_kg)[np.newaxis, ...], axis=(1, 2, 3, 4))
