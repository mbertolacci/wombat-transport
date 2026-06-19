from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wombat_transport.fields import TracerField


@dataclass(frozen=True)
class ComparisonMetrics:
    names: tuple[str, ...]
    max_abs_error: np.ndarray
    mean_abs_error: np.ndarray


def compare_to_time_slice(
    candidate: TracerField,
    reference: TracerField,
    *,
    reference_time_index: int = -1,
) -> ComparisonMetrics:
    """Compare a single-time candidate field to one reference time slice."""

    if candidate.names != reference.names:
        raise ValueError("candidate and reference tracer names do not match")
    if reference.data.shape[1] == 0:
        raise ValueError("reference field has no time records")

    reference_slice = reference.data[:, [reference_time_index], ...]
    if candidate.data.shape != reference_slice.shape:
        raise ValueError(
            f"candidate shape {candidate.data.shape} does not match "
            f"reference slice shape {reference_slice.shape}"
        )

    abs_error = np.abs(candidate.data - reference_slice)
    reduce_axes = tuple(range(1, abs_error.ndim))
    return ComparisonMetrics(
        names=candidate.names,
        max_abs_error=np.max(abs_error, axis=reduce_axes),
        mean_abs_error=np.mean(abs_error, axis=reduce_axes),
    )


def format_metrics(metrics: ComparisonMetrics, *, limit: int = 8) -> str:
    lines = ["tracer,max_abs_error,mean_abs_error"]
    for name, max_abs, mean_abs in zip(
        metrics.names[:limit],
        metrics.max_abs_error[:limit],
        metrics.mean_abs_error[:limit],
        strict=True,
    ):
        lines.append(f"{name},{max_abs:.8e},{mean_abs:.8e}")
    if len(metrics.names) > limit:
        lines.append(f"... {len(metrics.names) - limit} more tracers")
    return "\n".join(lines)
