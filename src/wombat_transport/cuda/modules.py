"""Loading of optional raw CUDA source modules."""

from __future__ import annotations

from importlib.resources import files
from typing import Any

from wombat_transport.cuda.runtime import require_cupy

STRICT_CUDA_OPTIONS = (
    "--std=c++11",
    "--fmad=false",
    "--prec-div=true",
    "--prec-sqrt=true",
)


def load_raw_module(
    source_name: str,
    *,
    name_expressions: tuple[str, ...],
    options: tuple[str, ...] = STRICT_CUDA_OPTIONS,
) -> Any:
    """Load one package-owned CUDA source file without importing CuPy eagerly."""

    source = files("wombat_transport.cuda.sources").joinpath(source_name).read_text()
    cupy = require_cupy()
    return cupy.RawModule(
        code=source,
        options=options,
        name_expressions=name_expressions,
    )
