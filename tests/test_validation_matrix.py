from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_validation_matrix_module():
    path = Path(__file__).parents[1] / "tools" / "run_validation_matrix.py"
    spec = importlib.util.spec_from_file_location("run_validation_matrix", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validation_matrix = _load_validation_matrix_module()


def test_thread_roots_remain_below_validation_work_directory():
    prefix = Path("validation_runs/work/obsoperator")

    assert validation_matrix._thread_root(prefix, 1) == Path("validation_runs/work/obsoperator_t1")
    assert validation_matrix._thread_root(prefix, 2) == Path("validation_runs/work/obsoperator_t2")
