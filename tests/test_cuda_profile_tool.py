from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


TOOL = Path(__file__).parents[1] / "tools" / "profile_cuda_run.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("profile_cuda_run", TOOL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cuda_profile_cli_parses_reproducibility_controls():
    tool = _load_tool()

    args = tool._parse_args(
        [
            "run.yml",
            "--dtype",
            "float32",
            "--steps",
            "24",
            "--warmup-steps",
            "2",
            "--block-width",
            "16",
            "--device",
            "1",
            "--nvtx",
            "--output",
            "profile.json",
        ]
    )

    assert args.config == Path("run.yml")
    assert args.dtype == "float32"
    assert args.steps == 24
    assert args.warmup_steps == 2
    assert args.block_width == 16
    assert args.device == 1
    assert args.nvtx
    assert args.output == Path("profile.json")


def test_cuda_profile_cli_rejects_nonpositive_steps():
    tool = _load_tool()

    with pytest.raises(SystemExit):
        tool._parse_args(["run.yml", "--steps", "0"])
