from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from wombat_transport.run_config import RunConfig


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
            "--simulation-end",
            "2014-09-03 00:00",
            "--block-width",
            "16",
            "--device",
            "1",
            "--nvtx",
            "--summary-only",
            "--output",
            "profile.json",
            "--run-dir",
            "profile-run",
        ]
    )

    assert args.config == Path("run.yml")
    assert args.dtype == "float32"
    assert args.steps == 24
    assert args.warmup_steps == 2
    assert args.simulation_end == "2014-09-03 00:00"
    assert args.block_width == 16
    assert args.device == 1
    assert args.nvtx
    assert args.summary_only
    assert args.output == Path("profile.json")
    assert args.run_dir == Path("profile-run")


def test_cuda_profile_cli_rejects_nonpositive_steps():
    tool = _load_tool()

    with pytest.raises(SystemExit):
        tool._parse_args(["run.yml", "--steps", "0"])


def test_cuda_profile_redirects_outputs_but_preserves_inputs(tmp_path):
    tool = _load_tool()
    source = tmp_path / "source"
    source.mkdir()
    config = RunConfig(
        name="profile",
        root=source,
        source_run_dir=source,
        species_database=source / "species.yml",
        initial_restart=source / "restart.nc4",
        grid_template=source / "restart.nc4",
        output_dir=source / "OutputDir",
        diagnostics={},
        comparison={},
        simulation={},
        meteorology={"root": "Met"},
        emissions="emissions.yml",
        outputs={
            "expid": "Original/GEOSChem",
            "obsoperator": {
                "activate": True,
                "input_file": "Obs/obsoperator-YYYYMMDD.yml",
                "output_file": "Original/Obs.YYYYMMDD.nc4",
                "restart_file": "Original/Restart.YYYYMMDD.nc4",
            },
        },
        logging={},
        validation={},
    )

    redirected = tool._redirect_config(
        config,
        tmp_path / "profile-output",
        name_suffix="timed",
    )

    assert redirected.root == (tmp_path / "profile-output").resolve()
    assert redirected.outputs["expid"].startswith(str(redirected.root))
    obsoperator = redirected.outputs["obsoperator"]
    assert obsoperator["input_file"] == str(
        (source / "Obs/obsoperator-YYYYMMDD.yml").resolve()
    )
    assert obsoperator["output_file"].startswith(str(redirected.root))
    assert obsoperator["restart_file"].startswith(str(redirected.root))
    assert redirected.emissions == str((source / "emissions.yml").resolve())
    assert redirected.meteorology["root"] == str((source / "Met").resolve())
    assert config.outputs["expid"] == "Original/GEOSChem"
    assert config.meteorology["root"] == "Met"
