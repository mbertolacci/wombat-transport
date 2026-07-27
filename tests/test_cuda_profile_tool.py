from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from wombat_transport.fields import TracerField
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
            "--random-initial-condition",
            "--random-seed",
            "17",
            "--random-relative-amplitude",
            "0.2",
            "--output-compression-algorithm",
            "zlib",
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
    assert args.random_initial_condition
    assert args.random_seed == 17
    assert args.random_relative_amplitude == 0.2
    assert args.output_compression_algorithm == "zlib"
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


def test_cuda_profile_random_initial_condition_is_reproducible():
    tool = _load_tool()
    data = np.full((1, 2, 2, 3, 2), 4.0e-4)
    field = TracerField.from_canonical(
        names=("A", "B"),
        data=data,
        units=("mol mol-1", "mol mol-1"),
        coords={},
    )

    first = tool._randomize_initial_field(
        field,
        seed=9,
        relative_amplitude=0.1,
    )
    second = tool._randomize_initial_field(
        field,
        seed=9,
        relative_amplitude=0.1,
    )

    np.testing.assert_array_equal(first.block_data, second.block_data)
    assert np.all(first.block_data >= 3.6e-4)
    assert np.all(first.block_data <= 4.4e-4)
    assert not np.array_equal(first.block_data, field.block_data)


def test_cuda_profile_overrides_all_science_output_compression(tmp_path):
    tool = _load_tool()
    config = RunConfig(
        name="profile",
        root=tmp_path,
        source_run_dir=tmp_path,
        species_database=tmp_path / "species.yml",
        initial_restart=None,
        grid_template=tmp_path / "restart.nc4",
        output_dir=tmp_path / "OutputDir",
        diagnostics={},
        comparison={},
        simulation={},
        meteorology={},
        emissions=None,
        outputs={
            "compression": {"algorithm": "blosc_zstd", "level": 1},
            "collections": {
                "Average": {
                    "compression": {
                        "algorithm": "blosc_zstd",
                        "level": 1,
                    }
                },
                "Restart": {},
            },
            "obsoperator": {
                "compression": {
                    "algorithm": "blosc_zstd",
                    "level": 1,
                }
            },
        },
        logging={},
        validation={},
    )

    overridden = tool._override_output_compression_algorithm(config, "zlib")

    assert overridden.outputs["compression"]["algorithm"] == "zlib"
    assert (
        overridden.outputs["collections"]["Average"]["compression"]["algorithm"]
        == "zlib"
    )
    assert overridden.outputs["collections"]["Restart"] == {}
    assert overridden.outputs["obsoperator"]["compression"]["algorithm"] == "zlib"
    assert config.outputs["compression"]["algorithm"] == "blosc_zstd"


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
    assert obsoperator["output_file"] == "obsoperator/Obs.YYYYMMDD.nc4"
    assert (
        obsoperator["restart_file"]
        == "obsoperator/Restart.YYYYMMDD.nc4"
    )
    assert redirected.emissions == str((source / "emissions.yml").resolve())
    assert redirected.meteorology["root"] == str((source / "Met").resolve())
    assert config.outputs["expid"] == "Original/GEOSChem"
    assert config.meteorology["root"] == "Met"
