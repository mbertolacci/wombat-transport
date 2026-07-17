from __future__ import annotations

from datetime import datetime, timedelta
import gzip
import os
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest
from yaml12 import read_yaml, write_yaml

import wombat_transport.obsoperator as obsoperator_module
import wombat_transport.obsoperator.input as obsoperator_input
import wombat_transport.obsoperator.sampling as obsoperator_sampling
import wombat_transport.obsoperator.writer as obsoperator_writer
from wombat_transport.fields import TracerField
from wombat_transport.grid import TransportGrid, geos_chem_horizontal_centers
from wombat_transport.obsoperator import (
    ObsOperatorConfig,
    ObsOperatorManager,
    expand_obsoperator_template,
    parse_obsoperator_config,
)
from wombat_transport.obsoperator.state import MAX_FIELD_NAME_LENGTH, MAX_ID_LENGTH
from wombat_transport.output import OutputSnapshot


START = datetime(2014, 9, 1)


def test_obsoperator_config_and_date_template():
    assert not parse_obsoperator_config({}).activate
    config = parse_obsoperator_config(
        {
            "obsoperator": {
                "activate": True,
                "verbose": True,
                "input_file": "obs-YYYYMMDD.yml.gz",
                "output_file": "out-YYYYMMDD_hhmmss.nc4",
                "restart_file": "restart-YYYYMMDD_hhmmss.nc4",
            }
        }
    )
    assert config.activate
    assert config.verbose
    assert expand_obsoperator_template(config.output_file or "", datetime(2014, 9, 2, 3, 4, 5)) == (
        "out-20140902_030405.nc4"
    )

    with pytest.raises(KeyError, match="input_file"):
        parse_obsoperator_config(
            {"obsoperator": {"activate": True, "output_file": "out.nc4", "restart_file": "restart.nc4"}}
        )
    with pytest.raises(KeyError, match="restart_file"):
        parse_obsoperator_config(
            {"obsoperator": {"activate": True, "input_file": "obs.yml", "output_file": "out.nc4"}}
        )
    with pytest.raises(TypeError, match="must be a mapping"):
        parse_obsoperator_config({"obsoperator": "on"})
    with pytest.raises(ValueError, match="no longer supported"):
        parse_obsoperator_config({"obsoperator": {"input_mode": "threaded"}})
    with pytest.raises(ValueError, match="no longer supported"):
        parse_obsoperator_config({"obsoperator": {"writer": "threaded"}})


def test_reference_manager_executes_one_array_kernel_for_all_entries_at_a_step(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WOMBAT_OBSOPERATOR_NUMBA", "0")
    _write_yaml(
        tmp_path / "obs-20140901.yml",
        {"entries": [_entry_raw(entry_id="first"), _entry_raw(entry_id="second")]},
    )
    original = obsoperator_sampling._sample_prepared_entries_kernel
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(obsoperator_sampling, "_sample_prepared_entries_kernel", counted)
    manager = _manager(tmp_path)
    manager.sample(step_start=START, time_index=0, snapshot=_snapshot())
    manager.close(boundary_time=START + timedelta(minutes=10))

    assert calls == 1


def test_numba_manager_matches_python_array_sampler_for_float64_accumulators(tmp_path: Path, monkeypatch):
    entries = []
    operators = [
        (
            {"type": "point", "unit": "grid_index", "longitude": 1, "latitude": 1},
            {"type": "point", "unit": "pressure_level", "value": 2},
        ),
        (
            {
                "type": "box",
                "unit": "grid_index",
                "longitude_start": 1,
                "longitude_end": 2,
                "latitude_start": 1,
                "latitude_end": 2,
                "weights": "equal",
            },
            {"type": "range", "unit": "altitude", "start": 0.0, "end": 1000.0, "weights": "normalized"},
        ),
        (
            {
                "type": "box",
                "unit": "grid_index",
                "longitude_start": 1,
                "longitude_end": 2,
                "latitude_start": 1,
                "latitude_end": 3,
                "weights": "normalized_area",
            },
            {
                "type": "range",
                "unit": "pressure",
                "start": 100.0,
                "end": 900.0,
                "weights": "normalized_pressure",
            },
        ),
        (
            {"type": "point", "unit": "degrees", "longitude": 0.0, "latitude": 0.0},
            {"type": "exact", "unit": "pressure", "values": [900.0, 500.0], "weights": [0.25, 0.75]},
        ),
    ]
    for index, (horizontal, vertical) in enumerate(operators):
        entry = _entry_raw(
            entry_id=f"operator-{index}",
            fields=["SpeciesConcVV_A", "SpeciesConcVV_B"],
            time={"type": "range", "unit": "time_index", "start": 0, "end": 1, "weights": "equal"},
        )
        entry["horizontal_operator"] = horizontal
        entry["vertical_operator"] = vertical
        entries.append(entry)

    results = {}
    for mode in ("0", "1"):
        run_dir = tmp_path / mode
        run_dir.mkdir()
        _write_yaml(run_dir / "obs-20140901.yml", {"entries": entries})
        monkeypatch.setenv("WOMBAT_OBSOPERATOR_NUMBA", mode)
        manager = _manager(run_dir)
        manager.sample(step_start=START, time_index=0, snapshot=_snapshot(horizontal_gradient=True))
        results[mode] = _manager_accumulators(manager)
        manager.close(boundary_time=START + timedelta(minutes=10))

    assert results["1"].keys() == results["0"].keys()
    for entry_id in results["0"]:
        np.testing.assert_array_equal(results["1"][entry_id], results["0"][entry_id])


def test_plain_and_gzip_yaml_parse_identically_and_deduplicate_fields(tmp_path: Path):
    raw = {
        "entries": [
            _entry_raw(
                entry_id="sample",
                fields=["SpeciesConcVV_A", "SpeciesConcVV_?ALL?", "SpeciesConcVV_A"],
            )
        ]
    }
    plain = tmp_path / "obs.yml"
    compressed = tmp_path / "obs.yml.gz"
    write_yaml(raw, plain)
    with gzip.open(compressed, "wt", encoding="utf-8") as handle:
        write_yaml(raw, handle)

    plain_state = _load(plain)
    gzip_state = _load(compressed)

    assert plain_state.field_names[0] == ("SpeciesConcVV_A", "SpeciesConcVV_B")
    np.testing.assert_array_equal(_entry_field_indices(plain_state, 0), np.array([0, 1]))
    np.testing.assert_array_equal(_entry_field_indices(gzip_state, 0), _entry_field_indices(plain_state, 0))
    np.testing.assert_array_equal(gzip_state.remaining_time_us, plain_state.remaining_time_us)


def test_array_loader_builds_flat_selection_and_schedule_tables(tmp_path: Path):
    raw = {
        "entries": [
            _entry_raw(
                entry_id="range",
                fields=["SpeciesConcVV_B", "SpeciesConcVV_?ALL?"],
                time={"type": "range", "unit": "time_index", "start": 0, "end": 2},
            ),
            _entry_raw(
                entry_id="exact",
                fields="SpeciesConcVV_A",
                time={"type": "point", "unit": "time_index", "time": 1},
            ),
        ]
    }
    raw["entries"][0]["horizontal_operator"] = {
        "type": "box",
        "unit": "grid_index",
        "longitude_start": 1,
        "longitude_end": 2,
        "latitude_start": 1,
        "latitude_end": 3,
        "weights": "normalized_area",
    }
    raw["entries"][1]["vertical_operator"] = {
        "type": "exact",
        "unit": "pressure_level",
        "values": [1, 3],
        "weights": [0.25, 0.75],
    }
    path = _write_yaml(tmp_path / "obs.yml", raw)
    state = _load(path)

    assert state.ids == ("range", "exact")
    assert state.field_names == (("SpeciesConcVV_B", "SpeciesConcVV_A"), ("SpeciesConcVV_A",))
    np.testing.assert_array_equal(state.prepared.entry_field_start, [0, 2])
    np.testing.assert_array_equal(state.prepared.entry_field_count, [2, 1])
    np.testing.assert_array_equal(state.prepared.field_indices, [1, 0, 0])
    np.testing.assert_array_equal(state.time_start, [0, 3])
    np.testing.assert_array_equal(state.time_count, [3, 1])
    np.testing.assert_array_equal(state.schedule_entry, [0, 0, 1, 0])
    np.testing.assert_array_equal(state.schedule_count, [1, 2, 1])
    np.testing.assert_array_equal(state.prepared.entry_exact_count, [0, 2])


def test_time_date_ranges_are_half_open_model_periods(tmp_path: Path):
    raw = _entry_raw()
    raw["time_operator"] = {
        "type": "range",
        "unit": "time",
        "start": [20140901, 5],
        "end": [20140901, 20],
        "weights": "normalized",
    }
    path = _write_yaml(tmp_path / "obs.yml", {"entries": [raw]})
    state = _load(path)

    np.testing.assert_array_equal(_entry_time_indices(state, 0), np.array([0, 1]))
    np.testing.assert_allclose(_entry_time_weights(state, 0), np.full(2, 1.0 / 2.0))


def test_time_date_exact_boundaries_map_to_period_ending_there(tmp_path: Path):
    entries = [
        _entry_raw(
            entry_id="hour",
            time={
                "type": "range",
                "unit": "time",
                "start": [20140901, 0],
                "end": [20140901, 100],
            },
        ),
        _entry_raw(
            entry_id="point",
            time={"type": "point", "unit": "time", "time": [20140901, 110]},
        ),
        _entry_raw(
            entry_id="degenerate-range",
            time={
                "type": "range",
                "unit": "time",
                "start": [20140901, 120],
                "end": [20140901, 120],
            },
        ),
    ]
    state = _load(_write_yaml(tmp_path / "obs.yml", {"entries": entries}))

    np.testing.assert_array_equal(_entry_time_indices(state, 0), np.arange(6))
    np.testing.assert_allclose(_entry_time_weights(state, 0), np.full(6, 1.0 / 6.0))
    np.testing.assert_array_equal(_entry_time_indices(state, 1), np.array([6]))
    np.testing.assert_array_equal(_entry_time_indices(state, 2), np.array([7]))


def test_yaml_clock_with_leading_zero_is_read_as_decimal(tmp_path: Path):
    path = tmp_path / "obs.yml"
    path.write_text(
        """entries:
  - id: leading-zero
    fields: SpeciesConcVV_A
    time_operator:
      type: point
      unit: time
      time: [20140901, 0052]
    horizontal_operator:
      type: point
      unit: grid_index
      longitude: 1
      latitude: 1
    vertical_operator:
      type: point
      unit: pressure_level
      value: 1
""",
        encoding="utf-8",
    )

    state = _load(path)

    np.testing.assert_array_equal(_entry_time_indices(state, 0), np.array([5]))


@pytest.mark.parametrize(
    ("vertical", "expected"),
    [
        ({"type": "point", "unit": "pressure_level", "value": 1}, 1.0),
        ({"type": "range", "unit": "pressure_level", "start": 1, "end": 3, "weights": "normalized"}, 2.0),
        ({"type": "range", "unit": "pressure_level", "start": 1, "end": 3, "weights": "equal"}, 6.0),
        (
            {"type": "range", "unit": "pressure_level", "start": 1, "end": 3, "weights": "normalized_pressure"},
            (300.0 + 800.0 + 897.0) / 999.0,
        ),
        (
            {"type": "range", "unit": "pressure_level", "start": 1, "end": 3, "weights": "pressure"},
            300.0 + 800.0 + 897.0,
        ),
        ({"type": "point", "unit": "pressure", "value": 500.0}, 2.0),
        ({"type": "point", "unit": "altitude", "value": 0.0}, 1.0),
        ({"type": "exact", "unit": "pressure_level", "values": [1, 3], "weights": [0.25, 0.75]}, 2.5),
    ],
)
def test_vertical_operator_modes(tmp_path: Path, vertical: dict, expected: float):
    raw = _entry_raw()
    raw["vertical_operator"] = vertical
    path = _write_yaml(tmp_path / "obs.yml", {"entries": [raw]})
    state = _load(path)
    sampled = _sample_state(state, _snapshot(), _grid())[0, :1]

    np.testing.assert_allclose(sampled, np.array([expected]))


def test_sampler_maps_global_tracer_indices_into_blocks(tmp_path: Path):
    path = _write_yaml(
        tmp_path / "obs.yml",
        {"entries": [_entry_raw(fields=["SpeciesConcVV_A", "SpeciesConcVV_B"])]},
    )
    state = _load(path)
    canonical = _snapshot()
    blocked = OutputSnapshot(
        timestamp=canonical.timestamp,
        state=canonical.state.reblock(1),
        delp_dry_hpa=canonical.delp_dry_hpa,
        forcing=canonical.forcing,
    )

    expected = _sample_state(state, canonical, _grid())
    actual = _sample_state(state, blocked, _grid())

    np.testing.assert_array_equal(actual, expected)


def test_horizontal_box_weight_modes_and_longitude_wrap(tmp_path: Path):
    raw = _entry_raw()
    raw["horizontal_operator"] = {
        "type": "box",
        "unit": "grid_index",
        "longitude_start": 1,
        "longitude_end": 2,
        "latitude_start": 1,
        "latitude_end": 2,
        "weights": "normalized",
    }
    path = _write_yaml(tmp_path / "obs.yml", {"entries": [raw]})
    state = _load(path)
    sampled = _sample_state(state, _snapshot(horizontal_gradient=True), _grid())[0, :1]
    np.testing.assert_allclose(sampled, np.array([(1.0 + 11.0 + 2.0 + 12.0) / 4.0]))

    raw["horizontal_operator"] = {
        "type": "point",
        "unit": "degrees",
        "longitude": 180.0,
        "latitude": -90.0,
    }
    _write_yaml(path, {"entries": [raw]})
    wrapped = _load(path)
    np.testing.assert_array_equal(wrapped.prepared.horizontal_lat, np.array([0]))
    np.testing.assert_array_equal(wrapped.prepared.horizontal_lon, np.array([0]))


@pytest.mark.parametrize(("latitude", "expected_index"), [(-90.0, 0), (-89.0, 1), (89.0, 90), (90.0, 90)])
def test_geos_polar_degree_boundaries(tmp_path: Path, latitude: float, expected_index: int):
    lat = np.concatenate(([-89.5], np.arange(-88.0, 90.0, 2.0), [89.5]))
    lon = np.arange(-180.0, 180.0, 2.5)
    grid = TransportGrid(
        lat_deg=lat,
        lon_deg=lon,
        lev=np.array([1.0, 2.0, 3.0]),
        area_m2=np.ones((lat.size, lon.size)),
        hyai_hpa=np.array([1000.0, 700.0, 300.0, 1.0]),
        hybi=np.zeros(4),
        template_path=Path("unused.nc4"),
    )
    raw = _entry_raw()
    raw["horizontal_operator"] = {
        "type": "point",
        "unit": "degrees",
        "longitude": -180.0,
        "latitude": latitude,
    }
    path = _write_yaml(tmp_path / "obs.yml", {"entries": [raw]})

    state = obsoperator_input._load_obsoperator_array_state(
        path,
        tracer_names=("A", "B"),
        grid=grid,
        simulation_start=START,
        transport_dt_s=600.0,
    )

    np.testing.assert_array_equal(state.prepared.horizontal_lat, np.array([expected_index]))
    np.testing.assert_array_equal(state.prepared.horizontal_lon, np.array([0]))


@pytest.mark.parametrize(
    ("latitude", "expected_index"),
    [(-90.0, 0), (-89.0, 0), (-88.0, 1), (86.0, 44), (88.0, 45), (90.0, 45)],
)
def test_geos_4x5_polar_degree_boundaries(tmp_path: Path, latitude: float, expected_index: int):
    lat, lon = geos_chem_horizontal_centers("4x5")
    grid = TransportGrid(
        lat_deg=lat,
        lon_deg=lon,
        lev=np.array([1.0, 2.0, 3.0]),
        area_m2=np.ones((lat.size, lon.size)),
        hyai_hpa=np.array([1000.0, 700.0, 300.0, 1.0]),
        hybi=np.zeros(4),
        template_path=Path("unused.nc4"),
    )
    raw = _entry_raw()
    raw["horizontal_operator"] = {
        "type": "point",
        "unit": "degrees",
        "longitude": 180.0,
        "latitude": latitude,
    }
    path = _write_yaml(tmp_path / "obs.yml", {"entries": [raw]})

    state = obsoperator_input._load_obsoperator_array_state(
        path,
        tracer_names=("A", "B"),
        grid=grid,
        simulation_start=START,
        transport_dt_s=600.0,
    )

    np.testing.assert_array_equal(state.prepared.horizontal_lat, np.array([expected_index]))
    np.testing.assert_array_equal(state.prepared.horizontal_lon, np.array([0]))


def test_parser_rejects_obsolete_species_and_invalid_values(tmp_path: Path):
    obsolete = _entry_raw()
    obsolete.pop("fields")
    obsolete["species"] = "A"
    path = _write_yaml(tmp_path / "obs.yml", {"entries": [obsolete]})
    with pytest.raises(ValueError, match="species is obsolete"):
        _load(path)

    invalid = _entry_raw(fields="SpeciesConcVV_MISSING")
    _write_yaml(path, {"entries": [invalid]})
    with pytest.raises(ValueError, match="unknown tracer"):
        _load(path)

    invalid = _entry_raw()
    invalid["vertical_operator"] = {
        "type": "exact",
        "unit": "pressure_level",
        "values": [1, 2],
        "weights": [1.0],
    }
    _write_yaml(path, {"entries": [invalid]})
    with pytest.raises(ValueError, match="same length"):
        _load(path)


def test_manager_writes_fortran_compatible_netcdf_and_first_step_sample(tmp_path: Path):
    raw = {
        "entries": [
            _entry_raw(entry_id="first", fields=["SpeciesConcVV_A", "SpeciesConcVV_B"]),
            _entry_raw(
                entry_id="average",
                time={"type": "range", "unit": "time_index", "start": 0, "end": 1},
            ),
        ]
    }
    _write_yaml(tmp_path / "obs-20140901.yml", raw)
    manager = _manager(tmp_path)

    manager.sample(step_start=START, time_index=0, snapshot=_snapshot())
    second = _snapshot(scale=3.0)
    manager.sample(step_start=START + timedelta(minutes=10), time_index=1, snapshot=second)
    manager.close(boundary_time=START + timedelta(minutes=20))

    output = tmp_path / "out-20140901_0000.nc4"
    with netCDF4.Dataset(output) as dataset:
        assert set(dataset.dimensions) == {"entries", "id_chars", "fields", "field_chars", "samples"}
        assert dataset.dimensions["entries"].isunlimited()
        assert dataset.dimensions["fields"].isunlimited()
        assert dataset.dimensions["samples"].isunlimited()
        assert len(dataset.dimensions["id_chars"]) == MAX_ID_LENGTH
        assert len(dataset.dimensions["field_chars"]) == MAX_FIELD_NAME_LENGTH
        assert dataset.variables["id"].dimensions == ("entries", "id_chars")
        assert dataset.variables["field"].dimensions == ("fields", "field_chars")
        assert dataset.variables["sample"].dtype == np.dtype("float32")
        assert dataset.variables["id_index"].dtype == np.dtype("int32")
        assert dataset.variables["sample"].filters()["zlib"]
        assert dataset.variables["sample"].filters()["complevel"] == 1
        assert dataset.variables["id"].chunking() == [256, MAX_ID_LENGTH]
        assert dataset.variables["field"].chunking() == [64, MAX_FIELD_NAME_LENGTH]
        assert dataset.variables["id_index"].chunking() == [16_384]
        assert dataset.variables["field_index"].chunking() == [16_384]
        assert dataset.variables["sample"].chunking() == [16_384]
        assert dataset.variables["sample"].description == "sample of the id and field"
        assert _decode_rows(dataset.variables["id"][:]) == ["first", "average"]
        assert _decode_rows(dataset.variables["field"][:]) == ["SpeciesConcVV_A", "SpeciesConcVV_B"]
        np.testing.assert_array_equal(dataset.variables["id_index"][:], np.array([1, 1, 2]))
        np.testing.assert_array_equal(dataset.variables["field_index"][:], np.array([1, 2, 1]))
        np.testing.assert_allclose(dataset.variables["sample"][:], np.array([1.0, 10.0, 2.0], dtype=np.float32))


def test_science_writer_stages_bounded_batches_and_flushes_remainder_on_close(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(obsoperator_writer, "SCIENCE_STAGE_ENTRIES", 2)
    output = tmp_path / "staged.nc4"
    writer = obsoperator_writer._ObsOperatorNetCDFWriter(output)
    path = _write_yaml(
        tmp_path / "obs.yml",
        {
            "entries": [
                _entry_raw(entry_id="one", fields=["SpeciesConcVV_A", "SpeciesConcVV_B"]),
                _entry_raw(entry_id="two", fields="SpeciesConcVV_B"),
                _entry_raw(entry_id="three", fields=["SpeciesConcVV_C", "SpeciesConcVV_A"]),
            ]
        },
    )
    state = obsoperator_input._load_obsoperator_array_state(
        path,
        tracer_names=("A", "B", "C"),
        grid=_grid(),
        simulation_start=START,
        transport_dt_s=600.0,
    )
    state.field_accumulator[:] = [1.0, 2.0, 3.0, 4.0, 5.0]

    writer.write_array_entries(((state, np.array([0])),))
    assert not output.exists()
    writer.write_array_entries(((state, np.array([1])),))
    assert output.exists()
    writer.write_array_entries(((state, np.array([2])),))
    writer.close()

    with netCDF4.Dataset(output) as dataset:
        assert _decode_rows(dataset.variables["id"][:]) == ["one", "two", "three"]
        assert _decode_rows(dataset.variables["field"][:]) == [
            "SpeciesConcVV_A",
            "SpeciesConcVV_B",
            "SpeciesConcVV_C",
        ]
        np.testing.assert_array_equal(dataset.variables["id_index"][:], [1, 1, 2, 3, 3])
        np.testing.assert_array_equal(dataset.variables["field_index"][:], [1, 2, 2, 3, 1])
        np.testing.assert_array_equal(dataset.variables["sample"][:], [1.0, 2.0, 3.0, 4.0, 5.0])


def test_manager_rotates_cross_day_entries_to_new_output(tmp_path: Path):
    first_day = _entry_raw(
        entry_id="cross-day",
        time={"type": "range", "unit": "time_index", "start": 0, "end": 1},
    )
    second_day = _entry_raw(
        entry_id="second-day",
        time={"type": "point", "unit": "time_index", "time": 1},
    )
    _write_yaml(tmp_path / "obs-20140901.yml", {"entries": [first_day]})
    _write_yaml(tmp_path / "obs-20140902.yml", {"entries": [second_day]})
    manager = _manager(tmp_path, dt_s=86400.0)

    manager.sample(step_start=START, time_index=0, snapshot=_snapshot())
    manager.sample(step_start=START + timedelta(days=1), time_index=1, snapshot=_snapshot(scale=3.0))
    manager.close(boundary_time=START + timedelta(days=2))

    assert not (tmp_path / "out-20140901_0000.nc4").exists()
    with netCDF4.Dataset(tmp_path / "out-20140902_0000.nc4") as dataset:
        assert _decode_rows(dataset.variables["id"][:]) == ["cross-day", "second-day"]
        np.testing.assert_allclose(dataset.variables["sample"][:], np.array([2.0, 3.0], dtype=np.float32))


def test_manager_skips_missing_day_and_restarts_incomplete_entry_without_partial_output(tmp_path: Path):
    manager = _manager(tmp_path)
    manager.sample(step_start=START, time_index=0, snapshot=_snapshot())
    manager.close(boundary_time=START + timedelta(minutes=10))
    assert not list(tmp_path.glob("out-*.nc4"))

    incomplete = _entry_raw(
        entry_id="partial",
        time={"type": "range", "unit": "time_index", "start": 0, "end": 5, "weights": "equal"},
    )
    _write_yaml(tmp_path / "obs-20140901.yml", {"entries": [incomplete]})
    manager = _manager(tmp_path)
    manager.sample(step_start=START, time_index=0, snapshot=_snapshot())
    manager.close(boundary_time=START + timedelta(minutes=10))
    assert not list(tmp_path.glob("out-*.nc4"))
    assert (tmp_path / "restart-20140901_001000.nc4").is_file()


@pytest.mark.parametrize(("weighting", "expected"), [("normalized", 2.0), ("equal", 6.0)])
def test_restart_continues_accumulator_without_partial_output_or_final_division(
    tmp_path: Path,
    weighting: str,
    expected: float,
):
    entry = _entry_raw(
        entry_id="continued",
        fields=["SpeciesConcVV_A", "SpeciesConcVV_B"],
        time={"type": "range", "unit": "time_index", "start": 0, "end": 2, "weights": weighting},
    )
    _write_yaml(tmp_path / "obs-20140901.yml", {"entries": [entry]})

    first = _manager(tmp_path)
    first.sample(step_start=START, time_index=0, snapshot=_snapshot(scale=1.0))
    first.close(boundary_time=START + timedelta(minutes=10))
    (tmp_path / "obs-20140901.yml").unlink()
    assert not list(tmp_path.glob("out-*.nc4"))

    first_restart = tmp_path / "restart-20140901_001000.nc4"
    with netCDF4.Dataset(first_restart) as dataset:
        expected_first = 1.0 / 3.0 if weighting == "normalized" else 1.0
        np.testing.assert_array_equal(
            dataset.variables["field_accumulator"][:],
            np.array([expected_first, 10.0 * expected_first], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            dataset.variables["remaining_time_us"][:],
            np.array([_time_us(START + timedelta(minutes=10)), _time_us(START + timedelta(minutes=20))]),
        )

    second = _manager(tmp_path, start=START + timedelta(minutes=10))
    second.sample(step_start=START + timedelta(minutes=10), time_index=0, snapshot=_snapshot(scale=2.0))
    second.close(boundary_time=START + timedelta(minutes=20))
    assert not list(tmp_path.glob("out-*.nc4"))

    third = _manager(tmp_path, start=START + timedelta(minutes=20))
    third.sample(step_start=START + timedelta(minutes=20), time_index=0, snapshot=_snapshot(scale=3.0))
    temporal_weights = [1.0 / 3.0] * 3 if weighting == "normalized" else [1.0] * 3
    expected_float64 = np.zeros(2, dtype=np.float64)
    for scale, weight in zip((1.0, 2.0, 3.0), temporal_weights, strict=True):
        expected_float64 += weight * np.array([scale, 10.0 * scale], dtype=np.float64)
    np.testing.assert_array_equal(_manager_accumulators(third)["continued"], expected_float64)
    third.close(boundary_time=START + timedelta(minutes=30))

    with netCDF4.Dataset(tmp_path / "out-20140901_0020.nc4") as dataset:
        assert _decode_rows(dataset.variables["id"][:]) == ["continued"]
        np.testing.assert_allclose(
            dataset.variables["sample"][:],
            np.array([expected, 10.0 * expected], dtype=np.float32),
        )
    with netCDF4.Dataset(tmp_path / "restart-20140901_003000.nc4") as dataset:
        assert len(dataset.dimensions["entries"]) == 0


def test_per_entry_operator_classes_and_compatibility_functions_are_removed():
    for name in (
        "ObsOperatorEntry",
        "TimeOperator",
        "HorizontalOperator",
        "VerticalOperator",
        "load_obsoperator_entries",
        "sample_obsoperator_entry",
    ):
        assert not hasattr(obsoperator_module, name)


def test_restart_schema_preserves_exact_operator_before_first_sample(tmp_path: Path):
    entry = _entry_raw(
        entry_id="exact",
        fields="SpeciesConcVV_B",
        time={"type": "point", "unit": "time_index", "time": 1},
    )
    entry["horizontal_operator"] = {
        "type": "box",
        "unit": "grid_index",
        "longitude_start": 1,
        "longitude_end": 2,
        "latitude_start": 1,
        "latitude_end": 3,
        "weights": "normalized_area",
    }
    entry["vertical_operator"] = {
        "type": "exact",
        "unit": "pressure_level",
        "values": [1, 3],
        "weights": [0.25, 0.75],
    }
    _write_yaml(tmp_path / "obs-20140901.yml", {"entries": [entry]})
    manager = _manager(tmp_path)
    manager.sample(step_start=START, time_index=0, snapshot=_snapshot())
    manager.close(boundary_time=START + timedelta(minutes=10))

    path = tmp_path / "restart-20140901_001000.nc4"
    with netCDF4.Dataset(path) as dataset:
        assert dataset.format == "Wombat ObsOperator restart"
        assert dataset.format_version == 2
        assert dataset.restart_time_us == _time_us(START + timedelta(minutes=10))
        assert dataset.transport_timestep_seconds == 600.0
        assert len(dataset.grid_signature) == 64
        assert set(dataset.dimensions) == {
            "entries", "entry_fields", "remaining_times", "vertical_values", "id_chars", "field_chars",
            "horizontal_bound", "vertical_bound",
        }
        assert dataset.variables["field_accumulator"].dtype == np.dtype("float64")
        assert dataset.variables["remaining_time_us"].dtype == np.dtype("int64")
        assert dataset.variables["remaining_time_weight"].dtype == np.dtype("float64")
        assert dataset.variables["field_accumulator"].filters()["zlib"]
        assert _decode_rows(dataset.variables["id"][:]) == ["exact"]
        assert _decode_rows(dataset.variables["field_name"][:]) == ["SpeciesConcVV_B"]
        np.testing.assert_array_equal(dataset.variables["horizontal_bounds"][:], np.array([[0, 1, 0, 2]]))
        np.testing.assert_array_equal(dataset.variables["vertical_value"][:], np.array([1.0, 3.0]))
        np.testing.assert_array_equal(dataset.variables["vertical_weight"][:], np.array([0.25, 0.75]))


@pytest.mark.parametrize(
    "vertical",
    [
        {"type": "point", "unit": "pressure_level", "value": 1},
        {"type": "range", "unit": "pressure_level", "start": 1, "end": 3, "weights": "equal"},
        {"type": "range", "unit": "pressure_level", "start": 1, "end": 3, "weights": "normalized"},
        {"type": "range", "unit": "pressure", "start": 100.0, "end": 900.0, "weights": "pressure"},
        {
            "type": "range",
            "unit": "pressure",
            "start": 100.0,
            "end": 900.0,
            "weights": "normalized_pressure",
        },
        {"type": "range", "unit": "altitude", "start": 0.0, "end": 1000.0, "weights": "normalized"},
        {"type": "exact", "unit": "pressure", "values": [900.0, 500.0], "weights": [0.25, 0.75]},
        {"type": "exact", "unit": "altitude", "values": [0.0, 1000.0], "weights": [0.5, 0.5]},
    ],
)
def test_restart_round_trip_preserves_vertical_operator_modes(tmp_path: Path, vertical: dict):
    entry = _entry_raw(time={"type": "point", "unit": "time_index", "time": 1})
    entry["vertical_operator"] = vertical
    _write_yaml(tmp_path / "obs-20140901.yml", {"entries": [entry]})
    expected = _sample_state(_load(tmp_path / "obs-20140901.yml"), _snapshot(), _grid())[0, :1]

    first = _manager(tmp_path)
    first.sample(step_start=START, time_index=0, snapshot=_snapshot())
    first.close(boundary_time=START + timedelta(minutes=10))
    (tmp_path / "obs-20140901.yml").unlink()
    second = _manager(tmp_path, start=START + timedelta(minutes=10))
    second.sample(step_start=START + timedelta(minutes=10), time_index=0, snapshot=_snapshot())
    second.close(boundary_time=START + timedelta(minutes=20))

    with netCDF4.Dataset(tmp_path / "out-20140901_0010.nc4") as dataset:
        np.testing.assert_allclose(dataset.variables["sample"][:], expected.astype(np.float32))


@pytest.mark.parametrize("weighting", ["area", "normalized_area", "normalized", "equal"])
def test_restart_round_trip_preserves_horizontal_weighting_modes(tmp_path: Path, weighting: str):
    entry = _entry_raw(time={"type": "point", "unit": "time_index", "time": 1})
    entry["horizontal_operator"] = {
        "type": "box",
        "unit": "grid_index",
        "longitude_start": 1,
        "longitude_end": 2,
        "latitude_start": 1,
        "latitude_end": 2,
        "weights": weighting,
    }
    _write_yaml(tmp_path / "obs-20140901.yml", {"entries": [entry]})
    expected = _sample_state(
        _load(tmp_path / "obs-20140901.yml"),
        _snapshot(horizontal_gradient=True),
        _grid(),
    )[0, :1]

    first = _manager(tmp_path)
    first.sample(step_start=START, time_index=0, snapshot=_snapshot())
    first.close(boundary_time=START + timedelta(minutes=10))
    (tmp_path / "obs-20140901.yml").unlink()
    second = _manager(tmp_path, start=START + timedelta(minutes=10))
    second.sample(
        step_start=START + timedelta(minutes=10),
        time_index=0,
        snapshot=_snapshot(horizontal_gradient=True),
    )
    second.close(boundary_time=START + timedelta(minutes=20))

    with netCDF4.Dataset(tmp_path / "out-20140901_0010.nc4") as dataset:
        np.testing.assert_allclose(dataset.variables["sample"][:], expected.astype(np.float32))


def test_restart_resumes_cross_midnight_entry_in_completion_day_output(tmp_path: Path):
    entry = _entry_raw(
        entry_id="cross-midnight",
        time={"type": "range", "unit": "time_index", "start": 0, "end": 1},
    )
    _write_yaml(tmp_path / "obs-20140901.yml", {"entries": [entry]})
    first = _manager(tmp_path, dt_s=86400.0)
    first.sample(step_start=START, time_index=0, snapshot=_snapshot())
    first.close(boundary_time=START + timedelta(days=1))
    assert not list(tmp_path.glob("out-*.nc4"))

    second = _manager(tmp_path, start=START + timedelta(days=1), dt_s=86400.0)
    second.sample(step_start=START + timedelta(days=1), time_index=0, snapshot=_snapshot(scale=3.0))
    second.close(boundary_time=START + timedelta(days=2))
    with netCDF4.Dataset(tmp_path / "out-20140902_0000.nc4") as dataset:
        np.testing.assert_allclose(dataset.variables["sample"][:], np.array([2.0], dtype=np.float32))


def test_restart_rejects_duplicate_id_from_daily_input(tmp_path: Path):
    entry = _entry_raw(
        entry_id="continued",
        time={"type": "range", "unit": "time_index", "start": 0, "end": 1},
    )
    _write_yaml(tmp_path / "obs-20140901.yml", {"entries": [entry]})
    first = _manager(tmp_path)
    first.sample(step_start=START, time_index=0, snapshot=_snapshot())
    first.close(boundary_time=START + timedelta(minutes=10))
    second = _manager(tmp_path, start=START + timedelta(minutes=10))
    with pytest.raises(ValueError, match="duplicate active ObsOperator id 'continued'"):
        second.sample(step_start=START + timedelta(minutes=10), time_index=0, snapshot=_snapshot())


def test_restart_rejects_incompatible_boundary_timestep_grid_and_fields(tmp_path: Path):
    entry = _entry_raw(
        entry_id="continued",
        time={"type": "range", "unit": "time_index", "start": 0, "end": 2},
    )
    _write_yaml(tmp_path / "obs-20140901.yml", {"entries": [entry]})
    manager = _manager(tmp_path)
    manager.sample(step_start=START, time_index=0, snapshot=_snapshot())
    manager.close(boundary_time=START + timedelta(minutes=10))
    restart = "restart-20140901_001000.nc4"

    with pytest.raises(ValueError, match="timestep changed"):
        _manager(tmp_path, start=START + timedelta(minutes=10), dt_s=300.0, restart_file=restart)
    changed_grid = _grid()
    changed_grid.area_m2[0, 0] += 1.0
    with pytest.raises(ValueError, match="grid changed"):
        _manager(tmp_path, start=START + timedelta(minutes=10), grid=changed_grid, restart_file=restart)
    with pytest.raises(ValueError, match="does not match simulation start"):
        _manager(tmp_path, start=START + timedelta(minutes=20), restart_file=restart)
    with pytest.raises(ValueError, match="missing field"):
        _manager(
            tmp_path,
            start=START + timedelta(minutes=10),
            tracer_names=("B",),
            restart_file=restart,
        )


def test_restart_maps_fields_by_name_across_tracer_reordering_and_additions(tmp_path: Path):
    entry = _entry_raw(
        entry_id="continued",
        fields=["SpeciesConcVV_A", "SpeciesConcVV_B"],
        time={"type": "range", "unit": "time_index", "start": 0, "end": 1},
    )
    _write_yaml(tmp_path / "obs-20140901.yml", {"entries": [entry]})
    manager = _manager(tmp_path)
    manager.sample(step_start=START, time_index=0, snapshot=_snapshot())
    manager.close(boundary_time=START + timedelta(minutes=10))

    restored = _manager(
        tmp_path,
        start=START + timedelta(minutes=10),
        tracer_names=("B", "EXTRA", "A"),
        restart_file="restart-20140901_001000.nc4",
    )
    np.testing.assert_array_equal(_entry_field_indices(restored._states[0], 0), np.array([2, 0]))


def test_restart_missing_policy_and_corrupt_offsets(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="restart missing"):
        _manager(tmp_path, restart_missing="error")

    entry = _entry_raw(
        time={"type": "range", "unit": "time_index", "start": 0, "end": 1},
    )
    _write_yaml(tmp_path / "obs-20140901.yml", {"entries": [entry]})
    manager = _manager(tmp_path)
    manager.sample(step_start=START, time_index=0, snapshot=_snapshot())
    manager.close(boundary_time=START + timedelta(minutes=10))
    restart_path = tmp_path / "restart-20140901_001000.nc4"
    with netCDF4.Dataset(restart_path, "r+") as dataset:
        dataset.variables["field_start"][0] = 1
    with pytest.raises(ValueError, match="invalid contiguous fields offsets"):
        _manager(
            tmp_path,
            start=START + timedelta(minutes=10),
            restart_file=restart_path.name,
        )


def test_local_geos_chem_obsoperator_output_parity_if_available():
    gc_path = os.environ.get("WOMBAT_GC_OBSOPERATOR_OUTPUT")
    wombat_path = os.environ.get("WOMBAT_OBSOPERATOR_OUTPUT")
    if not gc_path or not wombat_path or not Path(gc_path).is_file() or not Path(wombat_path).is_file():
        pytest.skip(
            "set WOMBAT_GC_OBSOPERATOR_OUTPUT and WOMBAT_OBSOPERATOR_OUTPUT to matching local NetCDF outputs"
        )

    with netCDF4.Dataset(gc_path) as expected, netCDF4.Dataset(wombat_path) as actual:
        assert list(actual.dimensions) == list(expected.dimensions)
        assert list(actual.variables) == list(expected.variables)
        for name in expected.variables:
            expected_variable = expected.variables[name]
            actual_variable = actual.variables[name]
            assert actual_variable.dimensions == expected_variable.dimensions
            assert actual_variable.dtype == expected_variable.dtype
            assert actual_variable.filters() == expected_variable.filters()
            assert actual_variable.ncattrs() == expected_variable.ncattrs()
            for attribute in expected_variable.ncattrs():
                assert getattr(actual_variable, attribute) == getattr(expected_variable, attribute)
        expected_samples = _obsoperator_output_samples(expected)
        actual_samples = _obsoperator_output_samples(actual)
        assert actual_samples.keys() == expected_samples.keys()
        keys = sorted(expected_samples)
        expected_values = np.asarray([expected_samples[key] for key in keys])
        actual_values = np.asarray([actual_samples[key] for key in keys])
        close = np.isclose(actual_values, expected_values, rtol=1.0e-6, atol=1.0e-12)
        if not np.all(close):
            absolute_error = np.abs(actual_values - expected_values)
            worst = np.argsort(absolute_error)[-5:][::-1]
            details = "; ".join(
                f"{keys[index]!r}: GC={expected_values[index]:.9g}, "
                f"Wombat={actual_values[index]:.9g}, abs={absolute_error[index]:.3g}"
                for index in worst
            )
            pytest.fail(
                f"{np.count_nonzero(~close)}/{len(keys)} canonical ObsOperator samples differ; "
                f"max abs error={np.max(absolute_error):.3g}; worst: {details}"
            )


def test_local_daily_input_contains_restartable_cross_day_entries_if_available(tmp_path: Path):
    input_dir = os.environ.get("WOMBAT_OBSOPERATOR_INPUT_DIR")
    if not input_dir:
        pytest.skip("set WOMBAT_OBSOPERATOR_INPUT_DIR to the directory containing real daily gzip inputs")
    path = Path(input_dir) / "obsoperator-20140901.yml.gz"
    if not path.is_file():
        pytest.skip(f"local ObsOperator input is unavailable: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        raw = read_yaml(handle)
    cross_day = [
        entry
        for entry in raw["entries"]
        if entry.get("time_operator", {}).get("type") == "range"
        and entry["time_operator"].get("end", [None])[0] == 20140902
    ]
    assert cross_day
    subset = _write_yaml(tmp_path / "cross-day.yml", {"entries": cross_day[:10]})
    state = obsoperator_input._load_obsoperator_array_state(
        subset,
        tracer_names=("CO2",),
        grid=_global_grid(),
        simulation_start=START,
        transport_dt_s=600.0,
    )
    assert state.entry_count
    assert all(
        state.remaining_time_us[_entry_time_slice(state, entry_index)][-1]
        >= _time_us(START + timedelta(days=1))
        for entry_index in range(state.entry_count)
    )


def _manager(
    tmp_path: Path,
    *,
    start: datetime = START,
    dt_s: float = 600.0,
    tracer_names: tuple[str, ...] = ("A", "B"),
    grid: TransportGrid | None = None,
    restart_file: str = "restart-YYYYMMDD_hhmmss.nc4",
    restart_missing: str = "ignore",
) -> ObsOperatorManager:
    return ObsOperatorManager(
        root=tmp_path,
        config=ObsOperatorConfig(
            activate=True,
            input_file="obs-YYYYMMDD.yml",
            output_file="out-YYYYMMDD_hhmm.nc4",
            restart_file=restart_file,
            restart_missing=restart_missing,
        ),
        start=start,
        transport_dt_s=dt_s,
        tracer_names=tracer_names,
        grid=grid or _grid(),
    )


def _load(path: Path):
    return obsoperator_input._load_obsoperator_array_state(
        path,
        tracer_names=("A", "B"),
        grid=_grid(),
        simulation_start=START,
        transport_dt_s=600.0,
    )


def _entry_field_slice(state, entry_index: int) -> slice:
    start = int(state.prepared.entry_field_start[entry_index])
    return slice(start, start + int(state.prepared.entry_field_count[entry_index]))


def _entry_field_indices(state, entry_index: int) -> np.ndarray:
    return state.prepared.field_indices[_entry_field_slice(state, entry_index)]


def _entry_time_slice(state, entry_index: int) -> slice:
    start = int(state.time_start[entry_index])
    return slice(start, start + int(state.time_count[entry_index]))


def _entry_time_indices(state, entry_index: int) -> np.ndarray:
    times = state.remaining_time_us[_entry_time_slice(state, entry_index)]
    return (times - _time_us(START)) // (600 * 1_000_000)


def _entry_time_weights(state, entry_index: int) -> np.ndarray:
    return state.remaining_time_weight[_entry_time_slice(state, entry_index)]


def _manager_accumulators(manager: ObsOperatorManager) -> dict[str, np.ndarray]:
    return {
        state.ids[entry_index]: state.field_accumulator[_entry_field_slice(state, entry_index)].copy()
        for state in manager._states
        for entry_index in range(state.entry_count)
    }


def _sample_state(state, snapshot: OutputSnapshot, grid: TransportGrid) -> np.ndarray:
    entries = np.arange(state.entry_count, dtype=np.int64)
    samples = np.empty((state.entry_count, state.prepared.max_field_count), dtype=np.float64)
    prepared = state.prepared
    state_bottom = np.asarray(
        snapshot.state.block_data[0, :, ::-1, :, :, :], dtype=np.float64
    )
    obsoperator_sampling._sample_prepared_entries_kernel(
        state_bottom,
        snapshot.state.block_width,
        np.asarray(snapshot.forcing.wet_surface_pressure_hpa[0], dtype=np.float64),
        np.asarray(snapshot.forcing.specific_humidity_kg_kg[0], dtype=np.float64),
        np.asarray(snapshot.forcing.temperature_k[0], dtype=np.float64),
        grid.hyai_hpa,
        grid.hybi,
        entries,
        prepared.entry_field_start,
        prepared.entry_field_count,
        prepared.entry_horizontal_start,
        prepared.entry_horizontal_count,
        prepared.entry_vertical_type,
        prepared.entry_vertical_unit,
        prepared.entry_vertical_weighting,
        prepared.entry_vertical_lower,
        prepared.entry_vertical_upper,
        prepared.entry_exact_start,
        prepared.entry_exact_count,
        prepared.field_indices,
        prepared.horizontal_lat,
        prepared.horizontal_lon,
        prepared.horizontal_weight,
        prepared.exact_value,
        prepared.exact_weight,
        samples,
    )
    return samples


def _grid() -> TransportGrid:
    lat = np.array([-45.0, 0.0, 45.0])
    lon = np.array([-180.0, -90.0, 0.0, 90.0])
    area = np.arange(1, lat.size * lon.size + 1, dtype=np.float64).reshape(lat.size, lon.size)
    return TransportGrid(
        lat_deg=lat,
        lon_deg=lon,
        lev=np.array([1.0, 2.0, 3.0]),
        area_m2=area,
        hyai_hpa=np.array([1000.0, 700.0, 300.0, 1.0]),
        hybi=np.zeros(4),
        template_path=Path("unused.nc4"),
    )


def _global_grid() -> TransportGrid:
    lat = np.concatenate(([-89.5], np.arange(-88.0, 90.0, 2.0), [89.5]))
    lon = np.arange(-180.0, 180.0, 2.5)
    return TransportGrid(
        lat_deg=lat,
        lon_deg=lon,
        lev=np.arange(1.0, 48.0),
        area_m2=np.ones((lat.size, lon.size)),
        hyai_hpa=np.linspace(1000.0, 0.0, 48),
        hybi=np.zeros(48),
        template_path=Path("unused.nc4"),
    )


def _snapshot(*, scale: float = 1.0, horizontal_gradient: bool = False) -> OutputSnapshot:
    grid = _grid()
    nlev, nlat, nlon = grid.shape
    bottom = np.zeros((nlev, nlat, nlon, 2), dtype=np.float64)
    for level in range(nlev):
        bottom[level, :, :, 0] = float(level + 1)
        bottom[level, :, :, 1] = float(10 * (level + 1))
    if horizontal_gradient:
        for lat_index in range(nlat):
            for lon_index in range(nlon):
                bottom[:, lat_index, lon_index, 0] += 10.0 * lon_index + lat_index
    bottom *= scale
    state = TracerField(
        names=("A", "B"),
        data=bottom[::-1][np.newaxis, ...],
        units=("mol mol-1 dry", "mol mol-1 dry"),
        coords={},
    )
    forcing = SimpleNamespace(
        wet_surface_pressure_hpa=np.full((1, nlat, nlon), 1000.0),
        specific_humidity_kg_kg=np.zeros((1, nlev, nlat, nlon)),
        temperature_k=np.full((1, nlev, nlat, nlon), 280.0),
    )
    return OutputSnapshot(
        timestamp=START + timedelta(minutes=10),
        state=state,
        delp_dry_hpa=np.ones((1, nlev, nlat, nlon)),
        forcing=forcing,  # type: ignore[arg-type]
    )


def _entry_raw(
    *,
    entry_id: str = "sample",
    fields="SpeciesConcVV_A",
    time: dict | None = None,
) -> dict:
    return {
        "id": entry_id,
        "fields": fields,
        "time_operator": time or {"type": "point", "unit": "time_index", "time": 0},
        "horizontal_operator": {
            "type": "point",
            "unit": "grid_index",
            "longitude": 1,
            "latitude": 1,
        },
        "vertical_operator": {"type": "point", "unit": "pressure_level", "value": 1},
    }


def _write_yaml(path: Path, raw: dict) -> Path:
    write_yaml(raw, path)
    return path


def _decode_rows(values: np.ndarray) -> list[str]:
    result = []
    for row in values:
        result.append(row.tobytes().split(b"\x00", 1)[0].decode("utf-8"))
    return result


def _filled_chars(values: np.ndarray) -> np.ndarray:
    if np.ma.isMaskedArray(values):
        return values.filled(b"\x00")
    return np.asarray(values)


def _obsoperator_output_samples(dataset: netCDF4.Dataset) -> dict[tuple[str, str], float]:
    ids = _decode_rows(_filled_chars(dataset.variables["id"][:]))
    fields = _decode_rows(_filled_chars(dataset.variables["field"][:]))
    id_index = np.asarray(dataset.variables["id_index"][:], dtype=np.int64) - 1
    field_index = np.asarray(dataset.variables["field_index"][:], dtype=np.int64) - 1
    samples = np.asarray(dataset.variables["sample"][:], dtype=np.float64)
    return {
        (ids[int(id_value)], fields[int(field_value)]): float(sample)
        for id_value, field_value, sample in zip(id_index, field_index, samples, strict=True)
    }


def _time_us(value: datetime) -> int:
    delta = value - datetime(1970, 1, 1)
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
