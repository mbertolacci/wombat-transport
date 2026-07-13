from __future__ import annotations

from datetime import datetime, timedelta
import gzip
import os
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest
import yaml

from wombat_transport.fields import TracerField
from wombat_transport.grid import TransportGrid
from wombat_transport.obsoperator import (
    MAX_FIELD_NAME_LENGTH,
    MAX_ID_LENGTH,
    ObsOperatorConfig,
    ObsOperatorManager,
    expand_obsoperator_template,
    load_obsoperator_entries,
    parse_obsoperator_config,
    sample_obsoperator_entry,
)
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
            }
        }
    )
    assert config.activate
    assert config.verbose
    assert expand_obsoperator_template(config.output_file or "", datetime(2014, 9, 2, 3, 4, 5)) == (
        "out-20140902_030405.nc4"
    )

    with pytest.raises(KeyError, match="input_file"):
        parse_obsoperator_config({"obsoperator": {"activate": True, "output_file": "out.nc4"}})
    with pytest.raises(TypeError, match="must be a mapping"):
        parse_obsoperator_config({"obsoperator": "on"})


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
    plain.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with gzip.open(compressed, "wt", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle)

    plain_entries = _load(plain)
    gzip_entries = _load(compressed)

    assert plain_entries[0].field_names == ("SpeciesConcVV_A", "SpeciesConcVV_B")
    np.testing.assert_array_equal(plain_entries[0].field_indices, np.array([0, 1]))
    np.testing.assert_array_equal(gzip_entries[0].field_indices, plain_entries[0].field_indices)
    np.testing.assert_array_equal(gzip_entries[0].time.indices, plain_entries[0].time.indices)


def test_time_date_values_are_zero_based_and_floor_to_timestep(tmp_path: Path):
    raw = _entry_raw()
    raw["time_operator"] = {
        "type": "range",
        "unit": "time",
        "start": [20140901, 5],
        "end": [20140901, 20],
        "weights": "normalized",
    }
    path = _write_yaml(tmp_path / "obs.yml", {"entries": [raw]})
    entry = _load(path)[0]

    np.testing.assert_array_equal(entry.time.indices, np.array([0, 1, 2]))
    np.testing.assert_allclose(entry.time.weights, np.full(3, 1.0 / 3.0))


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

    entry = _load(path)[0]

    np.testing.assert_array_equal(entry.time.indices, np.array([5]))


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
    entry = _load(path)[0]

    sampled = sample_obsoperator_entry(entry, _snapshot(), _grid())

    np.testing.assert_allclose(sampled, np.array([expected]))


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
    entry = _load(path)[0]
    sampled = sample_obsoperator_entry(entry, _snapshot(horizontal_gradient=True), _grid())
    np.testing.assert_allclose(sampled, np.array([(1.0 + 11.0 + 2.0 + 12.0) / 4.0]))

    raw["horizontal_operator"] = {
        "type": "point",
        "unit": "degrees",
        "longitude": 180.0,
        "latitude": -90.0,
    }
    _write_yaml(path, {"entries": [raw]})
    wrapped = _load(path)[0]
    np.testing.assert_array_equal(wrapped.horizontal.indices, np.array([[0, 0]]))


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

    entry = load_obsoperator_entries(
        path,
        tracer_names=("A", "B"),
        grid=grid,
        simulation_start=START,
        transport_dt_s=600.0,
    )[0]

    np.testing.assert_array_equal(entry.horizontal.indices, np.array([[expected_index, 0]]))


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
    manager.close()

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
        assert dataset.variables["sample"].description == "sample of the id and field"
        assert _decode_rows(dataset.variables["id"][:]) == ["first", "average"]
        assert _decode_rows(dataset.variables["field"][:]) == ["SpeciesConcVV_A", "SpeciesConcVV_B"]
        np.testing.assert_array_equal(dataset.variables["id_index"][:], np.array([1, 1, 2]))
        np.testing.assert_array_equal(dataset.variables["field_index"][:], np.array([1, 2, 1]))
        np.testing.assert_allclose(dataset.variables["sample"][:], np.array([1.0, 10.0, 2.0], dtype=np.float32))


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
    manager.close()

    assert not (tmp_path / "out-20140901_0000.nc4").exists()
    with netCDF4.Dataset(tmp_path / "out-20140902_0000.nc4") as dataset:
        assert _decode_rows(dataset.variables["id"][:]) == ["cross-day", "second-day"]
        np.testing.assert_allclose(dataset.variables["sample"][:], np.array([2.0, 3.0], dtype=np.float32))


def test_manager_skips_missing_day_and_writes_partial_entry_on_close(tmp_path: Path):
    manager = _manager(tmp_path)
    manager.sample(step_start=START, time_index=0, snapshot=_snapshot())
    manager.close()
    assert not list(tmp_path.glob("out-*.nc4"))

    incomplete = _entry_raw(
        entry_id="partial",
        time={"type": "range", "unit": "time_index", "start": 0, "end": 5, "weights": "equal"},
    )
    _write_yaml(tmp_path / "obs-20140901.yml", {"entries": [incomplete]})
    manager = _manager(tmp_path)
    manager.sample(step_start=START, time_index=0, snapshot=_snapshot())
    manager.close()
    with netCDF4.Dataset(tmp_path / "out-20140901_0000.nc4") as dataset:
        np.testing.assert_allclose(dataset.variables["sample"][:], np.array([1.0], dtype=np.float32))


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
        np.testing.assert_array_equal(_filled_chars(actual.variables["id"][:]), _filled_chars(expected.variables["id"][:]))
        np.testing.assert_array_equal(
            _filled_chars(actual.variables["field"][:]), _filled_chars(expected.variables["field"][:])
        )
        np.testing.assert_array_equal(actual.variables["id_index"][:], expected.variables["id_index"][:])
        np.testing.assert_array_equal(actual.variables["field_index"][:], expected.variables["field_index"][:])
        np.testing.assert_allclose(actual.variables["sample"][:], expected.variables["sample"][:], rtol=1.0e-6, atol=1.0e-12)


def _manager(tmp_path: Path, *, dt_s: float = 600.0) -> ObsOperatorManager:
    return ObsOperatorManager(
        root=tmp_path,
        config=ObsOperatorConfig(
            activate=True,
            input_file="obs-YYYYMMDD.yml",
            output_file="out-YYYYMMDD_hhmm.nc4",
        ),
        start=START,
        transport_dt_s=dt_s,
        tracer_names=("A", "B"),
        grid=_grid(),
    )


def _load(path: Path):
    return load_obsoperator_entries(
        path,
        tracer_names=("A", "B"),
        grid=_grid(),
        simulation_start=START,
        transport_dt_s=600.0,
    )


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
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
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
