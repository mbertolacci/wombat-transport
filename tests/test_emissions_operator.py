from __future__ import annotations

from datetime import datetime
from pathlib import Path

import netCDF4
import numpy as np
import yaml

from wombat_transport.emissions import EmissionsOperator
from wombat_transport.grid import TransportGrid
from wombat_transport.species import Species


def test_same_grid_surface_source_and_missing_species_are_exact(tmp_path):
    grid = _grid(lat=[-45.0, 45.0], lon=[45.0, 135.0, 225.0, 315.0], nlev=3)
    species = _species("A", "B")
    values = np.arange(8.0).reshape(2, 4) + 1.0
    _write_xy_file(tmp_path / "source.nc", values)
    config_path = _write_config(
        tmp_path,
        fields=[
            {
                "name": "field_a",
                "species": "A",
                "path_template": "source.nc",
                "variable": "emis",
                "frequency": "constant",
                "dimensions": "xy",
            }
        ],
    )

    emissions = EmissionsOperator.from_yaml(config_path, root=tmp_path, species=species, grid=grid).evaluate(
        datetime(2014, 9, 1)
    )

    assert emissions.names == ("A", "B")
    assert emissions.shape == (1, 3, 2, 4, 2)
    np.testing.assert_array_equal(emissions.data[0, :-1, :, :, :], np.zeros((2, 2, 4, 2)))
    np.testing.assert_array_equal(emissions.data[0, -1, :, :, 0], values)
    np.testing.assert_array_equal(emissions.data[0, :, :, :, 1], np.zeros((3, 2, 4)))


def test_from_mapping_accepts_inline_emissions_spec(tmp_path):
    grid = _grid(lat=[-45.0, 45.0], lon=[45.0, 135.0, 225.0, 315.0], nlev=1)
    values = np.arange(8.0).reshape(2, 4) + 1.0
    _write_xy_file(tmp_path / "source.nc", values)

    emissions = EmissionsOperator.from_mapping(
        {
            "unit_conversion": "none",
            "missing_species": "zero",
            "scales": {},
            "fields": [_field("field_a", "A", "source.nc")],
        },
        root=tmp_path,
        species=_species("A"),
        grid=grid,
    ).evaluate(datetime(2014, 9, 1))

    np.testing.assert_array_equal(emissions.data[0, -1, :, :, 0], values)


def test_evaluate_surface_flux_returns_compact_surface_field(tmp_path):
    grid = _grid(lat=[-45.0, 45.0], lon=[45.0, 135.0, 225.0, 315.0], nlev=3)
    values = np.arange(8.0).reshape(2, 4) + 1.0
    _write_xy_file(tmp_path / "source.nc", values)
    operator = EmissionsOperator.from_mapping(
        {
            "unit_conversion": "none",
            "missing_species": "zero",
            "scales": {},
            "fields": [_field("field_a", "A", "source.nc")],
        },
        root=tmp_path,
        species=_species("A", "B"),
        grid=grid,
    )

    surface = operator.evaluate_surface_flux(datetime(2014, 9, 1))
    full = surface.to_tracer_field(grid.shape[0])

    assert surface.names == ("A", "B")
    assert surface.shape == (2, 4, 2)
    np.testing.assert_array_equal(surface.data[:, :, 0], values)
    np.testing.assert_array_equal(surface.data[:, :, 1], np.zeros((2, 4)))
    np.testing.assert_array_equal(full.data[0, -1, :, :, 0], values)
    np.testing.assert_array_equal(full.data[0, :-1, :, :, :], np.zeros((2, 2, 4, 2)))


def test_shared_source_file_is_read_once_for_multiple_selected_fields(tmp_path, monkeypatch):
    grid = _grid(lat=[-45.0, 45.0], lon=[45.0, 135.0, 225.0, 315.0], nlev=1)
    values = np.stack(
        [
            np.full((2, 4), 10.0),
            np.full((2, 4), 20.0),
            np.full((2, 4), 30.0),
        ]
    )
    source_path = tmp_path / "source.nc"
    _write_time_npft_file(source_path, [datetime(2014, 9, 1)], values[np.newaxis, ...], npft=None)
    config_path = _write_config(
        tmp_path,
        fields=[
            {
                "name": "field_a",
                "species": "A",
                "path_template": "source.nc",
                "variable": "emis",
                "frequency": "hourly",
                "dimensions": "xy",
                "select": {"dimension": "npft", "value": 1},
            },
            {
                "name": "field_b",
                "species": "B",
                "path_template": "source.nc",
                "variable": "emis",
                "frequency": "hourly",
                "dimensions": "xy",
                "select": {"dimension": "npft", "value": 2},
            },
        ],
    )
    original_dataset = netCDF4.Dataset
    opened = []

    def counting_dataset(path, *args, **kwargs):
        if Path(path) == source_path:
            opened.append(Path(path))
        return original_dataset(path, *args, **kwargs)

    monkeypatch.setattr(netCDF4, "Dataset", counting_dataset)
    operator = EmissionsOperator.from_yaml(config_path, root=tmp_path, species=_species("A", "B"), grid=grid)

    emissions = operator.evaluate_surface_flux(datetime(2014, 9, 1))
    again = operator.evaluate_surface_flux(datetime(2014, 9, 1, 0, 30))

    assert len(opened) == 1
    np.testing.assert_array_equal(emissions.data[:, :, 0], np.full((2, 4), 10.0))
    np.testing.assert_array_equal(emissions.data[:, :, 1], np.full((2, 4), 20.0))
    np.testing.assert_array_equal(again.data, emissions.data)


def test_constant_file_and_multiple_scale_factors_are_multiplied(tmp_path):
    grid = _grid(lat=[-45.0, 45.0], lon=[45.0, 135.0, 225.0, 315.0], nlev=1)
    values = np.full((2, 4), 3.0)
    file_scale = np.arange(8.0).reshape(2, 4) + 1.0
    _write_xy_file(tmp_path / "source.nc", values)
    _write_xy_file(tmp_path / "scale.nc", file_scale, variable_name="scale")
    config_path = _write_config(
        tmp_path,
        scales={
            "mask": {
                "path_template": "scale.nc",
                "variable": "scale",
                "frequency": "constant",
                "dimensions": "xy",
            },
            "negative": {"value": -2.0},
        },
        fields=[
            {
                "name": "field_a",
                "species": "A",
                "path_template": "source.nc",
                "variable": "emis",
                "frequency": "constant",
                "dimensions": "xy",
                "scales": ["mask", "negative"],
            }
        ],
    )

    emissions = EmissionsOperator.from_yaml(config_path, root=tmp_path, species=_species("A"), grid=grid).evaluate(
        datetime(2014, 9, 1)
    )

    np.testing.assert_array_equal(emissions.data[0, -1, :, :, 0], values * file_scale * -2.0)


def test_multiple_entries_sum_and_species_order_follows_species_list(tmp_path):
    grid = _grid(lat=[-45.0, 45.0], lon=[45.0, 135.0, 225.0, 315.0], nlev=1)
    _write_xy_file(tmp_path / "first.nc", np.full((2, 4), 1.0))
    _write_xy_file(tmp_path / "second.nc", np.full((2, 4), 4.0))
    _write_xy_file(tmp_path / "third.nc", np.full((2, 4), 9.0))
    config_path = _write_config(
        tmp_path,
        fields=[
            _field("b_first", "B", "first.nc"),
            _field("a_field", "A", "third.nc"),
            _field("b_second", "B", "second.nc"),
        ],
    )

    emissions = EmissionsOperator.from_yaml(config_path, root=tmp_path, species=_species("A", "B"), grid=grid).evaluate(
        datetime(2014, 9, 1)
    )

    np.testing.assert_array_equal(emissions.data[0, -1, :, :, 0], np.full((2, 4), 9.0))
    np.testing.assert_array_equal(emissions.data[0, -1, :, :, 1], np.full((2, 4), 5.0))


def test_npft_selection_uses_one_based_member_even_with_coordinate(tmp_path):
    grid = _grid(lat=[-45.0, 45.0], lon=[45.0, 135.0, 225.0, 315.0], nlev=1)
    values = np.stack(
        [
            np.full((2, 4), 10.0),
            np.full((2, 4), 20.0),
            np.full((2, 4), 30.0),
            np.full((2, 4), 40.0),
        ]
    )
    _write_time_npft_file(tmp_path / "source.nc", [datetime(2014, 9, 1)], values[np.newaxis, ...], npft=[1, 4, 7, 9])
    config_path = _write_config(
        tmp_path,
        fields=[
            {
                "name": "field_a",
                "species": "A",
                "path_template": "source.nc",
                "variable": "emis",
                "frequency": "hourly",
                "dimensions": "xy",
                "select": {"dimension": "npft", "value": 4},
            }
        ],
    )

    emissions = EmissionsOperator.from_yaml(config_path, root=tmp_path, species=_species("A"), grid=grid).evaluate(
        datetime(2014, 9, 1)
    )

    np.testing.assert_array_equal(emissions.data[0, -1, :, :, 0], np.full((2, 4), 40.0))


def test_npft_selection_falls_back_to_one_based_index_without_coordinate(tmp_path):
    grid = _grid(lat=[-45.0, 45.0], lon=[45.0, 135.0, 225.0, 315.0], nlev=1)
    values = np.stack([np.full((2, 4), 10.0), np.full((2, 4), 20.0), np.full((2, 4), 30.0)])
    _write_time_npft_file(
        tmp_path / "source.nc",
        [datetime(2014, 9, 1)],
        values[np.newaxis, ...],
        npft=None,
    )
    config_path = _write_config(
        tmp_path,
        fields=[
            {
                "name": "field_a",
                "species": "A",
                "path_template": "source.nc",
                "variable": "emis",
                "frequency": "hourly",
                "dimensions": "xy",
                "select": {"dimension": "npft", "value": 2},
            }
        ],
    )

    emissions = EmissionsOperator.from_yaml(config_path, root=tmp_path, species=_species("A"), grid=grid).evaluate(
        datetime(2014, 9, 1)
    )

    np.testing.assert_array_equal(emissions.data[0, -1, :, :, 0], np.full((2, 4), 20.0))


def test_hourly_monthly_time_selection_and_path_template_expansion(tmp_path):
    grid = _grid(lat=[-45.0, 45.0], lon=[45.0, 135.0, 225.0, 315.0], nlev=1)
    path_dir = tmp_path / "2014" / "09"
    path_dir.mkdir(parents=True)
    hourly_values = np.stack([np.full((2, 4), 1.0), np.full((2, 4), 5.0), np.full((2, 4), 9.0)])
    _write_time_xy_file(
        path_dir / "hourly-2014-09-01-02.nc",
        [datetime(2014, 9, 1, 1), datetime(2014, 9, 1, 2), datetime(2014, 9, 1, 3)],
        hourly_values,
    )
    monthly_values = np.stack([np.full((2, 4), 11.0), np.full((2, 4), 17.0)])
    _write_time_xy_file(
        tmp_path / "monthly.nc",
        [datetime(2014, 9, 1), datetime(2014, 10, 1)],
        monthly_values,
    )
    config_path = _write_config(
        tmp_path,
        fields=[
            {
                "name": "hourly",
                "species": "A",
                "path_template": "$YYYY/$MM/hourly-$YYYY-$MM-$DD-$HH.nc",
                "variable": "emis",
                "frequency": "hourly",
                "dimensions": "xy",
            },
            {
                "name": "monthly",
                "species": "B",
                "path_template": "monthly.nc",
                "variable": "emis",
                "frequency": "monthly",
                "dimensions": "xy",
            },
        ],
    )

    emissions = EmissionsOperator.from_yaml(config_path, root=tmp_path, species=_species("A", "B"), grid=grid).evaluate(
        datetime(2014, 9, 1, 2, 30)
    )

    np.testing.assert_array_equal(emissions.data[0, -1, :, :, 0], np.full((2, 4), 5.0))
    np.testing.assert_array_equal(emissions.data[0, -1, :, :, 1], np.full((2, 4), 11.0))


def test_regridding_preserves_area_weighted_flux_mass(tmp_path):
    target_lat = np.array([-45.0, 45.0])
    target_lon = np.array([45.0, 135.0, 225.0, 315.0])
    source_lat = np.array([-67.5, -22.5, 22.5, 67.5])
    source_lon = np.array([22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5])
    grid = _grid(lat=target_lat, lon=target_lon, nlev=1)
    source_values = (np.arange(source_lat.size)[:, np.newaxis] + 2.0) * (
        np.arange(source_lon.size)[np.newaxis, :] + 1.0
    )
    _write_xy_file(tmp_path / "source.nc", source_values, lat=source_lat, lon=source_lon)
    config_path = _write_config(tmp_path, fields=[_field("field_a", "A", "source.nc")])

    emissions = EmissionsOperator.from_yaml(config_path, root=tmp_path, species=_species("A"), grid=grid).evaluate(
        datetime(2014, 9, 1)
    )

    target_values = emissions.data[0, -1, :, :, 0]
    expected = _expected_overlap_regrid(source_values, source_lat, source_lon, target_lat, target_lon)
    np.testing.assert_allclose(target_values, expected, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(
        np.sum(target_values * _cell_areas(target_lat, target_lon)),
        np.sum(source_values * _cell_areas(source_lat, source_lon)),
        rtol=1.0e-13,
    )


def test_regridding_averages_polar_cap_longitudes_like_hemco(tmp_path):
    target_lat = np.array([-89.5, 0.0, 89.5])
    target_lon = np.arange(8, dtype=np.float64) * 45.0 + 22.5
    source_lat = np.array([-89.5, 0.0, 89.5])
    source_lon = np.arange(4, dtype=np.float64) * 90.0 + 45.0
    grid = _grid(lat=target_lat, lon=target_lon, nlev=1)
    source_values = np.array(
        [
            [1.0, 3.0, 5.0, 7.0],
            [10.0, 20.0, 30.0, 40.0],
            [-8.0, -4.0, 4.0, 8.0],
        ],
        dtype=np.float64,
    )
    _write_xy_file(tmp_path / "source.nc", source_values, lat=source_lat, lon=source_lon)
    config_path = _write_config(tmp_path, fields=[_field("field_a", "A", "source.nc")])

    emissions = EmissionsOperator.from_yaml(config_path, root=tmp_path, species=_species("A"), grid=grid).evaluate(
        datetime(2014, 9, 1)
    )

    target_values = emissions.data[0, -1, :, :, 0]
    np.testing.assert_array_equal(target_values[0], np.full(target_lon.size, np.mean(source_values[0])))
    np.testing.assert_array_equal(target_values[-1], np.full(target_lon.size, np.mean(source_values[-1])))


def _species(*names: str) -> list[Species]:
    return [Species(name=name, molecular_weight_g=44.0, background_vv=0.0, full_name=name) for name in names]


def _grid(*, lat: list[float] | np.ndarray, lon: list[float] | np.ndarray, nlev: int) -> TransportGrid:
    lat_array = np.asarray(lat, dtype=np.float64)
    lon_array = np.asarray(lon, dtype=np.float64)
    return TransportGrid(
        lat_deg=lat_array,
        lon_deg=lon_array,
        lev=np.arange(nlev, 0, -1, dtype=np.float64),
        area_m2=_cell_areas(lat_array, lon_array),
        hyai_hpa=np.linspace(0.0, 1000.0, nlev + 1),
        hybi=np.linspace(0.0, 1.0, nlev + 1),
        template_path=Path("synthetic.nc"),
    )


def _field(name: str, species: str, path_template: str) -> dict[str, object]:
    return {
        "name": name,
        "species": species,
        "path_template": path_template,
        "variable": "emis",
        "frequency": "constant",
        "dimensions": "xy",
    }


def _write_config(
    root: Path,
    *,
    fields: list[dict[str, object]],
    scales: dict[str, dict[str, object]] | None = None,
) -> Path:
    path = root / "emissions.yml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "unit_conversion": "none",
                "missing_species": "zero",
                "scales": scales or {},
                "fields": fields,
            },
            handle,
            sort_keys=False,
        )
    return path


def _write_xy_file(
    path: Path,
    values: np.ndarray,
    *,
    variable_name: str = "emis",
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
) -> None:
    array = np.asarray(values, dtype=np.float64)
    lat_values = np.asarray(lat if lat is not None else [-45.0, 45.0], dtype=np.float64)
    lon_values = np.asarray(lon if lon is not None else [45.0, 135.0, 225.0, 315.0], dtype=np.float64)
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("lat", lat_values.size)
        dataset.createDimension("lon", lon_values.size)
        dataset.createVariable("lat", "f8", ("lat",))[:] = lat_values
        dataset.createVariable("lon", "f8", ("lon",))[:] = lon_values
        dataset.createVariable(variable_name, "f8", ("lat", "lon"))[:] = array


def _write_time_xy_file(path: Path, times: list[datetime], values: np.ndarray, *, variable_name: str = "emis") -> None:
    array = np.asarray(values, dtype=np.float64)
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", len(times))
        dataset.createDimension("lat", array.shape[1])
        dataset.createDimension("lon", array.shape[2])
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "hours since 2000-01-01 00:00:00 UTC"
        time[:] = netCDF4.date2num(times, time.units)
        dataset.createVariable("lat", "f8", ("lat",))[:] = [-45.0, 45.0]
        dataset.createVariable("lon", "f8", ("lon",))[:] = [45.0, 135.0, 225.0, 315.0]
        dataset.createVariable(variable_name, "f8", ("time", "lat", "lon"))[:] = array


def _write_time_npft_file(
    path: Path,
    times: list[datetime],
    values: np.ndarray,
    *,
    npft: list[int] | None,
) -> None:
    array = np.asarray(values, dtype=np.float64)
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", len(times))
        dataset.createDimension("npft", array.shape[1])
        dataset.createDimension("lat", array.shape[2])
        dataset.createDimension("lon", array.shape[3])
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "hours since 2000-01-01 00:00:00 UTC"
        time[:] = netCDF4.date2num(times, time.units)
        if npft is not None:
            dataset.createVariable("npft", "i4", ("npft",))[:] = npft
        dataset.createVariable("lat", "f8", ("lat",))[:] = [-45.0, 45.0]
        dataset.createVariable("lon", "f8", ("lon",))[:] = [45.0, 135.0, 225.0, 315.0]
        dataset.createVariable("emis", "f8", ("time", "npft", "lat", "lon"))[:] = array


def _expected_overlap_regrid(
    values: np.ndarray,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    source_area = _cell_areas(source_lat, source_lon)
    target_area = _cell_areas(target_lat, target_lon)
    result = np.zeros((target_lat.size, target_lon.size), dtype=np.float64)
    source_lat_low, source_lat_high = _lat_bounds(source_lat)
    target_lat_low, target_lat_high = _lat_bounds(target_lat)
    source_lon_low, source_lon_high = _lon_bounds(source_lon)
    target_lon_low, target_lon_high = _lon_bounds(target_lon)
    for tj in range(target_lat.size):
        for ti in range(target_lon.size):
            weighted = 0.0
            for sj in range(source_lat.size):
                lat_weight = _lat_overlap(
                    source_lat_low[sj],
                    source_lat_high[sj],
                    target_lat_low[tj],
                    target_lat_high[tj],
                )
                if lat_weight == 0.0:
                    continue
                for si in range(source_lon.size):
                    lon_weight = _lon_overlap(
                        source_lon_low[si],
                        source_lon_high[si],
                        target_lon_low[ti],
                        target_lon_high[ti],
                    )
                    weighted += values[sj, si] * lat_weight * lon_weight
            result[tj, ti] = weighted / target_area[tj, ti]
    source_total = np.sum(values * source_area)
    target_total = np.sum(result * target_area)
    np.testing.assert_allclose(target_total, source_total, rtol=1.0e-13)
    return result


def _cell_areas(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    low_lat, high_lat = _lat_bounds(lat)
    low_lon, high_lon = _lon_bounds(lon)
    lat_weight = np.sin(np.deg2rad(high_lat)) - np.sin(np.deg2rad(low_lat))
    lon_weight = high_lon - low_lon
    return lat_weight[:, np.newaxis] * lon_weight[np.newaxis, :]


def _lat_bounds(lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    midpoints = (lat[:-1] + lat[1:]) / 2.0
    bounds = np.empty(lat.size + 1, dtype=np.float64)
    bounds[1:-1] = midpoints
    bounds[0] = -90.0
    bounds[-1] = 90.0
    return bounds[:-1], bounds[1:]


def _lon_bounds(lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized = np.sort((lon + 360.0) % 360.0)
    step = float(np.median(np.diff(normalized)))
    low = normalized - step / 2.0
    high = normalized + step / 2.0
    low[0] = 0.0
    high[-1] = 360.0
    return low, high


def _lat_overlap(source_low: float, source_high: float, target_low: float, target_high: float) -> float:
    low = max(source_low, target_low)
    high = min(source_high, target_high)
    if high <= low:
        return 0.0
    return float(np.sin(np.deg2rad(high)) - np.sin(np.deg2rad(low)))


def _lon_overlap(source_low: float, source_high: float, target_low: float, target_high: float) -> float:
    low = max(source_low, target_low)
    high = min(source_high, target_high)
    return float(max(0.0, high - low))
