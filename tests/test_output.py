from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from wombat_transport.fields import TracerField
from wombat_transport.grid import load_transport_grid
from wombat_transport import history_accumulation
from wombat_transport.history_accumulation import accumulate_history_sum
from wombat_transport.io import FIXED_GRID, load_restart, load_species_conc
from wombat_transport.output import (
    HistoryOutputManager,
    OutputCollectionConfig,
    OutputStorageConfig,
    OutputSnapshot,
    expand_history_template,
    parse_history_interval,
    parse_output_storage,
    parse_output_writer,
    write_restart_collection,
    write_species_conc_collection,
)
from wombat_transport.species import Species

BASE_RESTART = "tests/fixtures/io_readers_2x25_v1/restart.nc4"


def test_history_interval_parses_monthly_daily_and_hourly_values():
    monthly = parse_history_interval("00000100 000000")
    daily = parse_history_interval("00000001 000000")
    three_hourly = parse_history_interval("00000000 030000")

    assert monthly.months == 1
    assert daily.days == 1
    assert three_hourly.seconds == 3 * 60 * 60
    assert monthly.add_to(datetime(2014, 9, 1)) == datetime(2014, 10, 1)


def test_history_template_expands_geos_chem_date_tokens():
    assert (
        expand_history_template("GEOSChem.Restart.%y4%m2%d2_%h2%n2z.nc4", datetime(2014, 9, 1, 3, 40))
        == "GEOSChem.Restart.20140901_0340z.nc4"
    )


def test_output_storage_defaults_and_validates():
    storage = parse_output_storage({})

    assert storage.dtype == "float32"
    assert storage.compression.enabled
    assert storage.compression.level == 1
    assert storage.compression.shuffle
    assert storage.chunking.rank1 is None
    assert storage.chunking.rank4 is None

    explicit = parse_output_storage(
        {
            "dtype": "float64",
            "compression": {"enabled": False, "level": 0, "shuffle": False},
            "chunking": {"rank3": [1, 2, 3], "rank4": [1, 1, 2, 3]},
        }
    )
    assert explicit.dtype == "float64"
    assert not explicit.compression.enabled
    assert explicit.chunking.rank3 == (1, 2, 3)
    assert explicit.chunking.rank4 == (1, 1, 2, 3)


def test_output_writer_defaults_and_validates():
    assert parse_output_writer({}).mode == "sync"
    assert parse_output_writer({"writer": "threaded"}).mode == "threaded"

    try:
        parse_output_writer({"writer": "process"})
    except ValueError as exc:
        assert "outputs.writer" in str(exc)
    else:
        raise AssertionError("accepted invalid output writer mode")


def test_history_accumulation_native_and_numba_are_bitwise_equal(monkeypatch):
    if not history_accumulation._NUMBA_AVAILABLE:
        pytest.skip("numba is unavailable")
    rng = np.random.default_rng(20140901)
    values = [rng.standard_normal((2, 3, 4, 5)) for _ in range(6)]
    native = rng.standard_normal((2, 3, 4, 5))
    accelerated = native.copy()

    monkeypatch.setenv("WOMBAT_HISTORY_NUMBA", "0")
    for value in values:
        accumulate_history_sum(native, value)
    monkeypatch.setenv("WOMBAT_HISTORY_NUMBA", "1")
    monkeypatch.setenv("WOMBAT_HISTORY_NUMBA_THREADS", "2")
    for value in values:
        accumulate_history_sum(accelerated, value)

    np.testing.assert_array_equal(accelerated, native)


def test_history_accumulation_uses_native_fallback_without_numba(monkeypatch):
    accumulator = np.arange(12, dtype=np.float64).reshape(3, 4)
    expected = accumulator + 0.25
    monkeypatch.setattr(history_accumulation, "_NUMBA_AVAILABLE", False)
    monkeypatch.setenv("WOMBAT_HISTORY_NUMBA", "1")

    accumulate_history_sum(accumulator, np.full_like(accumulator, 0.25))

    np.testing.assert_array_equal(accumulator, expected)


def test_output_storage_rejects_invalid_values():
    for raw in (
        {"dtype": "float16"},
        {"compression": {"level": 10}},
        {"chunking": {"rank4": [1, 2, 3]}},
        {"chunking": {"rank3": [1, 0, 3]}},
    ):
        try:
            parse_output_storage(raw)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"accepted invalid storage config {raw}")


def test_species_conc_writer_roundtrips_geos_chem_style_collection(tmp_path):
    first = _field(("A", "B"), values=(1.0, 10.0))
    second = _field(("A", "B"), values=(3.0, 30.0))
    output_path = tmp_path / "GEOSChem.SpeciesConcThreeHourly.20140901_0000z.nc4"

    write_species_conc_collection(
        output_path,
        [
            (datetime(2014, 9, 1, 0), first),
            (datetime(2014, 9, 1, 3), second),
        ],
        BASE_RESTART,
        title="GEOS-Chem diagnostic collection: SpeciesConcThreeHourly",
    )

    loaded = load_species_conc(output_path)
    assert loaded.names == ("A", "B")
    assert loaded.shape == (2, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"], 2)
    np.testing.assert_allclose(loaded.data[0, :, :, :, 0], first.data[0, :, :, :, 0])
    np.testing.assert_allclose(loaded.data[1, :, :, :, 1], second.data[0, :, :, :, 1])
    with netCDF4.Dataset(output_path) as dataset:
        assert dataset.title == "GEOS-Chem diagnostic collection: SpeciesConcThreeHourly"
        variable = dataset.variables["SpeciesConcVV_A"]
        assert variable.dtype == np.dtype("float32")
        assert variable.filters()["zlib"]
        assert variable.filters()["shuffle"]
        assert variable.filters()["complevel"] == 1
        assert variable.chunking() == [1, 1, FIXED_GRID["lat"], FIXED_GRID["lon"]]
        assert dataset.variables["time"].chunking() == [512]
        np.testing.assert_array_equal(dataset.variables["time"][:], np.array([0.0, 180.0]))


def test_species_conc_writer_reads_logical_tracers_from_blocks(tmp_path):
    field = _field(("A", "B"), values=(1.0, 10.0))
    blocked = field.reblock(1)
    output_path = tmp_path / "blocked.nc4"

    write_species_conc_collection(
        output_path,
        [(datetime(2014, 9, 1), blocked)],
        BASE_RESTART,
        title="blocked",
    )

    loaded = load_species_conc(output_path)
    np.testing.assert_allclose(loaded.data[0], field.data[0])


def test_species_conc_writer_honors_float64_dtype_and_explicit_chunks(tmp_path):
    field = _field(("A",), values=(1.0,))
    output_path = tmp_path / "float64_species_conc.nc4"
    storage = OutputStorageConfig(
        dtype="float64",
        chunking=parse_output_storage({"chunking": {"rank4": [1, 2, 91, 144]}}).chunking,
    )

    write_species_conc_collection(
        output_path,
        [(datetime(2014, 9, 1, 0), field)],
        BASE_RESTART,
        title="GEOS-Chem diagnostic collection: SpeciesConcThreeHourly",
        storage=storage,
    )

    with netCDF4.Dataset(output_path) as dataset:
        variable = dataset.variables["SpeciesConcVV_A"]
        assert variable.dtype == np.dtype("float64")
        assert variable.chunking() == [1, 2, FIXED_GRID["lat"], FIXED_GRID["lon"]]


def test_restart_writer_roundtrips_species_and_writes_met_fields(tmp_path):
    field = _field(("A",), values=(4.0,))
    level_values = np.arange(FIXED_GRID["lev"], dtype=np.float64).reshape(1, FIXED_GRID["lev"], 1, 1)
    delp = np.broadcast_to(
        10.0 + level_values,
        (1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"]),
    ).copy()
    sphu_start = np.broadcast_to(
        0.001 + level_values * 1.0e-5,
        (1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"]),
    ).copy()
    temperature_start = np.broadcast_to(
        250.0 + level_values,
        (1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"]),
    ).copy()
    forcing = SimpleNamespace(
        surface_pressure_pa=np.full((1, FIXED_GRID["lat"], FIXED_GRID["lon"]), 101000.0),
        surface_pressure_start_pa=np.full((1, FIXED_GRID["lat"], FIXED_GRID["lon"]), 100700.0),
        restart_surface_pressure_pa=np.full((1, FIXED_GRID["lat"], FIXED_GRID["lon"]), 101100.0),
        i3_start_wet_surface_pressure_hpa=np.full((1, FIXED_GRID["lat"], FIXED_GRID["lon"]), 1007.0),
        restart_wet_surface_pressure_hpa=np.full((1, FIXED_GRID["lat"], FIXED_GRID["lon"]), 1011.0),
        i3_start_dry_surface_pressure_hpa=np.full((1, FIXED_GRID["lat"], FIXED_GRID["lon"]), 1006.0),
        restart_dry_surface_pressure_hpa=np.full((1, FIXED_GRID["lat"], FIXED_GRID["lon"]), 1009.0),
        specific_humidity_kg_kg=np.full((1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"]), 0.002),
        i3_start_specific_humidity_kg_kg=sphu_start,
        restart_specific_humidity_kg_kg=np.full((1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"]), 0.005),
        temperature_k=np.full((1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"]), 280.0),
        i3_start_temperature_k=temperature_start,
        restart_temperature_k=np.full((1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"]), 281.0),
    )
    output_path = tmp_path / "GEOSChem.Restart.20140901_0010z.nc4"

    write_restart_collection(
        output_path,
        OutputSnapshot(
            timestamp=datetime(2014, 9, 1, 0, 10),
            state=field,
            delp_dry_hpa=delp,
            forcing=forcing,  # type: ignore[arg-type]
        ),
        BASE_RESTART,
        fields=("SpeciesRst_?ALL?", "Met_DELPDRY", "Met_PS1DRY", "Met_PS1WET", "Met_SPHU1", "Met_TMPU1"),
        title="GEOS-Chem diagnostic collection: Restart",
    )

    loaded = load_restart(output_path, [Species("A", 44.0, 0.0, "A")])
    np.testing.assert_allclose(loaded.data, field.data)
    with netCDF4.Dataset(output_path) as dataset:
        rst = dataset.variables["SpeciesRst_A"]
        assert rst.dtype == np.dtype("float32")
        assert rst.filters()["complevel"] == 1
        assert rst.chunking() == [1, 1, FIXED_GRID["lat"], FIXED_GRID["lon"]]
        assert dataset.variables["hyai"].dtype == np.dtype("float64")
        assert dataset.variables["hybi"].dtype == np.dtype("float64")
        assert dataset.variables["lat"].dtype == np.dtype("float64")
        assert dataset.variables["lon"].dtype == np.dtype("float64")
        assert dataset.variables["Met_DELPDRY"].shape == (1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"])
        assert dataset.variables["Met_PS1DRY"].shape == (1, FIXED_GRID["lat"], FIXED_GRID["lon"])
        assert dataset.variables["Met_PS1DRY"].chunking() == [1, FIXED_GRID["lat"], FIXED_GRID["lon"]]
        assert dataset.variables["Met_PS1WET"].shape == (1, FIXED_GRID["lat"], FIXED_GRID["lon"])
        np.testing.assert_allclose(dataset.variables["Met_DELPDRY"][:], delp)
        np.testing.assert_allclose(dataset.variables["Met_PS1DRY"][:], 1006.0)
        np.testing.assert_allclose(dataset.variables["Met_PS1WET"][:], 1007.0)
        assert dataset.variables["Met_SPHU1"].units == "g kg-1"
        np.testing.assert_allclose(dataset.variables["Met_SPHU1"][:], sphu_start * 1000.0)
        np.testing.assert_allclose(dataset.variables["Met_TMPU1"][:], temperature_start)
    written_grid = load_transport_grid(output_path)
    template_grid = load_transport_grid(BASE_RESTART)
    np.testing.assert_array_equal(written_grid.hyai_hpa, template_grid.hyai_hpa)
    np.testing.assert_array_equal(written_grid.hybi, template_grid.hybi)


def test_output_manager_uses_post_step_arithmetic_averages(tmp_path):
    manager = HistoryOutputManager(
        root=tmp_path,
        template_path=BASE_RESTART,
        expid="OutputDir/GEOSChem",
        collections=(
            OutputCollectionConfig(
                name="SpeciesConcThreeHourly",
                filename=None,
                template="%y4%m2%d2_%h2%n2z.nc4",
                frequency=parse_history_interval("00000000 003000"),
                duration=parse_history_interval("00000001 000000"),
                mode="time-averaged",
                fields=("SpeciesConcVV_?ADV?",),
            ),
        ),
        start=datetime(2014, 9, 1),
    )
    forcing = SimpleNamespace(
        surface_pressure_pa=np.full((1, FIXED_GRID["lat"], FIXED_GRID["lon"]), 101000.0),
        specific_humidity_kg_kg=np.zeros((1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"])),
        temperature_k=np.zeros((1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"])),
    )
    delp = np.ones((1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"]))

    manager.record_step(
        OutputSnapshot(datetime(2014, 9, 1, 0, 10), _field(("A",), values=(1.0,)), delp, forcing)  # type: ignore[arg-type]
    )
    manager.record_step(
        OutputSnapshot(datetime(2014, 9, 1, 0, 20), _field(("A",), values=(3.0,)), delp, forcing)  # type: ignore[arg-type]
    )
    manager.close()

    output = tmp_path / "OutputDir" / "GEOSChem.SpeciesConcThreeHourly.20140901_0000z.nc4"
    loaded = load_species_conc(output)
    np.testing.assert_allclose(loaded.data[0, :, :, :, 0], _field(("A",), values=(2.0,)).data[0, :, :, :, 0])


def test_output_manager_streams_species_conc_across_daily_files(tmp_path):
    manager = HistoryOutputManager(
        root=tmp_path,
        template_path=BASE_RESTART,
        expid="OutputDir/GEOSChem",
        collections=(
            OutputCollectionConfig(
                name="SpeciesConcHourly",
                filename=None,
                template="%y4%m2%d2_%h2%n2z.nc4",
                frequency=parse_history_interval("00000000 010000"),
                duration=parse_history_interval("00000001 000000"),
                mode="time-averaged",
                fields=("SpeciesConcVV_?ADV?",),
            ),
        ),
        start=datetime(2014, 9, 1),
    )
    forcing = SimpleNamespace(
        surface_pressure_pa=np.full((1, FIXED_GRID["lat"], FIXED_GRID["lon"]), 101000.0),
        specific_humidity_kg_kg=np.zeros((1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"])),
        temperature_k=np.zeros((1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"])),
    )
    delp = np.ones((1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"]))

    manager.record_step(
        OutputSnapshot(datetime(2014, 9, 1, 0, 10), _field(("A",), values=(1.0,)), delp, forcing)  # type: ignore[arg-type]
    )
    manager.record_step(
        OutputSnapshot(datetime(2014, 9, 2, 0, 10), _field(("A",), values=(2.0,)), delp, forcing)  # type: ignore[arg-type]
    )
    manager.close()

    first = tmp_path / "OutputDir" / "GEOSChem.SpeciesConcHourly.20140901_0000z.nc4"
    second = tmp_path / "OutputDir" / "GEOSChem.SpeciesConcHourly.20140902_0000z.nc4"
    np.testing.assert_allclose(load_species_conc(first).data[0, :, :, :, 0], _field(("A",), values=(1.0,)).data[0, :, :, :, 0])
    np.testing.assert_allclose(load_species_conc(second).data[0, :, :, :, 0], _field(("A",), values=(2.0,)).data[0, :, :, :, 0])


@pytest.mark.parametrize("history_numba", ("0", "1"))
def test_threaded_output_manager_matches_sync_species_conc(tmp_path, monkeypatch, history_numba):
    if history_numba == "1" and not history_accumulation._NUMBA_AVAILABLE:
        pytest.skip("numba is unavailable")
    monkeypatch.setenv("WOMBAT_HISTORY_NUMBA", history_numba)
    monkeypatch.setenv("WOMBAT_HISTORY_NUMBA_THREADS", "2")
    forcing = _forcing()
    delp = np.ones((1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"]))
    collection = OutputCollectionConfig(
        name="SpeciesConcHourly",
        filename=None,
        template="%y4%m2%d2_%h2%n2z.nc4",
        frequency=parse_history_interval("00000000 010000"),
        duration=parse_history_interval("00000001 000000"),
        mode="time-averaged",
        fields=("SpeciesConcVV_?ADV?",),
    )

    for writer_mode in ("sync", "threaded"):
        manager = HistoryOutputManager(
            root=tmp_path / writer_mode,
            template_path=BASE_RESTART,
            expid="OutputDir/GEOSChem",
            collections=(collection,),
            start=datetime(2014, 9, 1),
            writer=parse_output_writer({"writer": writer_mode}),
        )
        manager.record_step(
            OutputSnapshot(datetime(2014, 9, 1, 0, 10), _field(("A",), values=(1.0,)), delp, forcing)  # type: ignore[arg-type]
        )
        manager.record_step(
            OutputSnapshot(datetime(2014, 9, 1, 0, 20), _field(("A",), values=(3.0,)), delp, forcing)  # type: ignore[arg-type]
        )
        manager.record_step(
            OutputSnapshot(datetime(2014, 9, 1, 1, 10), _field(("A",), values=(5.0,)), delp, forcing)  # type: ignore[arg-type]
        )
        manager.close()

    sync = load_species_conc(
        tmp_path / "sync" / "OutputDir" / "GEOSChem.SpeciesConcHourly.20140901_0000z.nc4"
    )
    threaded = load_species_conc(
        tmp_path / "threaded" / "OutputDir" / "GEOSChem.SpeciesConcHourly.20140901_0000z.nc4"
    )
    np.testing.assert_array_equal(threaded.data, sync.data)
    assert threaded.names == sync.names


def test_threaded_output_manager_reraises_writer_errors(tmp_path):
    manager = HistoryOutputManager(
        root=tmp_path,
        template_path=tmp_path / "missing_template.nc4",
        expid="OutputDir/GEOSChem",
        collections=(
            OutputCollectionConfig(
                name="SpeciesConcHourly",
                filename=None,
                template="%y4%m2%d2_%h2%n2z.nc4",
                frequency=parse_history_interval("00000000 010000"),
                duration=parse_history_interval("00000001 000000"),
                mode="time-averaged",
                fields=("SpeciesConcVV_?ADV?",),
            ),
        ),
        start=datetime(2014, 9, 1),
        writer=parse_output_writer({"writer": "threaded"}),
    )
    forcing = _forcing()
    delp = np.ones((1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"]))
    manager.record_step(
        OutputSnapshot(datetime(2014, 9, 1, 0, 10), _field(("A",), values=(1.0,)), delp, forcing)  # type: ignore[arg-type]
    )

    try:
        manager.close()
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("threaded output writer error was not reraised")


def _forcing() -> SimpleNamespace:
    return SimpleNamespace(
        surface_pressure_pa=np.full((1, FIXED_GRID["lat"], FIXED_GRID["lon"]), 101000.0),
        specific_humidity_kg_kg=np.zeros((1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"])),
        temperature_k=np.zeros((1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"])),
    )


def _field(names: tuple[str, ...], *, values: tuple[float, ...]) -> TracerField:
    data = np.empty((1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"], len(names)), dtype=np.float64)
    for index, value in enumerate(values):
        data[..., index] = value + np.arange(FIXED_GRID["lev"], dtype=np.float64)[np.newaxis, :, np.newaxis, np.newaxis]
    return TracerField(
        names=names,
        data=data,
        units=tuple("mol mol-1 dry" for _ in names),
        coords={},
    )
