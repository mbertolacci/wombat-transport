from __future__ import annotations

import logging

import pytest

from wombat_transport.transport import numba_control


def test_tpcore_numba_defaults_to_enabled(monkeypatch):
    from wombat_transport.transport import tpcore
    from wombat_transport.transport.tpcore import _numba as tpcore_numba

    monkeypatch.delenv("WOMBAT_TPCORE_NUMBA", raising=False)
    monkeypatch.delenv("WOMBAT_NUMBA", raising=False)

    assert tpcore._numba_tpcore_mode() == "1"
    assert tpcore._numba_tpcore_enabled() is tpcore_numba._NUMBA_AVAILABLE
    assert tpcore._numba_tpcore_x_enabled() is tpcore_numba._NUMBA_AVAILABLE
    assert tpcore._numba_tpcore_y_enabled() is tpcore_numba._NUMBA_AVAILABLE
    assert tpcore._numba_tpcore_z_enabled() is tpcore_numba._NUMBA_AVAILABLE
    assert tpcore._numba_tpcore_prepass_enabled() is tpcore_numba._NUMBA_AVAILABLE


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "x", "all"])
def test_tpcore_numba_truthy_modes_enable_single_numba_path(monkeypatch, value):
    from wombat_transport.transport import tpcore
    from wombat_transport.transport.tpcore import _numba as tpcore_numba

    monkeypatch.setenv("WOMBAT_TPCORE_NUMBA", value)

    assert tpcore._numba_tpcore_enabled() is tpcore_numba._NUMBA_AVAILABLE
    assert tpcore._numba_tpcore_x_enabled() is tpcore_numba._NUMBA_AVAILABLE
    assert tpcore._numba_tpcore_y_enabled() is tpcore_numba._NUMBA_AVAILABLE
    assert tpcore._numba_tpcore_z_enabled() is tpcore_numba._NUMBA_AVAILABLE
    assert tpcore._numba_tpcore_prepass_enabled() is tpcore_numba._NUMBA_AVAILABLE


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "none"])
def test_tpcore_numba_falsey_modes_disable_numba(monkeypatch, value):
    from wombat_transport.transport import tpcore

    monkeypatch.setenv("WOMBAT_TPCORE_NUMBA", value)

    assert not tpcore._numba_tpcore_enabled()
    assert not tpcore._numba_tpcore_x_enabled()
    assert not tpcore._numba_tpcore_y_enabled()
    assert not tpcore._numba_tpcore_z_enabled()
    assert not tpcore._numba_tpcore_prepass_enabled()


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "none"])
def test_global_numba_falsey_modes_disable_all_transport_numba(monkeypatch, value):
    from wombat_transport.transport import convection
    from wombat_transport.transport import pbl
    from wombat_transport.transport import tpcore

    monkeypatch.delenv("WOMBAT_TPCORE_NUMBA", raising=False)
    monkeypatch.delenv("WOMBAT_VDIFF_NUMBA", raising=False)
    monkeypatch.delenv("WOMBAT_CONVECTION_NUMBA", raising=False)
    monkeypatch.setenv("WOMBAT_NUMBA", value)

    assert tpcore._numba_tpcore_mode() == value
    assert pbl._numba_vdiff_mode() == value
    assert convection._numba_convection_mode() == value
    assert not tpcore._numba_tpcore_enabled()
    assert not pbl._numba_vdiff_enabled()
    assert not convection._numba_convection_enabled()


def test_operator_numba_flags_override_global_flag(monkeypatch):
    from wombat_transport.transport import convection
    from wombat_transport.transport import pbl
    from wombat_transport.transport import tpcore
    from wombat_transport.transport.convection import _numba as convection_numba
    from wombat_transport.transport.pbl import _numba as pbl_numba
    from wombat_transport.transport.tpcore import _numba as tpcore_numba

    monkeypatch.setenv("WOMBAT_NUMBA", "0")
    monkeypatch.setenv("WOMBAT_TPCORE_NUMBA", "1")
    monkeypatch.setenv("WOMBAT_VDIFF_NUMBA", "1")
    monkeypatch.setenv("WOMBAT_CONVECTION_NUMBA", "1")

    assert tpcore._numba_tpcore_mode() == "1"
    assert pbl._numba_vdiff_mode() == "1"
    assert convection._numba_convection_mode() == "1"
    assert tpcore._numba_tpcore_enabled() is tpcore_numba._NUMBA_AVAILABLE
    assert pbl._numba_vdiff_enabled() is pbl_numba._NUMBA_AVAILABLE
    assert convection._numba_convection_enabled() is convection_numba._NUMBA_AVAILABLE


def test_operator_numba_falsey_flag_overrides_global_enabled(monkeypatch):
    from wombat_transport.transport import convection
    from wombat_transport.transport import pbl
    from wombat_transport.transport import tpcore

    monkeypatch.setenv("WOMBAT_NUMBA", "1")
    monkeypatch.setenv("WOMBAT_TPCORE_NUMBA", "0")
    monkeypatch.setenv("WOMBAT_VDIFF_NUMBA", "0")
    monkeypatch.setenv("WOMBAT_CONVECTION_NUMBA", "0")

    assert not tpcore._numba_tpcore_enabled()
    assert not pbl._numba_vdiff_enabled()
    assert not convection._numba_convection_enabled()


def test_unified_transport_requires_every_numba_operator(monkeypatch):
    monkeypatch.setattr(numba_control, "set_num_threads", lambda count: None)
    monkeypatch.setenv("WOMBAT_NUMBA", "1")
    assert numba_control.transport_numba_enabled()

    monkeypatch.setenv("WOMBAT_VDIFF_NUMBA", "0")
    assert not numba_control.transport_numba_enabled()


def test_transport_performance_warning_when_numba_unavailable(monkeypatch, caplog):
    monkeypatch.setattr(numba_control, "set_num_threads", None)
    monkeypatch.setattr(numba_control, "_transport_warning_emitted", False)

    logger = logging.getLogger("test.transport.performance")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        numba_control.warn_if_transport_numba_disabled(logger)
        numba_control.warn_if_transport_numba_disabled(logger)

    warnings = [record.message for record in caplog.records if "MAJOR PERFORMANCE WARNING" in record.message]
    assert len(warnings) == 1
    assert "Numba is unavailable" in warnings[0]


def test_transport_performance_warning_names_disabled_operator(monkeypatch, caplog):
    monkeypatch.setattr(numba_control, "set_num_threads", lambda count: None)
    monkeypatch.setattr(numba_control, "_transport_warning_emitted", False)
    monkeypatch.setenv("WOMBAT_NUMBA", "1")
    monkeypatch.setenv("WOMBAT_VDIFF_NUMBA", "0")

    logger = logging.getLogger("test.transport.disabled")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        numba_control.warn_if_transport_numba_disabled(logger)

    assert len(caplog.records) == 1
    assert "WOMBAT_VDIFF_NUMBA" in caplog.records[0].message


def test_transport_performance_warning_is_silent_when_all_paths_enabled(monkeypatch, caplog):
    monkeypatch.setattr(numba_control, "set_num_threads", lambda count: None)
    monkeypatch.setattr(numba_control, "_transport_warning_emitted", False)
    monkeypatch.delenv("WOMBAT_NUMBA", raising=False)
    for name in ("WOMBAT_TPCORE_NUMBA", "WOMBAT_VDIFF_NUMBA", "WOMBAT_CONVECTION_NUMBA"):
        monkeypatch.delenv(name, raising=False)

    logger = logging.getLogger("test.transport.enabled")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        numba_control.warn_if_transport_numba_disabled(logger)

    assert caplog.records == []
