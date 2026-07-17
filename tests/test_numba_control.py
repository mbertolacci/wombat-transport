from __future__ import annotations

import logging

import pytest

from wombat_transport import history_accumulation
from wombat_transport.obsoperator import sampling
from wombat_transport.transport import convection, pbl, tpcore
from wombat_transport.transport import numba_control
from wombat_transport.transport.convection import _operator as convection_numba
from wombat_transport.transport.pbl import _kernels as pbl_numba
from wombat_transport.transport.tpcore import _kernels as tpcore_numba


def test_numba_defaults_to_enabled_for_every_subsystem(monkeypatch):
    monkeypatch.delenv("WOMBAT_NUMBA", raising=False)

    assert numba_control.numba_mode() == "1"
    assert tpcore._numba_tpcore_enabled() is tpcore_numba._NUMBA_AVAILABLE
    assert pbl._numba_vdiff_enabled() is pbl_numba._NUMBA_AVAILABLE
    assert convection._numba_convection_enabled() is convection_numba._NUMBA_AVAILABLE
    assert history_accumulation._history_numba_enabled() is history_accumulation._NUMBA_AVAILABLE


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "x", "all"])
def test_global_truthy_modes_enable_every_subsystem(monkeypatch, value):
    monkeypatch.setenv("WOMBAT_NUMBA", value)

    assert tpcore._numba_tpcore_mode() == value
    assert pbl._numba_vdiff_mode() == value
    assert convection._numba_convection_mode() == value
    assert tpcore._numba_tpcore_enabled() is tpcore_numba._NUMBA_AVAILABLE
    assert pbl._numba_vdiff_enabled() is pbl_numba._NUMBA_AVAILABLE
    assert convection._numba_convection_enabled() is convection_numba._NUMBA_AVAILABLE
    assert history_accumulation._history_numba_enabled() is history_accumulation._NUMBA_AVAILABLE


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "none"])
def test_global_falsey_modes_disable_every_subsystem(monkeypatch, value):
    monkeypatch.setenv("WOMBAT_NUMBA", value)

    assert not tpcore._numba_tpcore_enabled()
    assert not pbl._numba_vdiff_enabled()
    assert not convection._numba_convection_enabled()
    assert not history_accumulation._history_numba_enabled()
    assert sampling.select_sampling_kernel() is sampling._sample_prepared_entries_kernel


@pytest.mark.parametrize("value", ["0", "-1", "invalid", "1.5"])
def test_numba_thread_count_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("WOMBAT_NUMBA_THREADS", value)

    with pytest.raises(ValueError, match="WOMBAT_NUMBA_THREADS must be a positive integer"):
        numba_control.numba_thread_count()


def test_numba_threads_are_configured_once_per_process(monkeypatch):
    calls = []
    monkeypatch.setattr(numba_control, "set_num_threads", calls.append)
    monkeypatch.setenv("WOMBAT_NUMBA_THREADS", "3")

    assert numba_control.configure_numba_threads(available=True) == 3
    monkeypatch.setenv("WOMBAT_NUMBA_THREADS", "7")
    assert numba_control.configure_numba_threads(available=True) == 3
    assert calls == [3]


def test_numba_threads_are_not_configured_when_unavailable(monkeypatch):
    calls = []
    monkeypatch.setattr(numba_control, "set_num_threads", calls.append)
    monkeypatch.setenv("WOMBAT_NUMBA_THREADS", "2")

    assert numba_control.configure_numba_threads(available=False) == 2
    assert calls == []


def test_numba_performance_warning_when_unavailable(monkeypatch, caplog):
    monkeypatch.setattr(numba_control, "set_num_threads", None)
    logger = logging.getLogger("test.numba.performance")

    with caplog.at_level(logging.WARNING, logger=logger.name):
        numba_control.warn_if_numba_disabled(logger)
        numba_control.warn_if_numba_disabled(logger)

    warnings = [record.message for record in caplog.records if "MAJOR PERFORMANCE WARNING" in record.message]
    assert len(warnings) == 1
    assert "Numba is unavailable" in warnings[0]


def test_numba_performance_warning_names_global_flag(monkeypatch, caplog):
    monkeypatch.setattr(numba_control, "set_num_threads", lambda count: None)
    monkeypatch.setenv("WOMBAT_NUMBA", "0")
    logger = logging.getLogger("test.numba.disabled")

    with caplog.at_level(logging.WARNING, logger=logger.name):
        numba_control.warn_if_numba_disabled(logger)

    assert len(caplog.records) == 1
    assert "WOMBAT_NUMBA" in caplog.records[0].message


def test_numba_performance_warning_is_silent_when_enabled(monkeypatch, caplog):
    monkeypatch.setattr(numba_control, "set_num_threads", lambda count: None)
    monkeypatch.delenv("WOMBAT_NUMBA", raising=False)
    logger = logging.getLogger("test.numba.enabled")

    with caplog.at_level(logging.WARNING, logger=logger.name):
        numba_control.warn_if_numba_disabled(logger)

    assert caplog.records == []
