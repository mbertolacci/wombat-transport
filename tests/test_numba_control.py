from __future__ import annotations

import logging

from wombat_transport.transport import numba_control


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
