from __future__ import annotations

import logging

from wombat_transport import yaml_io


def test_load_yaml_preserves_pyyaml_yaml_11_boolean_semantics():
    assert yaml_io.load_yaml("enabled: yes\n") == {"enabled": True}


def test_load_yaml12_uses_yaml12_parser(monkeypatch):
    calls = []

    def fake_parse(text):
        calls.append(text)
        return {"enabled": "yes"}

    monkeypatch.setattr(yaml_io, "_parse_yaml12", fake_parse)

    assert yaml_io.load_yaml12("enabled: yes\n") == {"enabled": "yes"}
    assert calls == ["enabled: yes\n"]


def test_load_yaml12_warns_once_when_falling_back_to_pyyaml(monkeypatch, caplog):
    monkeypatch.setattr(yaml_io, "_parse_yaml12", None)
    monkeypatch.setattr(yaml_io, "_yaml12_fallback_warned", False)

    with caplog.at_level(logging.WARNING, logger=yaml_io.__name__):
        first = yaml_io.load_yaml12("enabled: yes\n", source_name="observations.yml")
        second = yaml_io.load_yaml12("enabled: no\n", source_name="other.yml")

    assert first == {"enabled": True}
    assert second == {"enabled": False}
    warnings = [record.message for record in caplog.records if "MAJOR PERFORMANCE WARNING" in record.message]
    assert len(warnings) == 1
    assert "observations.yml" in warnings[0]
