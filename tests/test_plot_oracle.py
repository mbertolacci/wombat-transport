from __future__ import annotations

import numpy as np

from wombat_transport.plot_oracle import compute_metrics, render_dashboard_html, render_heatmap_svg


def test_render_heatmap_svg_includes_title_and_range():
    svg = render_heatmap_svg(
        np.array([[0.0, 1.0], [2.0, 3.0]]),
        title="Tiny field",
        units="ppm",
    )

    assert svg.startswith("<svg")
    assert "Tiny field" in svg
    assert "ppm" in svg
    assert "range 0 to 3" in svg


def test_compute_metrics_reports_zero_for_identical_difference():
    metrics = compute_metrics({"field": np.zeros((2, 3))})

    assert metrics["field_max_abs"] == 0.0
    assert metrics["field_mean_abs"] == 0.0


def test_compute_metrics_reports_shifted_values():
    metrics = compute_metrics({"field": np.array([-2.0, 1.0, 3.0])})

    assert metrics["field_max_abs"] == 3.0
    assert metrics["field_mean_abs"] == 2.0


def test_render_dashboard_html_references_expected_assets():
    html = render_dashboard_html(
        {
            "metrics": {"tracked_tracer": "CO2"},
            "numeric_metrics": {},
        }
    )

    assert "Wombat transport: current Python vs GEOS-Chem oracle" in html
    assert "assets/python_minus_oracle_surface.svg" in html
    assert "data/summary.json" in html
