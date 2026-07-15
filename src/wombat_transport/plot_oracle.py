from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from html import escape
from pathlib import Path

import netCDF4
import numpy as np

from wombat_transport.fields import TracerField, canonical_time_slice, transport_tracer_to_public4
from wombat_transport.gc_harness import (
    read_transport_step_output,
    run_pjc_harness,
    write_transport_step_input_from_config,
)
from wombat_transport.grid import load_transport_grid
from wombat_transport.io import initialize_tracers
from wombat_transport.run_config import (
    RunConfig,
    load_run_config,
    meteorology_initial_time_index,
    meteorology_root,
    simulation_start,
    transport_timestep_s,
)
from wombat_transport.transport import load_transport_forcing, run_transport_one_step


DEFAULT_BASE_CONFIG = Path("validation_runs/cases/realistic_restart_noemis/wombat/main/run.yml")
DEFAULT_RESIDUAL_CONFIG = Path("validation_runs/cases/residual_24tracer_emissions_1day/wombat/main/run.yml")
DEFAULT_EXECUTABLE = Path("tools/gc_harness/build/pjc_pfix_harness")
DEFAULT_OUTPUT = Path("/home/mgnb/public_html/wombat-transport")
DEFAULT_WORK_DIR = Path("/tmp/wombat_transport_plot_work")


@dataclass(frozen=True)
class ComparisonBundle:
    lon: np.ndarray
    lat: np.ndarray
    initial: np.ndarray
    oracle_after: np.ndarray
    python_after: np.ndarray
    oracle_xmass: np.ndarray
    oracle_ymass: np.ndarray
    python_xmass: np.ndarray
    python_ymass: np.ndarray
    oracle_surface_pressure: np.ndarray
    python_surface_pressure: np.ndarray
    tracer_name: str


def generate_dashboard(
    output_dir: str | Path,
    *,
    work_dir: str | Path = DEFAULT_WORK_DIR,
    base_config: str | Path = DEFAULT_BASE_CONFIG,
    residual_config: str | Path = DEFAULT_RESIDUAL_CONFIG,
    executable: str | Path = DEFAULT_EXECUTABLE,
    max_tracers: int = 1,
    skip_oracle_run: bool = False,
) -> Path:
    output = Path(output_dir)
    work = Path(work_dir)
    assets = output / "assets"
    data_dir = output / "data"
    if output.exists():
        shutil.rmtree(output)
    assets.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    base = _build_comparison_bundle(
        Path(base_config),
        work / "base",
        Path(executable),
        max_tracers=max_tracers,
        skip_oracle_run=skip_oracle_run,
    )
    residual_metrics = _residual_smoke_metrics(
        Path(residual_config),
        work / "residual",
        Path(executable),
        skip_oracle_run=skip_oracle_run,
    )
    summary = _write_assets(assets, base, residual_metrics)
    (data_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output / "index.html").write_text(render_dashboard_html(summary), encoding="utf-8")
    return output


def render_dashboard_html(summary: dict[str, object]) -> str:
    metric_rows = "\n".join(
        f"<tr><th>{escape(str(name))}</th><td>{escape(str(value))}</td></tr>"
        for name, value in summary["metrics"].items()  # type: ignore[index,union-attr]
    )
    cards = [
        ("Initial tracer", "assets/initial_surface.svg"),
        ("GEOS-Chem oracle after one step", "assets/oracle_after_surface.svg"),
        ("Current Python after one step", "assets/python_after_surface.svg"),
        ("GEOS-Chem one-step delta", "assets/oracle_delta_surface.svg"),
        ("Python one-step delta", "assets/python_delta_surface.svg"),
        ("Python minus GEOS-Chem after one step", "assets/python_minus_oracle_surface.svg"),
        ("Vertical mean profile", "assets/profile_overlay.svg"),
        ("Zonal mass flux: oracle", "assets/oracle_xmass_column.svg"),
        ("Zonal mass flux: current Python", "assets/python_xmass_column.svg"),
        ("Zonal mass flux difference", "assets/xmass_difference_column.svg"),
        ("Meridional mass flux: oracle", "assets/oracle_ymass_column.svg"),
        ("Meridional mass flux: current Python", "assets/python_ymass_column.svg"),
        ("Meridional mass flux difference", "assets/ymass_difference_column.svg"),
        ("Surface pressure comparison", "assets/surface_pressure_comparison.svg"),
    ]
    card_html = "\n".join(
        f'<section><h2>{escape(title)}</h2><img src="{src}" alt="{escape(title)}"></section>'
        for title, src in cards
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wombat transport Python vs GEOS-Chem oracle</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 0; background: #f4f6f8; color: #17202a; }}
header {{ background: #102a43; color: white; padding: 28px 34px; }}
main {{ max-width: 1120px; margin: 0 auto; padding: 24px; }}
section {{ background: white; margin: 18px 0; padding: 18px; border: 1px solid #d8dee4; border-radius: 8px; }}
h1 {{ margin: 0 0 8px; font-size: 28px; }}
h2 {{ margin: 0 0 12px; font-size: 20px; }}
p {{ line-height: 1.45; }}
img {{ width: 100%; height: auto; border: 1px solid #e5e7eb; }}
table {{ border-collapse: collapse; width: 100%; background: white; }}
th, td {{ text-align: left; padding: 9px 10px; border-bottom: 1px solid #e5e7eb; }}
th {{ width: 52%; }}
code {{ background: #eef2f6; padding: 2px 4px; border-radius: 3px; }}
pre {{ overflow-x: auto; background: #f8fafc; padding: 12px; }}
</style>
</head>
<body>
<header>
<h1>Wombat transport: current Python vs GEOS-Chem oracle</h1>
<p>Generated from the local one-step GEOS-Chem PJC+TPCORE harness and the current NumPy TPCORE path.</p>
</header>
<main>
<section>
<h2>What this tracks</h2>
<p>The GEOS-Chem oracle runs <code>DO_PJC_PFIX</code> followed by <code>TPCORE_FVDAS</code>. The Python side uses the GEOS-Chem-oriented NumPy TPCORE port, with branch limits reported separately from final-field errors.</p>
<table>{metric_rows}</table>
</section>
{card_html}
<section>
<h2>Reproduction command</h2>
<pre><code>tools/gc_harness/build_pjc_pfix_harness.sh
PYTHONDONTWRITEBYTECODE=1 python -m wombat_transport.plot_oracle --output /home/mgnb/public_html/wombat-transport</code></pre>
<p>Machine-readable metrics are in <code>data/summary.json</code>.</p>
</section>
</main>
</body>
</html>
"""


def compute_metrics(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        f"{name}_max_abs": float(np.max(np.abs(values)))
        for name, values in arrays.items()
    } | {
        f"{name}_mean_abs": float(np.mean(np.abs(values)))
        for name, values in arrays.items()
    }


def render_heatmap_svg(
    arr: np.ndarray,
    *,
    title: str,
    lon: np.ndarray | None = None,
    lat: np.ndarray | None = None,
    units: str = "",
    diverging: bool = False,
    symmetric: bool = False,
) -> str:
    arr = np.asarray(arr, dtype=float)
    arr_plot = arr[::-1, :]
    nlat, nlon = arr_plot.shape
    cell = 5
    plot_w = nlon * cell
    plot_h = nlat * cell
    margin_l = 54
    margin_t = 46
    margin_b = 42
    margin_r = 142
    svg_w = margin_l + plot_w + margin_r
    svg_h = margin_t + plot_h + margin_b
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    elif symmetric:
        vmax = float(np.max(np.abs(finite))) or 1.0
        vmin = -vmax
    else:
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
        if vmin == vmax:
            pad = abs(vmin) * 0.01 or 1.0
            vmin -= pad
            vmax += pad

    cmap = _div_color if diverging else _seq_color
    rects = []
    for j in range(nlat):
        for i in range(nlon):
            value = arr_plot[j, i]
            if np.isfinite(value):
                color = _rgb_hex(cmap((value - vmin) / (vmax - vmin)))
            else:
                color = "#eeeeee"
            rects.append(f'<rect x="{margin_l + i * cell}" y="{margin_t + j * cell}" width="{cell}" height="{cell}" fill="{color}"/>')

    bar_x = margin_l + plot_w + 34
    bar_y = margin_t
    bar_w = 18
    bar = []
    for k in range(plot_h):
        t = 1.0 - k / max(1, plot_h - 1)
        bar.append(f'<rect x="{bar_x}" y="{bar_y + k}" width="{bar_w}" height="1" fill="{_rgb_hex(cmap(t))}"/>')
    west = _fmt(float(lon[0])) + "E" if lon is not None else "west"
    east = _fmt(float(lon[-1])) + "E" if lon is not None else "east"
    north = _fmt(float(lat[-1])) + "N" if lat is not None else "north"
    south = _fmt(float(lat[0])) + "N" if lat is not None else "south"
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">
<style>text{{font-family:Arial,sans-serif;fill:#222}} .small{{font-size:12px}} .title{{font-size:18px;font-weight:700}}</style>
<rect width="100%" height="100%" fill="white"/>
<text class="title" x="{margin_l}" y="24">{escape(title)}</text>
<text class="small" x="{margin_l}" y="40">north-up map; {escape(units)}; range {_fmt(vmin)} to {_fmt(vmax)}</text>
<rect x="{margin_l}" y="{margin_t}" width="{plot_w}" height="{plot_h}" fill="#f8f8f8" stroke="#333"/>
{''.join(rects)}
<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#333"/>
<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#333"/>
<text class="small" x="{margin_l}" y="{margin_t + plot_h + 22}">{escape(west)}</text>
<text class="small" x="{margin_l + plot_w - 42}" y="{margin_t + plot_h + 22}">{escape(east)}</text>
<text class="small" x="8" y="{margin_t + 12}">{escape(north)}</text>
<text class="small" x="8" y="{margin_t + plot_h}">{escape(south)}</text>
{''.join(bar)}
<rect x="{bar_x}" y="{bar_y}" width="{bar_w}" height="{plot_h}" fill="none" stroke="#333"/>
<text class="small" x="{bar_x + 28}" y="{bar_y + 10}">{_fmt(vmax)}</text>
<text class="small" x="{bar_x + 28}" y="{bar_y + plot_h}">{_fmt(vmin)}</text>
</svg>'''


def render_profile_svg(
    profiles: list[tuple[str, np.ndarray, str]],
    *,
    title: str,
    units: str,
) -> str:
    nlev = len(profiles[0][1])
    width, height = 720, 520
    ml, mt, mr, mb = 78, 44, 30, 58
    pw, ph = width - ml - mr, height - mt - mb
    all_values = np.concatenate([np.asarray(values, dtype=float) for _, values, _ in profiles])
    xmin, xmax = float(np.min(all_values)), float(np.max(all_values))
    pad = (xmax - xmin) * 0.08 or 1.0
    xmin -= pad
    xmax += pad

    def xy(value: float, index: int) -> tuple[float, float]:
        return (
            ml + (value - xmin) / (xmax - xmin) * pw,
            mt + index / max(1, nlev - 1) * ph,
        )

    paths = []
    legend = []
    for legend_index, (name, values, color) in enumerate(profiles):
        parts = []
        for idx, value in enumerate(values):
            x, y = xy(float(value), idx)
            parts.append(("M" if idx == 0 else "L") + f"{x:.2f},{y:.2f}")
        paths.append(f'<path d="{" ".join(parts)}" fill="none" stroke="{color}" stroke-width="3"/>')
        legend.append(f'<text class="small" x="{ml + 12}" y="{mt + 20 + 18 * legend_index}" fill="{color}">{escape(name)}</text>')

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>text{{font-family:Arial,sans-serif;fill:#222}} .small{{font-size:12px}} .title{{font-size:18px;font-weight:700}}</style>
<rect width="100%" height="100%" fill="white"/>
<text class="title" x="{ml}" y="24">{escape(title)}</text>
<text class="small" x="{ml}" y="40">global mean by stored model level, {escape(units)}</text>
<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#333"/>
{''.join(paths)}
{''.join(legend)}
<text class="small" x="{ml}" y="{mt + ph + 28}">{_fmt(xmin)} {escape(units)}</text>
<text class="small" x="{ml + pw - 86}" y="{mt + ph + 28}">{_fmt(xmax)} {escape(units)}</text>
<text class="small" x="12" y="{mt + 10}">level 1</text>
<text class="small" x="12" y="{mt + ph}">level {nlev}</text>
</svg>'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Python-vs-GEOS-Chem oracle transport plots.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--residual-config", type=Path, default=DEFAULT_RESIDUAL_CONFIG)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--max-tracers", type=int, default=1)
    parser.add_argument("--skip-oracle-run", action="store_true")
    args = parser.parse_args(argv)
    output = generate_dashboard(
        args.output,
        work_dir=args.work_dir,
        base_config=args.base_config,
        residual_config=args.residual_config,
        executable=args.executable,
        max_tracers=args.max_tracers,
        skip_oracle_run=args.skip_oracle_run,
    )
    print(f"wrote_dashboard: {output}")
    return 0


def _build_comparison_bundle(
    config_path: Path,
    work_dir: Path,
    executable: Path,
    *,
    max_tracers: int,
    skip_oracle_run: bool,
) -> ComparisonBundle:
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / "transport_step_input.nc"
    output_path = work_dir / "transport_step_output.nc"
    if not skip_oracle_run:
        write_transport_step_input_from_config(config_path, input_path, max_tracers=max_tracers)
        run_pjc_harness(executable, input_path, output_path)

    with netCDF4.Dataset(input_path) as dataset:
        lon = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
        lat = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
        initial = np.asarray(dataset.variables["tracer_conc"][:], dtype=np.float64)
        names = _read_tracer_names(dataset)
    oracle = read_transport_step_output(output_path)
    config = load_run_config(config_path)
    python_result = _run_python_step(config, max_tracers=max_tracers)
    python_public = transport_tracer_to_public4(canonical_time_slice(python_result.state.data))
    python_surface_pressure = np.sum(python_result.delp_dry_hpa[0], axis=0) + 0.01
    return ComparisonBundle(
        lon=lon,
        lat=lat,
        initial=initial[0],
        oracle_after=oracle.tracer_conc_after[0],
        python_after=python_public[0],
        oracle_xmass=oracle.xmass_hpa,
        oracle_ymass=oracle.ymass_hpa,
        python_xmass=python_result.xmass_hpa[0],
        python_ymass=python_result.ymass_hpa[0],
        oracle_surface_pressure=oracle.surface_pressure_hpa,
        python_surface_pressure=python_surface_pressure,
        tracer_name=names[0] if names else "tracer_001",
    )


def _run_python_step(config: RunConfig, *, max_tracers: int):
    grid = load_transport_grid(config.grid_template)
    tracers = initialize_tracers(
        config.initial_restart,
        config.species_database,
        template_path=config.grid_template,
    )
    tracers = _limit_tracers(tracers, max_tracers)
    forcing = load_transport_forcing(
        meteorology_root(config),
        simulation_start(config),
        grid,
        time_index=meteorology_initial_time_index(config),
    )
    return run_transport_one_step(
        tracers,
        forcing,
        grid,
        dt_s=transport_timestep_s(config),
        include_flux_diagnostics=True,
    )


def _residual_smoke_metrics(
    config_path: Path,
    work_dir: Path,
    executable: Path,
    *,
    skip_oracle_run: bool,
) -> dict[str, float]:
    bundle = _build_comparison_bundle(
        config_path,
        work_dir,
        executable,
        max_tracers=2,
        skip_oracle_run=skip_oracle_run,
    )
    return {
        "residual_oracle_max_abs_change_ppm": float(np.max(np.abs((bundle.oracle_after - bundle.initial) * 1.0e6))),
        "residual_python_max_abs_change_ppm": float(np.max(np.abs((bundle.python_after - bundle.initial) * 1.0e6))),
    }


def _write_assets(assets: Path, bundle: ComparisonBundle, residual_metrics: dict[str, float]) -> dict[str, object]:
    initial_ppm = bundle.initial * 1.0e6
    oracle_ppm = bundle.oracle_after * 1.0e6
    python_ppm = bundle.python_after * 1.0e6
    oracle_delta = oracle_ppm - initial_ppm
    python_delta = python_ppm - initial_ppm
    after_diff = python_ppm - oracle_ppm
    oracle_x_col = np.sum(bundle.oracle_xmass, axis=0)
    python_x_col = np.sum(bundle.python_xmass, axis=0)
    oracle_y_col = np.sum(bundle.oracle_ymass, axis=0)
    python_y_col = np.sum(bundle.python_ymass, axis=0)

    heatmaps = [
        ("initial_surface.svg", initial_ppm[0], "Initial tracer, level 1", "ppm", False, False),
        ("oracle_after_surface.svg", oracle_ppm[0], "GEOS-Chem oracle after one step, level 1", "ppm", False, False),
        ("python_after_surface.svg", python_ppm[0], "Current Python after one step, level 1", "ppm", False, False),
        ("oracle_delta_surface.svg", oracle_delta[0], "GEOS-Chem one-step delta, level 1", "ppm", True, True),
        ("python_delta_surface.svg", python_delta[0], "Current Python one-step delta, level 1", "ppm", True, True),
        ("python_minus_oracle_surface.svg", after_diff[0], "Python minus GEOS-Chem after one step, level 1", "ppm", True, True),
        ("oracle_xmass_column.svg", oracle_x_col, "PJC xmass column sum: GEOS-Chem oracle", "hPa equivalent", True, True),
        ("python_xmass_column.svg", python_x_col, "xmass column sum: current Python", "hPa equivalent", True, True),
        ("xmass_difference_column.svg", python_x_col - oracle_x_col, "xmass column sum: Python minus oracle", "hPa equivalent", True, True),
        ("oracle_ymass_column.svg", oracle_y_col, "PJC ymass column sum: GEOS-Chem oracle", "hPa equivalent", True, True),
        ("python_ymass_column.svg", python_y_col, "ymass column sum: current Python", "hPa equivalent", True, True),
        ("ymass_difference_column.svg", python_y_col - oracle_y_col, "ymass column sum: Python minus oracle", "hPa equivalent", True, True),
    ]
    for filename, arr, title, units, diverging, symmetric in heatmaps:
        (assets / filename).write_text(
            render_heatmap_svg(
                arr,
                title=title,
                lon=bundle.lon,
                lat=bundle.lat,
                units=units,
                diverging=diverging,
                symmetric=symmetric,
            ),
            encoding="utf-8",
        )

    (assets / "surface_pressure_comparison.svg").write_text(
        render_profile_svg(
            [
                ("GEOS-Chem oracle surface pressure", np.mean(bundle.oracle_surface_pressure, axis=1), "#b2182b"),
                ("Python reconstructed surface pressure", np.mean(bundle.python_surface_pressure, axis=1), "#2166ac"),
            ],
            title="Zonal mean surface pressure after one step",
            units="hPa",
        ),
        encoding="utf-8",
    )
    (assets / "profile_overlay.svg").write_text(
        render_profile_svg(
            [
                ("initial", np.mean(initial_ppm, axis=(1, 2)), "#4d4d4d"),
                ("GEOS-Chem oracle", np.mean(oracle_ppm, axis=(1, 2)), "#b2182b"),
                ("current Python", np.mean(python_ppm, axis=(1, 2)), "#2166ac"),
            ],
            title=f"{bundle.tracer_name} vertical mean profile",
            units="ppm",
        ),
        encoding="utf-8",
    )

    numeric = compute_metrics(
        {
            "tracer_after_python_minus_oracle_ppm": after_diff,
            "tracer_delta_python_minus_oracle_ppm": python_delta - oracle_delta,
            "xmass_python_minus_oracle_hpa": bundle.python_xmass - bundle.oracle_xmass,
            "ymass_python_minus_oracle_hpa": bundle.python_ymass - bundle.oracle_ymass,
            "surface_pressure_python_minus_oracle_hpa": bundle.python_surface_pressure - bundle.oracle_surface_pressure,
        }
    )
    numeric.update(residual_metrics)
    metrics = {name: _fmt(value) for name, value in numeric.items()}
    metrics["tracked_tracer"] = bundle.tracer_name
    return {
        "metrics": metrics,
        "numeric_metrics": numeric,
    }


def _read_tracer_names(dataset: netCDF4.Dataset) -> tuple[str, ...]:
    if "tracer_name" not in dataset.variables:
        return ()
    names = []
    for name in netCDF4.chartostring(dataset.variables["tracer_name"][:]):
        if isinstance(name, bytes):
            names.append(name.decode("ascii", errors="replace").strip())
        else:
            names.append(str(name).strip())
    return tuple(names)


def _limit_tracers(tracers: TracerField, max_tracers: int) -> TracerField:
    return TracerField(
        names=tracers.names[:max_tracers],
        data=tracers.data[..., :max_tracers],
        units=tracers.units[:max_tracers],
        coords=tracers.coords,
    )


def _fmt(value: float) -> str:
    value = float(value)
    if value == 0.0 or 1.0e-3 <= abs(value) < 1.0e4:
        return f"{value:.6g}"
    return f"{value:.4e}"


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _rgb_hex(rgb: tuple[float, float, float]) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(round(c)))) for c in rgb)


def _seq_color(t: float) -> tuple[float, float, float]:
    return _interp_stops(
        [
            (0.00, (49, 54, 149)),
            (0.25, (69, 117, 180)),
            (0.50, (116, 173, 209)),
            (0.70, (171, 221, 164)),
            (0.85, (254, 224, 144)),
            (1.00, (215, 48, 39)),
        ],
        t,
    )


def _div_color(t: float) -> tuple[float, float, float]:
    return _interp_stops(
        [
            (0.00, (49, 54, 149)),
            (0.35, (116, 173, 209)),
            (0.50, (247, 247, 247)),
            (0.65, (244, 109, 67)),
            (1.00, (165, 0, 38)),
        ],
        t,
    )


def _interp_stops(stops: list[tuple[float, tuple[int, int, int]]], t: float) -> tuple[float, float, float]:
    t = max(0.0, min(1.0, float(t)))
    for idx in range(len(stops) - 1):
        t0, c0 = stops[idx]
        t1, c1 = stops[idx + 1]
        if t <= t1:
            local = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return tuple(_lerp(c0[channel], c1[channel], local) for channel in range(3))
    return stops[-1][1]


if __name__ == "__main__":
    raise SystemExit(main())
