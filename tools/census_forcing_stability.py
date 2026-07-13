from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np

from wombat_transport.grid import load_transport_grid
from wombat_transport.run_config import load_run_config, meteorology_root, simulation_start, transport_timestep_s
from wombat_transport.transport.forcing import TransportForcingProvider
from wombat_transport.transport.tpcore import build_tpcore_static_terms, setup_tpcore_terms


FIELDS = (
    "u_m_s",
    "v_m_s",
    "omega_pa_s",
    "dry_surface_pressure_start_hpa",
    "dry_surface_pressure_hpa",
    "temperature_k",
    "specific_humidity_kg_kg",
    "pbl_height_m",
    "sensible_heat_flux_w_m2",
    "latent_heat_flux_w_m2",
    "friction_velocity_m_s",
)
VDIFF_INPUT_FIELDS = (
    "u_m_s",
    "v_m_s",
    "dry_surface_pressure_start_hpa",
    "dry_surface_pressure_hpa",
    "temperature_k",
    "specific_humidity_kg_kg",
    "pbl_height_m",
    "sensible_heat_flux_w_m2",
    "latent_heat_flux_w_m2",
    "friction_velocity_m_s",
)
SETUP_FIELDS = (
    "xmass_hpa",
    "ymass_hpa",
    "delp1_hpa",
    "delpm_hpa",
    "delp2_hpa",
    "pu_hpa",
    "vertical_mass_flux_hpa",
    "cx",
    "cy",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Census adjacent-step forcing and TPCORE setup stability.")
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=36)
    args = parser.parse_args(argv)
    if args.steps < 2:
        parser.error("--steps must be at least 2")

    config = load_run_config(args.run_config)
    grid = load_transport_grid(config.grid_template)
    start = simulation_start(config)
    dt_s = transport_timestep_s(config)
    provider = TransportForcingProvider(meteorology_root(config), start, grid)
    static = build_tpcore_static_terms(
        area_m2=grid.area_m2,
        hyai_hpa=grid.hyai_hpa,
        hybi=grid.hybi,
        lat_deg=grid.lat_deg,
    )
    counts = defaultdict(lambda: [0, 0])
    previous = None
    previous_end = None
    previous_setup = None
    boundary_matches = 0
    vdiff_complete_matches = 0
    transitions = args.steps - 1
    for step in range(args.steps):
        forcing = provider.forcing_for_step(start + timedelta(seconds=step * dt_s), dt_s=dt_s)
        setup = setup_tpcore_terms(
            p1_hpa=forcing.dry_surface_pressure_start_hpa[0],
            p2_hpa=forcing.dry_surface_pressure_hpa[0],
            u_m_s=forcing.u_m_s[0],
            v_m_s=forcing.v_m_s[0],
            area_m2=grid.area_m2,
            hyai_hpa=grid.hyai_hpa,
            hybi=grid.hybi,
            lat_deg=grid.lat_deg,
            dt_s=dt_s,
            static_terms=static,
        )
        if previous is not None:
            for name in FIELDS:
                before = getattr(previous, name)
                after = getattr(forcing, name)
                counts[name][0] += int(np.shares_memory(before, after))
                counts[name][1] += int(np.array_equal(before, after))
            boundary_matches += int(np.array_equal(previous_end, forcing.dry_surface_pressure_start_hpa))
            vdiff_complete_matches += int(
                all(
                    np.array_equal(getattr(previous, name), getattr(forcing, name))
                    for name in VDIFF_INPUT_FIELDS
                )
            )
            for name in SETUP_FIELDS:
                counts[f"setup.{name}"][1] += int(
                    np.array_equal(getattr(previous_setup, name), getattr(setup, name))
                )
        previous = forcing
        previous_end = forcing.dry_surface_pressure_hpa
        previous_setup = setup
    writer = csv.writer(sys.stdout)
    writer.writerow(("component", "field", "shares_memory", "equal_values", "transitions"))
    for name in FIELDS:
        shares, equal = counts[name]
        writer.writerow(("forcing", name, shares, equal, transitions))
    writer.writerow(("forcing", "previous_p2_equals_next_p1", "", boundary_matches, transitions))
    writer.writerow(("vdiff", "complete_input_set", "", vdiff_complete_matches, transitions))
    for name in SETUP_FIELDS:
        writer.writerow(("tpcore_setup", name, "", counts[f"setup.{name}"][1], transitions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
