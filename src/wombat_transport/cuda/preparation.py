"""Resident CUDA meteorology interpolation and transport-plan preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from wombat_transport.cuda.forcing import CudaForcingStep
from wombat_transport.cuda.modules import load_raw_module
from wombat_transport.cuda.runtime import CudaRuntime
from wombat_transport.grid import TransportGrid
from wombat_transport.transport.convection._cuda import CudaConvectionPlan
from wombat_transport.transport.pbl._cuda import CudaVdiffPlan
from wombat_transport.transport.tpcore._cuda import CudaTpcorePlan
from wombat_transport.transport.tpcore.types import TpcoreStaticTerms


@dataclass(frozen=True)
class CudaStepMeteorology:
    wet_surface_pressure_start_hpa: Any
    wet_surface_pressure_hpa: Any
    dry_surface_pressure_start_hpa: Any
    dry_surface_pressure_hpa: Any
    specific_humidity_kg_kg: Any
    specific_humidity_after_kg_kg: Any
    temperature_k: Any


class CudaPlanPreparation:
    """Own static geometry, interpolated meteorology, and reusable plan arrays."""

    _TPCORE_CENTER_NAMES = (
        "delp1",
        "delp2",
        "pu",
        "xmass",
        "ymass",
        "vertical_mass_flux",
        "normalized_vertical_courant",
        "cx",
        "cy",
        "ua",
        "va",
    )

    def __init__(
        self,
        runtime: CudaRuntime,
        *,
        dtype: np.dtype[Any] | type[Any],
        grid: TransportGrid,
        tpcore_static_terms: TpcoreStaticTerms,
        initial_dry_surface_pressure_hpa: np.ndarray,
    ) -> None:
        self._runtime = runtime
        self._cupy = runtime.array_module
        self._dtype = np.dtype(dtype)
        self._grid = grid
        self._nlev, self._nlat, self._nlon = grid.shape
        center_shape = grid.shape
        horizontal_shape = grid.shape[1:]
        edge_shape = (self._nlev + 1,)
        geometry = tpcore_static_terms.pjc_geometry
        expressions = (
            "prepare_surface_endpoints",
            "average_surface_endpoint_poles",
            "interpolate_step_meteorology",
            "prepare_tpcore_double",
            "tpcore_prepare_pressure_fix",
            "tpcore_prepare_mass_flux",
            "tpcore_divergence_interior",
            "tpcore_divergence_poles",
            "tpcore_sum_vertical",
            "tpcore_prepare_pressure_correction",
            "tpcore_apply_pressure_correction",
            "tpcore_restore_pressure",
            "tpcore_prepare_pressure_terms",
            "tpcore_prepare_vertical_flux",
            "tpcore_prepare_cross_terms",
            "tpcore_prepare_jn_js",
            f"cast_plan_array<{'float' if self._dtype == np.dtype(np.float32) else 'double'}>",
            "compute_vdiff_start_level",
            "prepare_vdiff_double",
            f"prepare_convection_plan<{'float' if self._dtype == np.dtype(np.float32) else 'double'}>",
        )
        module = load_raw_module(
            "preparation.cu",
            name_expressions=expressions,
        )
        self._prepare_surface_endpoints = module.get_function(expressions[0])
        self._average_surface_poles = module.get_function(expressions[1])
        self._interpolate_meteorology = module.get_function(expressions[2])
        self._prepare_tpcore_serial = module.get_function(expressions[3])
        self._tpcore_pressure_fix = module.get_function(expressions[4])
        self._tpcore_mass_flux = module.get_function(expressions[5])
        self._tpcore_divergence_interior = module.get_function(expressions[6])
        self._tpcore_divergence_poles = module.get_function(expressions[7])
        self._tpcore_sum_vertical = module.get_function(expressions[8])
        self._tpcore_pressure_correction = module.get_function(expressions[9])
        self._tpcore_apply_correction = module.get_function(expressions[10])
        self._tpcore_restore_pressure = module.get_function(expressions[11])
        self._tpcore_pressure_terms = module.get_function(expressions[12])
        self._tpcore_vertical_flux = module.get_function(expressions[13])
        self._tpcore_cross_terms = module.get_function(expressions[14])
        self._tpcore_jn_js = module.get_function(expressions[15])
        self._cast_plan = module.get_function(expressions[16])
        self._compute_vdiff_start = module.get_function(expressions[17])
        self._prepare_vdiff = module.get_function(expressions[18])
        self._prepare_convection = module.get_function(expressions[19])

        self._area = runtime.to_device(grid.area_m2, dtype=np.float64)
        self._hyai = runtime.to_device(grid.hyai_hpa, dtype=np.float64)
        self._hybi = runtime.to_device(grid.hybi, dtype=np.float64)
        self._rel_area = runtime.to_device(geometry.rel_area, dtype=np.float64)
        self._geofac_double = runtime.to_device(geometry.geofac, dtype=np.float64)
        self._cose = runtime.to_device(geometry.cose, dtype=np.float64)
        self._cosp = runtime.to_device(geometry.cosp, dtype=np.float64)
        self._dap_geos = runtime.to_device(
            tpcore_static_terms.dap_geos_hpa,
            dtype=np.float64,
        )
        self._dbk_geos = runtime.to_device(
            tpcore_static_terms.dbk_geos,
            dtype=np.float64,
        )
        self._dap_top = runtime.to_device(
            tpcore_static_terms.dap_top_hpa,
            dtype=np.float64,
        )
        self._dbk_top = runtime.to_device(
            tpcore_static_terms.dbk_top,
            dtype=np.float64,
        )
        self._geofac_pc = float(geometry.geofac_pc)

        self._surface_endpoint = {
            name: runtime.empty(horizontal_shape, dtype=np.float64)
            for name in ("wet_start", "wet_end", "dry_start", "dry_end")
        }
        self._met = {
            name: runtime.empty(horizontal_shape, dtype=np.float64)
            for name in ("wet_start", "wet_end", "dry_start", "dry_end")
        }
        self._met["qv"] = runtime.empty(center_shape, dtype=np.float64)
        self._met["temperature"] = runtime.empty(center_shape, dtype=np.float64)
        initial_dry = np.asarray(initial_dry_surface_pressure_hpa, dtype=np.float64)
        if initial_dry.shape == (1, *horizontal_shape):
            initial_dry = initial_dry[0]
        if initial_dry.shape != horizontal_shape:
            raise ValueError(
                "initial CUDA dry surface pressure does not match the grid"
            )
        self._current_dry_surface = runtime.to_device(
            initial_dry,
            dtype=np.float64,
        )

        self._tpcore_double = {
            name: runtime.empty(center_shape, dtype=np.float64)
            for name in (
                *self._TPCORE_CENTER_NAMES,
                "delpm",
                "work3",
            )
        }
        self._tpcore_scratch = {
            "p1": runtime.empty(horizontal_shape, dtype=np.float64),
            "p2": runtime.empty(horizontal_shape, dtype=np.float64),
            "work2": runtime.empty(horizontal_shape, dtype=np.float64),
            "xfix": runtime.empty(horizontal_shape, dtype=np.float64),
            "mmfd": runtime.empty((self._nlat,), dtype=np.float64),
            "mmf": runtime.empty((self._nlat,), dtype=np.float64),
            "fxintegral": runtime.empty((self._nlon + 1,), dtype=np.float64),
        }
        self._jn = runtime.empty((self._nlev,), dtype=np.int64)
        self._js = runtime.empty((self._nlev,), dtype=np.int64)
        self._tpcore_output = (
            self._tpcore_double
            if self._dtype == np.dtype(np.float64)
            else {
                name: runtime.empty(center_shape, dtype=self._dtype)
                for name in self._TPCORE_CENTER_NAMES
            }
        )
        self._geofac = runtime.to_device(geometry.geofac, dtype=self._dtype)
        self._area_1d = runtime.to_device(
            grid.area_m2[:, 0],
            dtype=self._dtype,
        )
        self._plan = CudaTpcorePlan(
            delp1=self._tpcore_output["delp1"],
            delp2=self._tpcore_output["delp2"],
            pu=self._tpcore_output["pu"],
            xmass=self._tpcore_output["xmass"],
            ymass=self._tpcore_output["ymass"],
            vertical_mass_flux=self._tpcore_output["vertical_mass_flux"],
            normalized_vertical_courant=self._tpcore_output[
                "normalized_vertical_courant"
            ],
            cx=self._tpcore_output["cx"],
            cy=self._tpcore_output["cy"],
            geofac=self._geofac,
            geofac_pc=self._geofac_pc,
            ua=self._tpcore_output["ua"],
            va=self._tpcore_output["va"],
            jn=self._jn,
            js=self._js,
            area_1d=self._area_1d,
        )
        self._vdiff_double = {
            name: runtime.empty(center_shape, dtype=np.float64)
            for name in (
                "cch",
                "zeh",
                "termh",
                "rpdel",
                "dry_mass",
                "specific_humidity_after",
            )
        }
        self._vdiff_double.update(
            {
                name: runtime.empty(
                    (self._nlev + 1, self._nlat, self._nlon),
                    dtype=np.float64,
                )
                for name in ("cgs", "kvh", "potbar")
            }
        )
        self._vdiff_double.update(
            {
                name: runtime.empty(horizontal_shape, dtype=np.float64)
                for name in ("rrho", "tmp1")
            }
        )
        self._vdiff_output = (
            self._vdiff_double
            if self._dtype == np.dtype(np.float64)
            else {
                name: runtime.empty(values.shape, dtype=self._dtype)
                for name, values in self._vdiff_double.items()
            }
        )
        self._vdiff_start_level = runtime.empty((1,), dtype=np.int32)
        self._area_transport = runtime.to_device(
            grid.area_m2,
            dtype=self._dtype,
        )
        self._vdiff_plan: CudaVdiffPlan | None = None

        self._convection_arrays = {
            name: runtime.empty(center_shape, dtype=self._dtype)
            for name in (
                "cmfmc",
                "dtrain",
                "dqrcu",
                "reevapcn",
                "delp",
                "bmass",
            )
        }
        self._convection_plan: CudaConvectionPlan | None = None
        _ = edge_shape

    @property
    def tpcore_plan(self) -> CudaTpcorePlan:
        return self._plan

    @property
    def meteorology(self) -> CudaStepMeteorology:
        return CudaStepMeteorology(
            wet_surface_pressure_start_hpa=self._met["wet_start"],
            wet_surface_pressure_hpa=self._met["wet_end"],
            dry_surface_pressure_start_hpa=self._met["dry_start"],
            dry_surface_pressure_hpa=self._met["dry_end"],
            specific_humidity_kg_kg=self._met["qv"],
            specific_humidity_after_kg_kg=self._vdiff_double[
                "specific_humidity_after"
            ],
            temperature_k=self._met["temperature"],
        )

    @property
    def next_delp_dry_hpa(self) -> Any:
        """Canonical bottom-to-top dry pressure thickness for the step end."""

        return self._tpcore_double["delp2"][None, ::-1, :, :]

    @property
    def next_dry_air_mass_kg(self) -> Any:
        """Canonical bottom-to-top dry air mass for the step end."""

        return self._vdiff_double["dry_mass"][None, ::-1, :, :]

    @property
    def specific_humidity_after_kg_kg(self) -> Any:
        """Canonical bottom-to-top humidity after VDIFF."""

        return self._vdiff_double["specific_humidity_after"][
            None, ::-1, :, :
        ]

    def prepare_tpcore_step(
        self,
        forcing: CudaForcingStep,
        *,
        dt_s: float,
    ) -> CudaTpcorePlan:
        """Interpolate I3 fields and reproduce CPU strict TPCORE preparation."""

        horizontal_size = self._nlat * self._nlon
        center_size = self._nlev * horizontal_size
        threads = 128
        self._prepare_surface_endpoints(
            ((horizontal_size + threads - 1) // threads,),
            (threads,),
            (
                forcing.surface_pressure_start_pa,
                forcing.surface_pressure_end_pa,
                forcing.qv_start,
                forcing.qv_end,
                self._hyai,
                self._hybi,
                self._surface_endpoint["wet_start"],
                self._surface_endpoint["wet_end"],
                self._surface_endpoint["dry_start"],
                self._surface_endpoint["dry_end"],
                np.int32(self._nlev),
                np.int32(self._nlat),
                np.int32(self._nlon),
            ),
        )
        self._average_surface_poles(
            (1,),
            (1,),
            (
                self._surface_endpoint["wet_start"],
                self._surface_endpoint["wet_end"],
                self._surface_endpoint["dry_start"],
                self._surface_endpoint["dry_end"],
                self._area,
                np.int32(self._nlat),
                np.int32(self._nlon),
            ),
        )
        self._interpolate_meteorology(
            ((center_size + threads - 1) // threads,),
            (threads,),
            (
                self._surface_endpoint["wet_start"],
                self._surface_endpoint["wet_end"],
                self._surface_endpoint["dry_start"],
                self._surface_endpoint["dry_end"],
                forcing.qv_start,
                forcing.qv_end,
                forcing.temperature_start_k,
                forcing.temperature_end_k,
                np.float64(forcing.start_fraction),
                np.float64(forcing.end_fraction),
                np.float64(forcing.midpoint_fraction),
                self._met["wet_start"],
                self._met["wet_end"],
                self._met["dry_start"],
                self._met["dry_end"],
                self._met["qv"],
                self._met["temperature"],
                np.int32(self._nlev),
                np.int32(self._nlat),
                np.int32(self._nlon),
            ),
        )
        d = self._tpcore_double
        s = self._tpcore_scratch
        self._tpcore_pressure_fix(
            (1,),
            (1,),
            (
                self._current_dry_surface,
                self._met["dry_end"],
                self._rel_area,
                s["p1"],
                s["p2"],
                np.int32(self._nlat),
                np.int32(self._nlon),
            ),
        )
        center_blocks = (center_size + threads - 1) // threads
        horizontal_blocks = (horizontal_size + threads - 1) // threads
        level_blocks = (self._nlev + threads - 1) // threads
        self._tpcore_mass_flux(
            (center_blocks,),
            (threads,),
            (
                s["p1"],
                s["p2"],
                forcing.u_m_s,
                forcing.v_m_s,
                np.float64(dt_s),
                self._cosp,
                self._cose,
                self._dap_geos,
                self._dbk_geos,
                d["work3"],
                d["xmass"],
                d["ymass"],
                np.int32(self._nlev),
                np.int32(self._nlat),
                np.int32(self._nlon),
            ),
        )
        self._launch_divergence(
            d["xmass"],
            d["ymass"],
            d["work3"],
            bottom_reversed=True,
            center_blocks=center_blocks,
            level_blocks=level_blocks,
            threads=threads,
        )
        self._tpcore_sum_vertical(
            (horizontal_blocks,),
            (threads,),
            (
                d["work3"],
                s["work2"],
                np.int32(1),
                np.int32(self._nlev),
                np.int32(horizontal_size),
            ),
        )
        self._tpcore_pressure_correction(
            (1,),
            (1,),
            (
                s["p1"],
                s["p2"],
                s["work2"],
                self._rel_area,
                self._geofac_double,
                np.float64(self._geofac_pc),
                s["xfix"],
                s["mmfd"],
                s["mmf"],
                s["fxintegral"],
                np.int32(self._nlat),
                np.int32(self._nlon),
            ),
        )
        self._tpcore_apply_correction(
            (center_blocks,),
            (threads,),
            (
                d["xmass"],
                d["ymass"],
                s["xfix"],
                s["mmf"],
                self._dbk_geos,
                np.int32(self._nlev),
                np.int32(self._nlat),
                np.int32(self._nlon),
            ),
        )
        self._tpcore_restore_pressure(
            (1,),
            (1,),
            (
                self._current_dry_surface,
                self._met["dry_end"],
                self._rel_area,
                s["p1"],
                s["p2"],
                np.int32(self._nlat),
                np.int32(self._nlon),
            ),
        )
        self._tpcore_pressure_terms(
            (center_blocks,),
            (threads,),
            (
                s["p1"],
                s["p2"],
                d["xmass"],
                d["ymass"],
                self._cose,
                self._dap_top,
                self._dbk_top,
                d["delp1"],
                d["delpm"],
                d["delp2"],
                d["pu"],
                d["cx"],
                d["cy"],
                np.int32(self._nlev),
                np.int32(self._nlat),
                np.int32(self._nlon),
            ),
        )
        self._launch_divergence(
            d["xmass"],
            d["ymass"],
            d["work3"],
            bottom_reversed=False,
            center_blocks=center_blocks,
            level_blocks=level_blocks,
            threads=threads,
        )
        self._tpcore_vertical_flux(
            (horizontal_blocks,),
            (threads,),
            (
                d["work3"],
                self._dbk_top,
                d["delp1"],
                s["work2"],
                d["vertical_mass_flux"],
                d["normalized_vertical_courant"],
                np.int32(self._nlev),
                np.int32(horizontal_size),
            ),
        )
        self._tpcore_cross_terms(
            (center_blocks,),
            (threads,),
            (
                d["cx"],
                d["cy"],
                d["ua"],
                d["va"],
                np.int32(self._nlev),
                np.int32(self._nlat),
                np.int32(self._nlon),
            ),
        )
        self._tpcore_jn_js(
            (level_blocks,),
            (threads,),
            (
                d["cx"],
                self._jn,
                self._js,
                np.int32(self._nlev),
                np.int32(self._nlat),
                np.int32(self._nlon),
            ),
        )
        if self._dtype != np.dtype(np.float64):
            blocks = (center_size + threads - 1) // threads
            for name in self._TPCORE_CENTER_NAMES:
                self._cast_plan(
                    (blocks,),
                    (threads,),
                    (
                        d[name],
                        self._tpcore_output[name],
                        np.int64(center_size),
                    ),
                )
        self._cupy.copyto(self._current_dry_surface, self._met["dry_end"])
        return self._plan

    def _launch_divergence(
        self,
        xmass: Any,
        ymass: Any,
        output: Any,
        *,
        bottom_reversed: bool,
        center_blocks: int,
        level_blocks: int,
        threads: int,
    ) -> None:
        self._tpcore_divergence_interior(
            (center_blocks,),
            (threads,),
            (
                xmass,
                ymass,
                self._geofac_double,
                output,
                np.int32(bottom_reversed),
                np.int32(self._nlev),
                np.int32(self._nlat),
                np.int32(self._nlon),
            ),
        )
        self._tpcore_divergence_poles(
            (level_blocks,),
            (threads,),
            (
                ymass,
                np.float64(self._geofac_pc),
                output,
                np.int32(bottom_reversed),
                np.int32(self._nlev),
                np.int32(self._nlat),
                np.int32(self._nlon),
            ),
        )

    def prepare_vdiff_and_convection(
        self,
        forcing: CudaForcingStep,
        *,
        dt_s: float,
    ) -> tuple[CudaVdiffPlan, CudaConvectionPlan]:
        """Build VDIFF coefficients and oriented convection fields in place."""

        horizontal_size = self._nlat * self._nlon
        center_size = self._nlev * horizontal_size
        threads = 128
        self._compute_vdiff_start(
            (1,),
            (1,),
            (
                self._met["wet_end"],
                self._hyai,
                self._hybi,
                self._vdiff_start_level,
                np.int32(self._nlev),
                np.int32(self._nlat),
                np.int32(self._nlon),
            ),
        )
        v = self._vdiff_double
        self._prepare_vdiff(
            ((horizontal_size + threads - 1) // threads,),
            (threads,),
            (
                forcing.u_m_s,
                forcing.v_m_s,
                self._met["temperature"],
                self._met["qv"],
                self._met["wet_end"],
                forcing.pblh_m,
                forcing.hflux_w_m2,
                forcing.eflux_w_m2,
                forcing.ustar_m_s,
                self._area,
                self._hyai,
                self._hybi,
                self._tpcore_double["delp2"],
                np.float64(dt_s),
                self._vdiff_start_level,
                v["cch"],
                v["zeh"],
                v["termh"],
                v["cgs"],
                v["kvh"],
                v["potbar"],
                v["rpdel"],
                v["rrho"],
                v["tmp1"],
                v["dry_mass"],
                v["specific_humidity_after"],
                np.int32(self._nlev),
                np.int32(self._nlat),
                np.int32(self._nlon),
            ),
        )
        if self._dtype != np.dtype(np.float64):
            for name, source in self._vdiff_double.items():
                count = int(source.size)
                self._cast_plan(
                    ((count + threads - 1) // threads,),
                    (threads,),
                    (
                        source,
                        self._vdiff_output[name],
                        np.int64(count),
                    ),
                )
        self._vdiff_plan = CudaVdiffPlan(
            cch=self._vdiff_output["cch"],
            zeh=self._vdiff_output["zeh"],
            termh=self._vdiff_output["termh"],
            cgs=self._vdiff_output["cgs"],
            kvh=self._vdiff_output["kvh"],
            potbar=self._vdiff_output["potbar"],
            rpdel=self._vdiff_output["rpdel"],
            rrho=self._vdiff_output["rrho"],
            tmp1=self._vdiff_output["tmp1"],
            dry_mass=self._vdiff_output["dry_mass"],
            area_m2=self._area_transport,
            dt_s=float(dt_s),
            start_level=self._vdiff_start_level,
            specific_humidity_after=self._vdiff_output[
                "specific_humidity_after"
            ],
        )

        c = self._convection_arrays
        expression_count = center_size
        self._prepare_convection(
            ((expression_count + threads - 1) // threads,),
            (threads,),
            (
                forcing.cmfmc_kg_m2_s,
                forcing.dtrain_kg_m2_s,
                forcing.dqrcu_kg_kg_s,
                forcing.reevapcn_kg_kg_s,
                self._tpcore_double["delp2"],
                c["cmfmc"],
                c["dtrain"],
                c["dqrcu"],
                c["reevapcn"],
                c["delp"],
                c["bmass"],
                np.int32(self._nlev),
                np.int32(self._nlat),
                np.int32(self._nlon),
            ),
        )
        internal_steps = max(int(dt_s) // 300, 1)
        self._convection_plan = CudaConvectionPlan(
            cmfmc=c["cmfmc"],
            dtrain=c["dtrain"],
            delp_hpa=c["delp"],
            delp_dry=c["delp"],
            bmass=c["bmass"],
            dqrcu=c["dqrcu"],
            reevapcn=c["reevapcn"],
            area_m2=self._area_transport,
            reconstruct_conv_precip_flux=False,
            internal_steps=internal_steps,
            internal_dt_s=float(dt_s) / internal_steps,
        )
        return self._vdiff_plan, self._convection_plan
