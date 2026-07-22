#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
local_data_root="${WOMBAT_LOCAL_DATA_ROOT:-${repo_root}}"
if [[ ! -d "${local_data_root}/GCClassic" && -d "${repo_root}/../wombat-transport/GCClassic" ]]; then
  local_data_root="${repo_root}/../wombat-transport"
fi

gc_source_root="${GC_SOURCE_ROOT:-${local_data_root}/GCClassic}"
build_root="${GC_BUILD_ROOT:-${local_data_root}/validation_runs/work/realistic_restart_noemis/main/geoschem/build}"
out_dir="${GC_HARNESS_BUILD_DIR:-${repo_root}/tools/gc_harness/build}"
netcdf_prefix="${NETCDF_PREFIX:-/home/mgnb/miniconda3/envs/wombat-v3-forward}"
fc="${FC:-/usr/bin/f95}"

if [[ ! -f "${gc_source_root}/src/GEOS-Chem/GeosCore/vdiff_mod.F90" ]]; then
  echo "GEOS-Chem source not found under ${gc_source_root}; set GC_SOURCE_ROOT." >&2
  exit 2
fi
if [[ ! -d "${build_root}/src/GEOS-Chem/GeosCore" ]]; then
  echo "Compatible GEOS-Chem build tree not found at ${build_root}; set GC_BUILD_ROOT." >&2
  exit 2
fi

mkdir -p "${out_dir}"
src="${repo_root}/tools/gc_harness/gc_transport_frontier_harness.F90"
generated_vdiff="${out_dir}/vdiff_mod.frontier.F90"
generated_convection="${out_dir}/convection_mod.frontier.F90"
generated_pjc="${out_dir}/pjc_pfix_mod.frontier.F90"
exe="${out_dir}/gc_transport_frontier_harness"

"${repo_root}/tools/gc_harness/generate_pjc_frontier_source.py" \
  "${gc_source_root}/src/GEOS-Chem/GeosCore/pjc_pfix_mod.F90" \
  "${generated_pjc}" >/dev/null
"${repo_root}/tools/gc_harness/generate_vdiff_harness_source.py" \
  "${gc_source_root}/src/GEOS-Chem/GeosCore/vdiff_mod.F90" \
  "${generated_vdiff}" \
  --without-trace >/dev/null
"${repo_root}/tools/gc_harness/generate_convection_harness_source.py" \
  "${gc_source_root}/src/GEOS-Chem/GeosCore/convection_mod.F90" \
  "${generated_convection}" >/dev/null

common_compile_flags=(
  -O2
  -fopenmp
  -J"${out_dir}"
  -I"${out_dir}"
  -I"${build_root}/mod"
  -I"${netcdf_prefix}/include"
)
link_flags=(
  -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now
  -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,--allow-shlib-undefined
  -Wl,-rpath,"${netcdf_prefix}/lib"
  -Wl,-rpath-link,"${netcdf_prefix}/lib"
  -L"${netcdf_prefix}/lib"
  -L"${netcdf_prefix}/targets/x86_64-linux/lib"
  -L"${netcdf_prefix}/targets/x86_64-linux/lib/stubs"
)
gc_libs=(
  "${build_root}/src/GEOS-Chem/GeosCore/CMakeFiles/GeosCore.dir/cleanup.F90.o"
  "${build_root}/src/GEOS-Chem/GeosCore/libGeosCore.a"
  "${build_root}/src/GEOS-Chem/ObsPack/libObsPack.a"
  "${build_root}/src/GEOS-Chem/ObsOperator/libObsOperator.a"
  "${build_root}/src/GEOS-Chem/History/libHistory.a"
  "${build_root}/src/GEOS-Chem/KPP/fullchem/libKPP.a"
  "${build_root}/src/GEOS-Chem/GeosUtil/libGeosUtil.a"
  "${build_root}/src/GEOS-Chem/NcdfUtil/libNcdfUtil.a"
  "${build_root}/src/GEOS-Chem/GeosUtil/libJulDay.a"
  "${build_root}/src/GEOS-Chem/Headers/libHeaders.a"
  "${build_root}/src/GEOS-Chem/KPP/fullchem/libKPP_FirstPass.a"
  "${build_root}/src/Cloud-J/src/Core/libCloudJ_Core.a"
  "${build_root}/src/HETP/src/Core/libHETP_core.a"
  "${build_root}/src/HEMCO/src/Interfaces/Shared/libHCOI_Shared.a"
  "${build_root}/src/HEMCO/src/Extensions/libHCOX.a"
  "${build_root}/src/HEMCO/src/Core/libHCO.a"
  "${build_root}/src/HEMCO/src/Shared/GeosUtil/libGeosUtilHco.a"
  "${build_root}/src/HEMCO/src/Shared/NcdfUtil/libNcdfUtilHco.a"
  "${build_root}/src/HEMCO/src/Shared/GeosUtil/libJulDayHco.a"
  "${build_root}/src/HEMCO/src/Shared/Headers/libHeadersHco.a"
)

"${fc}" "${common_compile_flags[@]}" -cpp -c "${generated_pjc}" -o "${out_dir}/pjc_pfix_mod.frontier.o"
"${fc}" "${common_compile_flags[@]}" -cpp -c "${generated_vdiff}" -o "${out_dir}/vdiff_mod.frontier.o"
"${fc}" "${common_compile_flags[@]}" -cpp -c "${generated_convection}" -o "${out_dir}/convection_mod.frontier.o"
"${fc}" \
  "${common_compile_flags[@]}" \
  "${src}" \
  "${out_dir}/pjc_pfix_mod.frontier.o" \
  "${out_dir}/vdiff_mod.frontier.o" \
  "${out_dir}/convection_mod.frontier.o" \
  "${link_flags[@]}" \
  -Wl,--start-group \
  "${gc_libs[@]}" \
  -Wl,--end-group \
  "${netcdf_prefix}/lib/libnetcdff.so" \
  "${netcdf_prefix}/lib/libnetcdf.so" \
  -o "${exe}"

echo "${exe}"
