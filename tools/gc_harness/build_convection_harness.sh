#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_root="${repo_root}/base/build"
out_dir="${repo_root}/tools/gc_harness/build"
mkdir -p "${out_dir}"

fc="${FC:-/usr/bin/f95}"
src="${repo_root}/tools/gc_harness/convection_harness.F90"
generated_convection="${out_dir}/convection_mod.harness.F90"
exe="${out_dir}/convection_harness"
netcdf_prefix="${NETCDF_PREFIX:-/home/mgnb/miniconda3/envs/wombat-v3-forward}"

"${repo_root}/tools/gc_harness/generate_convection_harness_source.py" \
  "${repo_root}/GCClassic/src/GEOS-Chem/GeosCore/convection_mod.F90" \
  "${generated_convection}" >/dev/null

common_compile_flags=(
  -O2
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

"${fc}" "${common_compile_flags[@]}" -cpp -c "${generated_convection}" -o "${out_dir}/convection_mod.harness.o"

"${fc}" \
  "${common_compile_flags[@]}" \
  "${src}" \
  "${out_dir}/convection_mod.harness.o" \
  "${link_flags[@]}" \
  -Wl,--start-group \
  "${gc_libs[@]}" \
  -Wl,--end-group \
  -fopenmp \
  "${netcdf_prefix}/lib/libnetcdff.so" \
  "${netcdf_prefix}/lib/libnetcdf.so" \
  -o "${exe}"

echo "${exe}"
