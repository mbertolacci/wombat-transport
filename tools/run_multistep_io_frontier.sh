#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

python="${PYTHON:-.venv/bin/python}"
source_config="${SOURCE_CONFIG:-validation_runs/cases/residual_24tracer_emissions_2month_2x25/wombat/main/run.yml}"
obsoperator_dir="${OBSOPERATOR_DIR:-external_data/obsoperator}"
output_root="${OUT_DIR:-validation_runs/work/hpc_multistep_io_frontier}"
start_time="${START:-2014-09-01 00:00}"
end_time="${END:-2014-09-03 00:00}"
repeats="${REPEATS:-2}"
warmup_steps="${WARMUP_STEPS:-2}"
numa_node="${NUMA_NODE:-0}"
first_cpu="${FIRST_CPU:-0}"
socket_cores="${SOCKET_CORES:-40}"
cpu_list="${CPU_LIST:-}"
force="${FORCE:-0}"
dry_run="${DRY_RUN:-0}"
numba_cache="${NUMBA_CACHE_DIR:-${TMPDIR:-/tmp}/wombat-numba-cache}"

processes="${PROCESSES:-8}"
threads="${THREADS_PER_PROCESS:-5}"
read -r -a configured_tracer_totals <<<"${TRACER_COUNTS:-480 640 800}"

physical_cpus=()
if [[ -n "${cpu_list}" ]]; then
  read -r -a physical_cpus <<<"${cpu_list//,/ }"
fi

cpu_binding_for() {
  local offset="$1"
  local count="$2"
  if (( ${#physical_cpus[@]} )); then
    local selected=("${physical_cpus[@]:offset:count}")
    local IFS=,
    printf '%s' "${selected[*]}"
  else
    local start=$((first_cpu + offset))
    printf '%s-%s' "${start}" "$((start + count - 1))"
  fi
}

if [[ ! -x "${python}" ]]; then
  echo "Python executable not found: ${python}" >&2
  exit 2
fi
if [[ ! -f "${source_config}" ]]; then
  echo "Source config not found: ${source_config}" >&2
  exit 2
fi
if [[ ! -d "${obsoperator_dir}" ]]; then
  echo "ObsOperator input directory not found: ${obsoperator_dir}" >&2
  exit 2
fi
if ! command -v numactl >/dev/null 2>&1; then
  echo "numactl is required for CPU and memory binding" >&2
  exit 2
fi
for value in "${repeats}" "${warmup_steps}" "${first_cpu}" "${socket_cores}" "${processes}" "${threads}" \
  "${configured_tracer_totals[@]}" "${physical_cpus[@]}"; do
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "Expected a non-negative integer, got: ${value}" >&2
    exit 2
  fi
done
if (( repeats == 0 || processes == 0 || threads == 0 || ${#configured_tracer_totals[@]} == 0 )); then
  echo "REPEATS, PROCESSES, THREADS_PER_PROCESS, and TRACER_COUNTS must be positive" >&2
  exit 2
fi
for total_tracers in "${configured_tracer_totals[@]}"; do
  if (( total_tracers == 0 || total_tracers % processes != 0 )); then
    echo "Each tracer count must be positive and divisible by PROCESSES: ${total_tracers}" >&2
    exit 2
  fi
done

available_cores="${socket_cores}"
if (( ${#physical_cpus[@]} )); then
  available_cores="${#physical_cpus[@]}"
fi
if (( available_cores < processes * threads )); then
  echo "The experiment requires at least $((processes * threads)) configured physical CPUs" >&2
  exit 2
fi

mkdir -p "${output_root}" "${numba_cache}"
manifest="${output_root}/manifest.csv"
printf '%s\n' \
  "repetition,total_tracers,processes,threads_per_process,tracers_per_process,writer,start,end,case_dir" \
  >"${manifest}"

case_count=0
for ((repetition = 1; repetition <= repeats; repetition++)); do
  if (( repetition % 2 == 1 )); then
    tracer_totals=("${configured_tracer_totals[@]}")
  else
    tracer_totals=()
    for ((index = ${#configured_tracer_totals[@]} - 1; index >= 0; index--)); do
      tracer_totals+=("${configured_tracer_totals[index]}")
    done
  fi
  for total_tracers in "${tracer_totals[@]}"; do
    tracers_per_process=$((total_tracers / processes))
    writer="sync"
    case_name="r${repetition}_m${total_tracers}_p${processes}_t${threads}_b${tracers_per_process}"
    printf '%s\n' \
      "${repetition},${total_tracers},${processes},${threads},${tracers_per_process},${writer},${start_time},${end_time},${case_name}" \
      >>"${manifest}"
    case_count=$((case_count + 1))
  done
done

expected_cases=$((repeats * ${#configured_tracer_totals[@]}))
echo "Prepared ${case_count} cases in ${manifest}"
echo "Each rank runs real met, emissions, and ObsOperator for ${start_time} through ${end_time}, with daily SpeciesConc output."
if (( case_count != expected_cases )); then
  echo "Expected ${expected_cases} cases, generated ${case_count}" >&2
  exit 2
fi
if [[ "${dry_run}" == "1" ]]; then
  exit 0
fi

case_number=0
while IFS=, read -r repetition total_tracers rank_count rank_threads tracers_per_process writer case_start case_end case_name; do
  if [[ "${repetition}" == "repetition" ]]; then
    continue
  fi

  case_number=$((case_number + 1))
  case_dir="${output_root}/${case_name}"
  complete_marker="${case_dir}/.complete"
  mkdir -p "${case_dir}"

  if [[ -f "${complete_marker}" && "${force}" != "1" ]]; then
    echo "[${case_number}/${case_count}] already complete: ${case_name}"
    continue
  fi
  if [[ "${force}" == "1" ]]; then
    rm -f "${complete_marker}"
  fi

  echo "[${case_number}/${case_count}] repetition ${repetition}: ${total_tracers} tracers, "\
       "${rank_count} x ${rank_threads} threads, ${writer} output"

  pids=()
  for ((rank = 0; rank < rank_count; rank++)); do
    cpu_offset=$((rank * rank_threads))
    if (( cpu_offset + rank_threads > available_cores )); then
      echo "CPU assignment for rank ${rank} exceeds the configured physical CPUs" >&2
      exit 2
    fi
    cpu_binding="$(cpu_binding_for "${cpu_offset}" "${rank_threads}")"

    rank_dir="${case_dir}/rank_${rank}"
    rank_log="${case_dir}/rank_${rank}.log"
    field_offset=$((rank * tracers_per_process))
    mkdir -p "${rank_dir}"

    numactl \
      --physcpubind="${cpu_binding}" \
      --membind="${numa_node}" \
      env \
        NUMBA_NUM_THREADS="${rank_threads}" \
        WOMBAT_NUMBA_THREADS="${rank_threads}" \
        NUMBA_CACHE_DIR="${numba_cache}" \
        OMP_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        BLIS_NUM_THREADS=1 \
        NUMEXPR_NUM_THREADS=1 \
        VECLIB_MAXIMUM_THREADS=1 \
      "${python}" tools/profile_multistep_runtime.py \
        --source-config "${source_config}" \
        --output-dir "${rank_dir}" \
        --counts "${tracers_per_process}" \
        --field-offset "${field_offset}" \
        --start "${case_start}" \
        --end "${case_end}" \
        --warmup-steps "${warmup_steps}" \
        --outputs \
        --obsoperator-input-dir "${obsoperator_dir}" \
        >"${rank_log}" 2>&1 &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if (( failed != 0 )); then
    echo "Case failed: ${case_name}; inspect ${case_dir}/rank_*.log" >&2
    exit 1
  fi
  for ((rank = 0; rank < rank_count; rank++)); do
    if [[ ! -s "${case_dir}/rank_${rank}/summary.json" ]]; then
      echo "Case produced no summary for rank ${rank}: ${case_name}" >&2
      exit 1
    fi
  done

  touch "${complete_marker}"
done <"${manifest}"

echo "All ${case_count} cases completed."
echo "Summarize with: ${python} tools/summarize_multistep_io_frontier.py ${output_root}"
