#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

python="${PYTHON:-.venv/bin/python}"
run_config="${RUN_CONFIG:-validation_runs/cases/realistic_restart_noemis_2x25/wombat/main/run.yml}"
output_root="${OUT_DIR:-validation_runs/work/hpc_fixed_total_frontier}"
numa_node="${NUMA_NODE:-0}"
first_cpu="${FIRST_CPU:-0}"
socket_cores="${SOCKET_CORES:-40}"
cpu_list="${CPU_LIST:-}"
repeat="${REPEAT:-5}"
warmup="${WARMUP:-2}"
force="${FORCE:-0}"
dry_run="${DRY_RUN:-0}"
numba_cache="${NUMBA_CACHE_DIR:-${TMPDIR:-/tmp}/wombat-numba-cache}"

read -r -a total_tracer_counts <<<"${TRACER_COUNTS:-120 240 320 400 480 640 800}"
read -r -a core_budgets <<<"${CORE_BUDGETS:-8 16 20 24 32 40}"
read -r -a process_counts <<<"${PROCESS_COUNTS:-1 2 4 5 8 10 20 40}"

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
if [[ ! -f "${run_config}" ]]; then
  echo "Run config not found: ${run_config}" >&2
  exit 2
fi
if ! command -v numactl >/dev/null 2>&1; then
  echo "numactl is required for CPU and memory binding" >&2
  exit 2
fi
for value in "${first_cpu}" "${socket_cores}" "${repeat}" "${warmup}" \
  "${total_tracer_counts[@]}" "${core_budgets[@]}" "${process_counts[@]}" "${physical_cpus[@]}"; do
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "Expected a non-negative integer, got: ${value}" >&2
    exit 2
  fi
done
if (( repeat == 0 || ${#total_tracer_counts[@]} == 0 || ${#core_budgets[@]} == 0 || ${#process_counts[@]} == 0 )); then
  echo "REPEAT and all experiment dimensions must be non-empty and positive" >&2
  exit 2
fi
for value in "${total_tracer_counts[@]}" "${core_budgets[@]}" "${process_counts[@]}"; do
  if (( value == 0 )); then
    echo "Experiment dimensions must contain only positive integers" >&2
    exit 2
  fi
done

available_cores="${socket_cores}"
if (( ${#physical_cpus[@]} )); then
  available_cores="${#physical_cpus[@]}"
fi
for total_cores in "${core_budgets[@]}"; do
  if (( total_cores > available_cores )); then
    echo "Core budget ${total_cores} exceeds ${available_cores} configured physical CPUs" >&2
    exit 2
  fi
done

mkdir -p "${output_root}" "${numba_cache}"
manifest="${output_root}/manifest.csv"
printf '%s\n' \
  "total_tracers,total_cores,processes,threads_per_process,tracers_per_process,case_dir" \
  >"${manifest}"

case_count=0
for total_tracers in "${total_tracer_counts[@]}"; do
  for total_cores in "${core_budgets[@]}"; do
    for processes in "${process_counts[@]}"; do
      if (( total_tracers % processes != 0 || total_cores % processes != 0 )); then
        continue
      fi
      threads=$((total_cores / processes))
      tracers_per_process=$((total_tracers / processes))
      case_name="m${total_tracers}_c${total_cores}_p${processes}_t${threads}_b${tracers_per_process}"
      printf '%s\n' \
        "${total_tracers},${total_cores},${processes},${threads},${tracers_per_process},${case_name}" \
        >>"${manifest}"
      case_count=$((case_count + 1))
    done
  done
done

echo "Prepared ${case_count} cases in ${manifest}"
if (( case_count == 0 )); then
  echo "The configured dimensions generated no valid cases" >&2
  exit 2
fi
if [[ "${dry_run}" == "1" ]]; then
  exit 0
fi

case_number=0
while IFS=, read -r total_tracers total_cores processes threads tracers_per_process case_name; do
  if [[ "${total_tracers}" == "total_tracers" ]]; then
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

  echo "[${case_number}/${case_count}] ${total_tracers} tracers, ${total_cores} cores: "\
       "${processes} processes x ${threads} threads (${tracers_per_process} tracers/process)"

  pids=()
  for ((rank = 0; rank < processes; rank++)); do
    cpu_offset=$((rank * threads))
    if (( cpu_offset + threads > available_cores )); then
      echo "CPU assignment for rank ${rank} exceeds the configured physical CPUs" >&2
      exit 2
    fi
    cpu_binding="$(cpu_binding_for "${cpu_offset}" "${threads}")"

    rank_csv="${case_dir}/rank_${rank}.csv"
    rank_log="${case_dir}/rank_${rank}.log"
    numactl \
      --physcpubind="${cpu_binding}" \
      --membind="${numa_node}" \
      env \
        NUMBA_NUM_THREADS="${threads}" \
        WOMBAT_NUMBA_THREADS="${threads}" \
        NUMBA_CACHE_DIR="${numba_cache}" \
        OMP_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        BLIS_NUM_THREADS=1 \
        NUMEXPR_NUM_THREADS=1 \
        VECLIB_MAXIMUM_THREADS=1 \
      "${python}" tools/benchmark_transport_driver_scaling.py \
        --run-config "${run_config}" \
        --counts "${tracers_per_process}" \
        --repeat "${repeat}" \
        --warmup "${warmup}" \
        --output "${rank_csv}" \
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

  touch "${complete_marker}"
done <"${manifest}"

echo "All ${case_count} cases completed."
echo "Summarize with: ${python} tools/summarize_fixed_total_frontier.py ${output_root}"
