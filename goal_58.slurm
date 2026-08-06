#!/bin/bash
#SBATCH --job-name=hipac-goal58
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PREFIX="${PREFIX:-HiPAC2026crypto}"
GOAL_BITS="${GOAL_BITS:-58}"
# Slurm job limit 是 5 分鐘，預留 15 秒收集各 rank 的結果。
SEARCH_SECONDS="${SEARCH_SECONDS:-285}"
BATCH_NONCES="${BATCH_NONCES:-100000000}"
TASKS="${SLURM_NTASKS:-16}"

if [[ "${TASKS}" -ne 16 ]]; then
    echo "需要 16 個 task（2 nodes x 8 GPUs），目前是 ${TASKS}" >&2
    exit 2
fi
if ! [[ "${GOAL_BITS}" =~ ^[0-9]+$ && "${GOAL_BITS}" -gt 0 ]]; then
    echo "GOAL_BITS 必須是正整數" >&2
    exit 2
fi
if ! [[ "${SEARCH_SECONDS}" =~ ^[0-9]+$ && "${SEARCH_SECONDS}" -gt 0 ]]; then
    echo "SEARCH_SECONDS 必須是正整數" >&2
    exit 2
fi

if [[ -n "${WORKDIR:-}" ]]; then
    cd "${WORKDIR}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(dirname -- "${BASH_SOURCE[0]}")"
fi

if [[ -n "${CUDA_MODULE:-}" ]]; then
    if ! command -v module >/dev/null 2>&1; then
        echo "CUDA_MODULE 已設定，但找不到 module 指令" >&2
        exit 1
    fi
    module load "${CUDA_MODULE}"
elif command -v module >/dev/null 2>&1; then
    module load cuda
fi

if ! command -v nvcc >/dev/null 2>&1; then
    shopt -s nullglob
    cuda_roots=("${CUDA_HOME:-}" /usr/local/cuda /usr/local/cuda-*)
    for cuda_root in "${cuda_roots[@]}"; do
        if [[ -x "${cuda_root}/bin/nvcc" ]]; then
            export PATH="${cuda_root}/bin:${PATH}"
            break
        fi
    done
fi
command -v nvcc >/dev/null 2>&1 || {
    echo "找不到 nvcc" >&2
    exit 1
}

echo "=== build ==="
if [[ -n "${ARCH:-}" ]]; then
    make -B ARCH="${ARCH}"
else
    make -B
fi

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RUN_ROOT="${WORKDIR}/.goal58-${SLURM_JOB_ID}"
mkdir -p "${RUN_ROOT}"
export WORKDIR PREFIX RUN_ROOT SEARCH_SECONDS BATCH_NONCES

echo "=== formal goal search ==="
echo "prefix=${PREFIX}, goal_bits=${GOAL_BITS}, nodes=2, gpus=16"
echo "search_seconds=${SEARCH_SECONDS}, mode=continuous (no total)"

srun --ntasks=16 --ntasks-per-node=8 --gpus-per-task=1 \
    --kill-on-bad-exit=1 \
    --output="${RUN_ROOT}/rank-%t.out" \
    --error="${RUN_ROOT}/rank-%t.err" \
    bash -c '
        set -euo pipefail
        rank_dir="${RUN_ROOT}/rank-${SLURM_PROCID}"
        mkdir -p "${rank_dir}"
        cd "${rank_dir}"
        set +e
        timeout --signal=INT --kill-after=8s "${SEARCH_SECONDS}s" \
            env BATCH_NONCES="${BATCH_NONCES}" "${WORKDIR}/collision" \
            --start "$((SLURM_PROCID * 1000000000000))" "${PREFIX}"
        rc=$?
        set -e
        if [[ "${rc}" -ne 0 && "${rc}" -ne 124 && "${rc}" -ne 130 ]]; then
            exit "${rc}"
        fi
    '

shopt -s nullglob
log_files=("${RUN_ROOT}"/rank-*.out)
solution_files=("${RUN_ROOT}"/rank-*/solution_*.csv)
max_qbit=0
completed_nonces=0
if (( ${#log_files[@]} > 0 )); then
    max_qbit="$(grep -hEo '目前最多共同前綴 qbit：[0-9]+' "${log_files[@]}" \
        | awk -F '：' '{v=$2+0; if (v>max) max=v} END {print max+0}')"
    completed_nonces="$(grep -hEo '收到停止訊號，已完成 [0-9]+' "${log_files[@]}" \
        | awk '{sum+=$2} END {print sum+0}')"
fi

best_bits=0
best_file=""
for solution_file in "${solution_files[@]}"; do
    bits="${solution_file##*solution_}"
    bits="${bits%.csv}"
    if (( bits > best_bits )); then
        best_bits="${bits}"
        best_file="${solution_file}"
    fi
done
if [[ -n "${best_file}" ]]; then
    cp "${best_file}" "${WORKDIR}/solution_${best_bits}.csv"
fi

echo "=== goal summary ==="
echo "goal_bits=${GOAL_BITS}"
echo "current_max_qbit=${max_qbit}"
if (( max_qbit >= GOAL_BITS )); then
    echo "goal_reached=yes"
else
    echo "goal_reached=no"
fi
echo "completed_nonces=${completed_nonces}"
awk -v n="${completed_nonces}" -v s="${SEARCH_SECONDS}" \
    'BEGIN { printf "aggregate_throughput=%.2f MH/s (completed nonce / search time)\n", n / s / 1000000 }'
