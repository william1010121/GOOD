#!/bin/bash
#SBATCH --job-name=hipac-smoke
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# 可用 sbatch --export=ALL,PREFIX=HiPAC2026crypto,SMOKE_NONCES=100000000,
#   SMOKE_SECONDS=60 smoke_test.slurm 覆寫。
PREFIX="${PREFIX:-HiPAC2026crypto}"
SMOKE_NONCES="${SMOKE_NONCES:-100000000}"
SMOKE_SECONDS="${SMOKE_SECONDS:-60}"
EXACT_EPOCH="${EXACT_EPOCH:-0}"
EXACT_EPOCH_NONCES="${EXACT_EPOCH_NONCES:-1000000}"

if ! [[ "${SMOKE_NONCES}" =~ ^[0-9]+$ && "${SMOKE_NONCES}" -gt 0 ]]; then
    echo "SMOKE_NONCES 必須是正整數" >&2
    exit 2
fi
if ! [[ "${SMOKE_SECONDS}" =~ ^[0-9]+$ && "${SMOKE_SECONDS}" -gt 0 ]]; then
    echo "SMOKE_SECONDS 必須是正整數" >&2
    exit 2
fi

if [[ -n "${WORKDIR:-}" ]]; then
    cd "${WORKDIR}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    cd "${SCRIPT_DIR}"
fi

# 叢集若以 Environment Modules 管理 CUDA，可用 CUDA_MODULE 指定版本。
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
    echo "找不到 nvcc；請設定 CUDA_MODULE 或先載入 CUDA module" >&2
    exit 1
}

echo "=== build ==="
if [[ -n "${ARCH:-}" ]]; then
    make -B ARCH="${ARCH}"
else
    make -B
fi

TASKS="${SLURM_NTASKS:-16}"
if [[ "${TASKS}" -ne 16 ]]; then
    echo "需要 16 個 task（2 nodes x 8 GPUs），目前是 ${TASKS}" >&2
    exit 2
fi

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RUN_ROOT="${WORKDIR}/.slurm-smoke-${SLURM_JOB_ID}"
mkdir -p "${RUN_ROOT}"
export WORKDIR PREFIX SMOKE_NONCES SMOKE_SECONDS RUN_ROOT TASKS EXACT_EPOCH EXACT_EPOCH_NONCES

start_epoch="$(date +%s)"

echo "=== ${SMOKE_SECONDS}s multi-node smoke test ==="
echo "prefix=${PREFIX}, nodes=2, gpus=16, batch/rank=${SMOKE_NONCES}"
echo "one srun launch; each rank runs continuous mode for ${SMOKE_SECONDS}s"
if [[ "${EXACT_EPOCH}" == "1" ]]; then
    rm -rf "${RUN_ROOT}/exact"
    mkdir -p "${RUN_ROOT}/exact"
    echo "exact epoch mode: ${EXACT_EPOCH_NONCES} nonce/rank, 56-bit summary + cross-node exact merge"
fi
srun --ntasks=16 --ntasks-per-node=8 --gpus-per-task=1 \
    --kill-on-bad-exit=1 \
    --output="${RUN_ROOT}/rank-%t.out" \
    --error="${RUN_ROOT}/rank-%t.err" \
    bash -c '
        set -euo pipefail
        rank_dir="${RUN_ROOT}/rank-${SLURM_PROCID}"
        mkdir -p "${rank_dir}"
        cd "${rank_dir}"
        start_nonce="$((SLURM_PROCID * 1000000000000))"
        set +e
        if [[ "${EXACT_EPOCH}" == "1" ]]; then
            env EXACT_EPOCH_DIR="${RUN_ROOT}/exact" EXACT_DUMP_NONCES="${EXACT_EPOCH_NONCES}" \
                "${WORKDIR}/collision" --smoke --start "${start_nonce}" \
                "${PREFIX}" "${EXACT_EPOCH_NONCES}"
        else
            timeout --signal=INT --kill-after=8s "${SMOKE_SECONDS}s" \
                env BATCH_NONCES="${SMOKE_NONCES}" "${WORKDIR}/collision" \
                --start "${start_nonce}" "${PREFIX}"
        fi
        rc=$?
        set -e
        if [[ "${rc}" -ne 0 && "${rc}" -ne 124 && "${rc}" -ne 130 ]]; then
            exit "${rc}"
        fi
    '

end_epoch="$(date +%s)"
elapsed=$((end_epoch - start_epoch))
(( elapsed > 0 )) || elapsed=1

shopt -s nullglob
log_files=("${RUN_ROOT}"/rank-*.out)
solution_files=("${RUN_ROOT}"/rank-*/solution_*.csv)
max_qbit=0
total_nonces=0
if (( ${#log_files[@]} > 0 )); then
    max_qbit="$(awk '/目前最多共同前綴 qbit：/ { split($0, p, "："); split(p[2], q, "（"); v=q[1]+0; if (v > max) max=v } END { print max+0 }' "${log_files[@]}")"
    total_nonces="$(grep -hEo '收到停止訊號，已完成 [0-9]+' "${log_files[@]}" \
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

if [[ "${EXACT_EPOCH}" == "1" ]]; then
    python3 "${WORKDIR}/exact_epoch_merge.py" \
        --node0 "${RUN_ROOT}"/exact/node-0-*.bin \
        --node1 "${RUN_ROOT}"/exact/node-1-*.bin \
        --output "${WORKDIR}/exact_global.csv"
fi
if [[ -n "${best_file}" ]]; then
    cp "${best_file}" "${WORKDIR}/solution_${best_bits}.csv"
fi

echo "=== smoke summary ==="
echo "completed_runs=1"
echo "completed_nonces=${total_nonces}"
echo "elapsed_seconds=${elapsed}"
echo "current_max_qbit_prefix=${max_qbit}"
echo "global_best_scope=best among rank-local candidates"
echo "global_best_source=${best_file:-none}"
awk -v n="${total_nonces}" -v s="${elapsed}" -v g="${TASKS}" \
    'BEGIN { printf "aggregate_throughput=%.2f MH/s (16 GPUs, completed nonce / wall time)\nper_gpu_throughput=%.2f MH/s\n", n / s / 1000000, n / s / 1000000 / g }'
