#!/bin/bash
#SBATCH --job-name=hipac-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# 可用 sbatch --export=ALL,PREFIX=hipac_demo,SMOKE_NONCES=20000000,
#   SMOKE_SECONDS=60 smoke_test.slurm 覆寫。
PREFIX="${PREFIX:-hipac_demo}"
SMOKE_NONCES="${SMOKE_NONCES:-20000000}"
SMOKE_SECONDS="${SMOKE_SECONDS:-60}"

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

start_epoch="$(date +%s)"
deadline=$((start_epoch + SMOKE_SECONDS))
runs=0
total_nonces=0

echo "=== ${SMOKE_SECONDS}s smoke test ==="
echo "prefix=${PREFIX}, nonces/run=${SMOKE_NONCES}"
while true; do
    now="$(date +%s)"
    if (( runs > 0 && now >= deadline )); then
        break
    fi

    ./collision "${PREFIX}" "${SMOKE_NONCES}"
    runs=$((runs + 1))
    total_nonces=$((total_nonces + SMOKE_NONCES))
done

end_epoch="$(date +%s)"
elapsed=$((end_epoch - start_epoch))
(( elapsed > 0 )) || elapsed=1

echo "=== smoke summary ==="
echo "completed_runs=${runs}"
echo "completed_nonces=${total_nonces}"
echo "elapsed_seconds=${elapsed}"
awk -v n="${total_nonces}" -v s="${elapsed}" \
    'BEGIN { printf "aggregate_throughput=%.2f MH/s (completed nonce / wall time)\n", n / s / 1000000 }'
