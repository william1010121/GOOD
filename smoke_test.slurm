#!/bin/bash
# =====================================================================
# HiPAC 隱藏題1 — 正式搜尋（2 nodes × 8 GPU = 16 rank）
#
#   sbatch run_collision.slurm
#   sbatch --export=ALL,SEARCH_SECONDS=285,NONCE_START=0 run_collision.slurm
#
# 執行結構沿用 master 的 goal_58.slurm（已在本叢集驗證可跑）：
#   16 個 task、每 task 一張 GPU、nvcc 路徑自動搜尋、timeout 硬上限、
#   每個 rank 自己的目錄、最後彙整取最大 bits。
#
# ★ 唯一刻意不同的地方：--start 對所有 rank 都一樣。
#   舊版讓每個 rank 掃不同的 nonce 區段（PROCID * 1e12），那會切出 16 個
#   互不比較的池子。本版改成「所有 rank 掃同一段 nonce，但各自擁有不同的
#   hash partition」——最佳那對的共同前綴遠大於 partition 前綴，所以它們
#   必然落在同一個 partition、由唯一一個 rank 獨自持有，不需要跨 rank merge。
# =====================================================================
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

PREFIX="${PREFIX:-HiPAC2026crypto}"
# Slurm job limit 是 5 分鐘，預留 15 秒收集各 rank 的結果。
SEARCH_SECONDS="${SEARCH_SECONDS:-60}"
# 換一個值就是一輪完全獨立的搜尋（best-of-k）。所有 rank 必須用同一個值。
NONCE_START="${NONCE_START:-0}"
TASKS="${SLURM_NTASKS:-16}"

if [[ "${TASKS}" -ne 16 ]]; then
    echo "需要 16 個 task（2 nodes x 8 GPUs），目前是 ${TASKS}" >&2
    exit 2
fi

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${WORKDIR}"

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
    shopt -u nullglob
fi
command -v nvcc >/dev/null 2>&1 || { echo "找不到 nvcc" >&2; exit 1; }
echo "nvcc: $(command -v nvcc)"

echo "=== 硬體診斷（唯讀，不啟動 opensmd）==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

echo "=== build ==="
if [[ -n "${ARCH:-}" ]]; then
    make -B ARCH="${ARCH}"
else
    make -B
fi

RUN_ROOT="${WORKDIR}/.smoke-${SLURM_JOB_ID}"
mkdir -p "${RUN_ROOT}"
export WORKDIR PREFIX RUN_ROOT SEARCH_SECONDS NONCE_START

echo "=== search ==="
echo "prefix=${PREFIX}, nodes=2, gpus=16, search_seconds=${SEARCH_SECONDS}"
echo "nonce_start=${NONCE_START}（所有 rank 相同；各 rank 分到不同的 hash partition）"

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
        timeout --signal=INT --kill-after=10s "$((SEARCH_SECONDS + 20))s" \
            "${WORKDIR}/collision" \
            --seconds "${SEARCH_SECONDS}" \
            --start "${NONCE_START}" \
            "${PREFIX}"
        rc=$?
        set -e
        if [[ "${rc}" -ne 0 && "${rc}" -ne 124 && "${rc}" -ne 130 ]]; then
            exit "${rc}"
        fi
    '

echo "=== rank 0 完整輸出（含瓶頸診斷）==="
cat "${RUN_ROOT}/rank-0.out" || true

echo "=== 各 rank 摘要 ==="
grep -h "完成 .* 個 partition\|時間拆解" "${RUN_ROOT}"/rank-*.out || true

echo "=== 彙整最佳解 ==="
shopt -s nullglob
best_bits=0
best_file=""
for solution_file in "${RUN_ROOT}"/rank-*/solution_*.csv; do
    bits="${solution_file##*solution_}"
    bits="${bits%.csv}"
    if (( bits > best_bits )); then
        best_bits="${bits}"
        best_file="${solution_file}"
    fi
done

if [[ -z "${best_file}" ]]; then
    echo "沒有產生任何 solution CSV" >&2
    exit 1
fi

cp "${best_file}" "${WORKDIR}/solution_${best_bits}.csv"
echo "best_bits=${best_bits}"
echo "solution=${WORKDIR}/solution_${best_bits}.csv"
cat "${WORKDIR}/solution_${best_bits}.csv"
python3 "${WORKDIR}/verify_collision.py" --file "${WORKDIR}/solution_${best_bits}.csv"
