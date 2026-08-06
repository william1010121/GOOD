#!/bin/bash
# =====================================================================
# HiPAC 隱藏題1 — 正式搜尋（2 nodes × 8 GPU = 16 rank）
#
#   sbatch run_collision.slurm
#   sbatch --export=ALL,SEARCH_SECONDS=240,PREFIX=HiPAC2026crypto run_collision.slurm
#
# 每個 node 跑「一個」process，process 內用 OpenMP 一 thread 綁一張 GPU。
# 這是相對舊版最重要的改動：舊版每 20M nonce 就重啟一次 process，
# CUDA context 初始化吃掉 99.9% 的時間（10.5 GH/s 的卡端到端只有 4.44 MH/s）。
#
# 結果：每個 node 在自己的目錄寫 solution_<bits>.csv，最後彙整取最大。
# =====================================================================
#SBATCH --job-name=hipac-collision
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=00:05:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PREFIX="${PREFIX:-HiPAC2026crypto}"
SEARCH_SECONDS="${SEARCH_SECONDS:-240}"   # 生成階段秒數；Slurm 上限 5 分鐘，留收尾邊際
NONCE_START="${NONCE_START:-0}"           # 換一個值就是一輪獨立的 best-of-k

cd "${SLURM_SUBMIT_DIR}"

if [[ -n "${CUDA_MODULE:-}" ]]; then
    module load "${CUDA_MODULE}"
elif command -v module >/dev/null 2>&1; then
    module load cuda
fi

echo "=== 硬體診斷（唯讀，不啟動 opensmd）==="
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv || true
nvidia-smi topo -m || true

echo "=== build ==="
if [[ -n "${ARCH:-}" ]]; then
    make -B ARCH="${ARCH}"
else
    make -B
fi

RUN_ROOT="${SLURM_SUBMIT_DIR}/.run-${SLURM_JOB_ID}"
mkdir -p "${RUN_ROOT}"
export RUN_ROOT PREFIX SEARCH_SECONDS NONCE_START

echo "=== search: prefix=${PREFIX}, ${SEARCH_SECONDS}s, 2 nodes × 8 GPU ==="
srun --ntasks=2 --ntasks-per-node=1 --gres=gpu:8 \
    --output="${RUN_ROOT}/node-%t.out" --error="${RUN_ROOT}/node-%t.err" \
    bash -c '
        set -euo pipefail
        node_dir="${RUN_ROOT}/node-${SLURM_PROCID}"
        mkdir -p "${node_dir}"
        cd "${node_dir}"
        export OMP_NUM_THREADS=8
        exec "${SLURM_SUBMIT_DIR}/collision" \
            --seconds "${SEARCH_SECONDS}" \
            --start "${NONCE_START}" \
            "${PREFIX}"
    '

echo "=== 各 node 輸出 ==="
cat "${RUN_ROOT}"/node-*.out

echo "=== 彙整最佳解 ==="
best_bits=0
best_file=""
shopt -s nullglob
for f in "${RUN_ROOT}"/node-*/solution_*.csv; do
    bits="${f##*solution_}"; bits="${bits%.csv}"
    if (( bits > best_bits )); then best_bits="${bits}"; best_file="${f}"; fi
done

if [[ -n "${best_file}" ]]; then
    cp "${best_file}" "${SLURM_SUBMIT_DIR}/solution_${best_bits}.csv"
    echo "best_bits=${best_bits}"
    echo "solution=${SLURM_SUBMIT_DIR}/solution_${best_bits}.csv"
    cat "${SLURM_SUBMIT_DIR}/solution_${best_bits}.csv"
    python3 "${SLURM_SUBMIT_DIR}/verify_collision.py" \
        --file "${SLURM_SUBMIT_DIR}/solution_${best_bits}.csv"
else
    echo "沒有產生任何 solution CSV" >&2
    exit 1
fi
