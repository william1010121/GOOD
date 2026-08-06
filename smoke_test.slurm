#!/bin/bash
# =====================================================================
# 60 秒煙霧測試 —— 跟 run_collision.slurm 完全同一條路徑，只是時間短。
# 用來確認叢集上編得過、16 張卡都吃得到、CSV 驗得過。
#
#   sbatch smoke_test.slurm
#
# 正式跑請用 run_collision.slurm。
# =====================================================================
#SBATCH --job-name=hipac-smoke
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
SEARCH_SECONDS="${SEARCH_SECONDS:-60}"
NONCE_START="${NONCE_START:-0}"

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

RUN_ROOT="${SLURM_SUBMIT_DIR}/.smoke-${SLURM_JOB_ID}"
mkdir -p "${RUN_ROOT}"
export RUN_ROOT PREFIX SEARCH_SECONDS NONCE_START

echo "=== smoke: prefix=${PREFIX}, ${SEARCH_SECONDS}s, 2 nodes × 8 GPU ==="
srun --ntasks=2 --ntasks-per-node=1 --gres=gpu:8 \
    --output="${RUN_ROOT}/node-%t.out" --error="${RUN_ROOT}/node-%t.err" \
    bash -c '
        set -euo pipefail
        node_dir="${RUN_ROOT}/node-${SLURM_PROCID}"
        mkdir -p "${node_dir}"
        cd "${node_dir}"
        export OMP_NUM_THREADS=8
        exec "${SLURM_SUBMIT_DIR}/collision" \
            --seconds "${SEARCH_SECONDS}" --start "${NONCE_START}" "${PREFIX}"
    '

cat "${RUN_ROOT}"/node-*.out

echo "=== 彙整 ==="
best_bits=0; best_file=""
shopt -s nullglob
for f in "${RUN_ROOT}"/node-*/solution_*.csv; do
    bits="${f##*solution_}"; bits="${bits%.csv}"
    if (( bits > best_bits )); then best_bits="${bits}"; best_file="${f}"; fi
done
[[ -n "${best_file}" ]] || { echo "沒有產生任何 solution CSV" >&2; exit 1; }

cp "${best_file}" "${SLURM_SUBMIT_DIR}/solution_${best_bits}.csv"
echo "best_bits=${best_bits}"
python3 "${SLURM_SUBMIT_DIR}/verify_collision.py" \
    --file "${SLURM_SUBMIT_DIR}/solution_${best_bits}.csv"
