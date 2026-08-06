#!/bin/bash
#SBATCH --job-name=hipac-hybrid-smoke
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=112
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
export PATH=/usr/local/cuda/bin:/usr/mpi/gcc/openmpi-4.1.9a1/bin:${PATH}
export LD_LIBRARY_PATH=/usr/mpi/gcc/openmpi-4.1.9a1/lib:${LD_LIBRARY_PATH:-}
export PMIX_MCA_gds=hash

TOTAL="${TOTAL:-2000000000}"
PREFIX="${PREFIX:-HiPAC2026crypto}"
cd "${SLURM_SUBMIT_DIR}"

make -f Makefile.hybrid -B
rm -f solution_*.csv
echo "=== hybrid smoke: total=${TOTAL}, 2 nodes x 8 GPUs ==="
timeout 580 srun --mpi=pmix -n 2 --ntasks-per-node=1 \
    ./collision_hybrid "${PREFIX}" "${TOTAL}"

best="$(ls -1 solution_*.csv | sort -t_ -k2 -n | tail -1)"
python3 verify_collision.py --file "${best}"
