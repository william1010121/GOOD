# Smoke Test Research Log

## Latest completed result

- Host: `team2server1`
- GPU: NVIDIA H200, single GPU
- Slurm job: `480`
- Prefix: `hipac_demo`
- Workload: 11 runs × 20,000,000 nonce = 220,000,000 nonce
- Wall time: 62 seconds
- Aggregate end-to-end throughput: **3.55 MH/s**
- Per-run GPU hash throughput: approximately **10.32–10.83 GH/s**
- Per-run end-to-end throughput: approximately **3.53–4.09 MH/s**
- Best result: **51-bit common prefix**
- Verification: passed with `verify_collision.py`; `solution_51.csv` matched the claimed 51 bits.

The gap between GPU hash throughput and end-to-end throughput is mainly caused by host-side copy, `qsort`, and adjacent-pair scanning.

## Current 16-GPU test

The updated Slurm configuration uses 2 nodes × 8 GPUs, launches 16 one-GPU tasks, and assigns each task a distinct nonce start offset through `--start`. The prefix is now `HiPAC2026crypto`.

Job `485` was submitted from `/shared/guosw-shatest`, but was still pending on Slurm due to resource priority when this log was written. Therefore, no valid 16-GPU throughput number is recorded yet.
