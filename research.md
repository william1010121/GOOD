# Smoke Test Research Log

## Commit 對照

| Commit | 做了什麼 | Smoke test 結果 | 目前最好 qbit |
|---|---|---|---|
| `1c89360` | 建立初始 CUDA SHA-256 collision search、Makefile、README 與驗證工具。 | 沒有獨立 smoke 數據。 | 未記錄 |
| `6475841` | 新增單 GPU Slurm smoke、GPU timing 與吞吐量回報。 | Job `480`：單張 H200，220M nonce / 62 秒，整體 **3.55 MH/s**；GPU hash **10.32–10.83 GH/s**。 | **51 qbit** |
| `1b0b1d8` | 加入 continuous 模式、`--smoke`、`--start`、2 nodes × 8 GPUs 與 qbit 進度回報。 | Job `485`：16 GPU，2.24B nonce / 66 秒，aggregate **33.94 MH/s**，平均每 GPU **2.12 MH/s**。 | **55 qbit** |
| `2cd1427` | 新增 `research.md`，記錄前一輪 smoke 結果。 | 沿用 Job `480`：**3.55 MH/s**，220M nonce / 62 秒。沒有新增測試。 | **51 qbit** |
| `9750d50` | 將 CPU `qsort` 改成 CUB GPU stable radix sort，排序後只拷回一次結果。 | Job `491` 已提交，但目前仍在 Slurm 排隊，尚無有效優化後吞吐量。 | 尚未記錄 |

## 已完成 smoke 詳情

### Job 480 — baseline

- Host: `team2server1`
- GPU: 1 × NVIDIA H200
- Prefix: `hipac_demo`
- Workload: 11 runs × 20,000,000 nonce = 220,000,000 nonce
- Wall time: 62 seconds
- Aggregate end-to-end throughput: **3.55 MH/s**
- Best result: **51-bit common prefix**
- Verification: passed with `verify_collision.py`

### Job 485 — 16-GPU multi-node baseline

- Nodes: `team2server[1-2]`
- GPU tasks: 16（2 nodes × 8 GPUs）
- Prefix: `HiPAC2026crypto`
- Workload: 7 runs × 16 ranks × 20,000,000 nonce = 2,240,000,000 nonce
- Wall time: 66 seconds
- Aggregate end-to-end throughput: **33.94 MH/s**
- Average per-GPU throughput: **2.12 MH/s**
- Best result: **55-bit common prefix**
- Verification: passed with `verify_collision.py`

### Job 491 — GPU radix sort

Job `491` 使用 `9750d50` 的 CUB GPU radix sort 版本，prefix 為 `HiPAC2026crypto`，目前仍在等待 Slurm 資源，因此結果欄位暫不填入推測值。
