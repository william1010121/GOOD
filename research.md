# Smoke Test Research Log

## Commit 對照

| Commit | 做了什麼 | Smoke test 結果 | 目前最好 qbit |
|---|---|---|---|
| `1c89360` | 建立初始 CUDA SHA-256 collision search、Makefile、README 與驗證工具。 | 沒有獨立 smoke 數據。 | 未記錄 |
| `6475841` | 新增單 GPU Slurm smoke、GPU timing 與吞吐量回報。 | Job `480`：單張 H200，220M nonce / 62 秒，整體 **3.55 MH/s**；GPU hash **10.32–10.83 GH/s**。 | **51 qbit** |
| `1b0b1d8` | 加入 continuous 模式、`--smoke`、`--start`、2 nodes × 8 GPUs 與 qbit 進度回報。 | Job `485`：16 GPU，2.24B nonce / 66 秒，aggregate **33.94 MH/s**，平均每 GPU **2.12 MH/s**。 | **55 qbit** |
| `2cd1427` | 新增 `research.md`，記錄前一輪 smoke 結果。 | 沿用 Job `480`：**3.55 MH/s**，220M nonce / 62 秒。沒有新增測試。 | **51 qbit** |
| `9750d50` | 將 CPU `qsort` 改成 CUB GPU stable radix sort，排序後只拷回一次結果。 | Job `491`：16 GPU，4.48B nonce / 63 秒，aggregate **71.11 MH/s**，平均每 GPU **4.44 MH/s**。 | **55 qbit** |
| `701169f` | 將最佳 pair reduction 放到 GPU，並修正 smoke run 的 nonce range offset。 | Job `501` 在取得資源前被取消，沒有有效 smoke 結果。 | 未記錄 |
| `a7403d7` | 更新 100M batch 與 GPU reduction 的使用說明。 | 沒有新增 smoke。 | 沿用前一版本 |

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

- Nodes: `team2server[1-2]`
- GPU tasks: 16（2 nodes × 8 GPUs）
- Prefix: `HiPAC2026crypto`
- Workload: 14 runs × 16 ranks × 20,000,000 nonce = 4,480,000,000 nonce
- Wall time: 63 seconds
- Aggregate end-to-end throughput: **71.11 MH/s**
- Average per-GPU throughput: **4.44 MH/s**
- Per-rank completed-batch throughput: average **61.94 MH/s**, range **53.84–64.75 MH/s**
- Best result: **55-bit common prefix**
- Verification: passed with `verify_collision.py`

Compared with Job `485` on the previous CPU `qsort` version, aggregate throughput improved from **33.94 MH/s** to **71.11 MH/s** (**2.10×**, approximately **109.5% faster**).

### Job 496 — formal 58-bit goal

- Nodes: `team2server[1-2]`
- GPU tasks: 16（2 nodes × 8 GPUs）
- Prefix: `HiPAC2026crypto`
- Search time: 285 seconds
- Evaluated nonce: **965,580,000,000**
- Aggregate evaluated throughput: **3,388.00 MH/s**
- Best local batch result: **64 qbit**
- Goal 58 qbit: **reached**
- Note: 64 qbit 是 local batch/rank best，沒有做跨 rank hash 的 exact merge。

### Job 501 — canceled before start

Job `501` 在取得兩個 node 的 GPU 資源前被取消，沒有產生 Slurm log、rank output 或可用 smoke 數據，不能列入效能比較。

## Exact cross-rank merge 說明

目前每個 rank 只比較自己排序後的 hash：

```text
rank A: A1 ↔ A2 ↔ A3
rank B: B1 ↔ B2 ↔ B3
```

程式會找出 A 內與 B 內的最佳 pair，但沒有比較 `A1 ↔ B1`。因此目前的 qbit 是「所有 rank local best 中的最大值」，不是所有 nonce 合併後的 exact global best。

要做到 exact merge，必須把各 rank 的排序 hash/candidate 依 prefix 分區後交換，再在共同排序結果上比較；只收集每個 rank 的一組 `solution_*.csv` 不足以保證找出跨 rank 的最佳 pair。
