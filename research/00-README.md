# SHA-256 部分碰撞搜尋 — HiPAC 競賽研究筆記

> 針對 `mystery/q1_new_collision`（隱藏題 1）的方法研究。**研究方法，不做實作。**
> 目標機器：2 nodes × (2× Xeon Platinum 8480+ / 112C, 8× NVIDIA H200 SXM 141GB, 30TB NVMe RAID0, 8×400G IB ConnectX-7, Ubuntu 22.04.5) = **16× H200 / 224 核**。

---

## TL;DR

1. **題目本質**：`hash(nonce)=SHA-256(prefix‖nonce)`，找兩個不同 nonce 使雜湊**開頭相同 bit 最多**，10 分鐘限時。**分數 ≈ 2·log₂(N)**，N = 時限內能生成並比對的雜湊數。

2. **這不是密碼學題，是 HPC 吞吐題。** `sha256_block` 被凍結（不能優化 SHA 本身）；主辦方明說「比的是資料處理與平行化效率」。演算法人人都是生日攻擊，**決勝在 pipeline 端到端吞吐**。

3. **化簡成一句話**：*「10 分鐘內對盡量多的 SHA-256 輸出做分散式最近對 / 排序」*。
   - 記憶體不是根本上限——**prefix-partitioning** 讓峰值記憶體與 N 脫鉤，時間才是天花板。
   - Distinguished-Points 在此**上限 ~64 bit**（state 只有 64-bit nonce），不划算。
   - 沒有比 2·log₂(N) 更好的演算法（SHA 輸出均勻 + 函數凍結）→ **純吞吐競賽**。

4. **五個決定性選擇（依重要性，詳見 [03](03-winning-strategy.md)）**：
   1. **放大 N 用「重算 (regeneration)」不用「儲存」** —— 重算比 NVMe I/O 快 ~25×；`N=√(生成預算×H)`。**這是通往 80 的正解，不是 NVMe。**
   2. **每桶記憶體 H 拉大** —— 8-GPU 用 NVLink 協同排序把 H 做到 ~2^36（H×4→+1 bit，破 80 的關鍵）。
   3. **跨節點 SPLIT（各守一半前綴、零通訊）**；節點內用 NVLink 協同。
   4. **融合「算+過濾」進單一 kernel + 桶間重疊** —— GPU 全程不閒置（範本最大錯）。
   5. **最小化 bytes/key**（8~12B，桶前綴隱含不存）。

5. **目標分數（校正後）**：安全地板 **~74**（HBM-only），主線 **~79~81**（regen+協同+SPLIT，衝 80），追極限 **~82~83**。範本是 51 bit@20M。

6. **第一件該做的事**：microbenchmark 量 ① 固定 kernel 實際 GH/s（H200 INT32 減半，可能慢），② 8-GPU NVLink 協同排序（決定 H 能否到 2^36 → 能否破 80）。見 [10](10-open-questions-and-benchmarks.md)。

> 📝 數字更正：RTX 4090 SHA-256 是 **22 GH/s**（不是 50.9，那是 SHA-1）；通往 80 的路是**重算**不是 NVMe。早期草稿的這兩點已在 [02](02-math-birthday-and-budget.md)/[11](11-pipeline-design-80bit.md) 更正。

---

## 文件索引

| 檔案 | 內容 |
|---|---|
| [01-problem-analysis.md](01-problem-analysis.md) | 題目精確重述、可動/不可動邊界、化簡成分散式排序題、為何 DP/密碼分析不適用 |
| [02-math-birthday-and-budget.md](02-math-birthday-and-budget.md) | 最長共同前綴統計（2log₂N、variance）、N↔分數表、生成預算、**regeneration 的 N=√(預算×H)** |
| [03-winning-strategy.md](03-winning-strategy.md) | **策略總綱**：五個決定性選擇、分數地圖、不要做的事、執行順序 |
| [04-gpu-sort-and-partitioning.md](04-gpu-sort-and-partitioning.md) | CUB/thrust radix sort 吞吐、radix partitioning、掃描最近對、key 大小取捨 |
| [05-gpu-hashtable-closest-pair.md](05-gpu-hashtable-closest-pair.md) | warpcore/cuCollections/BGHT，融合 hash→插表→最近對，saturated 表的 log(M)+log(N) 陷阱 |
| [06-multigpu-intranode.md](06-multigpu-intranode.md) | 8×H200：OpenMP/NVLink/NCCL all-to-all、discard-vs-route、重疊 |
| [07-multinode-and-io.md](07-multinode-and-io.md) | 2 節點：CUDA-aware MPI + 400G IB、**SPLIT 零通訊**、**regeneration≫NVMe** 定量、容錯 |
| [08-fixed-kernel-throughput.md](08-fixed-kernel-throughput.md) | 在**不改 sha256_block** 下榨吞吐：launch config、stream、CUDA Graph、融合分桶、SASS 檢查 |
| [09-competition-intel.md](09-competition-intel.md) | 同類競賽題型/計分/驗證、基本分+排名分的策略含義、best-of-k、variance |
| [10-open-questions-and-benchmarks.md](10-open-questions-and-benchmarks.md) | **必做 microbenchmark 清單 + 決策樹 + 風險 + 上機 playbook** |
| [11-pipeline-design-80bit.md](11-pipeline-design-80bit.md) | ★ **衝 80 bit 的完整 pipeline**（regeneration 主線、H 階梯、時間預算） |

---

## 研究方法論備註

- 本輪**刻意不寫實作程式**，只研究方法、量化上限、備妥決策樹——因為在還沒實測 kernel/sort 吞吐前，任何 pipeline 參數都是猜的。
- 每個數字都標註「已查證 / 估計」，關鍵數字經**對抗式 fact-check** 後才落地。
- 正當性：本研究針對公開學術 HPC 競賽的雜湊搜尋**效能工程**，內容為公開密碼學文獻與 HPC 最佳化技術。
