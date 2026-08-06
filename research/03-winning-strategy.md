# 03 — 完整打法（策略總覽）

> 這是「先讀這頁就知道整盤怎麼下」的總綱。深入細節見各 sibling 檔；衝 80 的完整 pipeline 見 [11](11-pipeline-design-80bit.md)。

---

## 1. 一句話策略

> **把它當「10 分鐘內對盡量多的 SHA-256 輸出找最近對」的吞吐題。分數 ≈ 2·log₂(N)；放大 N 的正解是「重算 (regenerate)、不儲存」，配 prefix-partitioning 讓記憶體不設限。**

演算法人人都是生日攻擊（沒有更好的，見 [01 §3.4](01-problem-analysis.md)）→ **決勝全在工程吞吐**。

---

## 2. 五個決定性選擇（照重要性）

| # | 選擇 | 正解 | 為什麼 | 出處 |
|---|---|---|---|---|
| 1 | 放大 N 的方式 | **重算 (partition-by-regeneration)** | 重算比 NVMe I/O 快 ~25×；`N=√(預算×H)` | [07](07-multinode-and-io.md), [11](11-pipeline-design-80bit.md) |
| 2 | 每桶記憶體 H | **8-GPU NVLink 協同排序把 H 拉到 ~2^36** | H×4 → +1 bit；這是能否破 80 的關鍵 | [11 §3](11-pipeline-design-80bit.md) |
| 3 | 跨節點 | **SPLIT（node A=top-bit0 / B=1，零通訊）** | 最佳對必同 node；省 IB，好除錯 | [07 Part 2](07-multinode-and-io.md) |
| 4 | GPU 不閒置 | **融合「算+過濾」進單一 kernel + 桶間重疊** | 範本最大錯是排序時 GPU 全睡 | [08](08-fixed-kernel-throughput.md), [05](05-gpu-hashtable-closest-pair.md) |
| 5 | bytes/key | **8~12 B（桶前綴隱含不存）** | 每減半 +0.5 bit | [04](04-gpu-sort-and-partitioning.md) |

---

## 3. 不要做的事（省時間）

- ❌ **NVMe out-of-core**：external sort I/O-bound，頂多 ~76 bit，2^42 key=70TB 存不下。只拿來 checkpoint。
- ❌ **Distinguished Points / rho**：state=64-bit nonce → 上限 ~64 bit。
- ❌ **改 SHA / 密碼分析 / reduced-round**：`sha256_block` 凍結、全 64 輪、輸出均勻 → 無用武之地。
- ❌ **像範本那樣把全部 hash 複製回 host 用單執行緒 qsort**。
- ❌ **啟動 opensmd / 重裝 driver**（fabric 已配置）。

---

## 4. 分數地圖（校正後估計）

| 佈局 | 分數 | 定位 |
|---|---:|---|
| 範本（20M, 單卡, host qsort） | 51 | 起點 |
| HBM-only 單趟 + SPLIT | ~73–74 | **安全地板，先拿** |
| Regen + 每卡各自分桶 | ~77 | 過渡 |
| **Regen + 8-GPU 協同 + SPLIT** | **~79–81** | **主線，衝 80** |
| + 跨節點協同 / best-of-k | ~82–83 | 追極限 |

（生成率採校正值：H200 固定 kernel ~6~12 GH/s；因 `N∝√預算`，這些分數對生成率誤差很穩健。詳見 [02 §2](02-math-birthday-and-budget.md)。）

---

## 5. 執行順序

1. 上機先跑 [10](10-open-questions-and-benchmarks.md) 的 microbenchmark #1（kernel GH/s）、#2（NVLink 協同排序）。
2. 先交 **HBM-only 單趟 + SPLIT** 版 → 確保有分（~74）。
3. 升級 **regen + 8-GPU 協同排序**（[11](11-pipeline-design-80bit.md) 的完整 pipeline）→ 衝 80。
4. 行有餘力：跨節點協同 +1.5 bit，或 best-of-k 靠 ±1.9 bit variance 多撈 2~3 bit。
5. 全程留 30s 收尾邊際 + best-so-far checkpoint，避免踩 10 分鐘線。

---

## 6. 導覽

- 題目化簡與邊界 → [01](01-problem-analysis.md)
- 計分數學與算力預算 → [02](02-math-birthday-and-budget.md)
- **衝 80 的完整 pipeline** → [11](11-pipeline-design-80bit.md) ★
- 排序/分桶技術 → [04](04-gpu-sort-and-partitioning.md)｜GPU hash table → [05](05-gpu-hashtable-closest-pair.md)
- 多 GPU → [06](06-multigpu-intranode.md)｜多節點+I/O → [07](07-multinode-and-io.md)｜固定 kernel 榨吞吐 → [08](08-fixed-kernel-throughput.md)
- 競賽情報/計分 → [09](09-competition-intel.md)｜benchmark 清單+決策樹 → [10](10-open-questions-and-benchmarks.md)
