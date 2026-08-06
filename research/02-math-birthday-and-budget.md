# 02 — 計分數學 + 10 分鐘算力預算

> 本文把「分數能到幾 bit」釘死。所有工程決策都回到這張表。
> （本檔已隨題目落地重寫；先讀 [01-problem-analysis.md](01-problem-analysis.md) 的化簡。）

---

## Part 1 — 「最長共同前綴」的統計

### 1.1 期望值
N 個均勻隨機的 256-bit 雜湊，兩兩比對取最長共同前綴 `B*`：

- 任一對開頭相同 `b` bits 的機率 = `2^-b`
- 配對總數 ≈ `N²/2`
- 期望有 ≥ b bit 前綴的對數 `E_b = (N²/2)·2^-b`
- 令 `E_b ≈ 1` → **`B* ≈ 2·log₂(N) − 1`**

分佈近似 Gumbel（極值分佈）：
- **平均 ≈ 2·log₂(N) − 1**（更精確：mode 在 `2log₂N−1`，mean 再高約 0.83 bit）
- **標準差 ≈ 1.9 bits**

### 1.2 兩個直覺
1. **N 翻 4 倍 → 分數 +2 bit。** 多拿 1 分要 `N × √2 ≈ 1.41×`。指數尺度，越後面越貴。
2. **variance ≈ 1.9 bit 很可觀。** 「多跑幾次取最好」能白撈 2~3 bit（見 [09](09-competition-intel.md) 的 best-of-k 分析）。範本 20M 拿 51 bit（期望 47.5）就是 +1.9σ 的幸運例。

### 1.3 N ↔ 分數 對照表

| N | log₂(N) | 期望分數 `2log₂N−1` | 每筆 16B 的裸儲存量 |
|---:|---:|---:|---:|
| 2^24 (1.7×10⁷) | 24 | **47** | 256 MB |
| 2^28 (2.7×10⁸) | 28 | **55** | 4 GB |
| 2^30 (1.1×10⁹) | 30 | **59** | 16 GB |
| 2^32 (4.3×10⁹) | 32 | **63** | 64 GB |
| 2^34 (1.7×10¹⁰) | 34 | **67** | 256 GB |
| 2^36 (6.9×10¹⁰) | 36 | **71** | 1 TB |
| 2^38 (2.7×10¹¹) | 38 | **75** | 4 TB |
| 2^40 (1.1×10¹²) | 40 | **79** | 16 TB |
| 2^42 (4.4×10¹²) | 42 | **83** | 64 TB |
| 2^44 (1.8×10¹³) | 44 | **87** | 256 TB |

> 「裸儲存量」是**若把全部存下來**的量；用 prefix-partitioning 後峰值記憶體只需其 `1/2^p`（見 [01 §3.2](01-problem-analysis.md) 與 [04](04-gpu-sort-and-partitioning.md)）。所以這欄不是上限，只是提醒資料規模。

---

## Part 2 — 兩個吞吐上限：生成 vs 處理

分數由 `N = min(可生成量, 可處理量) × 是否重疊` 決定。10 分鐘 = 600 秒。

### 2.1 生成上限（SHA-256 hash rate，固定 kernel）

**硬體事實（已查證）：**
- H200 = GH100 die，132 SM，**INT32 只有 64 ALU/SM**（compute cap 9.0，FP32:INT32 = 2:1）[3]。SHA-256 是純 32-bit 整數/邏輯運算，瓶頸就是 INT32+邏輯吞吐（tensor core 完全無用）。
- 參照（**已更正**）：RTX 4090（Ada，128 INT32/SM）hashcat **SHA2-256 (mode 1400) = 22.0 GH/s** [4]。
  ⚠️ 常被引用的「50.9 GH/s」其實是 **SHA-1 (mode 100)**，不是 SHA-256——本題第一版筆記誤植，已改。（同卡 MD5=164、SHA1=50.6、SHA256=22.0 GH/s [4]）

⚠️ **重大不確定性**：H200 的 INT32/SM 只有 4090 的一半、時脈也較低（~1.98 vs ~2.5GHz），SM 數 132 vs 128 略多。純 INT32 吞吐比 ≈ (64×132×1.98)/(128×128×2.5) ≈ **0.41**。且**本題 kernel 凍結、不能用 hashcat 的手工最佳化**。所以 **單張 H200 的固定 kernel 單 SHA-256 估 ~6~12 GH/s，可能輸給一張 4090。** 這正是挖礦社群對 H100/H200「不划算」的共識來源。**必須用 microbenchmark 實測**（見 [10](10-open-questions-and-benchmarks.md)）。詳細推導見 [08](08-fixed-kernel-throughput.md)。

**粗估區間（待實測校正，誤差可達 2~3×）：**

| 情境 | 單 GPU | 16 GPU 叢集 | 600s 可生成量 |
|---|---:|---:|---:|
| 保守 | 4 GH/s | 64 GH/s | 3.8×10¹³ ≈ **2^45.1** |
| 中性 | 8 GH/s | 128 GH/s | 7.7×10¹³ ≈ **2^46.1** |
| 樂觀 | 12 GH/s | 192 GH/s | 1.2×10¹⁴ ≈ **2^46.7** |

→ **生成預算約 2^45~2^47。生成本身不是瓶頸——但它是 partition-by-regeneration 的主要成本（見下）。**

### 2.2 真正的天花板：**partition-by-regeneration**（重算，不儲存）

關鍵發現（經查證，詳見 [07](07-multinode-and-io.md)/[11](11-pipeline-design-80bit.md)）：**要放大 N，最佳做法不是把 hash 存起來排序，而是「重算」。** 因為在本機上，重算一個 hash 比「寫出去再讀回來」的 NVMe I/O **快 ~20~30×**。

機制：把雜湊空間依前綴切成 P 個 partition；每個 partition 重掃 nonce 範圍、只留落在此桶的（塞得進 HBM）、在 HBM 內 sort 找最佳、記錄、換下一桶。設每桶 resident 上限 = H：

```
達到 N 個 distinct hash 需 P = N/H 個 partition，總重算量 = P × N = N² / H
令 N²/H ≤ 生成預算  →  N = sqrt(生成預算 × H)
```

- H = 全叢集 HBM ≈ 2^37（2.2TB/16B，需節點內多 GPU 協同排序）
- 生成預算（校正後）≈ 2^45~2^47

| 生成預算 | H | N = √(預算×H) | 分數 ≈ 2·log₂N |
|---|---:|---:|---:|
| 2^45.1（保守） | 2^37 | ~2^41.0 | **~81** |
| 2^46.1（中性） | 2^37 | ~2^41.5 | **~82** |
| 2^46.7（樂觀） | 2^37 | ~2^41.9 | **~83** |

> 因 **N ∝ √預算**，生成率就算差 2× 也只搬動 N 約 0.5 bit（分數 ~1）→ 這條路線**非常穩健**。
> ⚠️ H 的取法影響大：若各 GPU 獨立分桶（H=單卡 2^33，不做多 GPU 協同排序），N 降到 ~2^39.5（分數 ~78）。要衝 80+ 需**節點內 8 GPU 協同排序**把 H 拉到 ~2^36.5。細節見 [11](11-pipeline-design-80bit.md)。

**其他佈局對照**（詳見 [07](07-multinode-and-io.md) Part 3）：
- HBM-only 單趟（最簡單、最穩）：N ≈ 2^37，分數 **~74**
- NVMe out-of-core（**不建議**，I/O bound）：N ≈ 2^38，分數 **~76**，且 2^42 需 70TB > 30TB 存不下

### 2.3 結論：目標分數區間

> **合理目標 80~83 bit（靠 partition-by-regeneration），保守地板 ~74~76（HBM-only），幸運 + best-of-k 再 +2~3 bit。**
> 決勝點不在演算法（人人都是生日攻擊），而在**「重算 vs 儲存」選對（選重算）、每桶 H 拉多大（多 GPU 協同排序）、bytes/key 壓多小、GPU 多不閒置、雙節點零通訊分割**。

---

## Part 3 — 記憶體階層（可用來裝多大的 partition）

| 層級 | 容量 | 頻寬（約） | 用途 |
|---|---:|---:|---|
| 單 GPU HBM3e | 141 GB | 4.8 TB/s | 排序/hash table 的工作區（單 partition） |
| 16×GPU HBM 合計 | ~2.2 TB | — | 全叢集常駐上限 |
| Host RAM /node | (待查，通常 1~2 TB) | ~300+ GB/s | overflow、DP 表、協調 |
| NVMe RAID0 /node | 30 TB | 讀寫數十 GB/s | out-of-core（10 分鐘內多半不划算，見 [07](07-multinode-and-io.md)） |

- 用 prefix-partitioning，把 partition 大小設成能塞單卡 HBM（例如 2^31~2^33 筆），即可處理遠大於 HBM 的 N。
- 每筆只存**前 ~96 bit 雜湊 + 8B nonce**（甚至只存 nonce 事後重算），把 bytes/key 壓到最小 = 直接換更多 bit。

---

## Part 4 — CPU 要不要參戰？

- Xeon 8480+ = Sapphire Rapids，**有 SHA-NI**[5]。單核 ≈ 25~30 MH/s，224 核 ≈ **~6 GH/s**，約為 GPU 叢集的 **<3%**。
- **結論**：CPU 生成貢獻可忽略。CPU 的真正價值 = **協調 shuffle、host 端 merge、I/O、跨節點通訊、維護 running best**。詳見 [06](06-multigpu-intranode.md)/[07](07-multinode-and-io.md)。

---

## Part 5 — 尺度感（sanity check）
- 比特幣全網 ≈ 800 EH/s ≈ 2^69.4/s；我們叢集 ≈ 2^38/s → 差 **~2^31（20 億倍）**。這說明為何談的是「~80 bit 部分碰撞」而非全 256。
- 全 256-bit 真碰撞需 2^128，**物理不可能** —— 本題只問「開頭最多幾 bit」，是完全可解的暴力搜尋。

---

## 參考來源
- [3] [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)（compute cap 9.0：FP32:INT32 = 2:1）
- [4] [Hashcat RTX 4090 benchmark](https://gist.github.com/Chick3nman/32e662a5bb63bc4f51b847bb422222fd)（**已核對**：MD5=164、SHA1(mode100)=50.6、**SHA2-256(mode1400)=22.0 GH/s**）
- [5] [Intel Xeon Platinum 8480+ 規格](https://www.intel.com/content/www/us/en/products/sku/231746/intel-xeon-platinum-8480-processor-105m-cache-2-00-ghz/specifications.html)
- H200 規格：[Spheron](https://www.spheron.network/blog/nvidia-h200-specs/) / [Lenovo Press](https://lenovopress.lenovo.com/lp1944-nvidia-h200-141gb-gpu)

> 數字警語：GPU 速率與排序吞吐皆為**上界推導 + 社群類比**，未經本機實測。所有時程都應用 [10](10-open-questions-and-benchmarks.md) 的 microbenchmark 校正（誤差可能 2~3×）。
