# 04 — GPU 大規模排序 + radix prefix-partitioning（closest-pair by hash-prefix）

> 本題化簡後是一個 **distributed GPU sort / closest-pair** 問題（見 [01](01-problem-analysis.md) §3、[02](02-math-birthday-and-budget.md)）。
> SHA 生成便宜、瓶頸在「把海量 (hash, nonce) 依 hash 前綴排好並找最近對」。本文專攻這一段：**用什麼 library、多快、記憶體怎麼算、full sort vs radix partitioning 怎麼選、跨 8 卡怎麼放**。
> 多 GPU 的「管線 / NVLink / route-vs-discard」在 [06](06-multigpu-intranode.md)，本文只給演算法與吞吐；兩者交叉引用。

標記慣例：**SOURCED** = 有 URL 佐證的事實；**⚠️ EST** = 本人估計 / 外推（未經本機實測）。

---

## TL;DR

1. **底層引擎已經是 solved problem**：CUB `DeviceRadixSort`（現行預設演算法為 **Onesweep**）就是目前最快的通用 GPU LSD radix sort，A100 上 32-bit keys-only 達 **29.4 GKey/s**（SOURCED）。`thrust::sort` 對整數鍵直接 dispatch 到同一個 CUB kernel，所以「thrust 還是 cub」在效能上等價，選 CUB 只為拿到 `begin_bit/end_bit` 與 temp-storage 控制權。
2. **radix sort 是 memory-bandwidth-bound**，時間 ≈ `passes × 2 × bytes_per_key × N / BW_eff`。**passes = ⌈排序位元數 / 8⌉**（Onesweep 每趟吃 8-bit digit）。→ **少存幾個位元 = 少幾趟 = 直接變快**。這是本題最重要的旋鈕。
3. **不要做 full sort**。closest-pair-by-prefix 只需要 MSD-first 把桶切到「桶內 ≤ 少數幾個元素」，期望 **passes ≈ ⌈log₂N / 8⌉ ≈ 5**（N=2^40），比 full 96-bit sort 的 12 趟快 ~2.4×。這就是 **radix partitioning** 打敗 full sort 的地方。
4. **partitioning 把記憶體從 N 解耦**：top-p bits 分 2^p 桶，每桶 `N/2^p` 個 key，把 p 開大到單桶塞得進 HBM（或塞得進一張卡的份額），全域最佳對保證落在同一桶（共享 top bits）→ 見 [01](01-problem-analysis.md) §3、[06](06-multigpu-intranode.md) §5.1。
5. **8× H200 單節點在 10 分鐘內的可排序 N**：in-HBM 上限 ~2^35（穩、分數 ~70 bit）；靠 NVMe out-of-core 再往上，**站得住的上限約 2^38（分數 ~76 bit）**；**2^40（分數 ~80 bit）是「賭 exotic ~100+ GB/s NVMe 陣列」的樂觀端，時間吃緊、多半超時**（out-of-core 變 I/O-bound，模型見 §6.2 修正版與 [06](06-multigpu-intranode.md)）。⚠️ EST，須實測校正。

---

## 1. CUB `DeviceRadixSort` / Onesweep：吞吐與 pass 數

### 1.1 演算法：Onesweep（現行 CUB 預設）

- **LSD radix sort**，每趟處理一個 **8-bit digit**（radix = 256）。**passes = ⌈key_bits / 8⌉**。 [SOURCED: [gpuopen Onesweep 解析](https://gpuopen.com/learn/boosting_gpu_radix_sort/)]

  | key 寬度 | passes（8-bit digit） |
  |---:|---:|
  | 8-bit | 1 |
  | 32-bit | 4 |
  | 64-bit | 8 |
  | 96-bit | 12 |
  | 128-bit | 16 |

- Onesweep 的關鍵改良：用 **single-pass prefix-sum + decoupled look-back**，把每個 digit-binning iteration 的 global 記憶體流量從古典的 `3n`（2 讀 1 寫）壓到 **`2n`（1 讀 1 寫）**。 [SOURCED: [gpuopen](https://gpuopen.com/learn/boosting_gpu_radix_sort/)、[Onesweep 論文 arXiv:2206.01784](https://arxiv.org/abs/2206.01784)]
- 已在 **CUDA 12 / CCCL 起成為 CUB `DeviceRadixSort` 的預設 backend**。

### 1.2 實測吞吐（keys-only, 32-bit, 均勻隨機）

| GPU | 演算法 | keys | 吞吐 | 記憶體頻寬 | 來源 |
|---|---|---:|---:|---:|---|
| V100 (0.9 TB/s) | Onesweep | — | **~16 GKey/s**（≈2× 當時 CUB） | HBM2 | SOURCED [GTC 2020 s21572] |
| A100-80GB (2.04 TB/s) | CUB (pre-Onesweep) | 256M | **19.6 GKey/s** | HBM2e | SOURCED [Onesweep 論文] |
| A100-80GB (2.04 TB/s) | **Onesweep** | 256M | **29.4 GKey/s**（1.5× CUB） | HBM2e | SOURCED [Onesweep 論文] |
| RTX 5090 (~1.79 TB/s) | libcusort(Onesweep) | 67M | **~43.6 GKey/s**（1.537 ms） | GDDR7 | SOURCED [libcusort] |
| **H200 (4.8 TB/s)** | Onesweep | — | **~60–75 GKey/s** | HBM3e | **⚠️ EST（見 §1.3）** |

- [SOURCED: A100 兩個數字、1.5× 加速 — [Onesweep 論文 arXiv:2206.01784](https://arxiv.org/abs/2206.01784) / [ResearchGate 361135714](https://www.researchgate.net/publication/361135714)]
- [SOURCED: RTX 5090 67M 元素 32-bit keys-only 1.537 ms（libcusort）vs 1.609 ms（CUB）、64-bit 6.103 ms vs 6.171 ms — [github.com/IlyaGrebnov/libcusort](https://github.com/IlyaGrebnov/libcusort)]。換算：67e6 / 1.537e-3 ≈ **43.6 GKey/s**（32-bit）、67e6 / 6.103e-3 ≈ **11.0 GKey/s**（64-bit）。

### 1.3 pass 數如何隨 key 寬度縮放（本題最重要）

radix sort 是 **bandwidth-bound**，用一個乾淨的模型就能外推所有情況：

```
sort_time  ≈  passes × 2 × B × N / BW_eff
throughput ≈  N / sort_time  =  BW_eff / (2 × B × passes)
```

其中 `B` = bytes/key（key-value 則 `B = key_bytes + value_bytes`）、`passes = ⌈sorted_bits/8⌉`、`2×` 來自 Onesweep 每趟 1 讀 1 寫。

**用 A100 校準 `BW_eff`**：29.4 GKey/s、B=4、passes=4 → `BW_eff = 29.4e9 × 2×4×4 ≈ 941 GB/s`，即 **~46% of A100 峰值 2.04 TB/s**（radix 的 scatter 有隨機性，40–50% 峰值是常態）。 [計算基於上表 SOURCED 數字]

> **⚠️ 用本文自己的 RTX 5090 數字交叉驗證，46% 其實偏保守**：同一個 `2n` 模型下，5090 的 43.6 GKey/s（B=4、passes=4）換算 `BW_eff = 43.6e9×2×4×4 ≈ 1.40 TB/s`，即 **~78% of 5090 峰值 1.79 TB/s**；64-bit 那筆（10.98 GKey/s、B=8、passes=8）換算同樣 ≈ 1.40 TB/s / **~78%**，兩個寬度一致。→ **A100（2022 Onesweep 論文）= 46%，但較新的 CUB / CUDA 13 在 Blackwell 上 = ~78%**。差異主要來自軟體版本與架構世代，不是 HBM vs GDDR7。H200（Hopper HBM3e、搭現行 CUB）合理效率應落在 **~55–75%**，比 46% 高。**結論：下方 H200 EST 以 46% 為底，更可能被「低估」而非高估；若實測到 ~70%，全表 H200 吞吐要 ×~1.5。** [計算基於上表兩筆 SOURCED 數字]

**外推 H200（⚠️ EST，**刻意採保守下限 ~46% 效率**、HBM3e 4.8 TB/s → `BW_eff ≈ 2.2 TB/s`；若採 ~70% 則 `BW_eff ≈ 3.4 TB/s`、下表全部 ×~1.5）：**

| 場景 | B (bytes) | passes | H200 吞吐（⚠️ EST） |
|---|---:|---:|---:|
| 32-bit keys-only | 4 | 4 | `2.2e12/(2·4·4)` ≈ **69 GKey/s** |
| 64-bit keys-only | 8 | 8 | `2.2e12/(2·8·8)` ≈ **17 GKey/s** |
| 32b key + 32b value | 8 | 4 | ≈ **34 GPair/s** |
| 96b key(12B) + 64b nonce(8B)，**full sort** | 20 | 12 | ≈ **4.6 GPair/s** |
| 同上但 **MSD 只切到 40 bit**（見 §3） | 20 | 5 | ≈ **11 GPair/s** |

> **直接結論**：從 32→64→96 bit，pass 數 4→8→12，吞吐幾乎等比下滑。**每多存 8 bit hash，就多一趟、慢一截。** 這正是 §5「minimize bytes/key」的量化依據。
> ⚠️ EST 警語：46% 效率是從 **A100（2022 論文）** 外推、當作保守下限；本文自己的 RTX 5090 SOURCED 數字換算 ~78%（見上方 blockquote），故誤差方向**偏「低估 H200」**。真實 H200 吞吐可能比上表高 1.5×（若效率 ~70%），也不排除被 latency 拖累而較低。**必須拿到機器後用 `nvbench`/CUB benchmark 實測**（見 [06](06-multigpu-intranode.md) §7）才能定案。

---

## 2. `thrust::sort` vs CUB；keys-only vs key-value；排序 (hashkey, nonce) pair

### 2.1 thrust 其實就是 cub

- `thrust::sort` / `thrust::sort_by_key` 對**算術型別鍵（`is_arithmetic`：整數 / 浮點）且 comparator 為 `thrust::less`/`greater`** 時，在 CUDA backend 直接走 radix path 呼叫 `cub::DeviceRadixSort::SortKeys` / `SortPairs`（判斷式 `can_use_primitive_sort = is_arithmetic<Key> && (CompareOp==less||greater)`；否則 fallback 到 merge sort）。 [SOURCED: [thrust/system/cuda/detail/sort.h](https://github.com/NVIDIA/thrust/blob/main/thrust/system/cuda/detail/sort.h)]
  - ⚠️ 注意：**自訂 POD/struct 或 packed 寬鍵（`uint4`、`__int128`）不是 `is_arithmetic`，thrust 高階 `sort` 會 fallback 到 merge sort，不會自動用 radix** → 本題的 packed key **必須直接呼叫 CUB**（見 §2.3、§3.3），這也是「選 CUB」的另一個硬理由。
- 效能等價。**選 CUB 而非 thrust 的唯二理由**：(1) `begin_bit/end_bit` 精準控制排序位元數（thrust 高階 API 不暴露）；(2) 明確的 `d_temp_storage` 兩段式呼叫，可預先分配、避免比賽時序中途 `cudaMalloc`。
- thrust 的價值在膠水：`transform` / `unique` / `reduce_by_key` 拿來做 dedup、adjacent-diff（§4）很順手。

### 2.2 keys-only vs key-value（本題必然是 key-value）

我們要保留「哪個 nonce」，所以是 **SortPairs（key = 截斷 hash，value = 8-byte nonce）**。代價：每趟多搬 value 的位元組，`B = key_bytes + value_bytes`（見 §1.3 模型）。

- 古典觀察：key-value 因為要同步搬 payload，比 keys-only 慢。文獻中 memory-efficient hybrid radix sort 相對 CUB 的加速在 **64/64 key-value 高達 4×**、32/32 為 2.32×（顯示 CUB 在寬 key-value 有改進空間，但 Onesweep 已補上大部分）。 [SOURCED: [Hybrid Radix Sort arXiv:1611.01137](https://arxiv.org/pdf/1611.01137)]
- **省 value 的技巧**：value 只放 **8-byte nonce**，不要把整顆 hash 當 value 拖著跑。

### 2.3 排序 (hashkey, nonce) 的實務打包

- **鍵取 hash 的高位**（MSB-first 的排序目標）。因為 CUB LSD 是對整個 key 做，若把 key 定成「hash 高 t bits 放進一個 `t`-bit 整數欄位」，排完後**相鄰即前綴最接近**。
- **打包成一個寬整數**：把 `(top-t-hash-bits, nonce)` 塞進 `unsigned __int128` 或 `uint2/uint4`，用 `begin_bit/end_bit` 只排 hash 那段位元。CUB 支援自訂寬度的 key 型別（含 128-bit）。
- **Structure of Arrays**：hash 鍵一個陣列、nonce value 一個陣列，`SortPairs` 一起 permute，cache 行為比 AoS 好。

---

## 3. 替代 full sort：radix PARTITIONING（本題的主力）

### 3.1 為什麼 partition 比 full global sort 便宜

closest-pair-by-prefix **不需要全序**，只需要「共享最長前綴的那一對相鄰」。做法：**MSD-first radix**，一趟切 8 個高位 bit 成 256 桶；對每個桶遞迴，**桶內元素數 ≤ 閾值（例如 ≤ 32）就停**，改在桶內暴力比對。

- 期望停止深度：桶要細到 ~O(1) 個元素，需要 `p ≈ log₂N` 個 bit → **passes ≈ ⌈log₂N / 8⌉**。
- N = 2^40 → **passes ≈ 5**，對比 full 96-bit LSD sort 的 12 趟 → **~2.4× 更快**（見 §1.3 表最後兩列：4.6 → 11 GPair/s）。⚠️ 此 2.4× 是**純 bandwidth 模型的上限**（只數 pass 數）：MSD 遞迴分桶有 histogram atomics、桶尾 load-imbalance、大量小 kernel 的 launch 開銷，實測加速通常 **< 2.4×**；未經本機 benchmark，屬推理值。
- **全域最佳對保證在某個深桶內或其相鄰桶**：兩個 hash 若共享 ≥ p 個高位 bit，MSD 到第 p 位時必落同桶；共享 < p 位的對前綴一定更短、不可能是贏家。→ 只需在「每個葉桶內」+「桶邊界相鄰對」找最近對。 [推理見 [01](01-problem-analysis.md) §3.3、§4]

### 3.2 怎麼定 p（桶大小 vs 記憶體）

設 top-p bits 分 `2^p` 桶，密碼學 hash 高位均勻 → **每桶期望 `N/2^p`，天然等大**（相對誤差 ~`1/√(N/2^p)`，見 [06](06-multigpu-intranode.md) §5.2）。兩種定 p 的準則：

| 目標 | 準則 | N=2^40 例 |
|---|---|---|
| 單桶塞進單卡 HBM | `N/2^p ≤ HBM_keys` | HBM 容 3.5e9(20B×2buf) → `p ≥ 9`（切 512 桶，每桶 2.1e9）|
| 桶內 O(1)、直接暴力 | `N/2^p ≤ ~32` | `p ≈ 35`，但這麼深不必一次做完 |
| 8 卡靜態分派 | `p = 3`（每卡一個高 3-bit 桶 = N/8） | 每卡 1.4e11 → 仍需卡內再 partition，見下 |

**實務兩段式**：先 `p=3` 把工作分到 8 卡（[06](06-multigpu-intranode.md) 的 all-to-all / route），每卡再對自己的 `N/8` 做 `p'` 加深到 in-HBM 塞得下，然後每個 in-HBM 子桶跑一次 `DeviceRadixSort`（只排剩餘位元）+ adjacent-scan（§4）。**partition 就是把 memory 從 N 解耦的機制**：任意大的 N，只要 p 夠大，單桶恆能塞進 HBM。 [見 [01](01-problem-analysis.md) §3.2]

### 3.3 partition 的兩種實作

- **CUB `DeviceRadixSort` + `begin_bit/end_bit`**：只排高 p bits 就等於「以 top-p 為鍵的穩定分桶」。`begin_bit/end_bit` 明確指定要排的位元子區間，**減少 pass 數、直接換取效能**。 [SOURCED: [cub::DeviceRadixSort API](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html)、[NVIDIA Forum begin/end bit](https://forums.developer.nvidia.com/t/setting-begin-end-bit-parameters-in-cub-sortpairs/54908)]
- **手寫 histogram + scatter**（multisplit 風格）：一趟 `atomicAdd` 算 2^p 桶直方圖 → `ExclusiveScan` 得 offset → scatter。對 `p ≤ 8` 一趟搞定，是 out-of-core / 跨卡 shuffle 的自然分桶原語（[06](06-multigpu-intranode.md) §3 的 NCCL all-to-all 就是一次 p=3 的 partition）。 [背景: [GPU Multisplit arXiv:1701.01189](https://arxiv.org/pdf/1701.01189)]
- **多 GPU 版現成品**：**RMG Sort**（見 §6）本質就是「MSB radix partition → 一次 all-to-all P2P swap → 各卡本地 sort」，正是我們要的骨架。

---

## 4. 排序後的 adjacent-scan：算 max common prefix

排好（或 MSD 分桶到葉）之後，**贏家對必為排序序列中的相鄰對**（或桶邊界相鄰對）。on-GPU 一趟掃描即可：

```
// 每個 thread i 處理相鄰對 (i, i+1)，各桶內 + 桶邊界
lcp[i] = clz128( key[i] XOR key[i+1] )   // 共同前綴長度 = XOR 後前導零數
```

- **關鍵原語**：`__clz`（32-bit）/ `__clzll`（64-bit）算前導零；128-bit key 就先比高 64 再比低 64。XOR 後的前導零數 = 兩 hash 的 common-prefix bit 數（也就是分數）。
- **求全域最大**：`cub::DeviceReduce::ArgMax`（或 `thrust::max_element`）一趟拿到最大 `lcp` 及其 index，回推兩個 nonce。 [API: [CUB DeviceReduce](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceReduce.html)]
- **成本可忽略**：一趟 `2n` 讀 + 一趟 reduce，相對 §1.3 的多趟 sort 是零頭。
- **跨桶/跨卡邊界**：相鄰桶的「最後一個 vs 下一桶第一個」也要比（贏家可能跨桶邊界）。跨卡時只需交換每卡邊界的幾個 key，通訊量極小。
- **⚠️ 陷阱**：nonce 必須 **DISTINCT**（題目要求 a≠b）。若截斷 hash 相同但那是同一 nonce（不可能，nonce 唯一）或兩不同 nonce 真的 hash 前綴全同，照算即可；務必在 adjacent-scan 排除 `nonce[i]==nonce[i+1]` 的自比（穩定排序 + dedup 後一般不會發生，但要防）。

---

## 5. 最小化 bytes/key：吞吐 / 記憶體權衡

由 §1.3 模型，`B` 同時決定 **sort 速度**（`∝1/B`）與 **單卡可容 N**（`∝1/B`）。所以 B 是雙重槓桿。

### 5.1 存多少 hash 位元才夠？

分數 ≈ `2·log₂N`（[01](01-problem-analysis.md) §3.1）。N=2^40 → 期望贏家共享 ~**80 bit**。要能「分辨出」這對，**截斷 hash 至少要存 ≥ 分數位元 + 安全裕度**：

| 儲存方案 | key_bytes | +nonce | B | 141GB 可容(×2 ping-pong) | log₂ | 說明 |
|---|---:|---:|---:|---:|---:|---|
| nonce-only（排完重算 hash 驗證） | 0 | 8 | 8 | ~8.8e9 | 2^33.1 | 最省，但 **radix 需要物化 hash 位元當 digit → 不可行於純 nonce**（見下） |
| top-64 hash + nonce | 8 | 8 | 16 | ~4.4e9 | 2^32.0 | 分數上限被 64 bit 卡住 |
| **top-96 hash + nonce** | 12 | 8 | 20 | ~3.5e9 | 2^31.7 | **甜蜜點**：對分數 ~80 有裕度，20B 對齊佳 |
| top-128 hash + nonce | 16 | 8 | 24 | ~2.9e9 | 2^31.5 | 保險，多一點 pass |
| full-256 hash + nonce | 32 | 8 | 40 | ~1.76e9 | 2^30.7 | **浪費**：低 160 bit 永遠用不到，白搬白排 |

（單卡可容 = `141e9 / (2·B)`，radix 需 double-buffer；數字與 [06](06-multigpu-intranode.md) §5.1 一致。）

### 5.2 「nonce-only 重算」到底能不能省？

- **不能直接用在 radix 的 sort key 上**：LSD/MSD radix 每趟要讀 key 的某 8-bit digit，這些 digit 必須已物化在記憶體。若只存 nonce，就得在**每趟**重算 hash 才拿得到 digit → 重算 `passes` 次，SHA 雖便宜但 ×5 趟 + 破壞 memory-coalescing，得不償失。
- **可行的省法**：只在**最後驗證/精修**階段重算。主排序用 `top-t hash + nonce`；找到候選對後重算完整 256-bit hash 確認真實 common-prefix（避免截斷造成的高估）。
- **實務最省且可行 = `top-80~96 bit + 8B nonce`**，即 16–20 B/key。再往下砍 hash 位元會直接砍掉可分辨的分數上限。

### 5.3 對齊與向量化

- 20B 不是 2 的冪，載入不佳；可 pad 到 **24B（`uint2`+`uint4` 或 3×`uint64`）** 或用 32B（`uint4`×2）換對齊。權衡：pad 增 B、降容量與速度。⚠️ EST：對 H200 HBM3e，coalesced 16B/32B 存取通常勝過緊湊但未對齊的 20B，建議實測兩者。

---

## 6. 單卡與 8 卡的可行 N 上限（幾分鐘預算）

### 6.1 單卡 in-HBM 上限

- HBM 141 GB、B=20B、radix 需 double-buffer(×2) → 單卡單趟 in-HBM `N_max ≈ 141e9/(2·20) ≈ 3.5e9 = 2^31.7`。
- 8 卡各持一份 → in-HBM 合計 `≈ 2.8e10 = 2^34.8`。**要超過 2^35 就得 out-of-core（host RAM / NVMe）**，此時瓶頸從 HBM 頻寬轉為 PCIe/NVMe，交給 [06](06-multigpu-intranode.md) 的 I/O 路線。

### 6.2 sort-time 對照表（8× H200，B=20B，MSD passes=⌈log₂N/8⌉，`BW_eff`合計 = 8×2.2 = 17.6 TB/s）

| N | log₂N | passes | 資料量 `passes·2·B·N` | 8-卡 sort 時間（HBM-BW 下限） | 落點 |
|---:|---:|---:|---:|---:|---|
| 2^30 (1.07e9) | 30 | 4 | 171 GB | **0.010 s** | in-HBM |
| 2^32 (4.3e9) | 32 | 4 | 687 GB | **0.039 s** | in-HBM |
| 2^33 (8.6e9) | 33 | 5 | 1.72 TB | **0.098 s** | in-HBM |
| 2^34 (1.7e10) | 34 | 5 | 3.4 TB | **0.20 s** | in-HBM（近上限）|
| 2^35 (3.4e10) | 35 | 5 | 6.9 TB | **0.39 s\*** | 超 in-HBM，需 spill |
| 2^36 (6.9e10) | 36 | 5 | 13.7 TB | **0.78 s\*** | NVMe/host 串流 |
| 2^38 (2.7e11) | 38 | 5 | 55 TB | **3.1 s\*** | **NVMe-bound** |
| 2^40 (1.1e12) | 40 | 5 | 220 TB | **12.5 s\*** | **NVMe-bound（實際數分鐘）** |

**\* 警語（⚠️ EST，且此處先前的估計太樂觀，已修正）**：`≥2^35` 已超過 8 卡合計 HBM（~1.1 TB），上表時間只是「若資料都在 HBM」的**下限**，與真實 out-of-core 時間差一到兩個數量級。真實時間由 NVMe I/O 決定，且**必須區分兩種做法**：
>
> - **naïve 全域外部排序（5 趟都過 NVMe）不可行**：N=2^40 光存一份 = `2^40×20B ≈ 22 TB`，5 趟 read+write = `5×2×22 ≈ 220 TB`；即便 RAID0 拉到 ~100 GB/s（需 exotic 多卡陣列，見 [Phison Apex 演示 ~114 GB/s read](https://www.notebookcheck.net/Phison-demos-Apex-PCIe-5-0-RAID-matrix-breaking-140-GB-s-SSD-transfer-speeds.1021334.0.html)）也要 **~37 分鐘**，實務 ~數十 GB/s 則 **1 小時以上** → 遠超 10 分鐘硬限。**先前寫「數分鐘」是錯的。**
> - **唯一可行路線 = partition-once（或兩次）→ 每桶進 HBM 內排**：外部流量壓到 ~2–3 趟（讀入 + scatter-write + 讀回）= `~44–66 TB`。在 ~50 GB/s 下，N=2^40 仍要 **~15–22 分鐘（超時）**；N=2^38（5.5 TB/份、`~11–16 TB`）約 **~4–6 分鐘（可行）**。
>
> → **10 分鐘內真正站得住的 out-of-core 上限約 N≈2^38；2^40 需要 ~100+ GB/s 的 exotic NVMe 陣列且時間仍吃緊，屬「賭 I/O」的樂觀端。** 完整 I/O 模型與管線重疊見 [06](06-multigpu-intranode.md)。

### 6.3 結論：10 分鐘可達 N 與分數

- **純 in-HBM（最穩、全程 4.8 TB/s）**：N ≤ ~2^35，分數 ~**70 bit**，sort 只吃 ~0.4 s，剩下時間全給生成 → 可把 N 推到 in-HBM 極限。
- **out-of-core（拚 N，賭 I/O）**：**站得住的是 N ~2^38、分數 ~76 bit**（~2–3 趟外部 partition，~50 GB/s 下 ~4–6 分鐘）；**N=2^40 / 分數 ~80 bit 需要 ~100+ GB/s 的 exotic NVMe 陣列、且仍逼近或超過 10 分鐘**（見 §6.2 修正）。任一情況都須與生成/shuffle 三段流水線重疊（[06](06-multigpu-intranode.md) §5.3）才划算。
- **雙節點 16 卡**：合計 HBM ~2.2 TB、合計 `BW_eff` ~35 TB/s，in-HBM 上限翻倍到 ~2^35.8；但跨節點要走 IB（8×400G），比 NVLink 慢一個量級 → partition 一定要 **MSB-first 讓每節點各擁一段前綴、只做一次跨節點 all-to-all**（RMG 思路），細節見 [06](06-multigpu-intranode.md)。

---

## 7. Library / repo / license 清單

| 專案 | 用途 | License | URL |
|---|---|---|---|
| **CUB**（`DeviceRadixSort`/`DeviceSegmentedSort`/`DeviceReduce`）| 單卡 radix sort、分段 sort、ArgMax | **BSD-3-Clause**（舊檔）；CCCL 新檔 **Apache-2.0 WITH LLVM-exception** | [nvidia.github.io/cccl/cub](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html) / [github.com/NVIDIA/cccl](https://github.com/NVIDIA/cccl/blob/main/LICENSE) |
| **Thrust**（`sort`/`sort_by_key`/`unique`）| 高階膠水，整數鍵 dispatch 到 CUB | **Apache-2.0**（含 LLVM-exception 新檔） | [github.com/NVIDIA/thrust](https://github.com/NVIDIA/thrust/blob/main/thrust/system/cuda/detail/sort.h) |
| **libcusort** | 單頭檔、CUB-相容 API、Onesweep+decoupled-lookback，略快於 CUB | **Apache-2.0** | [github.com/IlyaGrebnov/libcusort](https://github.com/IlyaGrebnov/libcusort) |
| **GPUSorting**（b0nes164） | OneSweep / segmented sort 參考實作（CUDA/D3D12） | MIT（⚠️ 待確認，repo 有 LICENSE 檔） | [github.com/b0nes164/GPUSorting](https://github.com/b0nes164/GPUSorting) |
| **RMG Sort** | 多 GPU MSB radix-partition sort（一次 all-to-all P2P swap） | **Apache-2.0** | [github.com/hpides/rmg-sort](https://github.com/hpides/rmg-sort) |
| **moderngpu** | header-only merge sort / **segmented sort**（桶內排序備選） | **BSD**（⚠️ 待確認具體版本條款） | [github.com/moderngpu/moderngpu](https://github.com/moderngpu/moderngpu) |

- [SOURCED: CUB BSD-3 / CCCL Apache-2.0 WITH LLVM-exception — [cccl/LICENSE](https://github.com/NVIDIA/cccl/blob/main/LICENSE)]
- [SOURCED: libcusort Apache-2.0、Onesweep+decoupled-lookback — [libcusort](https://github.com/IlyaGrebnov/libcusort)]
- [SOURCED: RMG Sort Apache-2.0、MSB partition、一次 all-to-all P2P、支援至 2×10^9 元素 — [github.com/hpides/rmg-sort](https://github.com/hpides/rmg-sort)、[BTW'23 論文 PDF](https://hpi.de/oldsite/fileadmin/user_upload/fachgebiete/rabl/publications/2023/rmg-sort-ilic.pdf)]

### RMG Sort 具體數字（多 GPU 骨架的參考）

- **一次 all-to-all P2P swap 時間 14–17 ms（2/4/8 GPU），占總排序 <8%** → 跨卡 shuffle 幾乎免費，瓶頸在各卡本地 sort。 [SOURCED: RMG 論文]
- 相對 merge-based 多 GPU sort 加速 **1.3–1.8×**；排除 CPU↔GPU 傳輸時，8 卡上對競品達 **2.7–9.2×**。 [SOURCED: RMG 論文]
- ⚠️ EST：RMG 論文用 A100 級硬體、32/64-bit 整數；H200 NVLink 4/NVSwitch 更快，P2P swap 只會更小占比 → 我們的 partition-then-local 骨架安全。NVLink 頻寬事實見 [06](06-multigpu-intranode.md) §2.1。

---

## 8. `DeviceSegmentedSort`：桶內批次排序

partition 完得到很多「段」（每個高位桶一段），要在**每段內**排剩餘位元。三個選項：

| 方法 | 適用 | 備註 |
|---|---|---|
| `cub::DeviceSegmentedSort` / `DeviceSegmentedRadixSort` | 段數多、段長不均 | 一次 kernel 排所有段，內部依段長選 warp/block/device 策略 | [SOURCED: [DeviceSegmentedRadixSort](https://gevtushenko.github.io/cccl/cub/api/structcub_1_1DeviceSegmentedRadixSort.html)] |
| moderngpu segmented sort | 段長極不均、要 locality sort | header-only merge-based | [moderngpu segsort](https://moderngpu.github.io/segsort.html) |
| **不排、直接暴力** | 段長已 ≤ ~32（MSD 切夠深）| `O(k²)` per 段但 k 小，省掉 sort overhead；配 §4 的 clz 直接算 lcp | ⚠️ EST：通常最快 |

**建議**：MSD partition 切到「段長 ≤ 32」就停，桶內直接 `O(k²)` 兩兩 `clz` 求 lcp（§4），**跳過段內 sort**。只有在段長仍大（p 開不夠深、記憶體不允許）時才回退 `DeviceSegmentedSort`。

---

## 附：與其他文件的關係

- **為什麼是「排序 / 最近對」問題**、分數 ≈ 2·log₂N、為何 DP/rho 不划算 → [01](01-problem-analysis.md)。
- **N 上限的算力/記憶體/位元預算**、生日數學 → [02](02-math-birthday-and-budget.md)。
- **8 卡怎麼放、NVLink all-to-all、route-vs-discard、HBM 預算、out-of-core I/O** → [06](06-multigpu-intranode.md)（本文的 partition 骨架在那裡落地成管線）。

> 數字警語：本文所有 H200 吞吐（§1.3、§6）為「A100 SOURCED 數字 + bandwidth 外推」，未經本機實測，誤差可能達 1.5–2×。拿到機器後第一件事：用 `nvbench` 對 `DeviceRadixSort` 跑 32/64/96-bit、keys-only vs pairs、256M–4G keys 的 sweep，校準 §1.3 的 `BW_eff`，再回頭修 §6 的可達 N。
