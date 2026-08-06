# 05 — 融合式 GPU Hash Table 最近點對（sort 的替代路線）

> 本文是 regime B（大量截斷碰撞 / 最近點對）的**第二條主力工程路線**：不先把全部雜湊存下來再排序，而是「**算完 SHA 立刻插入 hash table，在插入當下就地偵測碰撞並更新全域最佳**」。
> 對照組是 [04-algorithm-memory-birthday.md](04-algorithm-memory-birthday.md) 的 radix sort / 外部排序路線。數學上限與預算見 [02-math-birthday-and-budget.md](02-math-birthday-and-budget.md)。題目定義（前導匹配位元數 = 分數）見 [00-README.md](00-README.md)。
> 硬體：每 node 8× H200 SXM（GH100，132 SM，141GB HBM3e @ 4.8 TB/s），雙 node 共 16 GPU。

---

## TL;DR

1. **核心洞見**：本題只要最近點對的「前導匹配位元數」，**不需要輸出所有碰撞**。因此可以把 hash table 當成一個「線上最近點對偵測器」——插入時若目標桶已被占用，就把兩個 nonce 對應的完整雜湊拿來數共同前綴，`atomicMax` 到一個全域最佳計數器。**全程單次 pass，記憶體 = 表大小 M，而非資料量 N。**

2. **可用的成熟函式庫（全 Apache-2.0，header-only，可直接進 kernel）**：
   - **NVIDIA cuCollections (cuco)** — `static_map` 在 H100 實測 insert **87.5 GB/s** / find **134.6 GB/s**（50% load factor）。[S1][S3]
   - **warpcore** — 單張 GV100 實測**峰值（up to）1.6×10⁹ insert/s、4.3×10⁹ retrieval/s**。[S2][S5]
   - **BGHT（bucketed cuckoo, BCHT）** — 桶大小 B=16，load factor 可達 **0.991** 仍 100% 成功，插入平均探測 1.43 次。[S4][S6]

3. **最大陷阱**：若每個桶只留「一個代表元」且表被灌爆（N ≫ M），分數上限退化成 **log₂(M)+log₂(N)**，而非完整生日界 **2·log₂(N)**；差距 = **log₂(N/M)** 位元。⚠️ 這是本路線最容易踩雷、也最容易在賽後才發現「為什麼分數卡住」的地方。解法：**每桶留多個代表元** 或 **prefix-partition 多趟**（與 sort 路線同一個 radix 分割思想，見 §5）。

4. **何時選 hash table、何時選 sort**：hash table 贏在**低延遲、單 pass、省一次全量寫回**；sort 贏在**確定性達到 2·log₂(N)、無 load-factor 風險、頻寬利用率滿**。實務建議：**hash table 當「線上粗篩 + 高位分割」，sort 當「每個 partition 內的精算」**，兩者混用（見 §7）。

> ⚠️ **全文數字警語**：所有 ops/s 數字皆來自各函式庫論文/官方 blog 的**特定 GPU、特定 key/value 寬度、特定 load factor**的 benchmark，**與本題（key=64-bit、只插入不查詢、H200）並非同條件**。務必在拿到機器後用 microbenchmark 校正（見 [09-open-questions.md](09-open-questions.md)），誤差可能 2~3×。

---

## 1. 為什麼考慮 hash table 而非直接 sort

sort 路線（[04](04-algorithm-memory-birthday.md)）的資料流是：**generate 全部 N 個雜湊 → 寫回 HBM/NVMe → radix sort → 掃描相鄰對**。這需要至少一次「把 N 個 key 完整寫出再讀回」的全量記憶體來回。

hash table 路線把它壓成一次 pass：

| 步驟 | sort 路線 | fused hash-table 路線 |
|---|---|---|
| SHA 生成 | ✔（frozen `sha256_block`） | ✔（同一顆 kernel） |
| 中間儲存 | **寫 N × (key+nonce) 到 HBM/NVMe** | **只寫入表 M slots**（M 可 ≪ N） |
| 找最近對 | 全域排序 O(N) + 掃描 | 插入當下 `atomicMax`，**O(1)/key** |
| 全量讀回 | 需要（sort 本身多趟） | 不需要 |
| 記憶體 | 隨 N 線性 → 需 partition | = 表大小 M（天生 decouple） |

關鍵：**SHA 生成很便宜、最近點對搜尋才是瓶頸**（[00](00-README.md) KEY ANALYSIS (b)）。若 hash table 的 insert 吞吐能追上 SHA 吞吐，就能省掉整條 sort 管線。下面先看真實吞吐數字，再看它是否追得上。

**吞吐是否平衡（我方估算）**：單 H200 SHA-256 規劃基準 ~15 GH/s（[02](02-math-birthday-and-budget.md) §2.1，樂觀上界）；cuco insert 87.5 GB/s ÷ 8 byte/pair ≈ **~11×10⁹ insert/s**（我方換算）。**兩者同一數量級** → 融合式 kernel 的 insert 幾乎能吃下 SHA 的產出，不會讓 SHA 空轉。⚠️ 但這是把「H100 的 4+4 byte 查詢 benchmark」硬套到「H200 的 64-bit-key 純插入」，條件差異大，需實測。

---

## 2. GPU Hash Table 函式庫總覽（SOURCED）

三個都是開源、header-only、可 `#include` 進 device code 的 C++/CUDA 函式庫，皆 **Apache-2.0**（對學術競賽無授權障礙）。

| 函式庫 | 探測法 | 實測吞吐 | benchmark GPU | 授權 | Repo |
|---|---|---|---|---|---|
| **cuCollections (cuco)** | open addressing + linear probing（可切 double hashing）| insert **87.5 GB/s**、find **134.6 GB/s**（load 50%, 2²⁷ keys, 4+4 byte）[S1][S3] | H100-80GB-SXM | Apache-2.0 | github.com/NVIDIA/cuCollections [S3] |
| **warpcore** | open addressing，warp-cooperative | insert **1.6×10⁹/s**、retrieval **4.3×10⁹/s** [S2][S5] | GV100 (Titan V, Volta) | Apache-2.0 | github.com/sleeepyjack/warpcore [S2] |
| **BGHT / BCHT** | bucketed cuckoo（另有 power-of-two、iceberg）| load factor 可達 **0.991**、insert 平均探測 **1.43** [S4][S6] | Volta+（V100）| Apache-2.0 | github.com/owensgroup/BGHT [S6] |

### 2.1 各函式庫特性與對本題的適配

- **cuco `static_map`**：NVIDIA 官方維護、最活躍。限制 **key+value 合計 ≤ 8 byte**（用 `cuda::std::atomic<pair>` 單指令 CAS 整個 slot）。[S1] 高 load factor 時用 **cooperative group tile size = 4** 做群組探測。[S1] 另附 `static_set`、`bloom_filter`（blocked Bloom）、`hyperloglog`。[S3]
  - 對本題：8-byte slot 限制是硬傷——我們想同時存 top-k 雜湊位元（當 key）+ 64-bit nonce（當 value）會超過 8 byte。可退而求其次：**只存 nonce**（把 nonce 當 key），桶索引改用「雜湊高位」自訂 hash functor，碰撞時由 nonce 重算 SHA 取得完整雜湊。
- **warpcore**：key 支援 `uint32_t`/`uint64_t`，value 為任意 trivially-copyable 型別（無 8-byte 硬限）。[S2] 提供 `HashSet`、`SingleValueHashTable`、`CountingHashTable`、`MultiBucketHashTable`。**`MultiBucketHashTable`（每桶多值）正是 §5 recovery 需要的「每桶多代表元」結構。**
- **BGHT/BCHT**：static（建表後不增長），bucketed cuckoo 把整桶（B=16 key）用一個 128-byte cache line 一次讀入，warp/CG 平行比對——**空間效率最高（可撐到 99% load）**，適合「表要盡量塞滿以最大化 M」的場景。[S4]

### 2.2 warp-cooperative probing（三者共同的核心技巧）

GPU hash table 的效能關鍵不是「一個 thread 管一個 key」，而是 **一個 cooperative group（通常 4、8、16 或 32 lane）協作處理一個 key**：整組 lane 一次讀入一整桶（bucket）到暫存器，用 `__ballot_sync` / `__shfl_sync` 平行比對桶內所有 slot，命中或找空位只需一次 coalesced 記憶體交易。這把「隨機探測的 latency」攤平成「合併存取的 bandwidth」，是 BCHT 每桶 128 byte（16×8）恰好對齊 L2 sector 的原因。[S4]

- 對本題含意：**桶大小應對齊 128 byte（一條 cache line）**，讓每次探測都是滿載 coalesced 讀取，直接吃 H200 的 4.8 TB/s HBM3e。

---

## 3. 融合式方法：hash → insert → closest-pair（單趟）

資料流（每個 GPU thread/CG 處理一個 nonce 區間）：

```text
for nonce in my_range:
    h = sha256_block(base_words, nonce)     # frozen，256-bit 雜湊
    key = top_k_bits(h)                      # 取高 k 位元當桶索引
    old = table.insert_or_get(key, nonce)    # CAS 佔桶；若已占用回傳既有 nonce
    if old != EMPTY:
        h2 = sha256_block(base_words, old)   # 由既有 nonce 重算雜湊（SHA 便宜）
        p  = common_prefix_bits(h, h2)       # 數共同前導位元
        atomicMax(&global_best_bits, p)      # 更新全域最佳
```

要點：

1. **記憶體 = M slots，不是 N**。表只需存「桶索引 key + nonce」≈ 16 byte/slot（見 §4）。N 可遠大於 M。
2. **不需儲存完整 256-bit 雜湊**：碰撞時用既有 nonce 重跑 frozen `sha256_block` 重算，用算力換記憶體。SHA 便宜、碰撞事件稀疏，重算成本可忽略。
3. **`global_best_bits` 的 `atomicMax` 幾乎無爭用**：它是單調遞增、只在「刷新紀錄」時才寫，整場總更新次數 ~O(log N)（每提升 1 bit 才寫一次）。真正的爭用在 **insert 對 slot 的 `atomicCAS`**（見 §6）。
4. 需同時記下「達成最佳的那一對 (a, b)」以輸出 CSV：把 `atomicMax` 換成「CAS 一個 packed `<bits:16, idx:48>`」或維護 best 時另存一組 nonce（用 `atomicCAS` 迴圈保護）。

---

## 4. Slot 位元組成本與表容量

| 欄位 | 寬度 | 說明 |
|---|---|---|
| 桶索引 key（高位雜湊） | 32–64 bit | 用來分桶；存下來可省一次重算做比對 |
| nonce | 64 bit | 輸出 CSV 必需；也是重算雜湊的種子 |
| **合計/slot（我方估算）** | **~12–16 byte** | 若把 key 壓到 32 bit 則 12 byte |

單 H200 141GB HBM3e，扣掉 base words / kernel 常駐，可用 ~130GB：

- 16 byte/slot → **M ≈ 8.1×10⁹ = 2³³ slots / GPU**（load factor 1.0 理論上限）。
- 實務上 open addressing 要留 load factor ≤ 0.5~0.7（cuco benchmark 用 0.5 [S1]），或用 BCHT 撐到 0.99 [S4]。
- 16 GPU 合計 **M_total ≈ 1.3×10¹¹ = 2³⁷ slots**（BCHT @0.9 load）。

⚠️ **關鍵限制**：10 分鐘可生成 N ≈ 2⁴⁶（[02](02-math-birthday-and-budget.md) §3，樂觀）遠大於 M_total ≈ 2³⁷。**表一定會被灌爆 N ≫ M** → 直接觸發 §5 的退化陷阱。這正是為什麼「純單表」不夠、必須配 partition。

---

## 5. 核心陷阱：single-representative 飽和退化（與解法）

### 5.1 為什麼會退化

**完整生日界**：對 N 個 distinct 雜湊，最近點對的期望共同前綴 = **2·log₂(N)** 位元（[02](02-math-birthday-and-budget.md) Part 1；因為要比對的是全部 C(N,2) ≈ N²/2 對，log₂(N²)=2log₂(N)）。

**但單代表元表只做了 N 次比對，不是 N²/2 次**：每個 key 插入時只跟「它落到的那個桶裡既有的那 1 個代表元」比一次。表灌爆後（N ≫ M），每桶平均 N/M 個 key 掉進來，但只跟第一個代表元比；同桶後來的 N/M−1 個彼此之間的碰撞完全沒被檢查。

推導最佳分數：共發生 ~N 次比對事件，每次兩個雜湊「因為同桶」保證高 k=log₂(M) 位元相同；超過第 k 位之後兩者相對隨機，額外相同位元服從幾何分布，N 次事件的最大值 ≈ log₂(N)。故：

> **single-representative 最佳分數 ≈ log₂(M) + log₂(N)**

### 5.2 formula 對照

| 方案 | 有效比對數 | 期望最佳分數 | 相對完整界的損失 |
|---|---|---|---|
| 完整最近點對（sort / 每桶全存） | ~N²/2 | **2·log₂(N)** | 0（基準） |
| single-representative 飽和表 | ~N | **log₂(M) + log₂(N)** | **log₂(N/M)** |
| 每桶存 b 個代表元 | ~N·b/2（同桶內）| ≈ log₂(M) + log₂(N·b) 逼近 | 隨 b↑ 縮小 |
| prefix-partition P 趟（每趟表不飽和） | ~N²/2（分散在 P 趟）| **2·log₂(N)** | 0 |

- 損失 **log₂(N/M)**：以 §4 的 N≈2⁴⁶、M_total≈2³⁷ 計，**損失約 9 位元**——直接把可能的 ~92 分砍到 ~83 分。⚠️ 這是名次級的差距。
- 反過來看：**若 M ≥ N（表塞得下、load factor ≤ 1、幾乎不飽和），log₂(M)+log₂(N) ≈ 2·log₂(N)，零損失。** 退化只在「刻意用小表灌爆」時發生。

### 5.3 兩種 recovery

**(A) 每桶留多個代表元（few-per-bucket）**
用 `warpcore::MultiBucketHashTable` 或 BCHT（B=16）每桶存 b 個 nonce。插入時對桶內既有 b 個全部比一遍（warp-cooperative，一次 coalesced 讀入整桶）。有效比對數 ×b，把損失從 log₂(N/M) 降到 log₂(N/(M·b))。b=16 → 多回收 4 位元。**成本**：slot 記憶體 ×b，M 相應變小，需權衡。

**(B) prefix-partition 多趟（推薦，與 sort 路線同構）**
跑 P = ⌈N/M_fit⌉ 趟；第 i 趟只處理「完整雜湊高 p 位元 == i」的 nonce（p=log₂P）。每趟落入表的 key 數 ≈ N/2^p ≈ M_fit → **表不飽和 → 該分區內達成完整 2·log₂(N)**。
關鍵正確性保證（[00](00-README.md) KEY ANALYSIS (c)）：**全域最佳對必然共享最高位元，故一定落在同一個 partition 內**，分趟不會漏掉它。這與 [04](04-algorithm-memory-birthday.md) 的 radix 分割是**完全相同的 decouple 記憶體思想**，只是「桶內用 hash table 即時偵測」取代「桶內排序」。

> 結論：**融合式 hash table 要拿到完整 2·log₂(N)，必須配 prefix-partition。** 單靠一張大表灌爆會固定損失 log₂(N/M)。這一點與 sort 路線的記憶體結論一致——差別只在分區「內部」用 table 還是 sort。

---

## 6. Atomics：`atomicCAS`（搶 slot）vs `atomicMax`（全域最佳）

本路線有兩處 atomic，爭用特性天差地別：

| Atomic | 位置 | 爭用程度 | 瓶頸風險 |
|---|---|---|---|
| `atomicCAS` 佔桶 | 每個 key 插入時，散布在 M 個 slot | 中～高（灌爆時同桶 hot） | **主要成本**——隨機位址、L2/HBM 往返 |
| `atomicMax` 全域最佳 | 單一計數器 | **極低**（單調遞增、~O(log N) 次寫） | 可忽略 |

要點（SOURCED + 我方分析）：

- **atomic 硬體實作於 memory controller / L2，用 CAS 為基礎**；單一位址被數千 thread 爭用會**序列化**，吞吐崩塌。[S7] 因此**絕不要**讓所有 thread 去 atomic 同一位址。
- 本題天然避開了最壞情況：全域最佳只在破紀錄時寫，爭用可忽略。**危險的是 insert 的 slot 爭用**——當某高位前綴特別多 nonce 命中（表灌爆時的 hot bucket），該桶 CAS 會塞車。
- 減緩手法（通用最佳實務 [S7]）：**per-warp / per-block 先做本地聚合，再一次 atomic**。對本題可行做法：每個 warp 在 shared memory 維護一個小型局部最佳，warp 結束再 `atomicMax` 進全域——把全域 atomic 次數再降一個數量級。
- **HBM3e 頻寬**：H200 4.8 TB/s；Hopper L2 吞吐實測 ~4472 byte/clock、global memory 實測 ~1861 GB/s（H800 PCIe 樣本）[S8]。atomic 隨機存取吃的是 latency-bound 路徑，實際遠低於順序頻寬——**這是 hash table 相對 sort（順序、頻寬滿載）的結構性劣勢**。⚠️ H200 SXM 的 atomic 吞吐未見公開實測數字，需 microbenchmark。

---

## 7. Filters 當廉價預篩：「這個前綴看過沒？」

在插入昂貴的主表之前，可先用**近似成員查詢（AMQ）filter** 快速判斷「這個高位前綴是否已出現過」，命中才進主表比對。filter 每項只需幾 bit，可在 HBM 裡容納**遠多於主表**的前綴指紋。

| Filter | 型態 | 實測吞吐（GH200 Grace-Hopper，Hopper GPU + HBM3）| bits/item | 特性 | 來源 |
|---|---|---|---|---|---|
| **Cuckoo-GPU** | cuckoo filter | 正查詢 **~30×10⁹/s**、插入 **~11×10⁹/s**（95% load）| 16-bit 指紋，FPR ~0.045% | 支援刪除、飽和查詢頻寬 | [S9] |
| **GPU Blocked Bloom (GBBF)** | blocked Bloom | 插入 **~20×10⁹/s**（k-mer 資料）| 依 FPR 調 | 最快、但不支援刪除 | [S9] |
| **cuco `bloom_filter`** | blocked Bloom | （官方未列單獨數字）| 可調 | NVIDIA 官方、與 cuco 同生態 | [S3] |
| GPU Quotient Filter (GQF) / TCF | quotient / two-choice | 支援刪除但有效能懲罰 | 較省空間 | 動態場景 | [S9] |

對本題的用法與取捨：

- **用途**：把主表的桶索引 key 先丟 filter；`might_contain==false` 表示此高位前綴首見 → 直接寫主表、跳過比對。只有 filter 命中（可能碰撞）才做昂貴的重算 SHA + 比對。
- **省什麼**：省掉「首見前綴」的一次 slot CAS + 潛在重算，降低主表爭用。
- ⚠️ **但本題其實是「幾乎每個前綴都要比」**：既然目標就是找碰撞，飽和表下同桶命中率高，filter 過濾掉的多是「真的沒碰撞」的插入，效益不如它在「稀疏碰撞偵測」場景那麼大。**建議把 filter 定位成「hot-partition 的快速去重 / 決定要不要展開下一趟 partition」，而非核心比對器。** blocked Bloom（GBBF）因為插入最快、又不需刪除，最契合單 pass 全插入的工作型態。

---

## 8. Hash table 何時勝、Sort 何時勝

| 面向 | Fused hash table | Radix sort（[04](04-algorithm-memory-birthday.md)）|
|---|---|---|
| pass 數 | **1 趟**（generate=insert=detect 融合） | 多趟（generate→寫→sort→掃）|
| 記憶體來回 | 省一次全量寫回 | 需全量寫回 + sort 內部多趟 |
| 記憶體用量 | = 表大小 M（配 partition decouple） | = N（必須 partition decouple）|
| 達成完整 2·log₂(N) | **需配 partition，否則損失 log₂(N/M)**（§5）| 天然達成（排序後相鄰對即最近） |
| 記憶體存取型態 | **隨機 + atomic**（latency-bound，劣勢）| **順序 + coalesced**（bandwidth 滿載，優勢）|
| load-factor 風險 | 有（爆表退化、hot-bucket 爭用） | 無 |
| 確定性 / 可預測性 | 較差（atomic 爭用、探測長度浮動）| 高（cub/ModernGPU radix sort 吞吐穩定）|
| 成熟度 | cuco/warpcore/BGHT 皆生產級 | cub `DeviceRadixSort` 極成熟、A100/H100 可達數十億 key/s |

**綜合建議（我方判斷）**：

1. **主力仍建議 sort**（[04](04-algorithm-memory-birthday.md)）——它順序存取、頻寬滿載、確定達到 2·log₂(N)、無 load-factor 地雷，對「10 分鐘硬時限」的可預測性最好。
2. **hash table 當輔助**：用在 **(a) 線上粗篩**——一邊生成一邊即時抓「當前最佳」，讓每次 10 分鐘的 run 都有保底分數，不必等排序跑完；**(b) partition 內部的即時偵測**，當某分區資料量恰好 ≤ M_fit（不飽和），就地用 table 省掉排序。
3. **決勝點是 kernel 常數**（[00](00-README.md) TL;DR #3）：兩條路線都吃同一顆 frozen SHA kernel，最終名次多半由「insert/sort 能不能追上 SHA 生成、GPU 有沒有閒置」決定，而非演算法漸進複雜度。

---

## 9. 不確定性彙整（拿到機器必量）

| # | 不確定項 | 為何重要 | 量測方式 |
|---|---|---|---|
| 1 | ⚠️ cuco/warpcore/BGHT 的吞吐在 **H200 + 64-bit key + 純插入**條件下的真值 | 全文 ops/s 皆非同條件外推，可能差 2~3× | 直接跑各 repo benchmark，改成 8/16-byte slot、關掉 find |
| 2 | ⚠️ H200 SXM 的 **global atomic（CAS/Max）吞吐與爭用曲線** | 決定 insert 是否成為瓶頸 | 微基準：可調爭用度的 atomicCAS 掃描 |
| 3 | ⚠️ 融合 kernel 中 **SHA + insert 是否 register/occupancy 打架** | frozen SHA 已吃很多暫存器，再加探測邏輯可能降 occupancy | 比較「純 SHA」vs「SHA+insert」的達成 hash/s |
| 4 | ⚠️ hot-bucket 爭用在真實高位前綴分布下的嚴重度 | 決定要不要 per-warp 本地聚合 | 統計桶佔用直方圖 |
| 5 | filter 預篩對本飽和工作型態的實際淨效益（可能為負）| §7 存疑 | A/B：有/無 filter 的 end-to-end 分數 |

---

## 參考來源

- [S1] NVIDIA Technical Blog — *Maximizing Performance with Massively Parallel Hash Maps on GPUs*（cuco `static_map` H100 insert 87.5 GB/s / find 134.6 GB/s、linear probing、CG tile 4、key+value ≤ 8 byte、load 50%）: https://developer.nvidia.com/blog/maximizing-performance-with-massively-parallel-hash-maps-on-gpus/
- [S2] warpcore repo（Apache-2.0、key uint32/uint64、MultiBucketHashTable）: https://github.com/sleeepyjack/warpcore
- [S3] NVIDIA/cuCollections repo（Apache-2.0、static_map/static_set/bloom_filter/hyperloglog、linear probing 預設可切 double hashing）: https://github.com/NVIDIA/cuCollections
- [S4] BGHT 論文 *Better GPU Hash Tables*, Awad et al., APOCS（BCHT bucket B、load 0.991、平均探測 1.43）: https://arxiv.org/abs/2108.07232 / PDF https://arxiv.org/pdf/2108.07232
- [S5] warpcore 論文 *WarpCore: A Library for fast Hash Tables on GPUs*, Jünger et al., IEEE HiPC 2020（GV100 1.6×10⁹ insert/s、4.3×10⁹ retrieval/s）: https://arxiv.org/abs/2009.07914
- [S6] BGHT repo（Apache-2.0、bucketed cuckoo / power-of-two / iceberg、header-only）: https://github.com/owensgroup/BGHT ／專案頁 https://owensgroup.github.io/BGHT/
- [S7] CUDA atomic 效能與爭用（atomic 建於 CAS、單址序列化、per-warp/block 聚合最佳實務）: https://supercomputingblog.com/cuda/cuda-tutorial-5-performance-of-atomics/ ／ https://deepwiki.com/kis-balazs/CUDA-Research/6.2-atomic-operations-and-synchronization
- [S8] *Benchmarking and Dissecting the Nvidia Hopper GPU Architecture*, arXiv 2402.13499（Hopper L2 ~4472 byte/clk、global mem ~1861 GB/s 樣本）: https://arxiv.org/html/2402.13499v1
- [S9] *Cuckoo-GPU: Accelerating Cuckoo Filters on Modern GPUs*, arXiv 2603.15486（GH200：cuckoo filter 正查詢 ~30×10⁹/s、插入 ~11×10⁹/s；GBBF 插入 ~20×10⁹/s；16-bit 指紋、95% load、FPR ~0.045%）: https://arxiv.org/abs/2603.15486 / HTML https://arxiv.org/html/2603.15486

> 數字警語（重申）：[S1][S2][S5][S9] 的吞吐數字均為**各自 benchmark 的特定 GPU（GV100/H100/GH200）、特定 key/value 寬度、特定 load factor** 之結果，與本題（H200、64-bit key、單 pass 純插入、不查詢）**條件不同**，僅供量級參考，實測前不得寫進時程承諾。
