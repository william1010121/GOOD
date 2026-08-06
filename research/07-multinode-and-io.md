# 07 — 雙節點擴展 + I/O（multi-node scaling & I/O）

本文回答一個看似理所當然、實則反直覺的問題：**兩個 node 之間到底該不該通訊？**
結論先講：**在本題的參數下，「不通訊」（node A 拿 hash top-bit=0、node B 拿 top-bit=1，各自丟掉另一半）幾乎總是贏過「用 InfiniBand 路由把 key 洗牌」。** 而且 **NVMe out-of-core 幾乎確定不值得用**。以下逐項量化。

前置閱讀：[00-README.md](00-README.md)（regime 判定）、[02-math-birthday-and-budget.md](02-math-birthday-and-budget.md)（算力預算、`分數 ≈ 2·log2(N)`）。演算法細節見 sibling 檔（memory-式生日攻擊 / radix sort / closest-pair）。

> ⚠️ **本文所有數字分兩類**：標 `[來源]` 的是查證過的硬體/文獻數字；標 `【估計】` 的是我根據這些數字對本題推導的結果，誤差可能 2~3×，拿到機器後必須用 microbenchmark 校正（`ib_write_bw`、`nccl-tests`、`gdsio`、`cub::DeviceRadixSort`）。

---

## Part 0 — 一句話心智模型

題目的計分是 `分數 ≈ 2·log2(N)`，其中 **N = 你能「產生 + 排序 + 互相比較」的 distinct hash 數**。要放大 N，你會做 **prefix-partitioning（依 hash 前 p 個 bit 做 radix 分桶）**：因為全域最佳 pair 一定共享最高幾個 bit，所以它一定落在同一個 partition 內（[00](00-README.md) 的 KEY ANALYSIS (c)）。

**關鍵洞見（本文骨幹）**：把「hash 最高 1 個 bit」這個 partition 指派給兩個不同 node，**就是 prefix-partitioning 的最外層那一位**。所以「雙節點依 top-bit 拆分」不是額外的犧牲，它**本來就是演算法要做的第一層 radix**——只是把兩個桶放到兩台機器上。這讓「零通訊」變成幾乎免費，而不是一個 trade-off。

---

## Part 1 — CUDA-aware MPI + GPUDirect RDMA over 400G IB：實際頻寬與延遲

### 1.1 硬體上界

| 項目 | 數值 | 來源 |
|---|---|---|
| ConnectX-7 單埠 NDR | 400 Gb/s = **50 GB/s** 線速（4 lane × 100G PAM4 + RS-FEC） | [NVIDIA ConnectX-7 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/networking/infiniband-adapters/infiniband-connectx7-data-sheet.pdf) |
| NDR 小訊息延遲 | ~**0.9 µs** @ 8 B（純 fabric，未含軟體堆疊） | [Spheron networking guide](https://www.spheron.network/blog/gpu-networking-infiniband-roce-spectrum-x-guide/) |
| 每 node IB HCA | 8× 400G ConnectX-7 → 3200 Gb/s = **400 GB/s** raw 聚合 | 題目規格 |
| 額外 | 2× 200GbE BlueField-3（管理/儲存，非計算路徑） | 題目規格 |

**單埠實務可達（`ib_write_bw`）**：NDR400 扣掉協定/FEC/PCIe 開銷後單向約 **45~47 GB/s**【估計，依 HDR/EDR 慣例外推，需實測】。PCIe Gen5 x16 單卡上限 ~64 GB/s [來源同上 Spheron]，所以 PCIe **不是**單卡瓶頸，但 8 張卡共享的 CPU root complex 與 NUMA 拓撲會是。

### 1.2 CUDA-aware MPI + GPUDirect RDMA 的真實折損

「CUDA-aware」= MPI 可直接收送 GPU buffer；搭配 **GPUDirect RDMA** 時 HCA 直接 DMA GPU HBM，資料**不經過 host memory**，省一次 PCIe bounce 與 CPU copy [[Open MPI CUDA docs](https://docs.open-mpi.org/en/v5.0.x/tuning-apps/networking/cuda.html)]。

⚠️ 網路上最常被引用的 NVIDIA CUDA-aware MPI benchmark 是**古董**（MVAPICH2-1.9b、Tesla K20、FDR IB）：device-to-device 峰值僅 4.18 GB/s，延遲 18 µs [[NVIDIA blog](https://developer.nvidia.com/blog/benchmarking-cuda-aware-mpi/)]。**不要拿它規劃 NDR**。較新的參考點：

| 情境 | 可達頻寬 | 來源 |
|---|---|---|
| H100 node 內 8-GPU CUDA-aware MPI（走 NVLink）雙向 | ~**220 GB/s**（理論 300 GB/s 的 73%） | [FAU CUDA-Aware-MPI blog](https://blogs.fau.de/adityauj/2025/02/18/cuda-aware-mpi-part-1-understanding-node-topology-and-communication-bandwidth/) |
| GPUDirect RDMA `ib_write_bw`，65 KB 訊息，A100 + ConnectX（單 mlx5 埠） | 35.6 Gb/s = **4.45 GB/s**（注意：此為受限拓撲下的舊測，非 NDR 上限） | [Oracle GPUDirect RDMA lab](https://docs.oracle.com/en/learn/gpudirect-rdma-ib-write-bw/index.html) |
| GPUDirect RDMA MPI 小訊息延遲（實務含堆疊） | ~2~3 µs【估計】 | — |

**本題規劃基準【估計】**：跨 node、多軌（8 HCA 一起用）、大訊息（我們洗的是連續大 buffer，最理想）的 **2-node bisection 聚合頻寬 ≈ 300~370 GB/s**。取 **~350 GB/s** 當規劃值。多軌聚合能否吃滿要靠 UCX 的 `UCX_NET_DEVICES`/rail-binding 與 NUMA-aware GPU↔HCA 配對；預設常只用 1~2 軌 → **必須實測校正**。

### 1.3 給洗牌用的 collective

若真要路由，需要的是 **`MPI_Alltoallv`**（variable count，因為各 prefix 桶大小不均）。Alltoall 類 collective **不含算術**，CUDA-aware 版可全程不碰 host memory [[NVIDIA blog](https://developer.nvidia.com/blog/benchmarking-cuda-aware-mpi/)]。但 alltoall 是 **bisection-bound**：兩 node 之間就是一條對切線，能用的就是那 ~350 GB/s，無法靠更多 GPU 變快。

---

## Part 2 —「拆分（零通訊）」vs「路由（保留全部 N）」：定量對決

### 2.1 兩種方案

| 方案 | 做法 | 通訊 | 代價 | N |
|---|---|---|---|---|
| **SPLIT（拆分）** | node A 只留 hash top-bit=0，node B 只留 top-bit=1，產生後**立即丟棄**另一半 | **零** | ~2× 產生量（SHA 很便宜） | 兩 node 合計仍覆蓋全空間 |
| **ROUTE（路由）** | 全部產生後，用 `MPI_Alltoallv` 依 prefix 洗牌，每 node 排序自己那段 | 洗 ~N/2 個 key 過 IB | alltoall 佔用 wall-clock | 保留全部 N |

### 2.2 「+2 bits」這個直覺哪裡對、哪裡錯

天真算法：ROUTE 保留 N、SPLIT 只保留 N/2 → `2·log2(N) − 2·log2(N/2) = +2 bit`。**這只在「產生（generation）是瓶頸」時成立。**

但 [00](00-README.md)/[02](02-math-birthday-and-budget.md) 的 KEY ANALYSIS 明確說：**產生很便宜，瓶頸是 sort / closest-pair。** 一旦瓶頸在 sort：

- SPLIT 的「丟掉另一半」是**在 sort 之前用一個 top-bit 比較就過濾掉**（幾乎零成本的 pre-filter）。昂貴的 sort 階段只處理**留下來的** key。
- 所以兩個 node 的 sort 產能被 100% 用在有用的 key 上；**SPLIT 的 2× 產生penalty 落在便宜的階段，不吃 sort 產能**。
- 反過來，ROUTE 的 alltoall **直接從 600 秒裡扣掉時間**，那段時間本來可以拿去 sort。

**結論：當 sort 是瓶頸（本題如此），SPLIT 不但沒少 2 bit，反而因為省下 alltoall 時間而 ≥ ROUTE。** 天真的「+2 bit」是**產生受限 regime** 才有的東西。

### 2.3 crossover：ROUTE 什麼時候才值得？

ROUTE 唯一勝出的條件是**同時**滿足：(i) 產生量變成綁定資源（不是 sort），且 (ii) alltoall 時間相對 600 秒可忽略。量化 (ii)：

洗牌需搬 ~N/2 個 key，每 key 取 16 B（8 B nonce + 8 B hash 高位當 sort key）：

```
t_comm = (N/2 × 16 B) / B_net ,  B_net ≈ 3.5×10^11 B/s
```

| N | 需搬位元組 | t_comm @350 GB/s | 佔 600 s |
|---:|---:|---:|---:|
| 2^41 (2.2×10^12) | 1.76×10^13 B | ~50 s | 8% |
| 2^42 (4.4×10^12) | 3.5×10^13 B | ~100 s | 17% |
| 2^43 (8.8×10^12) | 7.0×10^13 B | ~200 s | 33% |
| 2^45 (3.5×10^13) | 2.8×10^14 B | ~800 s | **>100%（放不進去）** |

**crossover ≈ N ≈ 2^41.3**（alltoall 剛好吃掉 60 s ≈ 10% 預算）。**超過 2^41 路由就開始賠本**，而本題可達的 N（見 Part 3）就在 2^41~2^42，正好落在 ROUTE 開始不划算之處。

> 小優化：ROUTE 可只送 8 B nonce、接收端**重算 hash**（SHA 便宜），把通訊減半 → crossover 右移到 ~2^42.3。但這同時把「產生」拉回計算路徑，反而強化了「產生便宜、別為省產生而通訊」的論點。

### 2.4 SPLIT 為何是「免費的最外層 radix」

Part 0 的洞見在此收斂：prefix-partitioning 本來就要依 hash 前 p bit 分桶。**把第 1 個 bit（p 的最外層）指派給 2 個 node**，等於把演算法的第一層 radix 直接映射到硬體。每 node 對「自己那半 hash 空間」獨立做剩下的 partition-by-regeneration（見 Part 3）。**沒有任何一個 key 需要跨 node**，因為跨 node 的邊界恰好是 top-bit，而 top-bit 不同的兩個 hash 不可能是「共享最多前導 bit」的那一對（它們連第 1 個 bit 都不同 → 前導匹配 = 0）。

**唯一會需要 ROUTE 的設計**，是你選擇**把 key 存起來再洗**（out-of-core），而不是**重算**。下一節說明為何「重算」贏「存起來」，這也順帶埋葬了 ROUTE 與 NVMe。

---

## Part 3 — N 到底能到多少：三種資料佈局的定量比較

總 HBM = 16 × 141 GB = **2.256 TB** ≈ 1.4×10^11 個 16 B key ≈ **2^37 個 resident key**。
叢集產生率採 [08](08-fixed-kernel-throughput.md) 的**校正後**數字（⚠️ [02](02-math-birthday-and-budget.md) 舊表的「2.4×10^11 hash/s / 2^47」是**高估**：它源自把 RTX 4090 的 SHA-256 當成 50.9 GH/s，但原始 hashcat gist 顯示 mode 1400 SHA-256 只有 **~21.98 GH/s**，50.6 GH/s 其實是 **SHA-1（mode 100）**被誤記；且本題 kernel 凍結、無法用 hashcat 的 midstate/LOP3，實際更低）：
- **中心 ~1.2×10^11 hash/s** → 600 s ≈ 7.2×10^13 = **2^46.0**
- **樂觀 ~1.6×10^11** → **2^46.6**；**保守 ~0.8×10^11** → **2^45.4**

（原文以 2^47 為預算，下方各表已隨此下修約 0.5~1 bit。）

| 佈局 | 機制 | N 上限【估計】 | 分數 ≈ 2·log2(N) | 備註 |
|---|---|---:|---:|---|
| **HBM-only 單趟** | 產生填滿 HBM → radix sort → 掃相鄰 | ~2^37 | ~74 | 最簡單、最穩、10 分鐘內絕對做得完 |
| **NVMe out-of-core** | 產生→寫 NVMe→external merge sort | ~2^38 | ~76 | 只多 ~1~2 bit，複雜度暴增（見 Part 4） |
| **Partition-by-regeneration** | 每個 partition：重掃 nonce 範圍、只留落在此桶的、在 HBM 內 sort、記最佳、換下一桶 | **~2^41~2^42** | **~82~84** | **本題最佳**；記憶體與 N 解耦（[00](00-README.md) KEY ANALYSIS c） |

### 3.1 為何 regeneration 贏，且其成本公式

設每桶 resident 上限 H = 2^37（HBM）、要達 N 個 distinct hash → 需 **P = N/H 個 partition**。每趟要掃過所有 N 個 nonce 但只留 1/P：

```
總產生量 = P × N = N² / H
```

令 `N²/H ≤ 總產生預算`：

| 總產生預算（[08] 校正） | N²/2^37 上限 | N 上限 | 分數 |
|---|---|---:|---:|
| 2^46.6（樂觀） | N² ≤ 2^83.6 | ~2^41.8 | ~84 |
| 2^46.0（中心） | N² ≤ 2^83 | **~2^41.5** | **~83** |
| 2^45.4（保守） | N² ≤ 2^82.4 | ~2^41.2 | ~82 |

> ⚠️ **原文此表用 2^47 預算得到 N=2^42/分數 84，屬樂觀上緣。** 以 [08] 校正後的產生率（4090 SHA-256 實為 ~22 GH/s 而非 50.9，且凍結 kernel 更慢），中心值落在 **N≈2^41.5、分數 ~83**，樂觀才摸到 2^42/84。因為 **N ∝ √(產生預算)**，產生率的 2× 誤差只搬動 N 約 0.5 bit（分數 ~1），所以結論的**排序**（regeneration ≫ NVMe ≫ HBM-only）不受影響，但別把 84 當保證值。

**這比 HBM-only 多 ~7~9 bit，比 NVMe 多 ~5~7 bit，而且完全不需要儲存、不需要跨 node 通訊。** SPLIT 天然契合：兩 node 各自跑 partition-by-regeneration，只是每 node 的最外層桶固定（top-bit=0 / =1）、各覆蓋一半 hash 空間，兩半互斥 → **合計有效比較空間 ≈ 全叢集的 ~2^41.5（樂觀 ~2^42）**，與全叢集不拆分做 regeneration 同級（最佳 pair 必在同一半，故拆分不損分）。

> ⚠️ 上式假設「重算 hash 的成本 ≈ 產生成本」且 partition filter 幾乎免費。若 SHA kernel 實測比 [02](02-math-birthday-and-budget.md) 樂觀值慢（該檔已警告 H200 INT32 減半、可能只有 ~10~20 GH/s/GPU），N 會等比下修：N ∝ sqrt(產生率)。這是**最大單一不確定性**，務必先跑 kernel microbenchmark。

### 3.2 GPU sort 產能（sanity check 瓶頸端）

| 參考點 | 數值 | 來源 |
|---|---|---|
| `cub::DeviceRadixSort`，32-bit key，GTX Titan | 1.41 G keys/s | [cub-users](https://groups.google.com/g/cub-users/c/UHrtrIjNC90) |
| 改良 radix，256M × 32-bit key，A100 | 29.4 G keys/s | [Stehle & Jacobsen, arXiv:1611.01137](https://arxiv.org/abs/1611.01137) |
| 64-bit key-value（vs 32-bit）| 約慢 ~4× | 同上 |

【估計】H200 對 uint64 key 的 `DeviceRadixSort` 大約 **4~10 G keys/s/GPU**；16 GPU 聚合 ~**64~160 G keys/s ≈ 2^36~2^37/s**。600 s → 可 sort ~2^45~2^46 次 key-touch。因為 partition-by-regeneration 每個 key 只 sort 一次（在它的桶內），sort 總量 = N ≈ 2^42 << sort 預算 → **確認 sort 不會比產生更綁定**，瓶頸仍是「重算」的產生量。→ 支撐 3.1 的 N ≈ 2^42。

---

## Part 4 — NVMe（30TB RAID0）當 out-of-core 溢位：值得嗎？（幾乎不）

### 4.1 NVMe 頻寬事實

| 參考點 | 數值 | 來源 |
|---|---|---|
| 單顆 Gen4 NVMe 循序讀 | ~7 GB/s | [Spheron GDS guide](https://www.spheron.network/blog/gpu-direct-storage-nvme-ai-training-inference-guide/) |
| 4 顆 Gen4 循序讀 | ~28 GB/s | 同上 |
| 企業級 RAID array（HighPoint SSD7580 等）| 可達 ~28.5 GB/s | [DapuStor](https://en.dapustor.com/news/41.html) |
| GPUDirect Storage（cuFile/gdsio）單機實測 | ~37.5 GB/s；3-node ~50 GB/s（A100，H100 節點 16 NVMe 可 ~4× A100） | [Oracle GDS blog](https://blogs.oracle.com/cloud-infrastructure/accelerate-ai-ml-workloads-oci-nvidia-ibm) |

**本題 30TB RAID0 規劃頻寬【估計】≈ 30~50 GB/s 循序**（RAID0 實務只達 ~1.7× 而非 2× 線性 [[eTeknix/PC Tech](https://www.eteknix.com/year-nvme-raid-0-real-world-setup/)]；GDS 可繞過 CPU bounce 拉高）。取 **40 GB/s** 當規劃值。⚠️ 30TB 是**容量**不是**頻寬**，兩者常被混淆。

### 4.2 為何不值得——external sort 是 I/O-bound

10 分鐘內 NVMe 單向能吞 `40 GB/s × 600 s = 24 TB`。但 external merge sort 至少要 **write runs → read+write merge → read**，實務約 **4× 資料量的 I/O**。故可外排的資料量：

```
D ≤ (40 GB/s × 600 s) / 4 ≈ 6 TB ≈ 3.75×10^11 個 16 B key ≈ 2^38.5
```

**→ NVMe out-of-core 頂多讓 N ≈ 2^38，只比 HBM-only 的 2^37 多 ~1 bit，比 partition-by-regeneration 的 ~2^41.5 少 ~3~4 bit（少 ~7 分）**，而且：

1. 若真要存下 N=2^42 個 key = 4.4×10^12 × 16 B = **70 TB > 30TB 容量**，根本存不下。
2. 就算存得下，`70 TB × 4 / 40 GB/s ≈ 7000 s >> 600 s`，**I/O 綁死**。

### 4.3 crossover / 門檻

| 工作集大小 | 最佳佈局 |
|---|---|
| ≤ 2^37（≤ 2.25 TB，塞得進 HBM）| HBM-only 單趟 |
| 2^37 ~ 2^38.5 | 理論上 NVMe 有微弱優勢，但複雜度不划算 → **仍用 regeneration** |
| > 2^38.5 | **只能靠 partition-by-regeneration**（重算比 I/O 便宜） |

**判準一句話**：只要「重算一個 hash 的時間 < 把它寫出去再讀回來的 I/O 時間」，就永遠選重算。單位對齊（皆取 per-node）後：**單 node 產生 ~6×10^10 hash/s（[08] 校正中心值）vs 單 node NVMe 40 GB/s÷16B = 2.5×10^9 key/s I/O → 重算比 NVMe I/O 快 ~24×**（保守 ~16×、樂觀 ~32×）。⚠️ 原文寫「~100×」是拿**叢集**產生率（且用了 [02] 高估的 2.4×10^11）去比**單 node** I/O，單位不一致；改用同級單位並校正產生率後約為 **~20~30×**。**即使 H200 INT32 減半到最悲觀（見 [02]/[08]），此比值仍遠 ≫1 → NVMe 不用當計算路徑。**

**NVMe 的正當用途**（保留）：(a) checkpoint best-so-far（Part 6，資料量 tiny）、(b) 若 kernel 實測極慢導致重算變貴、工作集又恰落 2^37~2^38.5 的窄縫，才考慮；此為 fallback，非主線。

---

## Part 5 — NCCL over IB vs MPI

| 面向 | CUDA-aware MPI（OpenMPI/MPICH + UCX） | NCCL |
|---|---|---|
| 拓撲感知 | 需手動調 rail/NUMA binding | 自動選 NVLink + IB，內建 SHARP in-network reduction |
| 大 buffer 集合通訊 | 好 | **最佳**（8-node H100 NDR allreduce ~**350 GB/s** [[Spheron](https://www.spheron.network/blog/gpu-networking-infiniband-roce-spectrum-x-guide/)]）|
| 不規則 / variable-count | **`MPI_Alltoallv` 原生支援** | 需用 `ncclSend/Recv` group 手刻，無原生 alltoallv |
| 小訊息延遲 | sub-µs ~ 數 µs | 較高（為吞吐最佳化）|
| 容錯 / MAXLOC reduce | MPI 原生 `MPI_MAXLOC` | 無對應語意，要自己做 |

**本題建議**：
- **主線（SPLIT）根本不需要任何跨 node collective** → MPI 只用來 spawn/協調 + 最後一次 `MPI_MAXLOC` 合併兩 node 的 best pair（16 B，可忽略）。
- **若走 ROUTE（不建議）**：桶大小不均 → 需 `MPI_Alltoallv`（NCCL 無原生對應）→ **選 CUDA-aware OpenMPI + UCX**。NCCL 的優勢在規則的 allreduce/allgather，對本題的不規則洗牌**無用武之地**。
- 兩者都經 UCX/GPUDirect RDMA 走同一條 IB，底層頻寬相同；差別在**語意契合度**。

---

## Part 6 — 單次 10 分鐘 run 內的容錯 / robustness

10 分鐘 run + 可重複提交、取最佳 → 容錯需求**極低**，但仍值得做最低限度保險：

1. **best-so-far checkpoint**：全域最佳 pair 只有 (nonce_a, nonce_b, matched_bits) = 24 B。每 rank 每 ~10~30 s 把 local best 寫 host memory / NVMe（tiny，零效能影響）。run 意外中止時已有可提交結果。
2. **最後合併**：`MPI_Allreduce` 搭 `MPI_MAXLOC`（以 matched_bits 為 key）在收尾一次拿到全域最佳；成本可忽略。
3. **GPU/rank 掉一個不致命**：因為 SPLIT 下每 GPU 獨立掃自己的 partition 子集，掉一個只損失該子集覆蓋（少幾個桶 → N 略降、分數降 <1 bit），**不需重跑**。設計成「桶佇列 + 動態領取」可讓存活 GPU 接手。
4. **watchdog / 10 分鐘硬上限**：留 ~20~30 s 收尾邊界（flush、MAXLOC、寫 CSV），避免踩線被判超時。CSV 輸出格式為 frozen，務必最後才寫、且冪等。
5. **不要**在 run 中做需要跨 node 同步的 barrier-heavy 流程——單點慢節點（straggler）會拖垮全體；SPLIT 天然無此問題（兩 node 完全獨立直到收尾）。

---

## Part 7 — 運維注意事項（IB fabric 已預配）

- ⚠️ **不要啟動 `opensmd`（OpenSM subnet manager）**。題目已說明 IB fabric 已配置好；多開一個 SM 會與既有 SM 衝突、可能重整 subnet 造成全叢集抖動甚至斷線。只做**唯讀**檢查：`ibstat`、`ibv_devinfo`、`ibdiagnet`（唯讀）、`nvidia-smi topo -m`（確認 GPU↔HCA NUMA 配對）。
- NVIDIA + IB driver 已預裝 → **不要重裝** MLNX_OFED / DOCA，避免破壞既有 fabric 狀態。
- GPUDirect RDMA 需 `nvidia-peermem`（或新版 `nvidia-peerdirect`）module 已載入 → `lsmod | grep -E 'nvidia_peermem|nv_peer_mem'` 確認即可，別動設定。
- 多軌綁定：驗證 `UCX_NET_DEVICES=mlx5_0:1,...` 涵蓋全部 8 埠，以及每 GPU 綁到同 NUMA 的 HCA（`nvidia-smi topo -m` 看 `PIX/PXB` vs `SYS`）；**這是 ROUTE 若真要用時能否吃滿 ~350 GB/s 的關鍵**，但對 SPLIT 主線不影響。

---

## 總結 — 給比賽當下的決策

| 問題 | 答案 |
|---|---|
| 兩 node 要通訊嗎？ | **不要。** node A=top-bit0 / node B=top-bit1，各自獨立到收尾。零 alltoall。 |
| 這樣不是少 2 bit 嗎？ | **不會。** 「少 2 bit」只在產生受限時成立；本題 sort/重算受限，SPLIT 的 2× 產生落在便宜階段，且省下 alltoall 時間 → SPLIT ≥ ROUTE。crossover ≈ N≈2^41.3，本題可達 N 就在其上方。 |
| N 能到多少？ | **中心 ~2^41.5（分數 ~83），範圍 2^41~2^42（82~84）**，靠 partition-by-regeneration（記憶體與 N 解耦），不需儲存。【估計，隨 SHA kernel 實測率縮放；產生預算採 [08] 校正值 2^45.4~2^46.6，非 [02] 舊表的 2^47】 |
| 要用 NVMe out-of-core 嗎？ | **不要。** external sort I/O-bound，頂多 N≈2^38（+1 bit），且 N=2^42 需 70 TB > 30 TB 容量。重算比 I/O 快 ~20~30×（同級單位、校正後）。NVMe 只拿來 checkpoint。 |
| NCCL 還是 MPI？ | 主線兩者都幾乎不用；只用 `MPI_MAXLOC` 收尾。若被迫 ROUTE → CUDA-aware OpenMPI 的 `MPI_Alltoallv`（NCCL 無原生 alltoallv）。 |
| opensmd？ | **絕對不啟動**，fabric 已配置。 |

---

## 參考來源

- [NVIDIA ConnectX-7 NDR 400G datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/networking/infiniband-adapters/infiniband-connectx7-data-sheet.pdf)
- [Spheron — GPU Networking: InfiniBand vs RoCE vs Spectrum-X (2026)](https://www.spheron.network/blog/gpu-networking-infiniband-roce-spectrum-x-guide/)（NDR 延遲 0.9µs、8-node NDR allreduce ~350 GB/s、PCIe Gen5 64 GB/s）
- [NVIDIA — Benchmarking CUDA-Aware MPI](https://developer.nvidia.com/blog/benchmarking-cuda-aware-mpi/)（古董 FDR/K20 數字，僅作方法說明）
- [FAU — CUDA-Aware-MPI Part 1](https://blogs.fau.de/adityauj/2025/02/18/cuda-aware-mpi-part-1-understanding-node-topology-and-communication-bandwidth/)（H100 node 內 ~220 GB/s）
- [Open MPI 5.0.x — CUDA / GPUDirect docs](https://docs.open-mpi.org/en/v5.0.x/tuning-apps/networking/cuda.html)
- [Oracle — GPUDirect RDMA IB write bandwidth lab](https://docs.oracle.com/en/learn/gpudirect-rdma-ib-write-bw/index.html)（`ib_write_bw` 方法與 A100 樣本輸出）
- [Oracle — OCI + Magnum IO GPUDirect Storage](https://blogs.oracle.com/cloud-infrastructure/accelerate-ai-ml-workloads-oci-nvidia-ibm)（GDS ~37.5~50 GB/s）
- [Spheron — GPU Direct Storage NVMe guide (2026)](https://www.spheron.network/blog/gpu-direct-storage-nvme-ai-training-inference-guide/)（Gen4 NVMe ~7 GB/s、4 顆 ~28 GB/s）
- [DapuStor — enterprise NVMe RAID benchmark](https://en.dapustor.com/news/41.html)（~28.5 GB/s array）
- [eTeknix — NVMe RAID0 real-world](https://www.eteknix.com/year-nvme-raid-0-real-world-setup/)（RAID0 實務 ~1.7× 線性）
- [Stehle & Jacobsen — Memory Bandwidth-Efficient Hybrid Radix Sort on GPUs, arXiv:1611.01137](https://arxiv.org/abs/1611.01137)（A100 29.4 Gkey/s；64-bit ~4× 慢）
- [cub-users — DeviceRadixSort speed](https://groups.google.com/g/cub-users/c/UHrtrIjNC90)

> 數字警語：本文所有跨 node / I/O / sort 的絕對數字皆為**上界推導 + 文獻類比**，未經本機實測。三個必做 microbenchmark：(1) `ib_write_bw`/`nccl-tests` 量 2-node 多軌 bisection；(2) `gdsio` 量 30TB RAID0 GDS 頻寬；(3) `cub::DeviceRadixSort` 量 H200 uint64 key/s。任一與此處差 2× 以上，Part 3 的 N 與 Part 2 的 crossover 都要等比重算。
