# 06 — 單節點 8× H200 的多 GPU 管線

> 本文只談**單一 node 內把 8 張 H200 榨乾**。跨 node（IB、GPUDirect RDMA、雙節點 all-to-all）在 [06-multinode-and-io.md](06-multinode-and-io.md)；SHA kernel 常數在 [05-sha256-kernel-optimization.md](05-sha256-kernel-optimization.md)；為什麼瓶頸是 sort/closest-pair 而非 hash 生成，見 [02-math-birthday-and-budget.md](02-math-birthday-and-budget.md) 與 [04-algorithm-memory-birthday.md](04-algorithm-memory-birthday.md)。

本題已定案為 **distributed GPU sort / closest-pair 吞吐問題**：分數 ≈ `2·log2(N)`，N = 10 分鐘內能「生成並互相比較」的 distinct hash 數。單節點的工程目標只有一句話：**讓 8 張 H200 共同貢獻到同一個有效 N，且 partition / sort 的資料流不要被 GPU 間搬移拖垮。**

---

## TL;DR

1. **控制平面用 OpenMP，一個 CPU thread 綁一張 GPU**（`#pragma omp parallel num_threads(8)` + `cudaSetDevice`）。這是題目本身就暗示的最小結構，比 8-rank MPI 在單節點更省事、共享 host pinned buffer 更自然。
2. **資料平面：radix by top-p hash bits，第 i 張 GPU 擁有 partition i。** 需要一次 **all-to-all shuffle** 把每張卡生成的 hash 路由到「擁有它 top-bits」的那張卡。NVSwitch 全互連讓這件事在節點內幾乎免費。
3. **NVLink 4 / NVSwitch gen3**：每 GPU 900 GB/s 雙向、8-GPU baseboard bisection **3.6 TB/s**（SOURCED）。P2P copy 走 NVLink，不碰 PCIe/CPU。
4. **route（NVLink all-to-all，保留全部 N）vs discard（每卡只留自己 partition、其餘丟掉靠重生）**：在本機參數下 **route 完勝**。crossover 落在 key size `b* ≈ 8·BW_eff/g ≈ 180 bytes`，而我們 key 只有 16~40 bytes，遠在 route 一側。⚠️ 這結論對 `g`（生成率）與 `BW_eff` 敏感，見 §4。
5. HBM 141GB/卡 放不下單一 partition 的全部 key → partition 深度 `p` 要開到讓 `N/2^p` 塞得進 HBM，或 sort 走 out-of-core（NVMe）。overlap：GPU i 生成的同時 GPU j 在 sort（不同 stream / 不同卡天然平行）。
6. MPS / multi-stream 用來讓「生成 kernel」與「sort/scan kernel」在同卡上 concurrent，填滿 SM occupancy 的空隙。

---

## 1. 控制結構：OpenMP one-thread-per-GPU

CUDA runtime 是 **per-thread state**：每條 host thread 各自維護「current device」，`cudaSetDevice()` 之後該 thread 的所有 API 呼叫都導向那張卡，直到再次呼叫（SOURCED，見 [CUDA Programming Guide §Multi-GPU](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)）。經典陷阱：spawn 出來的 thread 忘了 `cudaSetDevice`，全跑到 device 0（SOURCED，[CUDA Pro Tip: Always Set the Current Device](https://developer.nvidia.com/blog/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs)）。

標準骨架（**這是本題該用的最小結構**，<10 行示意）：

```c
#pragma omp parallel num_threads(8)          // 一 thread 綁一卡
{
    int dev = omp_get_thread_num();
    cudaSetDevice(dev);                       // 每 thread 必做，否則全跑 device 0
    cudaStream_t s; cudaStreamCreate(&s);
    generate_hashes<<<grid, blk, 0, s>>>(...);   // 本卡 nonce slice → hash
    partition_by_top_bits<<<...,0,s>>>(...);     // 本地 radix 分桶
    #pragma omp barrier                       // 全卡生成完才進 shuffle
    nccl_all_to_all(sendbuf, recvbuf, s);     // 路由到 owner GPU（§3）
    sort_and_closest_pair<<<...,0,s>>>(...);   // 本卡對自己 partition 做 closest-pair
}
```

- **為何 OpenMP 而非 MPI（單節點內）**：8 條 thread 共用同一位址空間 → host pinned staging buffer、NCCL communicator、結果彙整都不用跨 process。跨 node 才需要 MPI（見 [06-multinode-and-io.md](06-multinode-and-io.md)，屆時常見做法是 **MPI × node，OpenMP/NCCL × node 內 GPU**）。
- **kernel launch 非阻塞** → 一條 thread 可以連續對自己那張卡塞多個 async 呼叫再去做別的；不同卡天然 overlap（SOURCED，OLCF Multi-GPU workshop）。
- 每卡各建 stream；跨卡同步只在需要交換資料時用 `#pragma omp barrier` + `cudaStreamSynchronize` 或 NCCL 的 stream 序。

---

## 2. NVLink 4 / NVSwitch 拓撲與 P2P

### 2.1 頻寬事實（SOURCED）

H200 SXM 與 H100 SXM 用**同一套 NVLink 4 / NVSwitch gen3**（H200 只換 HBM，互連不變）。

| 項目 | 數值 | 來源 |
|---|---|---|
| NVLink 4 links / GPU | 18 條 | [fibermall NVLink evolution](https://www.fibermall.com/blog/nvidia-nvlink-and-nvswitch-evolution.htm) |
| 每 link 雙向頻寬 | 50 GB/s（25 GB/s ×2 方向） | 同上 |
| 每 GPU 總 NVLink 頻寬 | **900 GB/s 雙向**（≈450 GB/s 單向） | [Lenovo Press lp1944](https://lenovopress.lenovo.com/lp1944-nvidia-h200-141gb-gpu) |
| 8-GPU HGX baseboard bisection | **3.6 TB/s** | [NVSwitch HotChips 2022](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf) |
| NVSwitch gen3 chip / baseboard | 4 顆，全互連 full-mesh | 同上 |

**關鍵拓撲事實**：8 張卡透過 NVSwitch 形成 **full all-to-all**，任兩卡之間都是 900 GB/s，且**所有卡可同時對所有卡讀寫**、不經 CPU/PCIe（SOURCED，[runpod H200 guide](https://www.runpod.io/articles/guides/nvidia-h200-gpu)）。這正是我們 shuffle 想要的形狀——不是 ring、不是 tree，是真 crossbar。

### 2.2 P2P 直接複製

- `cudaDeviceCanAccessPeer()` 查詢 → `cudaDeviceEnablePeerAccess()` 開啟後，一張卡的 kernel 可直接讀另一張卡的 global memory，或 `cudaMemcpyPeerAsync()` 走 NVLink 直達（SOURCED，[CUDA Programming Guide Multi-GPU](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)）。
- 非 NVSwitch 系統每卡最多 8 條 peer 連線；**HGX/NVSwitch 系統無此限制**（fabric 代管），8 卡可同時全互連（SOURCED，同上）。
- 對「少量、規則」的 pairwise 搬移，手寫 `cudaMemcpyPeerAsync` 就夠；**但 all-to-all shuffle 請用 NCCL**（§3），因為 NCCL 會做 chunk / pipeline / 多 link 聚合，手刻很難打平。

> ⚠️ **估計，非量測**：900 GB/s 是 peak。實際 all-to-all pattern 因為 8×7 條流爭用 NVSwitch，per-GPU 有效吞吐通常掉到 peak 的 35~55%。拿到機器請跑 `nccl-tests` 的 `alltoall_perf` 量真值（見 §7）。

---

## 3. NCCL 做 all-to-all shuffle

### 3.1 為什麼需要 shuffle

radix 策略：取 hash 的 top-`p` bits 當 partition id，partition i 歸 GPU i 所有（8 卡 → `p≥3`）。**全域最佳 pair 保證落在同一 partition**（它們 top bits 相同才會前導位元多），所以只要每張卡在自己 partition 內做 closest-pair，就不會漏掉全域答案（見 [04-algorithm-memory-birthday.md](04-algorithm-memory-birthday.md) 的 partition 正確性論證）。

但每張卡是從自己的 nonce slice 生成 hash，生成出來的 top-bits 是**均勻隨機**的 → 一張卡手上約 `1/8` 屬於自己、`7/8` 屬於別人。要把這 `7/8` 送到 owner → 這就是 **all-to-all shuffle**。

### 3.2 NCCL 實作模式（SOURCED）

NCCL 2.8+ 用 `ncclSend`/`ncclRecv` 包在 `ncclGroupStart/End` 內來組出 AllToAll[v]（SOURCED，[NCCL P2P docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html)、[NVIDIA/nccl PR#316](https://github.com/NVIDIA/nccl/pull/316)）。官方 all-to-all 樣板：

```c
ncclGroupStart();
for (int r = 0; r < nranks; r++) {          // nranks = 8
  ncclSend(sendbuff[r], counts[r], ncclUint64, r, comm, stream);
  ncclRecv(recvbuff[r], rcounts[r], ncclUint64, r, comm, stream);
}
ncclGroupEnd();                             // group 內對不同 peer 併發、不互相阻塞
```

- 桶大小不等 → 用 **AllToAll_v**（counts 每個 rank 不同）。先做一次 `ncclAllGather` 交換各桶 count，才知道 recv 佈局。
- group 內對「不同 peer」的呼叫可獨立併發推進；對「同一 peer」的多個呼叫保序（SOURCED，NCCL P2P docs）。
- **repo / license**：[github.com/NVIDIA/nccl](https://github.com/NVIDIA/nccl)，**BSD-3-Clause**；測速工具 [github.com/NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests)。

### 3.3 預期頻寬

| 指標 | 值 | 來源 / 註 |
|---|---|---|
| H100/H200 NVLink4 AllToAll bus bandwidth（nccl-tests） | ~340 GB/s | ⚠️ 二手，[GTC S51111 / NCCL 講義](https://juser.fz-juelich.de/record/1019178/files/02-NCCL_NVSHMEM.pdf) 類數據 |
| all-to-all 的本質 | **受 bisection bandwidth 主宰，是最難 scale 的 collective** | SOURCED，[Demystifying NCCL, arXiv:2507.04786](https://arxiv.org/html/2507.04786v1) |

> ⚠️ **本文規劃基準**：節點內 all-to-all 取 **per-GPU 有效單向 ≈ 300~350 GB/s**（`BW_eff`）。這是保守中間值；必量測校正。

---

## 4. route vs discard：核心工程抉擇

兩條路都能讓每張卡最終持有 `N/8` 個「屬於自己 partition」的 distinct hash（全系統合計 N，最後每卡本地 closest-pair）：

- **route**：8 卡各生成 `N/8`（合計生成 N），做一次 all-to-all 把錯位的 `7/8` 送到 owner。**保留全部 N，付出 shuffle 通訊。**
- **discard / regenerate**：每卡只留生成結果中屬於自己 partition 的 `1/8`，其餘丟掉。要湊滿 `N/8` 保留量，每卡得生成 `N`（合計生成 **8N**）。**零通訊，付出 8× 生成。**

### 4.1 crossover 公式（YOUR ESTIMATE）

設 `g` = 每卡生成率（hash/s）、`BW_eff` = 每卡 all-to-all 有效單向頻寬（B/s）、`b` = 每 key 位元組數。以「共同基線（各卡最終持有 N/8）」之外的**額外成本**比較：

| 方法 | 額外成本（每卡） |
|---|---|
| discard | 多生成 `7N/8` → 時間 `(7/8)·N/g` |
| route | shuffle 送出 `7N/64` keys → 時間 `(7N·b/64)/BW_eff` |

令兩者相等，解出 crossover key size：

```
b* = 8 · BW_eff / g
```

- `b < b*` → **route 較快**；`b > b*` → discard 較快。
- 代入 `BW_eff = 3.3×10^11 B/s`、`g = 1.5×10^10 hash/s`（[02](02-math-birthday-and-budget.md) 樂觀基準）：
  **`b* ≈ 8 × 3.3e11 / 1.5e10 ≈ 176 bytes`。**
- 我們的 key 只有 **16~40 bytes**（截斷 hash + 8-byte nonce）→ **遠在 route 一側，route 完勝。**

### 4.2 具體時間（N = 2^40 ≈ 1.1×10^12，b = 40 B，8 卡）

| 方法 | 每卡額外工作 | 時間 |
|---|---|---|
| route（shuffle） | 送出 `7N/64 ≈ 1.2×10^11` keys × 40B = 4.8 TB | 4.8e12 / 3.3e11 ≈ **~15 s** |
| discard（重生） | 多生成 `7N/8 ≈ 9.6×10^11` hash | 9.6e11 / 1.5e10 ≈ **~64 s** |

在 600 s 總預算下，route 的 shuffle 只吃 ~2.5%，discard 的重生吃 ~11%。**route 明顯划算，且它保留全部 N（分數直接受 N 影響）。**

### 4.3 敏感度（何時 discard 反而好）— ⚠️ 全為估計

- 若 SHA 生成比預期慢很多（`g` 掉到 ~2×10^9，即 [02](02-math-birthday-and-budget.md) 警告的「H200 INT32 減半、可能輸 4090」情境）→ `b*` 升到 ~1300 bytes，route 更是唯一解（重生變超貴）。
- 若 NVLink 被別的流量佔滿導致 `BW_eff` 崩到 ~30 GB/s → `b*` 掉到 ~16 bytes，才勉強接近我們的 key size；此時可考慮 **hybrid**：本地先 dedup + 只 route 桶邊界附近的 key。
- **結論：在任何合理參數下，單節點內都該 route（NVLink 太快、生成沒那麼便宜到能 8×）。** discard 只在「跨 node、IB 頻寬遠低於 NVLink」時才需要重新評估 → 見 [06-multinode-and-io.md](06-multinode-and-io.md)。

---

## 5. 負載平衡、HBM 預算、overlap

### 5.1 每卡 HBM 預算（141 GB）

| key 格式 | bytes | 141GB 可容 keys | 對應 log2 |
|---|---:|---:|---:|
| 截斷 hash(8B) + nonce(8B) | 16 | ~8.8×10^9 | 2^33.0 |
| hash(32B) + nonce(8B) | 40 | ~3.5×10^9 | 2^31.7 |

單一 partition 的目標量 `N/8`（若 N=2^40 則 `≈1.4×10^11`）**遠超單卡 HBM**。兩條出路：
1. **加深 partition**：`p` 開到讓 `N/2^p` ≤ HBM 容量，再把多個 sub-partition 依序在同卡上跑（時間換空間）。partition 把 memory 從 N 解耦（見 [04](04-algorithm-memory-birthday.md)）。
2. **out-of-core sort**：HBM 放不下就 spill 到 host RAM / NVMe RAID0（30TB），走 [06-multinode-and-io.md](06-multinode-and-io.md) 的 I/O 路線。
- **建議：先把 p 開到「單卡單趟塞得進 HBM」**，讓 sort 全程 on-HBM（4.8 TB/s 頻寬，radix sort 才快），避免 spill。

### 5.2 load balancing

- top-p bits 對密碼學 hash 是均勻的 → **桶天然等大**（相對誤差 ~`1/sqrt(N/2^p)`，N 大時可忽略）。不需要動態 rebalance。
- 若 N 不是 8 的整除、或想用 `p>3` 再把 sub-bucket round-robin 給 8 卡：靜態分配即可，因為每桶期望大小相同。
- 唯一要防的是 **skew 來自 nonce slice 切法**，不是 hash 分佈——把 64-bit nonce 空間平均切 8 段給 8 卡即可，每段生成量相同。

### 5.3 overlap generation 與 sort（跨卡與同卡）

- **跨卡天然 overlap**：kernel launch 非阻塞，GPU i 在 sort 的同時 GPU j 可在 generate（不同 thread、不同卡）。用 double-buffer + `omp barrier` 只在 shuffle 邊界對齊。
- **同卡 overlap**：generate（INT32/ALU-bound）與 radix sort 的 scatter（memory-bound）資源互補 → 放不同 stream 可部分重疊，吃滿 SM 空隙。
- **pipeline 分段**：把 N 切成多個 chunk，chunk k 在 sort 時 chunk k+1 在 generate、chunk k-1 在 shuffle → 三段流水線，隱藏 shuffle 的 ~15 s。

---

## 6. MPS / streams 併發

- **multi-stream（首選、零額外部署）**：同一 process 內每卡開多條 stream，讓 generate / partition / sort kernel 併發，填 occupancy 空隙。本題單 process 8-thread 8-GPU，multi-stream 已足夠。
- **MPS（Multi-Process Service）**：讓**多個 process** 的 kernel block 空間共享同一張 GPU、甚至共用 SM（SOURCED，[NVIDIA MPS docs / Medium 說明](https://sagar-parmar.medium.com/demystifying-nvidia-mps-how-multi-process-service-improves-gpu-sharing-and-performance-9f633878318a)）。可用 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` 限制單 client 佔比（SOURCED，同上）。
  - 對本題**價值有限**：我們是單 process 多 thread 架構，kernel 夠大就能填滿 GPU，不太需要跨 process 拼併發。
  - **MPS 唯一可能用途**：若改成「每 GPU 一個獨立 process」的 MPI 風格、且各 process 的 kernel 都偏小 → MPS 避免 context time-slicing 的浪費（SOURCED，同上）。
  - ⚠️ 注意 MPS **不提供跨 process 記憶體保護 / error isolation**（SOURCED，同上）；競賽短跑可接受。
- **建議**：先 multi-stream + 大 kernel 榨滿單卡；只有量到「單卡 occupancy 上不去、kernel 太小」時才引入 MPS。

---

## 7. 拿到機器後必量（microbenchmark）

| # | 量什麼 | 工具 | 為何 |
|---|---|---|---|
| 1 | 單卡 SHA-256 生成率 `g` | 自寫 kernel | 決定 §4 的 `b*`、決定 route/discard（[02](02-math-birthday-and-budget.md) 警告 H200 可能很慢） |
| 2 | 8-GPU all-to-all `BW_eff` | `nccl-tests/alltoall_perf` | 決定 shuffle 時間、驗證 3.6 TB/s bisection 落地率 |
| 3 | 單卡 on-HBM radix sort keys/s | cub/CUB `DeviceRadixSort` | 決定真正瓶頸段 |
| 4 | P2P `cudaMemcpyPeerAsync` 實測 GB/s | `p2pBandwidthLatencyTest`（CUDA samples） | 驗證 NVLink 直達 |
| 5 | generate/sort 同卡 overlap 收益 | Nsight Systems | 決定要不要 pipeline / MPS |

---

## 參考來源

**SOURCED（附 URL）**
- NVLink4 18 links / 50 GB/s per link：[fibermall — NVLink & NVSwitch Evolution](https://www.fibermall.com/blog/nvidia-nvlink-and-nvswitch-evolution.htm)
- H200 900 GB/s NVLink：[Lenovo Press lp1944 — ThinkSystem H200](https://lenovopress.lenovo.com/lp1944-nvidia-h200-141gb-gpu)
- 8-GPU 3.6 TB/s bisection / NVSwitch gen3：[NVSwitch HotChips 2022](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf)
- NVSwitch full all-to-all、無 CPU/PCIe：[RunPod — NVIDIA H200 Guide](https://www.runpod.io/articles/guides/nvidia-h200-gpu)
- CUDA 多 GPU / per-thread device state / peer access：[CUDA Programming Guide — Multi-GPU Systems](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)
- cudaSetDevice per-thread 陷阱：[NVIDIA — CUDA Pro Tip: Always Set the Current Device](https://developer.nvidia.com/blog/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs)
- NCCL all-to-all（ncclSend/Recv + Group）：[NCCL User Guide — Point-to-point](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html)、[NVIDIA/nccl PR#316](https://github.com/NVIDIA/nccl/pull/316)
- NCCL repo（BSD-3-Clause）/ 測速：[github.com/NVIDIA/nccl](https://github.com/NVIDIA/nccl)、[github.com/NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests)
- all-to-all 受 bisection 主宰：[Demystifying NCCL, arXiv:2507.04786](https://arxiv.org/html/2507.04786v1)
- NCCL 講義（AllToAll bus bw 類數據）：[FZ-Jülich NCCL/NVSHMEM 講義](https://juser.fz-juelich.de/record/1019178/files/02-NCCL_NVSHMEM.pdf)
- MPS 併發 / thread percentage / 無記憶體保護：[Demystifying NVIDIA MPS](https://sagar-parmar.medium.com/demystifying-nvidia-mps-how-multi-process-service-improves-gpu-sharing-and-performance-9f633878318a)
- OpenMP one-thread-per-GPU、kernel launch 非阻塞：[OLCF — Programming Multi-GPU Nodes](https://www.olcf.ornl.gov/wp-content/uploads/2018/11/multi-gpu-workshop.pdf)

**YOUR ESTIMATE / ⚠️ 需量測**
- `BW_eff` 300~350 GB/s（peak 900 的 35~55%）；`g` = 1.5×10^10 hash/s（[02](02-math-birthday-and-budget.md) 樂觀端，H200 可能大幅偏低）；crossover `b* ≈ 176 bytes`；§4.2 的 15 s / 64 s 具體時間；§5.1 HBM 容量換算。以上皆須用 §7 清單校正（誤差可能 2~3×）。
