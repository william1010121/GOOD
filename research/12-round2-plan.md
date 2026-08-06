# 12 — 第二輪計畫：從 ~76 衝到 80 bits

> 本文接在隊友 Job 496 / 503（`research.md`）與本輪改寫之後。
> 目標：定位目前的瓶頸、算出通往 80 的缺口、給出唯一有效的那條路。

---

## 1. 先把分數公式釘死

設每次能「同時排序」的筆數 = **H**，時限內全叢集能生成的雜湊數 = **G**：

```
期望最長共同前綴 ≈ log₂(H) + log₂(G) − 1
```

**這不只是一個經驗式，它是給定（記憶體 H、吞吐 G）之下的上界。** 論證：
流過去的 G 個雜湊，每一個最多只能跟當下常駐的 H 筆比對，
所以可檢查的配對總數 ≤ `G × H`，期望最佳前綴 = `log₂(G×H)`。
排序（每筆進桶排一次、掃相鄰對）剛好達到這個上界。

推論：**沒有演算法可以繞過它。唯二的槓桿是 H 和 G。**
partition-by-regeneration 的價值不是「更聰明」，而是它讓峰值記憶體與掃描量脫鉤，
使得 H 可以就是「單次排序容量」而不是「總資料量」。

---

## 2. 目前的瓶頸在哪（用隊友的實測數字反推）

### 2.1 舊版把「生成」和「排序」綁成 1:1

Job 503 每個 100M batch 的流程是：算完 100M 個雜湊 → **全部**寫進 HBM →
兩次 stable radix sort（先 lo 後 hi）→ GPU reduce。

估算單張 H200 的每批耗時：

| 階段 | 資料量 | 估時 |
|---|---|---|
| hash 100M（10.5 GH/s） | 寫 100M × 24 B = 2.4 GB | ~10 ms |
| SortPairs #1（key 8 B + value 16 B，8 passes） | `8×2×24 B×1e8` = 38 GB | ~17 ms |
| SortPairs #2 | 同上 | ~17 ms |
| pack / reduce | 幾趟 2.4 GB | ~5 ms |
| **合計** | | **~50 ms → 2.0 GH/s** |

**實測 2.23 GH/s**（Job 503 per-rank completed-batch）。模型吻合。

→ **排序吃掉約 70% 的時間，真正在算雜湊的只有 20%。**
原因很單純：**產生多少就排序多少**。

### 2.2 分數還被「池子被切碎」再砍一刀

Job 496 評估了 9.66×10¹¹ 個 nonce。若這些是**同一個池子**，
期望分數 = `2·log₂(9.66e11) − 1 ≈ 78.6`。實得 **64**，少了約 14.6 bits。

隊友自己在 `research.md` 已經指出原因：

> 目前的 qbit 是「所有 rank local best 中的最大值」，不是所有 nonce 合併後的
> exact global best…要做到 exact merge，必須把各 rank 的排序 hash/candidate
> 依 prefix 分區後交換。

每個 rank、每個 batch 各排各的，batch 之間、rank 之間從來沒有互相比較過。
192 個獨立的 100M 小池子，不是一個 1.9×10¹¹ 的大池子。

---

## 3. 本輪（第一輪）解了什麼

| 問題 | 解法 | 效果 |
|---|---|---|
| 排序吃 70% 時間 | **生成+過濾融合成單一 kernel**：只留落在本 partition 的 1/P，排序量變成生成量的 1/P | H200 上 P≈256 ⇒ 排序降到約 0.5%，有效吞吐 2.23 → ~10.4 GH/s |
| 池子被切碎 | **各 rank 掃同一段 nonce、擁有不同的 hash partition** | 全域最佳那對共同前綴 ≫ p ⇒ 前 p bits 必相同 ⇒ 必落在同一個 partition ⇒ **由唯一一個 rank 獨自持有** |
| process 重啟開銷 | 單一長時間 process，一個 node 只付一次 CUDA context 成本 | Job 491 的 4.44 MH/s 端到端 → 消失 |

第三列特別值得講：**這個設計讓 exact cross-rank merge 這件事直接不存在。**
不需要交換任何 key，因為 prefix 分區在生成的當下就做完了。
隊友文件裡說的「必須依 prefix 分區後交換」——我們把分區提前到最前面，交換就免了。

### 預期分數（285 秒、16×H200、H = 3.67 G 筆/桶）

```
log₂(G) = log₂(16 × 10.4e9 × 285) = 45.5
log₂(H) = log₂(3.67e9)            = 31.8
分數 ≈ 31.8 + 45.5 − 1 ≈ 76.3
```

對照隊友目前的 64 → **約 +12 bits**。

---

## 4. 通往 80 的缺口

要 80 bits 需要 `log₂(H) + log₂(G) = 81`。

G 這一側幾乎榨乾了：16 張 H200 的 SHA-256 峰值是硬體給定的，
**時間翻倍只值 +1 bit**（285 s → 570 s 只有 +1.0）。

所以缺口全在 H：

| H 的來源 | log₂(H) | 285 s 的分數 |
|---|---:|---:|
| 單卡 HBM（現況） | 31.8 | 76.3 |
| **節點內 8 卡 NVLink 協同排序** | **34.9** | **79.4** |
| 再加跨節點 16 卡（走 IB） | 35.9 | 80.4 |

**結論：第二輪唯一該做的事是節點內 8 卡協同排序（+3.1 bits）。**

---

## 5. 第二輪設計：節點內 8 卡協同排序

### 5.1 資料流（每個 partition）

```
① 8 張卡各掃 span 的 1/8，融合 kernel 過濾出屬於本 partition 的 key
   → 每卡手上 n 筆，但這 n 筆的「次高位 bits」是均勻散布的
② 依 key 的接下來 3 bits 做本地 radix 分桶（8 個 sub-bucket）
③ 一次 NVLink P2P all-to-all：sub-bucket i 全部送到 GPU i
   → 每卡收到 ~n 筆，且這 n 筆共享相同的 (partition prefix + 3 bits)
④ 每卡對收到的 n 筆做 cub::DeviceRadixSort（純本地）
⑤ 每卡掃相鄰對 + 交換 sub-bucket 邊界的一筆，取全域最佳
```

正確性和單卡版一樣：③ 之後，共享 `p+3` bits 的 key 全部落在同一張卡上，
而最佳對的共同前綴 ≫ p+3，所以它一定在某張卡的本地資料裡。

### 5.2 記憶體帳（單卡 143.8 GB，實測 `nvidia-smi` 給 143771 MiB）

只需要兩個 n×16 B 的緩衝區，排序時當 ping-pong 重用：

| 用途 | 大小 |
|---|---|
| Buffer A：本地生成+過濾的結果 | n × 16 B |
| Buffer B：all-to-all 收到的資料 | n × 16 B |
| 排序 | A/B 互當 ping-pong，不另外配 |

```
n = 143.8e9 × 0.88 / 32 ≈ 3.95 G 筆/卡
H = 8 × 3.95e9 = 3.16e10  →  log₂(H) = 34.9      （+3.1 bits）
```

### 5.3 all-to-all 成本

每卡送出 `7/8 × 3.95e9 × 16 B ≈ 55 GB`。NVSwitch 全互連、
per-GPU 有效單向頻寬保守取 300 GB/s → **約 0.18 秒**。

同一個 partition 的生成時間 = `S / (8 × 10.4 GH/s)`，
以 P≈64、S = P×H = 2.0e12 計 ≈ **24 秒**。

→ **shuffle 佔不到 1%**，完全划算。這也是研究 [06](06-multigpu-intranode.md) §4
「route vs discard」算出 crossover 在 176 bytes/key、而我們只有 16 bytes 的直接印證。

### 5.4 實作要點

- 控制結構不變：一個 process 一個 node，OpenMP 一 thread 一卡（已經是現況）。
- P2P：`cudaDeviceCanAccessPeer` → `cudaDeviceEnablePeerAccess`，
  然後 `cudaMemcpyPeerAsync`。8 卡規則搬移手寫就夠，不必引入 NCCL。
- 分桶用「histogram + exclusive scan + scatter」一趟做完（p'=3，只有 8 個桶）。
- 需要一次 `#pragma omp barrier` 對齊 shuffle 前後。
- 桶大小天然等大（密碼學雜湊高位均勻），相對誤差 ~`1/√n` ≈ 1.6e-5，不需要動態平衡。

**拓樸已確認（2026-08-06 實測）**：`nvidia-smi topo -m` 顯示節點內
**任兩張 GPU 之間都是 `NV18`**，即 NVSwitch 全互連、每對 18 條 bonded NVLink。
沒有任何一對走 PCIe/SYS。→ **8 卡協同排序的前提成立，這條路可以直接做。**

剩下要量的只有實際 P2P 頻寬（CUDA sample 的 `p2pBandwidthLatencyTest`），
用來把 §5.3 的 300 GB/s 保守估計換成真值。

另外從同一份 topo 看到：每張 GPU 各自 `PIX` 對應一張 mlx5 NIC
（GPU0↔NIC0、GPU1↔NIC3、GPU2↔NIC4、GPU3↔NIC5、GPU4↔NIC6…），
共 10 張 NIC。若之後要做跨節點協同（§7 選配），
GPU↔HCA 的 NUMA 配對已經是理想的 1:1，多軌能吃滿。

---

## 6. 幾乎免費的 +1.5～2.5 bits：best-of-k

分數是 Gumbel 分佈，**標準差約 1.9 bits**。同樣的 H 和 G 多跑幾輪、
每輪用不同的 `--start`（完全獨立的 nonce 範圍），取最好的一次：

| 輪數 k | 期望增益 |
|---|---|
| 2 | +1.1 |
| 3 | +1.6 |
| 5 | +2.2 |

**這需要的程式改動是零**——`--start` 已經實作，Slurm 的 `NONCE_START` 也已經暴露。
只要換個值重送就是一輪全新的獨立搜尋，而且比賽本來就取最佳的一次提交。

---

## 7. 第二輪的路線圖

| 步驟 | 增益 | 累計 | 風險 |
|---|---:|---:|---|
| 本輪（單卡分桶，285 s） | — | ~76 | 已驗證 |
| 8 卡 NVLink 協同排序 | +3.1 | ~79.4 | 中：要寫 P2P shuffle |
| best-of-3（換 `--start` 重送） | +1.6 | **~81** | 無：零改動 |
| （選配）跨節點 16 卡協同 | +1.0 | ~82 | 高：要 MPI + IB |
| （選配）爭取 10 分鐘視窗 | +1.0 | — | 看主辦方 |

**先做 best-of-k**（零成本，先把 ~76 的分數用 3 輪推到 ~78），
**再做 8 卡協同排序**（真正的 +3）。兩者相加穩過 80。

---

## 8. 本輪要從 log 帶回來的三個數字

第二輪的設計原本有三個未知數，其中拓樸已經確認，剩兩個：

1. **H200 實際可用 HBM 給出的 H** — 預期 3.6～3.9 G 筆。決定 §5.2 的記憶體帳。
2. **融合 kernel 的實際 GH/s** — 預期 ~10 GH/s（純 bench 是 10.5）。
   若明顯低於 10，代表 warp-aggregated atomic 的 scatter 寫入有爭用，要先修。
3. ~~**`nvidia-smi topo -m` 的拓樸**~~ — **已確認：全 `NV18`，NVSwitch 全互連。**
   8 卡協同排序沒有拓樸上的阻礙。

程式收尾會直接印出「以目前 G，要 80 bits 需要 log₂(H) = ?」，
不用手算就知道缺口。

---

> 數字來源：G 用隊友 Job 480 實測的單卡 10.32–10.83 GH/s；
> H 用 `nvidia-smi` 實測的 143771 MiB；排序時間用 bandwidth 模型
> 並已與 Job 503 的 2.23 GH/s 對過（誤差 <12%）。
> NVLink 有效頻寬 300 GB/s 為保守估計，**未實測**，須用
> `p2pBandwidthLatencyTest` 校正。
