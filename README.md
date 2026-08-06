# 隱藏題1：SHA-256 部分碰撞搜尋

## 題目

對每個 64 位元整數 `nonce` 定義 `hash(nonce) = SHA-256( prefix(ASCII) ‖ nonce(8 bytes 大端) )`，
找出兩個**不同的** nonce，讓兩個雜湊**開頭相同的 bits 最多**。正式 prefix 是 `HiPAC2026crypto`。

計分：共同前綴達 **58 bits** 得基礎分 4 分，未達為 0 分；再依 bits 排名加 0～6 分。

---

## 怎麼跑

```bash
make
sbatch run_collision.slurm          # 2 nodes × 8 GPU，跑 240 秒
```

結果會在提交目錄產生 `solution_<bits>.csv`，並自動用 `verify_collision.py` 驗過。

可調參數：

```bash
sbatch --export=ALL,SEARCH_SECONDS=240,PREFIX=HiPAC2026crypto,NONCE_START=0 run_collision.slurm
```

- `SEARCH_SECONDS` — 生成階段秒數。Slurm 上限設在 5 分鐘，預設 240 秒留收尾邊際。
- `NONCE_START` — 換一個值就是一輪**完全獨立**的搜尋。分數的標準差約 1.9 bits，
  多跑幾輪取最好可以多撈 2～3 bits。

單機直接跑（會用光本機所有 GPU）：

```bash
./collision --seconds 240 HiPAC2026crypto
./collision --bench HiPAC2026crypto      # 只量 SHA-256 吞吐
```

---

## 演算法

分數 ≈ `2·log₂(N)`，N 是能互相比對的雜湊數。記憶體裝不下 N，
所以用 **partition-by-regeneration**：

1. 把雜湊空間依「前 p bits」切成 `P = 2^p` 個 partition。
2. 每個 partition 重掃整段 nonce 範圍，**只留落在該桶的**，在 GPU 內排序、掃相鄰對。
3. 換下一個 partition。

正確性：全域最佳那對的共同前綴遠大於 p，所以它們的前 p bits 必然相同、必在同一桶，
分桶不會漏掉答案。

設每桶能容納 H 筆、時限內全叢集能生成 G 個雜湊：

```
期望分數 ≈ log₂(H) + log₂(G) − 1
```

這個式子**與掃描範圍 S 無關**（只要 S 大到 partition 跑不完），所以猜錯 GPU 速度
不會毀掉這一趟，只會少做幾個桶。程式啟動時會自己量吞吐、自己決定 P 和 S。

### 相對範本改了什麼

| # | 改動 | 為什麼 |
|---|---|---|
| 1 | **單一長時間執行的 process** | 舊版每 20M nonce 重啟一次，CUDA context 初始化吃掉 99.9% 的時間——實測 10.5 GH/s 的卡端到端只剩 4.44 MH/s |
| 2 | **生成 + 分桶融合成單一 kernel** | 雜湊不再完整落地 HBM 又讀回，省一整趟頻寬，也讓峰值記憶體與掃描量脫鉤 |
| 3 | **排序與比對全在 GPU** | CUB radix sort + 相鄰掃描，不再複製回主機 qsort |
| 4 | **一個 process 開滿本機 8 張 GPU** | OpenMP 一 thread 綁一張卡，一個 node 只付一次啟動成本 |
| 5 | **跨 batch 共用比較池** | 舊版每批各排各的，batch 之間從不互相比較，白白損失 log₂(批數) bits |

註：刻意**不做** sort 與生成的重疊。排序只佔每個 partition 約 0.5% 的時間，
但雙緩衝會讓 H 減半（= 少 1 分），不划算。

---

## 輸出格式（不可修改）

程式找到結果時產生 `solution_<共同前綴bits>.csv`，這是評分系統讀取的檔案：

```
prefix,nonce_a,nonce_b,match_bits
HiPAC2026crypto,1643364251,4721853729,63
```

`write_solution()` 標明「請勿修改」，維持原樣。`match_bits` 是用凍結的
`sha256_block` 重算完整 256-bit 雜湊後算出的**精確值**，與 `verify_collision.py` 一致。

驗證：

```bash
python3 verify_collision.py --file solution_63.csv
```

---

## 不可修改區

`sha256_block()`、`build_base_words()`、`put_nonce()`、`write_solution()` 與 CSV 格式
全部維持原樣。開放區（kernel 排程、記憶體配置、排序、多 GPU）已完全重寫。

---

## 調校旋鈕

| 旋鈕 | 預設 | 說明 |
|---|---|---|
| `MAX_SORT_ITEMS` | 無上限 | 單次 CUB 排序筆數上限。CUDA 12.9 的 `cub/detail/choose_offset.cuh` 對 `size_t` 會選 64-bit offset，所以沒有 2^31 限制，預設就吃滿顯存 |
| `-DHIPAC_TPB=` | 256 | 每 block thread 數 |
| `--partitions` / `--span` | 自動 | 強制覆寫自動推算的 P 與 S |

---

## 已知實測與推估

| 硬體 | 設定 | 結果 |
|---|---|---|
| 1 × RTX 5060 Laptop | 25 秒 · 2.7 GH/s · H = 0.18 G | **63 bits**（模型預測 62.5） |
| 1 × H200 | 原始 SHA-256 吞吐 | 10.3–10.8 GH/s |

16 × H200 推估（H = 3.5 G 筆/桶）：

| 生成秒數 | log₂(G) | 期望分數 |
|---|---|---|
| 240 s | 45.2 | **~76 bits** |
| 540 s | 46.4 | **~77 bits** |

### 通往 80 bits

分數 = `log₂(H) + log₂(G) − 1`。G 由硬體與時間決定，時間翻倍只值 +1 分。
**缺口全在 H。** 單卡 HBM 已吃滿，唯一的放大途徑是**節點內 8 卡用 NVLink 協同排序**：

| 改動 | 增益 | 累計 |
|---|---|---|
| 現況（單卡各自分桶，240 s） | — | ~76 |
| 跑滿 540 s | +1.2 | ~77 |
| 節點內 8 卡協同排序（H ×8） | +3.0 | **~80** |

協同排序的成本：每個桶要做一次 NVLink all-to-all 把 key 送到 owner GPU。
桶大小 2.9 G 筆 × 16 B = 464 GB，以 ~300 GB/s 有效頻寬約 1.5 秒，
相對每桶約 22 秒的生成時間約 7%，划算。
