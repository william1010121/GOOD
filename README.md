# 隱藏題1：SHA-256 部分碰撞搜尋 — 參賽說明

## 題目在做什麼

雜湊函數的重要性質之一是「抗碰撞」：要找到兩個不同的輸入卻得到相同的雜湊值，
理論上極其困難。但如果只要求「**開頭一部分相同**」，難度就會大幅下降 ——
這就是所謂的**生日攻擊**。

本題給定一個固定字串 `prefix`，對每個 64 位元整數 `nonce` 定義：

```
hash(nonce) = SHA-256( prefix(ASCII) ‖ nonce(8 bytes 大端) )
```

你的任務是找出**兩個不同的** nonce `a` 與 `b`，讓 `hash(a)` 與 `hash(b)`
**開頭有最多相同的 bits**。相同的 bits 越多，分數越高。

### 範例

```
prefix = "hipac_demo"

nonce_a = 13644422  →  429a8698aa53 6775ad1d418b19e7a371...
nonce_b = 18436129  →  429a8698aa53 757e2b88e18b43e4a445...
                       └── 共同前綴 51 bits ──┘
```

這組的成績就是「共同前綴 = 51 bits」。

---

## 範本程式在做什麼

`collision.cu` 已經是一支**可以正確執行**的程式，分三步：

1. **GPU 平行計算雜湊** —— 每個執行緒算一個 nonce 的 SHA-256，
   取雜湊的前 128 bits 當作排序鍵。
2. **GPU radix sort** —— 用 CUB stable radix sort 先排低 64 bits、再排高 64 bits。
3. **GPU 相鄰配對 reduction** —— 在 GPU 找出本 batch 最佳配對，CPU 只拷回一筆 candidate。

> 這支範本**能跑，但沒有特別快**。讓它更快就是你的任務。

---

## 如何編譯

需要 NVIDIA GPU 與 CUDA Toolkit（`nvcc`）。

```bash
make
```

`make` 會自動偵測 GPU 架構。偵測不到時（例如在沒有 GPU 的登入節點編譯），
會自動改編「多架構通用版」，換到哪張卡都能跑，只是編譯久一點。

想手動指定：

```bash
make ARCH=sm_90     # H200 / H100
make ARCH=sm_80     # A100
```

編譯出問題時，先跑這個把環境資訊印出來：

```bash
make info
```

也可以完全不用 make，直接編：

```bash
nvcc -O3 -Xcompiler -fopenmp -arch=sm_90 collision.cu -o collision
```

> `-Xcompiler -fopenmp` 範本本身用不到，但優化方向之一是多 GPU，
> 而多 GPU 最自然的寫法就是 OpenMP，先開著省得之後卡在連結錯誤。

---

## 如何執行

一般模式不使用 `total`，會持續以固定 batch 掃描，直到收到 `Ctrl-C`；每批完成後會回報目前找到的最高共同前綴 qbit 數量。每批大小可用 `BATCH_NONCES` 調整：

```bash
BATCH_NONCES=100000000 ./collision HiPAC2026crypto
```

只有明確加上 `--smoke` 才使用固定的 `total`，並在完成後結束：

```bash
./collision --smoke HiPAC2026crypto 100000000
```

### Slurm 1 分鐘 smoke test

`smoke_test.slurm` 會先編譯，再以 2 個 node、每個 node 8 張 GPU（共 16 張）執行 `--smoke`，重複完成固定大小的掃描批次直到達到 60 秒，最後輸出整體彙總吞吐量與目前最高 qbit 前綴：

```bash
sbatch --export=ALL,PREFIX=HiPAC2026crypto,SMOKE_NONCES=100000000,SMOKE_SECONDS=60 smoke_test.slurm
```

若叢集沒有自動找到 CUDA，可設定 `CUDA_MODULE`；若要指定 GPU 架構，可設定 `ARCH=sm_90`。工作目錄預設使用 `SLURM_SUBMIT_DIR`，也可用 `WORKDIR` 覆寫。

輸出範例：

```
使用 GPU: NVIDIA H200（132 個 SM）
prefix = "HiPAC2026crypto"    每批掃描 100000000 個 nonce

需要記憶體：GPU 約 16 GB（含 CUB sort/reduce temp），CPU 只有一筆 candidate

[batch 1] GPU 計算 100000000 個雜湊 …
[batch 1] GPU radix sort（lo → hi）+ pair reduction …

目前最多共同前綴 qbit：51
nonce_a : 13644422
nonce_b : 18436129
耗時    : 0.5 秒
答案已寫入 solution_51.csv

驗證指令：
  python3 verify_collision.py -p HiPAC2026crypto -a 13644422 -b 18436129
```

## 如何驗證答案

```bash
python3 verify_collision.py -p <prefix> -a <nonce_a> -b <nonce_b>
```

或直接讀繳交檔（會一併檢查 `match_bits` 是否與實際相符）：

```bash
python3 verify_collision.py --file solution_51.csv
```

---

## ⚠ 輸出檔案格式（不可修改）

程式找到結果時會產生一個 CSV 檔，**這就是評分系統讀取的檔案**：

- **檔名**：`solution_<共同前綴bits>.csv`，例如 `solution_51.csv`
- **內容**：第一行欄位名，第二行資料

```
prefix,nonce_a,nonce_b,match_bits
hipac_demo,13644422,18436129,51
```

規則：

- 檔名規則、欄位名、欄位順序都不可更動。
- 程式裡的 `write_solution()` 函式已標明「**請勿修改**」，請保持原樣。
- 你可以自由改寫程式的其他部分（甚至完全重寫演算法），但**只要輸出的 CSV
  格式對不上，評分系統就讀不到，該筆提交以 0 分計**。

---

## 可以動哪些地方（優化方向）

### 哪些不能改

程式分成兩區，中間有分隔標記：

- **【不可修改區】** —— `sha256_block()`、`build_base_words()`、`put_nonce()`。
  本題比的是資料處理與平行化的效率，不是密碼學實作，所以雜湊部分統一固定。
- **【開放區】** —— 其餘全部隨你改寫。

### 三個方向

程式裡標了 `★ 優化點` 的三個地方：

- **（最重要）** 複製回主機並排序這一步是瓶頸，整段期間 GPU 閒置。
- 範本一次把所有雜湊都存起來，記憶體會成為掃描量的上限。每筆 24 bytes，掃 10 億個需要 24 GB。
- 範本只用 1 張 GPU，比賽機器有多張。

開放區可以自由改寫，甚至完全重寫 —— 只要：

- **不可修改區**維持原樣
- 提交的 `(nonce_a, nonce_b)` 是**兩個不同的值**，且能通過 `verify_collision.py`
- 輸出的 CSV 格式不變（見上一節）
- 程式在 10 分鐘內執行完畢

---

## 繳交與計分

- **繳交內容**：程式產生的 `solution_<bits>.csv`（共同前綴越長越好）。
- **輸出格式不可修改**，格式錯誤該筆以 0 分計。
- **比賽時間內可重複提交**，取最佳的一次。
- 計分方式以主辦方公告為準（基本分 + 依名次的排名分）。
