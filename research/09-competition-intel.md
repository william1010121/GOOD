# 09 — 競賽情境、計分規則與驗證慣例

> 這份文件回答的不是「怎麼算得快」，而是「**在這個比賽的規則下，什麼行為會得分**」。
> 演算法上限見 [02-math-birthday-and-budget.md](02-math-birthday-and-budget.md)；題型判斷見 01-problem-taxonomy.md。
> 本文所有「事實」都附 URL；所有推測都用 ⚠️ 標記。

---

## TL;DR — 規則面的五個結論

1. **這題幾乎確定是 HiPAC 國網盃「隱藏題」**（第五屆 2026，決賽 8/4–8/6）。硬體規格（2 node、8× H200 SXM 141GB、Xeon 8480+、ConnectX-7）與官網 2026 決賽環境**完全吻合**。 [S1][S2]
2. **隱藏題在總分只佔 10%**（官網明列）[S1]，但題內計分是 **基本分 + 排名分**：基本分只要交出合法 CSV 就拿到，排名分按 bit 數與其他隊比序。→ **多拿幾個 bit 的邊際價值在「排名分」，不在「基本分」**。
3. **10 分鐘硬上限 + 可重複提交、取最佳** 是關鍵規則。max-common-prefix 的分數是**隨機變數**（近似 Gumbel，特徵尺度 β = 1/ln2 ≈ **1.44 bit**），所以「跑很多次、每次換隨機 nonce 起點、留最佳」在期望上能白賺 **β·ln k** 個 bit（k=10 次 ≈ +3.3 bit）。這是本題**最容易被忽略的免費分數**。
4. **verify_collision.py + 凍結 CSV 格式 = 反作弊與可重現的錨**。你交的兩個 nonce 會被主辦端**重新雜湊驗證**，前導相同 bit 數由驗證程式重算，不採信你自報的分數。→ 任何「預先算好、比賽時貼答案」的路線要看 prefix 是否賽時才公布（見 §4）。
5. 公開世界裡**沒有**跟這題一模一樣的 benchmark，但**近親很多**：BTCCollider（distinguished-point 找 HASH160 partial collision）、hashcat（歷屆國網盃隱藏題就是它）、以及 GPU radix sort 論文（Onesweep / RMG Sort）。URL 見 §6。

---

## 1. 這是哪個競賽？歷屆同類題型

### 1.1 賽事定位（已查證）

本題的 `hipac/` 目錄名、`基本分/排名分` 用語、`verify_collision.py`、以及硬體規格，指向 **HiPAC 國網盃應用程式效能優化競賽**（NCHC 國家高速網路與計算中心主辦，2022 起每年一屆）。 [S3]

- **第五屆（2026）決賽 8/4–8/6**，於 NCHC 新竹總部實體舉行；每隊 **≥ 2 node**，每 node **8× H200 SXM 141GB + 2× Xeon 8480+ + ConnectX-7**，2TB DDR5。 [S1][S2]
- **隱藏題（Hidden Problem）於 8/6 08:30 公布、11:00 截止**（約 2.5 小時作答窗）。 [S1]
  - ⚠️ 注意：本文撰寫日期即 2026-08-06。若「題目已釋出」，代表你正處在這個 2.5 小時窗內，10 分鐘/run 的上限是**這 2.5 小時內可重跑很多次**的意思，不是只能跑一次。**時間預算規劃見 §5。**

> ⚠️ **不確定**：官網 2026 頁面**未公開**隱藏題的技術內容（SHA-256/nonce/collision 字樣在公開頁面上查不到，屬正常，因為隱藏題設計上就是賽前保密）[S1]。「這是 HiPAC 隱藏題」是**高信度推論**（硬體逐項吻合 + 目錄名 + 計分用語），但非官網白紙黑字。

### 1.2 歷屆「雜湊 / 暴力搜尋 / 排序」相關題（已查證）

| 屆/年 | 賽事 | 題目 | 方法 | 計分/繳交 | 來源 |
|---|---|---|---|---|---|
| 第三屆 2024 | HiPAC 隱藏題 | 由已知 hash 值反推原文（MD5 / MD5crypt / bcrypt / SHA-256 混合，~2500 筆）| 字典攻擊 / 暴力 / 規則生成，實務上用 **hashcat** | 交 `result.csv`（欄位 `hash,plaintext`）；賽時持續提交、取最佳 | [S4][S5] |
| 第三屆 2024 | HiPAC 決賽 | PIConGPU（電漿 PIC）、NAMD（分子動力）、HPL/HPCG 調校 | GPU 平行 + 效能調校 | 各題賽時提交佐證、**取單題最佳成績** | [S6] |
| 第五屆 2026 | HiPAC 決賽 | 量子（Qiskit/cuQuantum）、LLM（vLLM）、Quantum ESPRESSO、CFD PINNs、HPL/HPL-MxP | 效能優化 | 佐證提交 + 口頭報告 | [S1] |

**跨賽事近親（非 NCHC，供計分/驗證慣例對照）：**

| 賽事 | 與本題相關的慣例 | 來源 |
|---|---|---|
| ISC / SC **Student Cluster Competition (SCC)** | benchmark 分數 + 挑戰完成 + 訪談的**綜合計分**；benchmark 有固定時間窗、同硬體配置、**功耗上限**規則 | [S7][S8] |
| **ASC** Student Supercomputer Challenge | HPL/HPCG + 指定 AI/HPC 應用最佳化；曾有 MD5 crypt cracker on petascale 之類的密碼破解研究背景 | [S9] |
| **APAC HPC-AI** | 應用最佳化（HOOMD-blue、Llama 2）+ 創新獎；跨國隊在遠端叢集競速 | [S10] |

> **takeaway**：國網盃隱藏題的歷史模式是「**給你一個可暴力/可搜尋的密碼學小題，用 GPU 把吞吐拉滿，交 CSV，取最佳成績**」。2024 是 hashcat 反推 hash；2026 換成「**找 partial collision（前導相同 bit 最多）**」，是同一家族的自然升級——從「反推固定答案」變成「**最佳化一個連續分數**」，這正好把題目變成 [02](02-math-birthday-and-budget.md) 講的 throughput 競速。

---

## 2. 計分方式：基本分 + 排名分，以及它的策略含意

### 2.1 兩層計分結構

README 明示本題計分為 **基本分 + 排名分**。對照官網「隱藏題佔總分 10%」[S1]，合理拆解（⚠️ 以下權重比例為推測，官網未公開隱藏題內部細分）：

| 分項 | 觸發條件 | 特性 | 策略 |
|---|---|---|---|
| **基本分** | 交出**格式正確、能通過 verify 的 CSV**（兩個 distinct nonce） | 二元（有/無），門檻低 | **先確保拿到**：極早期就跑一個「保證能交」的 baseline（哪怕只有 40 bit），把基本分鎖進口袋，再去衝排名分 |
| **排名分** | 你的 bit 數在所有隊中的**名次** | 階梯狀、非線性；差 1 名可能差好幾分 | 這裡才是主戰場。**bit 數不是連續換分，是換名次** |

### 2.2 策略含意（本題最重要的三點）

1. **「幾個 bit」可以跳好幾個名次。**
   分數 ≈ 2·log₂(N)（見 [02](02-math-birthday-and-budget.md)）。要多 1 bit，N 要 ×√2；多 2 bit，N 要 ×2。各隊的 N 通常擠在同一個數量級（大家都有 8–16 張 H200），所以**最終 bit 數會擠在很窄的區間**（可能全場只差 3–6 bit）。在這種擠壓區間，**+1 bit 就是跳名次**。→ 排名分的邊際報酬遠高於基本分，工程投資全押吞吐與 §5 的多跑策略。

2. **「取最佳」規則 = 偏好高變異（variance-seeking）。**
   既然留的是**最好的一次**，而不是平均，那麼「期望值低但變異大」的策略反而好。具體見 §5 的 Gumbel 分析：同樣的算力，**跑 10 次留最佳** 比 **跑 1 次跑好** 期望上多約 3 bit，成本只是把 10 分鐘切成多段。

3. **不要為了基本分過度保守。**
   基本分是門檻分（交得出就有），一旦鎖定就別再管它。所有剩餘時間都該拿去衝排名分。常見錯誤是把時間花在「讓 pipeline 更穩」而不是「讓 N 更大 / 多跑幾次」。

> ⚠️ 若官網之後公布排名分其實是**線性換算 bit**（而非階梯名次），則第 1 點的「跳名次」誇張效果會減弱，但「多 1 bit 仍是純賺」不變，策略方向不變。

---

## 3. verify_collision.py 與凍結 CSV 格式的角色

### 3.1 驗證是「重算」，不是「採信」

主辦提供 `verify_collision.py`（README 已凍結）。其職責幾乎一定是：

1. 讀你的 CSV → 取出兩個 nonce a、b；
2. **用官方 prefix 重新算** `SHA-256(prefix || a_be64)` 與 `SHA-256(prefix || b_be64)`（同 [00-README.md](00-README.md) 描述的單一 512-bit block）；
3. 檢查 **a ≠ b**；
4. **重新數兩個 hash 的前導相同 bit 數** → 這才是你的分數。

**含意：**
- 你**自報的 bit 數不算數**，一切以 verify 重算為準。→ 你的 kernel 裡數 leading-bit 的邏輯只要「足夠好到選出候選對」即可，最終分數由官方程式定案；但**務必在本地用同一支 verify_collision.py 跑過再交**，避免 CSV 欄位/位元組序/端序（big-endian nonce）對不上而變 0 分。
- **big-endian 8-byte nonce** 是凍結細節（`put_nonce()`）。你在 GPU 內部無論用什麼佈局，**輸出 CSV 前必須還原成官方 big-endian 表示**，否則 verify 會用不同的 nonce 重算出完全不同的 hash → 前導 bit 數暴跌。這是最容易 self-own 的地方。⚠️

### 3.2 凍結 CSV 格式 = 反作弊與可重現錨

- CSV 格式凍結（欄位、順序不可改）呼應 2024 年 `result.csv`（`hash,plaintext`）的慣例 [S5]。本題的兩欄應為兩個 nonce（十進位或十六進位，依 verify 期望——**以 verify_collision.py 的 parser 為準，別猜**）。
- **反預算（anti-precompute）機制**：本題防「賽前算好貼答案」靠的是 **prefix 賽時才公布**。只要 `prefix_ASCII` 是隱藏題公布當下才給的字串，任何預先建好的 nonce→hash 表就作廢（因為 hash 依賴 prefix）。→ **這也是為什麼 distinguished-point / rho 的預算表在這裡沒有跨 run 複用價值**（見 [02](02-math-birthday-and-budget.md) 對 DP 於本題受限的討論）。
  - ⚠️ 若 prefix 其實是固定/賽前已知，則預算合法且有巨大優勢；**開賽第一件事就是確認 prefix 何時給、是否每 run 相同**。
- **可重現性**：因為 SHA-256 對 nonce 完全確定，你的兩個 nonce 是**自證的**——主辦不需要你的程式碼就能驗分。這降低了「程式碼被要求重跑」的風險，但**口頭報告仍佔總分**（2026 決賽簡報 15%）[S1]，所以方法要能講清楚。

---

## 4. 操作建議：如何把 10 分鐘 × 很多次 打好

### 4.1 時間預算的正確心智模型

- **10 分鐘是「單一 run」的牆鐘上限**，不是總預算。作答窗（~2.5 小時，§1.1）內可重跑很多次，**留最佳**。
- 單 run 內部再切三段：**(a) 生成+比較的穩態吞吐**（主體 8–9 分鐘）、**(b) 收尾選最佳對 + 還原 big-endian + 寫 CSV**（要留 buffer，寫爆時間 = 0 分）、**(c) 本地 verify 自檢**。
- ⚠️ **啟動成本**：CUDA context、多 node MPI/NCCL 建立、IB queue pair 建立都吃秒級時間。10 分鐘的 run 若花 90 秒暖機，等於少 15% 產出。→ **暖機一次、run 多次**（同一個常駐 process 內重複執行搜尋，換隨機 nonce 區段），別每次重啟。

### 4.2 「多跑留最佳」為什麼是免費的 bit——Gumbel 分析

設單 run 能生成並互比 N 個 distinct hash。任一對的前導相同 bit 數 L 服從幾何分布 `P(L ≥ k) = 2^{-k}`。全部 M ≈ N²/2 對取最大，**最大值近似 Gumbel 分布**：

- **位置（眾數）** μ ≈ log₂M = 2·log₂N（這就是 [02](02-math-birthday-and-budget.md) 的 2·log₂N）。
- **尺度** β = 1/ln2 ≈ **1.4427 bit**（連續近似）。
- **平均** ≈ μ + γ·β ≈ μ + 0.83 bit（γ = 0.5772 為 Euler–Mascheroni 常數）。
- **標準差** = (π/√6)·β ≈ **1.85 bit**（連續近似；離散幾何實際略小，量級即 README 說的「±1.4 bit」）。 [S11][S12]

> **關鍵**：因為分數是隨機變數，**單一 run 的結果本身就有 ~±1.5–2 bit 的抖動**。這代表：
> - 對 k 個**獨立** run 取最佳，期望值 ≈ μ + β·(γ + ln k)。相對單 run 平均，**淨賺 β·ln k**：
>
> | run 次數 k | 相對單 run 平均的期望增益 β·ln k |
> |---:|---:|
> | 2 | +1.0 bit |
> | 5 | +2.3 bit |
> | 10 | +3.3 bit |
> | 20 | +4.3 bit |
>
> ⚠️ 這些增益數字是**理論期望**，且**只有在各 run 彼此獨立時成立**。

### 4.3 讓各 run 獨立（否則上表全部歸零）

如果你每次跑**完全一樣的確定性搜尋**（同 nonce 起點、同分割），每次會得到**完全相同的兩個 nonce、完全相同的分數** → 重跑毫無意義。要吃到 §4.2 的免費 bit，**每個 run 必須是獨立樣本**：

- **換 nonce 掃描區段**：每 run 用不同的 64-bit nonce 起點（隨機 offset 或不重疊的 stride），使每 run 覆蓋不同的 N 個點。
- **或換 partition 種子**：若用 prefix-partitioning（[02](02-math-birthday-and-budget.md)、04-algorithm-memory-birthday.md），改變分割映射/取樣。
- 記錄每 run 的 (seed, best_bits, nonce_a, nonce_b)，**維護一個 running best**，最後只交那個 CSV。

### 4.4 一個實務 run-loop 骨架（≤10 行示意，非完整程式）

```text
warmup_once()                      # CUDA/MPI/NCCL/IB 只建立一次
best = (-1, None, None)
while wall_time < window_deadline:
    seed = random_64bit()
    a, b, bits = search_one_run(seed, budget=9.5min)   # 生成N + closest-pair
    if bits > best.bits: best = (bits, a, b)           # running best
    log(seed, bits, a, b)
verify_locally(best); write_csv(best)                  # 交前務必本地 verify
```

### 4.5 風險清單（會直接變 0 分或掉名次）

| 風險 | 後果 | 防法 |
|---|---|---|
| CSV 端序/格式對不上 verify | 分數暴跌或 0 | 交前用官方 `verify_collision.py` 自檢 |
| a == b（誤交同一 nonce） | verify 拒收 | 選對時強制 a ≠ b |
| 收尾寫檔超時 | 整個 run 作廢 | 留 buffer、收尾非同步、先寫 baseline 再覆蓋 |
| 每 run 不獨立 | 多跑沒增益 | 每 run 換隨機 nonce 起點（§4.3）|
| 只跑 1 次「求穩」 | 放棄 §4.2 的 +2~3 bit | 多跑留最佳 |
| 硬體故障補時 | 2024 規則：僅正常時段補、上限 30 分 [S6] | 確認 2026 是否同規則；別把命押在補時 |

---

## 5. 公開 GitHub 專案：類似的 SHA-256 partial collision / HPC

> 這些**不能直接抄**（本題 kernel/CSV 凍結），但對「closest-pair / distinguished-point / GPU sort / hash 吞吐」的工程手法有直接參考價值。授權欄供合規判斷。

| 專案 | 內容 | 授權 | URL |
|---|---|---|---|
| **BTCCollider** | 用 distinguished-point 找兩個 HASH160 前導相同 bit 的 partial collision，多 GPU CUDA——**與本題目標最接近的公開實作** | 見 repo | https://github.com/JeanLucPons/BTCCollider |
| **sha256-truncation-collision-study** | GPU（A30 / CUDA 12.2）Monte-Carlo 量測 32/64-bit 截斷 SHA-256 碰撞率 vs 生日理論——**驗證方法學參考** | Apache-2.0 | https://github.com/cgkinyua/sha256-truncation-collision-study |
| **hashcat** | 歷屆國網盃隱藏題（2024）實際用的工具；SHA2-256 GPU 吞吐 benchmark 的事實基準 | MIT | https://github.com/hashcat/hashcat |
| **CudaSHA256 (Horkyze)** | 最小可讀的 CUDA SHA-256 參考 | 見 repo | https://github.com/Horkyze/CudaSHA256 |
| **shacuda (quantumish)** | 另一個 GPU SHA-256 | 見 repo | https://github.com/quantumish/shacuda |
| **NVIDIA CUB / CCCL** | `DeviceRadixSort`——closest-pair 前的 sort 主力（見下方吞吐） | Apache-2.0 (NVIDIA) | https://github.com/NVIDIA/cccl |
| **Onesweep (論文)** | 單 GPU LSD radix sort，A100 上 **29.4 GKey/s**（256M×32-bit），較 CUB 快 ~1.5× | 論文 | https://arxiv.org/abs/2206.01784 |
| **RMG Sort (論文)** | radix-partition 多 GPU 排序，**DGX A100（8×）> 10 G keys/s** 端到端、近線性擴展——**多 GPU closest-pair 的直接藍圖** | 論文 | https://hpi.de/oldsite/fileadmin/user_upload/fachgebiete/rabl/publications/2023/rmg-sort-ilic.pdf |

**吞吐錨點（供 [02](02-math-birthday-and-budget.md) 交叉驗算 sort 瓶頸）：**
- 單 A100 radix sort ~**29.4 GKey/s**（32-bit key）[S13]。H200 記憶體頻寬 4.8 TB/s ≈ A100 (2.0 TB/s) 的 2.4×，⚠️ **推測** 單 H200 32-bit radix sort 落在 **~40–60 GKey/s**（sort 為 memory-bound，隨頻寬近線性；未實測，須 microbenchmark）。
- 8× GPU 多卡 sort ~**10 G keys/s+**（DGX A100 實測）[S14]；本機 16× H200 + NVLink + IB，⚠️ **推測** all-to-all radix partition 後端到端可達 **數十 G keys/s**（瓶頸轉為 IB/NVLink all-to-all 頻寬，見 06-multinode-and-io.md）。

> ⚠️ **重要**：這些 sort 吞吐是「32-bit key」數字。本題 hash key 較寬（要比 leading bits，實務上可先用 top-64 或 top-128 bit 當 sort key），**每 key 位元組數翻倍 → sort 吞吐約砍半**。規劃時用「有效 GKey/s = 頻寬 / (key bytes × passes)」自算，別直接套 32-bit 數字。

---

## 6. 與其他文件的連結

- **能達到幾 bit（N 的上限）** → [02-math-birthday-and-budget.md](02-math-birthday-and-budget.md)（2·log₂N、算力預算）。
- **為什麼 closest-pair 是瓶頸、prefix-partitioning 如何解耦記憶體** → 04-algorithm-memory-birthday.md。
- **多 node all-to-all / NVLink / IB / GPUDirect RDMA 的 sort 擴展** → 06-multinode-and-io.md。
- **SHA-256 kernel 已凍結**（本題不能微調），但吞吐估算仍見 05-sha256-kernel-optimization.md。
- **拿到題目 30 分鐘內要確認的事**（prefix 何時給、每 run 是否同 prefix、CSV 欄位）→ 01-problem-taxonomy.md + 本文 §3–§4。

---

## 參考來源

**已查證（附 URL）：**
- [S1] 第五屆國網盃 2026 競賽說明（4 應用題 + HPL/HPL-MxP + 隱藏題 10% + 決賽 8/4–8/6、隱藏題 8/6 08:30–11:00、2 node/8×H200 141GB/Xeon 8480+/ConnectX-7）— https://event1.nchc.org.tw/2026/hipac/details.html ；首頁 https://event1.nchc.org.tw/2026/hipac/
- [S2] 2025 第四屆國網盃（決賽形式、雲端設備、清大公告）— https://dsa.site.nthu.edu.tw/p/406-1266-288364,r6890.php?Lang=zh-tw
- [S3] NCHC 競賽與人培（國網盃沿革、Taiwan Student Cluster Competition 2011–2017）— https://www.nchc.org.tw/Page?itemid=115&mid=209
- [S4] HiPAC 2024 CUDACOLA 隊參賽報告（2500 筆 MD5/MD5crypt/bcrypt、用 hashcat、curl 提交、切字典平行）— https://hackmd.io/@jiazheng/SycI9vV90
- [S5] 第五屆官網 2024 成果頁（第三屆隱藏題：由 hash 反推原文、給 hash.txt/wordlist.txt、交 result.csv 欄位 `hash,plaintext`、含 MD5/SHA-256）— https://event1.nchc.org.tw/2026/hipac/showcase3.html
- [S6] 第三屆國網盃 2024 競賽說明（計分權重、隱藏題 8/8 09:30–11:30 兩小時窗、取單題最佳、故障補時上限 30 分）— https://event1.nchc.org.tw/2024/hipac/details.html
- [S7] SCC23 benchmark 提交說明（HPL/HPCG/MLPerf-BERT + STREAM/OSU、時間窗、功耗上限）— https://scc23-benchmarking.readthedocs.io/en/latest/
- [S8] SC26 Student Cluster Competition（綜合計分：benchmark + 挑戰 + 訪談）— https://sc26.supercomputing.org/students/student-cluster-competition/
- [S9] ASC Student Supercomputer Challenge — https://www.asc-events.net/StudentChallenge/index.html
- [S10] 2024 APAC HPC-AI Competition（HOOMD-blue、Llama 2、遠端叢集競速）— https://www.hpcadvisorycouncil.com/events/2024/APAC-AI-HPC/
- [S11] Gumbel 分布 E[X]=μ+βγ、Var=(π²/6)β²（極值分布，對應 max-of-geometric）— https://www.randomservices.org/random/special/ExtremeValue.html
- [S12] 隨機 bitstring 的最長共同前綴期望 ≈ 2·log₂n + O(1)（LCP 陣列最大值）— https://arxiv.org/pdf/1801.04425
- [S13] Onesweep：A100 上 32-bit radix sort 29.4 GKey/s，較 CUB ~1.5× — https://arxiv.org/abs/2206.01784
- [S14] RMG Sort：DGX A100（8 GPU）> 10 G keys/s、近線性擴展 — https://hpi.de/oldsite/fileadmin/user_upload/fachgebiete/rabl/publications/2023/rmg-sort-ilic.pdf
- [S15] BTCCollider（distinguished-point 找 HASH160 partial collision，多 GPU CUDA）— https://github.com/JeanLucPons/BTCCollider

> **數字警語**：本文的「隱藏題內部計分細分（基本分/排名分權重）」「H200 sort 吞吐」「多跑增益 β·ln k」皆為**推測或理論期望**，已逐處以 ⚠️ 標記。開賽後第一時間要用 `verify_collision.py` 與 microbenchmark 校正（見 01-problem-taxonomy.md 的開賽檢查清單）。
