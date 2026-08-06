# 08 — 榨乾「凍結 SHA-256 kernel」的吞吐量

> 前提：`sha256_block()`（device function）、`build_base_words()`、`put_nonce()`（host helpers）**必須逐字沿用、不可改**。
> 因此本文**不碰壓縮函數本身**（無 midstate、無 early-exit、無手工 LOP3 重排 — 那些屬於 [05-sha256-kernel-optimization.md](05-sha256-kernel-optimization.md) 的假想空間，本題被凍結封死）。
> 本文只談 **OPEN 的部分**：`hash_kernel` wrapper、launch config、記憶體佈局、把 bucketing 融進 kernel、multi-GPU/multi-node 的產生端。演算法上限與預算見 [02-math-birthday-and-budget.md](02-math-birthday-and-budget.md)；closest-pair / sort 見 [04-algorithm-memory-birthday.md](04-algorithm-memory-birthday.md)。

本題經 [00-README.md](00-README.md) 與提示分析已定調為 **分散式 GPU sort / closest-pair 吞吐問題**：分數 ≈ `2·log2(N)`，N = 10 分鐘內能**產生並比較**的 distinct hash 數。SHA 產生便宜、closest-pair 是瓶頸。**但 kernel 常數項仍決定 N 的上限**，而凍結 `sha256_block` 之後，wrapper 的寫法就是唯一能動的常數。以下逐項拆解。

---

## Part 0 — 先講結論（凍結情境下值得做 / 不值得做）

| 手段 | 對本題（600s、單一 512-bit block、compute-bound）的價值 | 理由 |
|---|---|---|
| **把 bucketing/insert 融進 hash kernel** | ★★★★★ 最高 | hash 不落 HBM、不再讀回；省一整趟 write-all + read-all，見 Part 2 |
| grid sizing = 132 SM 的倍數 + grid-stride/persistent | ★★★★ | 一次填滿、單次 launch，launch overhead 直接歸零 |
| `__launch_bounds__` 掃 occupancy | ★★★ | SHA-256 register 很吃，occupancy 可能被 register 卡住；需實測 |
| threads/block 掃參數（128/256/512） | ★★★ | 便宜、影響 occupancy 與 ILP |
| 儲存 top-96 vs top-128 bits | ★★ | 頻寬省 25%，但對齊變醜；見 Part 4 |
| 多 CUDA streams 疊 compute / DtoH | ★★ | 本題 H2D≈0；只在 bucket buffer 回吐時有用 |
| **CUDA Graphs / persistent-kernel 省 launch overhead** | ★☆ | ⚠️ 本題 kernel 每次跑數 ms~秒級，launch overhead（µs 級）幾乎可忽略；用 persistent kernel 是為了程式簡潔而非省 overhead |
| CPU (SHA-NI, 224 core) 當額外產生器 | ★☆ | <3~5% GPU 算力，不划算；改當 sort/IO coordinator |
| 改 `sha256_block` 內部（LOP3/midstate…） | ✗ 禁止 | 凍結。只能**檢查** nvcc 產生的 SASS 好不好，見 Part 5 |

---

## Part 1 — wrapper 還剩哪些 levers

`sha256_block` 是黑盒，但呼叫它的 `hash_kernel` 是我們寫的。可動的旋鈕：

### 1.1 Thread / block / grid 配置（Hopper 事實）
Hopper (compute capability 9.0, GH100) 每 SM 的硬體限制（[Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)）：

| 資源 | 每 SM 上限 |
|---|---:|
| 最大常駐 threads | 2048 |
| 最大常駐 warps | 64 |
| 最大常駐 blocks | 32 |
| 32-bit registers | 65536 |
| shared memory | 228 KB（opt-in，預設 48KB/block） |
| **INT32 ALU** | **64 / SM**（FP32:INT32 = 2:1，與 Ampere 同） |

H200 = 132 SM。SHA-256 是純 32-bit 整數 + 邏輯 + rotate，瓶頸就是那 **64 INT32/SM**（源自 [02](02-math-birthday-and-budget.md) 已查證的事實）。

**Grid sizing 原則**：block 數取 **132 的倍數**避免尾端 SM 空轉。不要用「N/threads」這種隨資料量爆量的 grid；改用 **grid-stride loop**（[CUDA Pro Tip: Grid-Stride Loops](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)）或 **persistent kernel**：只 launch `occupancy × 132` 個 block，每 thread 用一個 64-bit counter 當 nonce base、以 `total_threads` 為 stride 掃過整個 nonce 空間：

```cuda
// 示意（<=10 行）：persistent grid-stride，nonce = 全域 thread id + i*stride
for (uint64_t n = gid; n < N_TOTAL; n += stride) {
    build_base_words(w, prefix); put_nonce(w, n);   // 凍結 helper
    sha256_block(h, w);                              // 凍結
    emit_top_bits(h, n);                             // 我們寫：見 Part 2
}
```
如此整個 600s 只需**一次 launch**，`grid=stride/threads` 恆定，launch overhead 問題自動消失。

### 1.2 `__launch_bounds__` 與 occupancy
SHA-256 全展開 kernel 通常吃 **~48~96 registers**（8 個 working vars + 16 個 W schedule + 暫存；⚠️ 實際值須 `ptxas -v` 量）。若 register/thread = 64，則每 SM 可容 `65536/64 = 1024` threads = 50% occupancy；若 = 96 則只剩 682 threads ≈ 33%。

在 wrapper 上加 `__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)` 讓 ptxas 反推 register 上限（[CUDA C++ Language Extensions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)）：
- 指定後編譯器會把 register 壓到「能塞下 minBlocksPerSM 個 block」的量；壓過頭會 **spill 到 local memory**（反而變慢）。
- 兩參數都給時，編譯器也可能**反向拉高** register 到上限以減少指令、增 ILP。

⚠️ **重要判斷**：SHA-256 是 latency-tolerant 的長算術鏈，**高 occupancy 不必然更快**——單 thread 內 64 輪的 ILP 已能餵飽 pipeline。實務常見「50% occupancy 但滿速」。**必做**：掃 `__launch_bounds__(128,?)`、`(256,?)`、`(512,?)` 各配 minBlocks 1~4，量實際 GH/s，不要迷信 occupancy 數字。

### 1.3 CUDA streams / CUDA Graphs / persistent kernel
- **H2D 幾乎為零**：nonce 由 device 端 counter 產生，不需傳輸；只有 `prefix`（幾十 bytes）與最終結果要搬。所以「overlap HtoD with compute」在本題**不是重點**。
- **DtoH overlap 才有用**：若 bucket buffer 分批回吐到 host/NVMe（partition 溢出時），用 **2~4 條 stream + pinned memory** 讓「第 i 批的 DtoH copy」疊「第 i+1 批的 compute」。
- **CUDA Graphs**：把「launch + memcpy + launch」序列錄成一個 graph 重播，可把每次 launch 的 ~2~5µs host 成本壓到 ~1.3µs（[Kernel Fusion / Launch Overhead, NVIDIA](https://developer.nvidia.com/blog/kernel-fusion-in-nvidia-cuda-optimizing-memory-traffic-and-launch-overhead)、[Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)）。
  > ⚠️ **本題判斷**：只有在「很多顆小 kernel」時 launch overhead 才會主導。本題每顆 kernel 跑數 ms 到整段 600s，launch overhead 佔比 < 0.01%。**因此 persistent kernel（單次 launch）就已經解決問題，CUDA Graphs 是錦上添花、非必要。** 若最後採「多批次 + 每批一 launch」設計，才回頭用 Graphs 收尾。

---

## Part 2 — 把 bucketing 融進 hash kernel（本文最重要的一招）

**問題**：naive pipeline 是 (1) kernel 算 hash → 寫 N 筆到 HBM；(2) 讀回 N 筆做 radix sort（多趟 read/write）；(3) 掃描相鄰找 closest pair。步驟 (1) 的「寫全部 hash」+ 步驟 (2) 的「讀回」是純浪費的一趟往返。

**解法（wrapper 改寫，允許）**：在 `sha256_block` 回傳 `h[8]` 後，**不把整個 hash 寫出**，而是當場：
1. 取 top-p bits 當 partition id（radix by top bits，見 [04](04-algorithm-memory-birthday.md)）。
2. `atomicAdd` 到該 partition 的 counter 取得 slot。
3. 只寫 **(reduced key = 次高位 bits, nonce)** 進那個 partition 的 buffer。

如此 hash 值**從不完整落地 HBM**，每個 nonce 只寫一次到它該去的桶。這正是 [prefix-partitioning 讓記憶體與 N 解耦](02-math-birthday-and-budget.md) 的產生端實作，且把「寫全 hash + 讀回」兩趟省成一趟。

**頻寬帳（ESTIMATE ⚠️）**：假設單 H200 產生 ~10 GH/s（見 Part 3，修正後中心值），每筆存 16 bytes（nonce 8B + top-64 key 8B）：
- 寫出頻寬 = `10e9 × 16 = 160 GB/s`，對 H200 的 **4.8 TB/s HBM3e** 只佔 **~3.3%** → 產生端寫出**不是**瓶頸。
- 但若 naive 版還要「讀回 160 GB/s + radix 每趟再 read+write」，一次 radix pass 就多 ~2× 全資料往返。**融合省掉的是 sort 前那趟 write+read（~320 GB/s 等效流量）**，更重要的是省掉 HBM 容量壓力（hash 不需同時全部存活）。
- ⚠️ 真正吃頻寬的是 `atomicAdd` 造成的 partition counter 競爭與 scatter 寫入的非 coalesced pattern。**必測**：用 warp-aggregated atomics（先在 warp 內 `__ballot`/`__popc` 聚合再一次 atomic）降低 counter 爭用。

**tradeoff**：融合後 kernel 的 register/shared 用量上升（要放 partition 邏輯），可能微幅拉低 hash 速率；但省下的頻寬 + 容量遠大於此。與 sort 階段的介面見 [04](04-algorithm-memory-birthday.md)、[06-multinode-and-io.md](06-multinode-and-io.md)。

---

## Part 3 — 單 SHA-256 hash rate 估算（H200，全展開 per-thread kernel）

> ⚠️ **以下全為 ESTIMATE，未經本機實測，誤差可能 2~3×。所有時程規劃須以 microbenchmark 校正（見 [09-open-questions.md]）。**

### 3.1 錨點：RTX 4090 hashcat（SOURCED）
[Hashcat v6.2.6 on RTX 4090](https://gist.github.com/Chick3nman/32e662a5bb63bc4f51b847bb422222fd)（實測、可查證）：

| Hash-Mode | 演算法 | 速率 |
|---|---|---:|
| 1400 | SHA2-256 | **21975.5 MH/s ≈ 21.98 GH/s** (`Accel:32 Loops:512 Thr:512`) |
| 1410 | sha256($pass.$salt) | 21.96 GH/s |
| 1430 | sha256(utf16le) | 22.00 GH/s |

> ⚠️ **與 [02](02-math-birthday-and-budget.md) 的數字不一致**：該文引用「RTX 4090 SHA2-256 ≈ 50.9 GH/s」。**本次直接抓原始 gist，mode 1400 實為 ~21.98 GH/s**。50.9 GH/s 較可能是 **SHA-1（mode 100，~50 GH/s 量級）** 被誤記為 SHA-256，或指某超頻/多卡配置。**本文採用可查證的 21.98 GH/s 作為 4090 SHA-256 錨點；此差異需回頭修正 [02] 的推導（此處僅標記，不改動該檔）。**
> 另注：hashcat 是**高度手工優化**的 kernel（含 midstate、跳過與 nonce 無關的前段運算、LOP3 手排）。本題 `sha256_block` 被凍結，**無法用這些技巧**，所以我們的實際速率應 **低於** hashcat 同硬體。

### 3.2 由 INT32 吞吐比推 H200
純 INT32/邏輯吞吐上界：

| GPU | SM | INT32/SM/clk | Boost | INT32 ops/s | 相對 4090 |
|---|---:|---:|---:|---:|---:|
| RTX 4090 (Ada, cc 8.9) | 128 | **64** | ~2.52 GHz | `128×64×2.52e9 ≈ 2.06e13` | 1.00 |
| **H200 (GH100, cc 9.0)** | 132 | **64** | ~1.98 GHz | `132×64×1.98e9 ≈ 1.67e13` | **~0.81** |

> 🔧 **已修正（本次 fact-check）**：舊表把 4090 的 INT32 寫成 **128/SM**，那是**錯的**。Ada（與 Ampere GA10x、Hopper 相同）每 SM 的 128 顆是 **FP32** CUDA core；整數/邏輯/shift 的可並行吞吐只有 **64/SM/clk**——每個 SM partition 的第二條 datapath 只能「跑 FP32 *或* INT32、不能同時」，純整數工作負載（SHA-256）永遠只吃得到 64/SM（來源：[NVIDIA Ampere Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)、[Ada GPU Architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)、CUDA Programming Guide「Arithmetic Instructions」throughput 表 cc 8.9/9.0 的 32-bit int add = 64/SM）。這也和本文 **Part 1.1「INT32 ALU 64/SM，與 Ampere 同」**自洽——舊 Part 3.2 的 128 是內部矛盾。**修正後兩架構 INT32/SM 相同，比值由 0.40 → ~0.81。**

**H200 每卡單 SHA-256 估算**（兩條路交叉驗證，皆已用修正後 0.81 比值）：
- **路徑 A（hashcat 錨點 × 吞吐比 × 凍結罰則）**：`21.98 × 0.81 × (0.5~0.9 凍結效率) ≈ 8.9 ~ 16 GH/s`。
- **路徑 B（理論頂 downscale）**：H200 整數頂 `1.67e13 / (~1000~1200 int ops/block) ≈ 14~15 GH/s` × 實務效率 40~70% = 6~10 GH/s。
- **合併採用範圍：單 H200 ≈ 7 ~ 14 GH/s，中心值取 ~10 GH/s。** ⚠️ ESTIMATE（先前 5~10 GH/s 是因 0.40 誤比值而低估）。

> ⚠️ **仍存在的風險（方向對、幅度已修正）**：H200 與 4090 的 **INT32/SM 其實相同（都是 64）**，並非「一半」；H200 之所以每卡略輸，只因 **boost 時脈較低（1.98 vs 2.52 GHz）**，而 132 vs 128 SM 幾乎抵不回來——淨值約 **0.81×**（即單卡 SHA-256 約比一張 4090 慢 ~20%，**不是慢一半**）。挖礦社群「H100/H200 純 SHA 每卡 hashrate 不如 4090、且功耗/成本不划算」的共識仍成立，但原因是時脈與 cost/perf，不是 INT32 減半。**這仍是全隊第一要量的 microbenchmark。**

### 3.3 叢集與 600s 可產生量（ESTIMATE ⚠️）

| 層級 | 卡數 | 單卡(GH/s) | 合計 (hash/s) | 600s 累計 | log2 |
|---|---:|---:|---:|---:|---:|
| 悲觀 (7 GH/s) | 16 | 7 | 1.12e11 | 6.72e13 | 2^45.9 |
| 中心 (10 GH/s) | 16 | 10 | 1.6e11 | 9.6e13 | 2^46.4 |
| 樂觀 (14 GH/s) | 16 | 14 | 2.24e11 | 1.34e14 | 2^46.9 |

- 16×H200 兩節點：**~1.1e11 ~ 2.2e11 hash/s ≈ 2^36.7 ~ 2^37.7 /s**（與 [02] 樂觀的 2.4e11 同量級；先前用 0.40 誤比值造成的低估已於本次修正）。
- **600s 可產生 ~2^45.9 ~ 2^46.9 個 hash。** 若能全部 distinct 且完成 closest-pair，分數上界 `2·log2(N) ≈ 92 ~ 94 bits` 相同 leading bits（與 [02] 的「10 分鐘 ~94 bit」量級吻合）。
- ⚠️ 實際分數會被 **closest-pair/sort 吞吐**（[04]）與 **multi-node all-to-all**（[06]）進一步吃掉，產生端不是唯一瓶頸。

---

## Part 4 — 存 top-96 vs top-128 bits（頻寬 / 容量 tradeoff）

closest-pair 只需比較 hash 的「前導 bits」。存多少決定頻寬與能分辨到幾 bit。

| 方案 | key bytes | +nonce(8B) | 每筆 | 相對頻寬 | 能分辨的最大 match |
|---|---:|---:|---:|---:|---|
| top-64 | 8 | 8 | 16 B | 0.80× | ≤64 bit（不夠，會大量 tie） |
| **top-96** | 12 | 8 | **20 B** | **1.00×**（基準） | ≤96 bit |
| top-128 | 16 | 8 | 24 B | 1.20× | ≤128 bit |
| top-96 packed | 12 | 8→packed | ~18~20 B | ~0.9× | ≤96 bit，對齊醜 |

分析：
- 本題 10 分鐘可達 match 上限 ~92~94 bit（Part 3，修正後）。**top-96 剛好覆蓋此上界**，且若運氣好超過 96 bit 也頂到天花板。
- **top-128 給 +32 bit headroom**，代價是 **+20% 頻寬 / 容量**（24B vs 20B）。且 `2×uint64` 對齊乾淨、radix sort 直接吃 128-bit key。
- ⚠️ **建議**：預設 **top-128（乾淨對齊、無風險）**；只有在 microbenchmark 證明 scatter 寫入或 sort 為頻寬瓶頸時，才降到 **top-96 packed** 換 ~20% 流量。分辨度損失（96 vs 128）在本題可達 bit 數下**幾乎不影響分數**，因為你根本到不了 96 bit 的比較深度。
- 也可只在 partition buffer 存 **次高位 key**（top bits 已隱含在 partition id 裡），再省幾 bit —— 見 [04](04-algorithm-memory-birthday.md) 的 key 壓縮討論。

---

## Part 5 — nvcc 到底有沒有幫凍結 macro 出 LOP3 / funnel-shift？（只檢查，不改）

`sha256_block` 凍結，但我們仍該確認 **編譯器有沒有把它編好**，以校正 Part 3 的效率假設。SHA-256 兩個關鍵可省指令的地方：
- **Ch/Maj**：`Ch=(e&f)^(~e&g)`、`Maj=(a&b)^(a&c)^(b&c)` 各可壓成**單一 `LOP3`**（3 輸入查表邏輯）。
- **Rotate**：`ROTR(x,n)` 可用**單一 `SHF`（funnel shift）** 完成，而非 `SHL|SHR|OR` 三指令。

**如何檢查（描述 how，不實作）**（[CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)）：
1. 編譯保留中繼：`nvcc -arch=sm_90a -Xptxas -v -lineinfo ...` → 看 `ptxas` 印出的 **registers / spill**。
2. 反組譯 SASS：`cuobjdump -sass a.out`（吃 host binary 或 cubin），或 `nvdisasm -c a.cubin`（只吃 cubin，輸出更細）。
3. 在 `sha256_block` 的 SASS 裡 grep：
   - 出現大量 **`LOP3.LUT`** → Ch/Maj 已融合（好）。
   - 出現 **`SHF.R`/`SHF.L`** → rotate 走 funnel-shift（好）。
   - 若看到成串 `SHL … ; SHR … ; LOP3/OR …` 手動組 rotate、或 Ch/Maj 被拆成多個 `AND`+`XOR` → 編譯器**沒**優化好。
4. 也看 `IADD3`（三輸入加法，SHA 的加法鏈理想形態）與有無 `LDL/STL`（local load/store = register spill，壞）。

⚠️ 由於程式凍結，即使發現編譯不理想也**不能改原始碼**；可調的只有**編譯旗標**（`-O3`、`-arch=sm_90a` 是否用了原生 Hopper ISA、`--maxrregcount` 是否誤壓了 register 導致 spill）。把 SASS 檢查結果回饋到 Part 1.2 的 `__launch_bounds__` 掃參數。

---

## Part 6 — 224 核 CPU（SHA-NI）：當產生器 vs 當協調者

Xeon 8480+ (Sapphire Rapids) 有 **SHA-NI**（`sha256rnds2` 等）與 AVX-512（源自 [02](02-math-birthday-and-budget.md) 已查證）。

**當額外 hash 產生器？**（ESTIMATE ⚠️）
- SHA-NI 單核 ~25~30 MH/s（64-byte 單 block）；224 核 → **~6 GH/s**。
- 相對 16×H200 的 ~1.1e11~2.2e11（修正後）：CPU 佔 **~2.7~5.5%**，與 [02] 的 <3% 同量級。
- 用 `intel-ipsec-mb` 的 AVX-512 multi-buffer 可再拉一點，量級不變。
- ⚠️ **判斷**：把 CPU 也丟去產 hash，會與其「協調 sort/IO」的職責搶記憶體頻寬與 CPU 週期，得不償失。**除非產生端完全 GPU-bound 且 CPU 閒置**，否則不值得。

**更好的用途（推薦）**：
1. 主持 **partition buffer 的 host-side merge / 去重 / 落 NVMe**（30TB RAID0 的 I/O 排程）。
2. 協調 **multi-node all-to-all**（把同 partition id 的 key 聚到同一張卡）——見 [06-multinode-and-io.md](06-multinode-and-io.md) 的 GPUDirect RDMA / NCCL 討論。
3. 跑 CUB/Thrust 的 host 端 driver、CUDA Graph 錄製、以及最後 closest-pair 結果的 CSV（凍結格式）輸出。

---

## Part 7 — sort/insert 端會用到的既成庫（產生端交棒對象）

融合 kernel 產出 partition buffer 後，排序/掃描由這些庫接手（詳見 [04](04-algorithm-memory-birthday.md)）：

| 庫 | 用途 | Repo / License |
|---|---|---|
| **CUB `DeviceRadixSort`** | 單卡 device-wide radix sort，已有 **SM90 (Hopper) tuning policy** | [nvidia.github.io/cccl/cub](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html) / [github.com/NVIDIA/cccl](https://github.com/NVIDIA/cccl) — **Apache-2.0 (w/ LLVM exceptions)** |
| **Thrust** | 高階 sort/unique/reduce，快速原型 | 同 CCCL repo，Apache-2.0 |
| **libcusort** | 單 header、CUB 相容 API 的快速 radix sort | [github.com/IlyaGrebnov/libcusort](https://github.com/IlyaGrebnov/libcusort) — MIT |
| **CUB `DeviceMergeSort` / `BlockRadixSort`** | partition 內排序、warp/block 級 | 同 CCCL |

⚠️ **CUB radix sort 吞吐（ESTIMATE，且偏樂觀）**：文獻未給精確 H100/H200 keys/s 數字（[CCCL 文件](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html) 只述「work-complexity 線性、大輸入時飽和」）。radix sort 為記憶體頻寬綁定，粗估 `keys/s ≈ HBM_BW / (passes × bytes/key × 2)`；以 H200 4.8TB/s、128-bit key（16B）、假設 `LSB radix` 只需 ~4 passes 估得 `4.8e12 / (4 × 16 × 2) ≈ 3.75e10 keys/s`。
> 🔧 **本次 fact-check 的警告**：那個 **~4 passes 假設對「完整 128-bit key」是過度樂觀**。CUB `DeviceRadixSort` 用 ~7~8-bit radix digit，**完整 128-bit key 實際約 16~19 passes**（除非只 sort 到必要的位元深度），且 key+value 每 pass 都要整包搬。同時可對照的實測基準：[libcusort](https://github.com/IlyaGrebnov/libcusort) 記載 V100 sort 64M 個 **uint32** 達 16 GKeys/s、且「比 cub::DeviceRadixSort 快 >2×」（即 CUB 於 V100、32-bit key ≈ 7~8 GKeys/s）。Hopper 頻寬約 V100 的 3~5×，故 CUB 於 **32-bit key** 約 20~40 GKeys/s，但 **128-bit key 因 passes×bytes 大增，實際很可能只有 ~6~10 GKeys/s**，比 3.75e10 低數倍。**必測**。
> 此值直接決定 closest-pair 是否追得上 Part 3 的產生速率（修正後 ~1.6e11/s）——若 sort 實際只有 ~1e10 量級，**sort 才是真瓶頸，產生端算再快也沒用**（呼應 [00]/[04] 的定調；sort 越慢，「sort 是瓶頸」的結論越成立）。

---

## 參考來源
- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) — SM 資源上限、compute cap 9.0、INT32:FP32=1:2
- [NVIDIA Ampere Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/) — GA10x/Ada SM 每 partition：一條 datapath 16 FP32、另一條 16 FP32+16 INT32（第二條 FP32 與 INT32 互斥），故 **INT32 = 64/SM/clk**（Ada 同）
- [NVIDIA Ada GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) — 128 FP32 CUDA core/SM、64 INT32/SM
- [CUDA C++ Language Extensions（`__launch_bounds__`）](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)
- [CUDA Pro Tip: Grid-Stride Loops](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)
- [Kernel Fusion / Launch Overhead（NVIDIA blog）](https://developer.nvidia.com/blog/kernel-fusion-in-nvidia-cuda-optimizing-memory-traffic-and-launch-overhead)
- [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) — graph replay 把 launch 成本壓到 ~1.3µs
- [Hashcat v6.2.6 RTX 4090 benchmark（原始 gist）](https://gist.github.com/Chick3nman/32e662a5bb63bc4f51b847bb422222fd) — SHA2-256 mode 1400 = 21.98 GH/s
- [CUB DeviceRadixSort](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html) / [CCCL repo](https://github.com/NVIDIA/cccl)（Apache-2.0）
- [libcusort](https://github.com/IlyaGrebnov/libcusort)（MIT）
- [CUDA Binary Utilities（cuobjdump / nvdisasm）](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)

> 數字警語：Part 3、4、6、7 的所有 GH/s、GB/s、keys/s 皆為**上界推導 + hashcat 類比 + 頻寬模型**，未經本機實測；凍結 kernel 的實際效率、H200 INT32 減半的真實衝擊、CUB 在 Hopper 的實測吞吐，**三者都必須在拿到機器後第一時間 microbenchmark**（見 [09-open-questions.md]），誤差可達 2~3×。
