# 01 — 題目精確分析與問題化簡

> 本文把 `mystery/q1_new_collision` 的題目**精確重述**，釘死「可以動 / 不可以動」的邊界，
> 並把它化簡成一個乾淨的 HPC 問題。**這是整個策略的地基**，讀完就知道該把工程力氣花在哪。

---

## 1. 題目精確重述

給定固定字串 `prefix`，對每個 64-bit 整數 `nonce`：

```
hash(nonce) = SHA-256( prefix(ASCII) ‖ nonce(8 bytes, big-endian) )
```

因為 `prefix ≤ 47 bytes`、`+8 bytes nonce`、`+padding+length`，訊息**恰好是單一 512-bit 區塊**（`build_base_words` 檢查 `len ≤ 55`，見 [collision.cu:78](../mystery/q1_new_collision/collision.cu)）。

**任務**：找出兩個**不同**的 nonce `a ≠ b`，使 `hash(a)` 與 `hash(b)` 的**開頭相同 bit 數最多**。
**分數 = 該相同 bit 數**（越大越好）。

**驗證**：`verify_collision.py`，計分 = `256 - (hash_a XOR hash_b).bit_length()`（見 [verify_collision.py:28](../mystery/q1_new_collision/verify_collision.py)）。

---

## 2. 邊界：可以動什麼、不可以動什麼

| 區域 | 內容 | 檔案位置 | 可否修改 |
|---|---|---|---|
| **雜湊核心** | `sha256_block()` | [collision.cu:52](../mystery/q1_new_collision/collision.cu) | ❌ 不可 |
| **訊息組法** | `build_base_words()` | [collision.cu:74](../mystery/q1_new_collision/collision.cu) | ❌ 不可 |
| **nonce 擺放** | `put_nonce()` | [collision.cu:101](../mystery/q1_new_collision/collision.cu) | ❌ 不可 |
| **輸出格式** | `write_solution()` / CSV | [collision.cu:182](../mystery/q1_new_collision/collision.cu) | ❌ 不可 |
| **其餘全部** | kernel 排程、記憶體、排序、比對、多 GPU、多節點、I/O | — | ✅ 開放 |

### 這條邊界的三個直接推論

1. **不能碰密碼學最佳化。** midstate 預算、early-exit、手工 LOP3/funnel-shift 塞進壓縮函數 —— 全部禁止，因為 `sha256_block` 必須原樣呼叫。挖礦/hashcat 那套「把 SHA 本身榨快」的技巧在本題**大部分用不上**。
   - ⚠️ 唯一例外：`hash_kernel`（呼叫 `sha256_block` 的**外層 wrapper**）是開放的。所以 launch config、佔用率、stream、CUDA Graph、以及**把「算完直接分桶/插表」融合進同一個 kernel**（省去把所有 hash 寫回 HBM 再讀出）都是合法且重要的槓桿。詳見 [08](08-fixed-kernel-throughput.md)。

2. **這不是密碼分析題。** 全 64 輪、函數固定、輸出是均勻亂數 → 沒有任何差分路徑 / SAT / reduced-round 手法可用（那些只對「可改 IV / 可減輪」的情境有意義）。本題就是**純暴力生日攻擊**。

3. **範本的瓶頸就是出題者要你解的題。** 範本（[collision.cu:196](../mystery/q1_new_collision/collision.cu)）的三個弱點被明白標成 `★ 優化點`：
   - 把**全部** hash 複製回 host，再用**單執行緒 `qsort`** 排序 → 排序期間 GPU 全閒置。
   - 一次把所有 hash 存起來（每筆 24 bytes）→ 記憶體上限卡住掃描量。
   - 只用**1 張 GPU**。
   - （額外 bug：範本用 `uint32_t` 當索引與 `(uint32_t)n`，[collision.cu:240](../mystery/q1_new_collision/collision.cu)，**單趟實際上限 < 2^32**；重寫開放區時必須改成 64-bit 索引 + 分批。）

---

## 3. 核心化簡：這是一個「分散式排序 / 最近對」問題

### 3.1 分數 ≈ 2·log₂(N)

設你在時限內能**產生並互相比對**的 distinct hash 數量為 `N`。
N 個均勻隨機字串中，任一對開頭相同 `b` bits 的機率是 `2^-b`；配對數約 `N²/2`。
期望「最長共同前綴」`B*` 落在 `(N²/2)·2^(-B*) ≈ 1` 之處：

```
B* ≈ 2·log₂(N) − 1     （標準差 ≈ 1.9 bits，Gumbel 分佈）
```

**每讓 N 翻 4 倍（+2 bit 算力/儲存），分數 +2 bit。** 反過來，多拿 1 分要 N × √2。
（Sanity check：範本 N=2×10⁷≈2^24.25，公式給 ~47.5 bit，範本示範拿到 51 bit —— 在 +2σ 幸運範圍內，吻合。）

### 3.2 關鍵洞察：**分數由 N 決定，而 N 由「吞吐」決定，不是由「記憶體」決定**

天真想法：「要比對 N 個就要存 N 個 → 記憶體是天花板」。**錯。**
用 **prefix-partitioning（依 hash 前 p bits 做 radix 分割）** 可以把記憶體和 N 脫鉤：

- 把整個雜湊空間依「前 p bits」切成 `2^p` 個 partition，每個 partition 只有 `N/2^p` 筆，可塞進 HBM。
- **全域最佳的那一對，兩個元素的前 p bits 一定相同（因為它們共同前綴 > p）→ 必落在同一個 partition。**
- 因此只要在每個 partition 內部各自找最近對，就能找到全域最佳，**完全達到 2·log₂(N)**，而峰值記憶體只需 `N/2^p`。

> 結論：**記憶體不是根本上限，時間才是。** 真正的天花板是「10 分鐘內能把多少 hash 推過 生成→分割→排序→掃描 這條 pipeline」。這正是一道**分散式排序 / 最近對（closest-pair）吞吐題**。

### 3.3 為什麼 Distinguished Points / rho 在本題不划算

VW distinguished-points 碰撞搜尋是「低記憶體」神器，但它攜帶的 state 就是 **64-bit 的 nonce**，所以它找到的碰撞**上限 ~64 bit**（前 64 bit 相等）。
而只要排序 pipeline 能處理 `N > 2^32`（存 2^32 筆 ×16B = 64GB，單張 H200 就吃得下），分數就 `> 64 bit`，直接贏過 DP。
→ **本題不用 DP**（僅在 09 存檔備查其推導）。

### 3.4 有沒有比 2·log₂(N) 更好的演算法？

**沒有。** SHA-256 輸出是均勻隨機、函數又被凍結，輸出端毫無可利用的結構；輸入 nonce 雖自由但不影響輸出分佈。
所以 `2·log₂(N)` 是資訊理論上限，**本題 100% 是吞吐競賽**——誰的 pipeline 端到端最快、誰能把 N 推最大，誰的分數最高。這與主辦方說的「比的是資料處理與平行化的效率」完全一致。

---

## 4. 約束清單（重寫開放區時必須滿足）

- ⏱ **單次執行 ≤ 10 分鐘**。可重複提交，取最佳（→ 每次跑都拚更大的 N；且 ±1.9 bit 的 variance 讓「多跑幾次取最好」能多撈 2~3 bit，見 [09](09-competition-intel.md)）。
- 🔒 `sha256_block` / `build_base_words` / `put_nonce` / CSV 格式**原樣不動**。
- ✔ 提交的 `(a,b)` 必須**相異**且能通過 `verify_collision.py`。
- 📄 輸出檔名 `solution_<bits>.csv`、欄位順序固定，錯了 0 分。
- 🧮 重寫開放區要修掉範本的 32-bit 索引限制（改 64-bit + 分批）。

---

## 5. 一句話策略

> **把它當成「10 分鐘內對盡可能多的 SHA-256 輸出做分散式排序 / 最近對」的 HPC 吞吐題來打。**
> 工程重點依序是：① 生成與排序/分桶的**融合與重疊**（GPU 不閒置）、② **prefix-partitioning** 讓記憶體不設限、③ **8×GPU + 2×node** 的分割與 shuffle、④ 最小化 bytes/key。
> 詳細打法見 [03-winning-strategy.md](03-winning-strategy.md)。
