# 10 — 必做 microbenchmark + 拿到機器後的決策樹

> 所有理論數字都是**上界推導**，誤差可能 2~3×。這份是「一坐上機器就照做」的清單：先量幾個關鍵數字，用它們選定 pipeline 佈局。**量測比任何理論值錢。**

---

## Part 1 — 優先 microbenchmark（照順序做）

| # | 量什麼 | 指令/方法 | 決定什麼 | 門檻判準 |
|---|---|---|---|---|
| **1** | **固定 kernel 單卡 SHA-256 速率** | 用原封不動的 `sha256_block` 寫個只算不存的 kernel，計時 N/秒（單卡 + 16 卡） | 生成預算 → N（`N∝√預算`，影響 ~1 bit/2×） | <4 GH/s：時程吃緊；>10：舒服 |
| **2** | **8-GPU NVLink 協同排序 keys/s + all-to-all 開銷** | `nccl-tests` (alltoall)、`cub::DeviceRadixSort` 多卡雛形 | **能否把 H 拉到 2^36 → 能否破 80** | all-to-all >300GB/s & 開銷<15% → 可行 |
| **3** | **RAM 實際容量** | `free -g`（每 node） | RAM staging 上限（H 替代路徑） | 2TB/node → H~2^37 可行 |
| **4** | `cub::DeviceRadixSort` uint64 單卡 keys/s | cub 內建 benchmark | sort 是否跟得上重算（通常跟得上） | >4 GKeys/s 即足 |
| **5** | NVLink 拓撲/頻寬 | `nvidia-smi topo -m`（找 NV# / NVSwitch） | 協同排序前提 | 全 NV18/NVSwitch → 理想 |
| 6 | NVMe 循序讀/寫（僅為排除） | `fio` 大區塊 | 確認 NVMe 不值得當計算路徑 | 供 [07](07-multinode-and-io.md) 對照 |
| 7 | IB 多軌 bisection（僅 ROUTE 才需要） | `ib_write_bw`, `nccl-tests` | SPLIT 主線用不到 | — |

**運維紅線**：⚠️ **不要啟動 `opensmd`**（IB fabric 已配置，多開 SM 會打架）。只做唯讀檢查（`ibstat`, `nvidia-smi topo -m`）。不要重裝 driver/OFED。

---

## Part 2 — 決策樹（用量到的數字選佈局）

```
量 #1 (kernel GH/s) 與 #2 (協同排序可行?)
│
├─ 先做 HBM-only 單趟版（不管上面結果都先做）───────► 安全地板 ~73-74，穩拿分
│
├─ #2 協同排序可行 (all-to-all>300GB/s, 開銷<15%)?
│   ├─ 是 ► 主線：regen + 8-GPU 協同(H~2^36) + SPLIT ──► ~79-81（衝 80）
│   │        再看 #1：
│   │        ├─ kernel 快(>10GH/s) ► N 抓 2^40+/node，摸 ~81
│   │        └─ kernel 慢(<5GH/s)  ► N 降，守 ~78-79 + best-of-k
│   └─ 否（協同排序難寫/慢）
│        └─ 用 RAM staging (H~2^37, 見 #3) 或 每卡各自分桶(H~2^33)
│               ├─ RAM staging ok ► ~80-81
│               └─ 只能各卡獨立 ► ~77，best-of-k 撈到 ~79
│
└─ 行有餘力 ► 跨節點協同 +1.5 bit(~82) 或 best-of-k 多跑幾輪
```

**best-of-k**：分數 variance ≈ 1.9 bit（Gumbel）。同樣 N 多跑 k 次取最好，期望 +`1.9×(視 k)`：k=4 約 +2.5 bit、k=10 約 +3.5 bit。因單趟主線已吃滿 10 分鐘，best-of-k 適用於「HBM-only 快版」或「稍微縮小 N 換多跑幾輪」。詳見 [09](09-competition-intel.md)。

---

## Part 3 — 風險登記表

| 風險 | 機率 | 影響 | 緩解 |
|---|---|---|---|
| 固定 kernel 比預期慢（INT32 減半） | 中 | N 降（但 √ 關係，僅 ~1bit/2×） | 先量 #1；`N∝√率` 讓它不致命 |
| 8-GPU 協同排序難寫/超時 | 中 | 卡在 ~77 上不去 | 備援 RAM staging；或先交 HBM-only |
| 踩 10 分鐘硬上限 | 中 | 該筆 0 分 | 留 30s 收尾；best-so-far checkpoint；watchdog |
| CSV 格式寫錯 | 低 | 0 分 | 用 frozen `write_solution`，最後才寫、冪等 |
| 範本 32-bit 索引沒改 | 中 | 單趟 <2^32 上限 | 重寫開放區用 64-bit 索引 + 分批 |
| 誤啟 opensmd / 重裝 driver | 低 | 全叢集抖動 | 只唯讀檢查 |
| nonce 重複 → verify 失敗 | 低 | 該筆 0 分 | 掃描時排除相同 nonce（範本已有） |

---

## Part 4 — 尚待確認的問題（拿到完整賽制後）

1. **計分細節**：基本分 + 排名分的實際權重？（差 1~2 bit 能跳幾名 → 決定要不要拚跨節點 +1.5bit）
2. **prefix 是否賽前公布**？固定 prefix → 可先暖機調參；賽中才給 → pipeline 要能秒啟動。
3. **是否禁止預計算**？本題 nonce→hash 無法預計算跨 prefix，但確認規則。
4. **提交次數上限 / 冷卻時間**？決定 best-of-k 能跑幾輪。
5. **是否有多題共享機時**？影響資源分配。
6. RAM 到底 2TB total 還是 2TB/node？（直接改 H 上限）

---

## Part 5 — 30 分鐘上機 playbook

1. `nvidia-smi topo -m`、`ibstat`、`free -g`、`nvcc --list-gpu-code`（唯讀，1 min）
2. 編範本 `make ARCH=sm_90`，跑 `./collision hipac_demo 20000000` 確認基線（2 min）
3. 跑 microbenchmark #1（kernel GH/s，單卡+16卡）（5 min）
4. 跑 #2（NVLink 協同排序雛形）+ #4（cub sort）（10 min）
5. 用 Part 2 決策樹選佈局（2 min）
6. 先交一版 HBM-only 確保有分（剩餘時間）
7. 依 #1/#2 結果決定是否上 regen 主線衝 80

---

> 一句話：**先量 kernel 速度與 NVLink 協同排序這兩個數字，其餘佈局選擇都由它們推出來。** 主線是 partition-by-regeneration；NVMe 別碰。
