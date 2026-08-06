// =====================================================================
// HiPAC 隱藏題1 — SHA-256 部分碰撞搜尋
//
//   給定固定字串 prefix，對每個 64-bit 整數 nonce 計算：
//       hash(nonce) = SHA-256( prefix(ASCII) ‖ nonce(8 bytes 大端) )
//
//   目標：找出兩個「不同」的 nonce a、b，讓 hash(a) 與 hash(b)
//         開頭有最多相同的 bits。相同的 bits 越多，分數越高。
//
//  【不可修改區】 —— SHA-256、訊息組法、nonce 擺放位置
//  【開放區】    —— partition-by-regeneration + GPU radix sort + 多 GPU
//
//  演算法（開放區）
//  ----------------
//  掃 N 個雜湊，期望最長共同前綴 ≈ 2·log2(N) − 1。但記憶體裝不下 N，
//  所以用 prefix-partitioning：把雜湊空間依「前 p bits」切成 P = 2^p 桶，
//  每個桶重掃整段 nonce 範圍、只留落在該桶的、在 GPU 內排序後掃相鄰對。
//  全域最佳那對的共同前綴遠大於 p ⇒ 前 p bits 必相同 ⇒ 必在同一桶，
//  所以分桶不會漏答案。
//
//  設每桶能容納 H 筆、時限內全叢集能生成 G 個雜湊，可推出
//      期望分數 ≈ log2(H) + log2(G) − 1
//  這個式子與掃描範圍 S 無關（只要 S 大到 partition 跑不完），所以
//  「猜錯 GPU 速度」不會毀掉這一趟 —— 只會少做幾個桶而已。
//
//  相對範本的四個改動（依重要性）：
//   1. 單一長時間執行的 process：範本每 20M nonce 就重啟一次，CUDA context
//      初始化與 srun 啟動吃掉 99.9% 的時間（實測 10.5 GH/s 的卡端到端只有
//      4.44 MH/s）。這裡一次 launch 掃數百億個 nonce。
//   2. 生成與分桶融合進同一個 kernel：雜湊不再完整落地 HBM 又讀回。
//   3. 排序與比對全在 GPU（CUB radix sort + 相鄰掃描），不回主機 qsort。
//   4. 一個 process 開滿本機所有 GPU（OpenMP 一 thread 一卡）。
// =====================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <signal.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#ifdef _OPENMP
#include <omp.h>
#endif

#define CUDA_CHECK(call) do {                                              \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
        fprintf(stderr, "[CUDA錯誤] %s (%s:%d)\n",                         \
                cudaGetErrorString(_e), __FILE__, __LINE__);              \
        exit(1);                                                           \
    }                                                                      \
} while (0)


// =====================================================================
// ★★★ 雜湊計算區 不可修改 ★★★
// =====================================================================
__constant__ uint32_t K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

#define ROTR(x,n)  (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x,y,z)  (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x)  (ROTR(x,2)  ^ ROTR(x,13) ^ ROTR(x,22))
#define EP1(x)  (ROTR(x,6)  ^ ROTR(x,11) ^ ROTR(x,25))
#define SIG0(x) (ROTR(x,7)  ^ ROTR(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTR(x,17) ^ ROTR(x,19) ^ ((x) >> 10))

// 標準 SHA-256 壓縮函式（訊息固定為單一 512-bit 區塊）
__device__ void sha256_block(const uint32_t in[16], uint32_t out[8]) {
    uint32_t w[64];
#pragma unroll
    for (int i = 0; i < 16; i++) w[i] = in[i];
#pragma unroll
    for (int i = 16; i < 64; i++)
        w[i] = SIG1(w[i-2]) + w[i-7] + SIG0(w[i-15]) + w[i-16];

    uint32_t a=0x6a09e667,b=0xbb67ae85,c=0x3c6ef372,d=0xa54ff53a,
             e=0x510e527f,f=0x9b05688c,g=0x1f83d9ab,h=0x5be0cd19;
#pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + EP1(e) + CH(e,f,g) + K[i] + w[i];
        uint32_t t2 = EP0(a) + MAJ(a,b,c);
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    out[0]=a+0x6a09e667; out[1]=b+0xbb67ae85; out[2]=c+0x3c6ef372; out[3]=d+0xa54ff53a;
    out[4]=e+0x510e527f; out[5]=f+0x9b05688c; out[6]=g+0x1f83d9ab; out[7]=h+0x5be0cd19;
}

// 主機端：把 prefix + padding + 長度 組成 SHA-256 的 w[0..15]，
// nonce 的 8 bytes 先留 0，交給 kernel 填。
static void build_base_words(const char *prefix, uint32_t w[16],
                             int *nonce_word, int *nonce_shift) {
    size_t plen = strlen(prefix);
    size_t len  = plen + 8;                 // +8 = nonce
    if (len > 55) {
        fprintf(stderr, "prefix 太長（需 <= 47 bytes）\n");
        exit(1);
    }

    unsigned char block[64] = {0};
    memcpy(block, prefix, plen);
    block[len] = 0x80;
    uint64_t bits = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++) block[63 - i] = (unsigned char)(bits >> (8 * i));

    for (int i = 0; i < 16; i++)
        w[i] = ((uint32_t)block[i*4] << 24) | ((uint32_t)block[i*4+1] << 16)
             | ((uint32_t)block[i*4+2] << 8) |  (uint32_t)block[i*4+3];

    int off = (int)plen;
    *nonce_word  = off / 4;
    *nonce_shift = (off % 4) * 8;
    for (int b = off; b < off + 8; b++)      // nonce 的 8 bytes 清成 0
        w[b/4] &= ~(0xFFu << (24 - 8 * (b % 4)));
}

// 裝置端：把 8 bytes 的 nonce 填進 w[]（可能跨越三個 word）
__device__ __forceinline__ void put_nonce(uint32_t w[16], int nonce_word,
                                          int nonce_shift, uint64_t nonce) {
    uint32_t nhi = (uint32_t)(nonce >> 32);
    uint32_t nlo = (uint32_t)nonce;
    if (nonce_shift == 0) {
        w[nonce_word]     = nhi;
        w[nonce_word + 1] = nlo;
    } else {
        w[nonce_word]     |= nhi >> nonce_shift;
        w[nonce_word + 1]  = (nhi << (32 - nonce_shift)) | (nlo >> nonce_shift);
        w[nonce_word + 2] |= nlo << (32 - nonce_shift);
    }
}
// =====================================================================
// ★★★ 不可修改區　結束 ★★★
//   以下全部開放：kernel 排程、記憶體配置、排序、多 GPU… 隨你改寫。
// =====================================================================


// ---------------------------------------------------------------------
// 組態
// ---------------------------------------------------------------------
#ifndef HIPAC_TPB
#define HIPAC_TPB 256                 // 每 block thread 數（可用 -DHIPAC_TPB= 覆寫）
#endif
#define WARP_MASK 0xFFFFFFFFu

// 單次 CUB radix sort 的筆數上限 = 顯存能裝多少就多少。
// CUDA 12.9 的 cub/detail/choose_offset.cuh 對 sizeof(NumItemsT) > 4 會選
// unsigned long long 當 offset，所以傳 size_t 沒有 2^31 限制。
// 需要人為壓低時用環境變數 MAX_SORT_ITEMS。
#define DEFAULT_MAX_SORT_ITEMS (~0ULL)

__constant__ uint32_t c_base_words[16];


// ---------------------------------------------------------------------
// Kernel 1：純生成，量測本卡 SHA-256 吞吐（不寫出資料）
// ---------------------------------------------------------------------
__global__ __launch_bounds__(HIPAC_TPB)
void bench_kernel(int nonce_word, int nonce_shift,
                  uint64_t start_nonce, uint64_t count, uint32_t *sink) {
    const uint64_t gid    = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    uint32_t acc = 0;
    for (uint64_t i = gid; i < count; i += stride) {
        uint32_t w[16];
#pragma unroll
        for (int k = 0; k < 16; k++) w[k] = c_base_words[k];
        put_nonce(w, nonce_word, nonce_shift, start_nonce + i);
        uint32_t h[8];
        sha256_block(w, h);
        acc ^= h[0];
    }
    sink[gid & 1023u] = acc;          // 防止整段運算被最佳化掉
}


// ---------------------------------------------------------------------
// Kernel 2（核心）：生成 + 分桶過濾，融合成單一 kernel
//
// 雜湊「從不完整落地」：算完當場判斷是否屬於本 partition，只有屬於的
// 才寫出 (key, nonce)。省掉範本「寫全部雜湊再讀回排序」的一整趟往返，
// 也讓峰值記憶體與掃描量脫鉤。
//
// key = 雜湊的 bit [p, p+64)。前 p bits 是 partition id，同桶內必然相同、
// 不必存 ⇒ 共同前綴 = p + clz64(keyA ^ keyB)，可分辨到 p+64 bits。
//
// 寫出用 warp-aggregated atomics：整個 warp 只做一次 atomicAdd。
// ---------------------------------------------------------------------
__global__ __launch_bounds__(HIPAC_TPB)
void gen_filter_kernel(int nonce_word, int nonce_shift,
                       uint64_t start_nonce, uint64_t count,
                       int part_bits, uint64_t part_id,
                       uint64_t * __restrict__ out_key,
                       uint64_t * __restrict__ out_nonce,
                       unsigned long long *counter,
                       unsigned long long capacity) {
    const uint64_t gid    = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    const unsigned lane   = threadIdx.x & 31u;
    const int      shift  = 64 - part_bits;          // part_bits ∈ [1,32]

    // 迴圈界線只跟 count 有關（整個 warp 一致），warp 不會提早解散，
    // 下面的 __ballot_sync / __shfl_sync 才合法。
    for (uint64_t base = 0; base < count; base += stride) {
        uint64_t i    = base + gid;
        bool     keep = false;
        uint64_t key = 0, nonce = 0;

        if (i < count) {
            nonce = start_nonce + i;
            uint32_t w[16];
#pragma unroll
            for (int k = 0; k < 16; k++) w[k] = c_base_words[k];
            put_nonce(w, nonce_word, nonce_shift, nonce);
            uint32_t h[8];
            sha256_block(w, h);

            uint64_t h0 = ((uint64_t)h[0] << 32) | h[1];   // 雜湊前 64 bits
            uint64_t h1 = ((uint64_t)h[2] << 32) | h[3];   // 次 64 bits
            if ((h0 >> shift) == part_id) {
                keep = true;
                key  = (h0 << part_bits) | (h1 >> shift);
            }
        }

        unsigned ballot = __ballot_sync(WARP_MASK, keep);
        if (ballot) {
            const int leader = __ffs((int)ballot) - 1;
            unsigned long long b = 0;
            if ((int)lane == leader)
                b = atomicAdd(counter, (unsigned long long)__popc(ballot));
            b = __shfl_sync(WARP_MASK, b, leader);
            if (keep) {
                unsigned long long slot =
                    b + (unsigned long long)__popc(ballot & ((1u << lane) - 1u));
                if (slot < capacity) {
                    out_key[slot]   = key;
                    out_nonce[slot] = nonce;
                }
            }
        }
    }
}


// ---------------------------------------------------------------------
// Kernel 3：排序後掃相鄰對，找共同前綴最長的一對。
// 打包成 ((bits+1) << 32) | index 後 atomicMax；先做 warp reduce，
// 每個 warp 只碰一次全域 atomic。
// ---------------------------------------------------------------------
__global__ void scan_pairs_kernel(const uint64_t * __restrict__ keys,
                                  const uint64_t * __restrict__ nonces,
                                  uint64_t n, unsigned long long *best) {
    const uint64_t gid    = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    unsigned long long local = 0;

    for (uint64_t i = gid; i + 1 < n; i += stride) {
        if (nonces[i] == nonces[i + 1]) continue;      // 必須是兩個不同 nonce
        unsigned bits = (unsigned)__clzll((long long)(keys[i] ^ keys[i + 1]));
        unsigned long long cand =
            ((unsigned long long)(bits + 1u) << 32) | (unsigned long long)(uint32_t)i;
        if (cand > local) local = cand;
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        unsigned long long other = __shfl_down_sync(WARP_MASK, local, off);
        if (other > local) local = other;
    }
    if ((threadIdx.x & 31u) == 0 && local != 0) atomicMax(best, local);
}


// ---------------------------------------------------------------------
// Kernel 4：對指定 nonce 算出完整 256-bit 雜湊，供最終精確驗算。
// GPU 排序鍵只到 p+64 bits，最後這一步用凍結的 sha256_block 重算完整
// 雜湊，得到與 verify_collision.py 完全一致的 match_bits。
// ---------------------------------------------------------------------
__global__ void full_hash_kernel(int nonce_word, int nonce_shift,
                                 const uint64_t *nonces, uint32_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t w[16];
#pragma unroll
    for (int k = 0; k < 16; k++) w[k] = c_base_words[k];
    put_nonce(w, nonce_word, nonce_shift, nonces[i]);
    uint32_t h[8];
    sha256_block(w, h);
#pragma unroll
    for (int k = 0; k < 8; k++) out[i * 8 + k] = h[k];
}


// ---------------------------------------------------------------------
// 主機端小工具
// ---------------------------------------------------------------------
static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

// 完整 256-bit 共同前綴，等價於 verify_collision.py 的
// 256 - (hash_a ^ hash_b).bit_length()
static int exact_common_prefix_bits(const uint32_t a[8], const uint32_t b[8]) {
    for (int i = 0; i < 8; i++) {
        uint32_t x = a[i] ^ b[i];
        if (x) return i * 32 + (int)__builtin_clz(x);
    }
    return 256;
}

static volatile sig_atomic_t stop_requested = 0;
static void request_stop(int signal_number) { (void)signal_number; stop_requested = 1; }


// =====================================================================
// ★★★ 輸出檔案格式 —— 請勿修改 ★★★
//
//   找到結果時輸出 CSV，檔名與內容格式固定：
//
//       檔名  ： solution_<共同前綴bits>.csv     例如 solution_51.csv
//       內容  ：
//           prefix,nonce_a,nonce_b,match_bits
//           hipac_demo,13644422,18436129,51
//
//   評分系統會讀取這個檔案。檔名規則、欄位順序、欄位名都不可更動，
//   否則該筆提交以 0 分計。
// =====================================================================
static void write_solution(const char *prefix, uint64_t nonce_a, uint64_t nonce_b,
                           int match_bits) {
    char filename[128];
    sprintf(filename, "solution_%d.csv", match_bits);
    FILE *fp = fopen(filename, "w");
    if (!fp) { fprintf(stderr, "無法建立檔案 %s\n", filename); exit(1); }
    fprintf(fp, "prefix,nonce_a,nonce_b,match_bits\n");
    fprintf(fp, "%s,%llu,%llu,%d\n", prefix,
            (unsigned long long)nonce_a, (unsigned long long)nonce_b, match_bits);
    fclose(fp);
    printf("答案已寫入 %s\n", filename);
}


// ---------------------------------------------------------------------
// 執行期組態與跨執行緒共用狀態
// ---------------------------------------------------------------------
typedef struct {
    const char *prefix;
    int      nonce_word, nonce_shift;
    uint64_t nonce_start;      // 掃描起點
    uint64_t span;             // S：每個 partition 要重掃的 nonce 數
    int      part_bits;        // p，P = 2^p
    uint64_t partitions;       // P
    int      world_ranks;      // 全叢集 GPU 總數
    double   deadline;         // 生成階段截止時刻（monotonic）
    int      smoke;
} RunCfg;

typedef struct { int bits; uint64_t a, b; int valid; } BestPair;
static BestPair g_best = {0, 0, 0, 0};

// 登記全域最佳（bits 必須已是精確值），必要時寫 CSV
static void publish_best(const char *prefix, int bits, uint64_t a, uint64_t b) {
#ifdef _OPENMP
#pragma omp critical(hipac_best)
#endif
    {
        if (bits > g_best.bits) {
            g_best.bits = bits; g_best.a = a; g_best.b = b; g_best.valid = 1;
            write_solution(prefix, a, b, bits);
            fflush(stdout);
        }
    }
}


// ---------------------------------------------------------------------
// 每張 GPU 的狀態
// ---------------------------------------------------------------------
typedef struct {
    int      dev, rank;
    int      grid, scan_grid;
    uint64_t capacity;               // 單一 partition 能容納的筆數
    uint64_t *d_key[2], *d_val[2];
    void     *d_temp;
    size_t    temp_bytes;
    unsigned long long *d_counter, *d_best;
    uint32_t *d_sink;
    uint64_t *d_pair;
    uint32_t *d_pair_hash;
    double    rate;                  // hash/s
    // 統計 / 瓶頸診斷
    uint64_t  hashes, parts, kept_total;
    double    gen_s, sort_s, scan_s;
    int       best_bits;
    uint64_t  best_a, best_b;
    uint64_t  overflow;
    size_t    free_bytes, total_bytes;
} DevCtx;

static uint64_t env_u64(const char *name, uint64_t fallback) {
    const char *s = getenv(name);
    if (!s || !*s) return fallback;
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (*end != '\0' || v == 0) return fallback;
    return (uint64_t)v;
}

// 量測本卡 SHA-256 吞吐（hash/s）
static double measure_rate(DevCtx *c, int nonce_word, int nonce_shift, uint64_t start) {
    uint64_t probe = 1ULL << 24;
    double   secs  = 0.0;
    for (int round = 0; round < 4; round++) {
        double t0 = monotonic_seconds();
        bench_kernel<<<c->grid, HIPAC_TPB>>>(nonce_word, nonce_shift, start, probe, c->d_sink);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        secs = monotonic_seconds() - t0;
        if (secs > 0.25) break;
        double scale = (secs > 1e-4) ? (0.35 / secs) : 16.0;
        if (scale > 32.0) scale = 32.0;
        probe = (uint64_t)((double)probe * scale) + (1ULL << 20);
    }
    return (secs > 0.0) ? ((double)probe / secs) : 0.0;
}

// 依剩餘顯存決定「單一 partition 最多能排幾筆」。
// 每筆 32 B：key 雙緩衝 2×8 + value 雙緩衝 2×8。這個 n 就是分數公式裡的 H。
static uint64_t plan_capacity(DevCtx *c, uint64_t max_sort_items, size_t *temp_out) {
    CUDA_CHECK(cudaMemGetInfo(&c->free_bytes, &c->total_bytes));
    size_t budget = (size_t)((double)c->free_bytes * 0.88);   // 留給 context / 碎片

    uint64_t n = (uint64_t)(budget / 34);
    if (n > max_sort_items) n = max_sort_items;

    size_t tb = 0;
    for (int it = 0; it < 24; it++) {
        cub::DoubleBuffer<uint64_t> kb(NULL, NULL), vb(NULL, NULL);
        tb = 0;
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(NULL, tb, kb, vb, (size_t)n));
        if ((size_t)n * 32 + tb <= budget) break;
        n = (uint64_t)((double)n * 0.94);
    }
    *temp_out = tb;
    return n;
}

static void dev_alloc(DevCtx *c) {
    size_t bytes = (size_t)c->capacity * sizeof(uint64_t);
    CUDA_CHECK(cudaMalloc(&c->d_key[0], bytes));
    CUDA_CHECK(cudaMalloc(&c->d_key[1], bytes));
    CUDA_CHECK(cudaMalloc(&c->d_val[0], bytes));
    CUDA_CHECK(cudaMalloc(&c->d_val[1], bytes));

    cub::DoubleBuffer<uint64_t> kb(c->d_key[0], c->d_key[1]);
    cub::DoubleBuffer<uint64_t> vb(c->d_val[0], c->d_val[1]);
    c->temp_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(NULL, c->temp_bytes, kb, vb,
                                               (size_t)c->capacity));
    CUDA_CHECK(cudaMalloc(&c->d_temp, c->temp_bytes));
    CUDA_CHECK(cudaMalloc(&c->d_counter, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&c->d_best,    sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&c->d_pair,      2 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&c->d_pair_hash, 16 * sizeof(uint32_t)));
}

// 用凍結的 sha256_block 重算完整雜湊，回傳精確共同前綴 bit 數
static int exact_bits_for_pair(DevCtx *c, int nonce_word, int nonce_shift,
                               uint64_t a, uint64_t b) {
    uint64_t pair[2] = { a, b };
    uint32_t hh[16];
    CUDA_CHECK(cudaMemcpy(c->d_pair, pair, sizeof(pair), cudaMemcpyHostToDevice));
    full_hash_kernel<<<1, 32>>>(nonce_word, nonce_shift, c->d_pair, c->d_pair_hash, 2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(hh, c->d_pair_hash, sizeof(hh), cudaMemcpyDeviceToHost));
    return exact_common_prefix_bits(hh, hh + 8);
}


// ---------------------------------------------------------------------
// 單一 partition：重掃 span 個 nonce → 過濾 → 排序 → 掃相鄰對
// ---------------------------------------------------------------------
static uint64_t run_partition(DevCtx *c, const RunCfg *cfg,
                              uint64_t span_start, uint64_t part_id,
                              uint64_t chunk_size, int *gpu_bits,
                              uint64_t *best_a, uint64_t *best_b,
                              uint64_t *scanned_out) {
    const double t_part_start = monotonic_seconds();
    CUDA_CHECK(cudaMemset(c->d_counter, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(c->d_best,    0, sizeof(unsigned long long)));
    *gpu_bits = 0;

    uint64_t done = 0;
    while (done < cfg->span) {
        if (stop_requested || monotonic_seconds() >= cfg->deadline) break;
        uint64_t chunk = cfg->span - done;
        if (chunk > chunk_size) chunk = chunk_size;

        gen_filter_kernel<<<c->grid, HIPAC_TPB>>>(
            cfg->nonce_word, cfg->nonce_shift,
            span_start + done, chunk,
            cfg->part_bits, part_id,
            c->d_key[0], c->d_val[0], c->d_counter, c->capacity);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        done += chunk;
    }
    *scanned_out = done;
    const double t_gen_end = monotonic_seconds();
    c->gen_s += t_gen_end - t_part_start;

    unsigned long long produced = 0;
    CUDA_CHECK(cudaMemcpy(&produced, c->d_counter, sizeof(produced), cudaMemcpyDeviceToHost));
    if (produced > c->capacity) c->overflow += produced - c->capacity;
    uint64_t n = (produced > c->capacity) ? c->capacity : (uint64_t)produced;
    c->kept_total += n;
    if (n < 2) return n;

    cub::DoubleBuffer<uint64_t> kb(c->d_key[0], c->d_key[1]);
    cub::DoubleBuffer<uint64_t> vb(c->d_val[0], c->d_val[1]);
    size_t tb = c->temp_bytes;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(c->d_temp, tb, kb, vb, (size_t)n));
    CUDA_CHECK(cudaDeviceSynchronize());
    const double t_sort_end = monotonic_seconds();
    c->sort_s += t_sort_end - t_gen_end;

    scan_pairs_kernel<<<c->scan_grid, HIPAC_TPB>>>(kb.Current(), vb.Current(), n, c->d_best);
    CUDA_CHECK(cudaGetLastError());

    unsigned long long packed = 0;
    CUDA_CHECK(cudaMemcpy(&packed, c->d_best, sizeof(packed), cudaMemcpyDeviceToHost));
    c->scan_s += monotonic_seconds() - t_sort_end;
    if (packed == 0) return n;

    unsigned key_bits = (unsigned)(packed >> 32) - 1u;
    uint64_t idx      = (uint64_t)(uint32_t)packed;
    uint64_t pair[2];
    CUDA_CHECK(cudaMemcpy(pair, vb.Current() + idx, 2 * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));

    *gpu_bits = cfg->part_bits + (int)key_bits;
    *best_a   = pair[0];
    *best_b   = pair[1];
    return n;
}


// ---------------------------------------------------------------------
// 一張 GPU 的完整工作：領 partition、跑完、回報
// ---------------------------------------------------------------------
static void run_device(DevCtx *c, const RunCfg *cfg) {
    const double t0 = monotonic_seconds();
    uint64_t chunk_size = (uint64_t)(c->rate * 2.0);      // 每次 launch 約 2 秒
    if (chunk_size < (1ULL << 24)) chunk_size = 1ULL << 24;

    for (uint64_t epoch = 0; !stop_requested && monotonic_seconds() < cfg->deadline; epoch++) {
        // 正常情況 partition 用不完；只有低估 GPU 速度時才會進到 epoch >= 1，
        // 此時換一段全新的 nonce 範圍（等同多跑一次獨立的 best-of-k）。
        uint64_t span_start = cfg->nonce_start + epoch * cfg->span;

        for (uint64_t k = 0;; k++) {
            uint64_t part = (uint64_t)c->rank + k * (uint64_t)cfg->world_ranks;
            if (part >= cfg->partitions) break;
            if (stop_requested || monotonic_seconds() >= cfg->deadline) break;

            int      gbits = 0;
            uint64_t pa = 0, pb = 0, scanned = 0;
            double   ts = monotonic_seconds();
            uint64_t kept = run_partition(c, cfg, span_start, part, chunk_size,
                                          &gbits, &pa, &pb, &scanned);
            double   te = monotonic_seconds();
            c->hashes += scanned;
            c->parts++;

            if (gbits > c->best_bits && pa != pb) {
                int exact = exact_bits_for_pair(c, cfg->nonce_word, cfg->nonce_shift, pa, pb);
                if (exact > c->best_bits) {
                    c->best_bits = exact; c->best_a = pa; c->best_b = pb;
                    publish_best(cfg->prefix, exact, pa, pb);
                }
            }

            printf("[rank %2d] epoch %llu  partition %llu/%llu  掃 %.2f G → 留 %.1f M 筆  "
                   "%.1f 秒  %.2f GH/s  本卡最佳 %d bits\n",
                   c->rank, (unsigned long long)epoch,
                   (unsigned long long)part, (unsigned long long)cfg->partitions,
                   (double)scanned / 1e9, (double)kept / 1e6, te - ts,
                   (te > ts) ? ((double)scanned / (te - ts) / 1e9) : 0.0, c->best_bits);
            fflush(stdout);
        }
        if (cfg->smoke) break;                     // smoke：掃完一輪就結束
    }

    double dt = monotonic_seconds() - t0;
    printf("[rank %2d] 完成 %llu 個 partition，%.2f G hash / %.1f 秒 = %.2f GH/s，"
           "最佳 %d bits (a=%llu b=%llu)%s\n",
           c->rank, (unsigned long long)c->parts, (double)c->hashes / 1e9, dt,
           (dt > 0) ? ((double)c->hashes / dt / 1e9) : 0.0,
           c->best_bits, (unsigned long long)c->best_a, (unsigned long long)c->best_b,
           c->overflow ? "  ⚠ 有桶溢位（請調小 --span）" : "");
    printf("[rank %2d] 時間拆解：生成 %.1fs (%.1f%%)  排序 %.1fs (%.1f%%)  掃描 %.1fs (%.1f%%)  "
           "其他 %.1fs\n",
           c->rank, c->gen_s,  100.0 * c->gen_s  / (dt > 0 ? dt : 1),
                    c->sort_s, 100.0 * c->sort_s / (dt > 0 ? dt : 1),
                    c->scan_s, 100.0 * c->scan_s / (dt > 0 ? dt : 1),
           dt - c->gen_s - c->sort_s - c->scan_s);
    fflush(stdout);
}


// ---------------------------------------------------------------------
static void print_usage(const char *program) {
    fprintf(stderr,
      "用法：\n"
      "  %s [選項] [prefix]                       正式模式：跑到時間用完為止\n"
      "  %s --smoke [選項] [prefix] [total]       smoke：掃固定 total 個 nonce 後結束\n"
      "  %s --bench [prefix]                      只量 SHA-256 吞吐\n"
      "\n選項：\n"
      "  --seconds N     生成階段秒數上限（預設 540；比賽硬上限 600）\n"
      "  --start N       nonce 掃描起點（不同 run 用不同值 = best-of-k）\n"
      "  --ranks N       全叢集 GPU 總數（預設 SLURM_NTASKS × 本機 GPU 數）\n"
      "  --rank-base N   本 process 第一張 GPU 的 global rank（預設 SLURM_PROCID × GPU 數）\n"
      "  --partitions N  強制 partition 數（必須是 2 的冪）\n"
      "  --span N        強制每個 partition 重掃的 nonce 數\n"
      "\n環境變數：MAX_SORT_ITEMS（單次排序筆數上限，預設 %llu）\n",
      program, program, program, (unsigned long long)DEFAULT_MAX_SORT_ITEMS);
}

static uint64_t parse_u64_value(const char *text, const char *name, int allow_zero) {
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (text[0] == '\0' || *end != '\0' || (!allow_zero && value == 0)) {
        fprintf(stderr, "%s 必須是正整數：%s\n", name, text);
        exit(2);
    }
    return (uint64_t)value;
}

int main(int argc, char **argv) {
    int      smoke = 0, bench_only = 0;
    double   seconds = 540.0;
    uint64_t nonce_start = 0, force_parts = 0, force_span = 0;
    int      world_ranks = 0, rank_base = -1;

    int argi = 1;
    while (argi < argc && argv[argi][0] == '-') {
        const char *a = argv[argi];
        if      (!strcmp(a, "--smoke"))       { smoke = 1;      argi++; }
        else if (!strcmp(a, "--bench"))       { bench_only = 1; argi++; }
        else if (!strcmp(a, "--seconds")    && argi + 1 < argc) { seconds     = atof(argv[argi+1]); argi += 2; }
        else if (!strcmp(a, "--start")      && argi + 1 < argc) { nonce_start = parse_u64_value(argv[argi+1], "start", 1); argi += 2; }
        else if (!strcmp(a, "--ranks")      && argi + 1 < argc) { world_ranks = (int)parse_u64_value(argv[argi+1], "ranks", 0); argi += 2; }
        else if (!strcmp(a, "--rank-base")  && argi + 1 < argc) { rank_base   = (int)parse_u64_value(argv[argi+1], "rank-base", 1); argi += 2; }
        else if (!strcmp(a, "--partitions") && argi + 1 < argc) { force_parts = parse_u64_value(argv[argi+1], "partitions", 0); argi += 2; }
        else if (!strcmp(a, "--span")       && argi + 1 < argc) { force_span  = parse_u64_value(argv[argi+1], "span", 0); argi += 2; }
        else { print_usage(argv[0]); return 2; }
    }

    const char *prefix = (argi < argc) ? argv[argi++] : "hipac_demo";
    uint64_t    total  = 0;
    if (smoke) total = (argi < argc) ? parse_u64_value(argv[argi++], "total", 0) : 20000000ULL;
    if (argi != argc)   { print_usage(argv[0]); return 2; }
    if (seconds <= 0.0) { fprintf(stderr, "--seconds 必須為正數\n"); return 2; }

    uint32_t base_words[16];
    int nonce_word, nonce_shift;
    build_base_words(prefix, base_words, &nonce_word, &nonce_shift);

    int ngpu = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ngpu));
    if (ngpu <= 0) { fprintf(stderr, "找不到 CUDA 裝置\n"); return 1; }

    // 叢集拓樸：一個 process 用光本機可見的 GPU，process 由 Slurm 排（一 node 一個）
    const char *procid = getenv("SLURM_PROCID");
    const char *ntasks = getenv("SLURM_NTASKS");
    int my_proc = procid ? atoi(procid) : 0;
    int nprocs  = ntasks ? atoi(ntasks) : 1;
    if (nprocs <= 0) nprocs = 1;
    if (world_ranks <= 0) world_ranks = nprocs * ngpu;
    if (rank_base   <  0) rank_base   = my_proc * ngpu;

    printf("prefix = \"%s\"   本機 GPU %d 張   全叢集 %d rank   本機 rank_base = %d\n",
           prefix, ngpu, world_ranks, rank_base);
    fflush(stdout);

    const uint64_t max_sort_items = env_u64("MAX_SORT_ITEMS", DEFAULT_MAX_SORT_ITEMS);
    DevCtx *ctx = (DevCtx *)calloc((size_t)ngpu, sizeof(DevCtx));
    if (!ctx) { fprintf(stderr, "主機記憶體不足\n"); return 1; }

    // ---- 階段 1：每張卡量吞吐 + 決定單桶容量 ----
#ifdef _OPENMP
#pragma omp parallel num_threads(ngpu)
#endif
    {
#ifdef _OPENMP
        int t = omp_get_thread_num();
#else
        int t = 0;
#endif
        DevCtx *c = &ctx[t];
        c->dev = t; c->rank = rank_base + t;
        CUDA_CHECK(cudaSetDevice(c->dev));
        CUDA_CHECK(cudaMemcpyToSymbol(c_base_words, base_words, sizeof(base_words)));
        CUDA_CHECK(cudaMalloc(&c->d_sink, 1024 * sizeof(uint32_t)));

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, c->dev));
        int blocks_per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &blocks_per_sm, gen_filter_kernel, HIPAC_TPB, 0));
        if (blocks_per_sm < 1) blocks_per_sm = 1;
        c->grid      = prop.multiProcessorCount * blocks_per_sm;
        c->scan_grid = prop.multiProcessorCount * 8;

        size_t temp_bytes_probe = 0;
        c->rate     = measure_rate(c, nonce_word, nonce_shift, nonce_start);
        c->capacity = plan_capacity(c, max_sort_items, &temp_bytes_probe);

#ifdef _OPENMP
#pragma omp critical(hipac_print)
#endif
        {
            printf("[rank %2d] %s  %d SM  grid=%d (%d blk/SM)  吞吐 %.2f GH/s\n",
                   c->rank, prop.name, prop.multiProcessorCount,
                   c->grid, blocks_per_sm, c->rate / 1e9);
            printf("[rank %2d] HBM %.1f/%.1f GB 可用  單桶容量 H = %.3f G 筆 (log2=%.2f)  "
                   "buf %.1f GB + CUB temp %.2f GB\n",
                   c->rank, (double)c->free_bytes / 1e9, (double)c->total_bytes / 1e9,
                   (double)c->capacity / 1e9, log2((double)c->capacity),
                   (double)c->capacity * 32 / 1e9, (double)temp_bytes_probe / 1e9);
            fflush(stdout);
        }
    }

    if (bench_only) {
        double sum = 0.0;
        for (int i = 0; i < ngpu; i++) sum += ctx[i].rate;
        printf("\n本機合計 %.2f GH/s（%d 張 GPU）\n", sum / 1e9, ngpu);
        printf("推估全叢集（%d rank）：%.2f GH/s\n", world_ranks, sum / ngpu * world_ranks / 1e9);
        return 0;
    }

    // ---- 階段 2：用最保守的一張卡決定 S 與 P，讓所有 rank 用同一組參數 ----
    double   rate_min = ctx[0].rate;
    uint64_t cap_min  = ctx[0].capacity;
    for (int i = 1; i < ngpu; i++) {
        if (ctx[i].rate     < rate_min) rate_min = ctx[i].rate;
        if (ctx[i].capacity < cap_min)  cap_min  = ctx[i].capacity;
    }
    if (rate_min <= 0.0) { fprintf(stderr, "吞吐量測失敗\n"); return 1; }

    // 配置滿容量，但 span 只填到 96%：期望落桶數 = 0.96×capacity，
    // 相對標準差 ~1/sqrt(2e9) ≈ 2e-5，溢位機率趨近於零。
    uint64_t capacity = cap_min;
    uint64_t fill     = (uint64_t)((double)capacity * 0.96);
    if (fill < 1024) fill = 1024;

    uint64_t partitions, span;
    if (smoke) {
        span = total;
        uint64_t need = (span + fill - 1) / fill;
        partitions = 2;
        while (partitions < need || partitions < (uint64_t)world_ranks) partitions <<= 1;
    } else {
        // 分數 ≈ log2(每桶筆數) + log2(總生成量) − 1，與 S 無關；
        // 只要 S 大到 partition 跑不完即可 ⇒ 理論值再乘 1.25 安全係數。
        double budget  = rate_min * seconds * (double)world_ranks;
        double p_ideal = 0.5 * log2(budget / (double)fill) + 0.32;
        int p = (int)ceil(p_ideal);
        if (p < 1)  p = 1;
        if (p > 32) p = 32;
        partitions = 1ULL << p;
        while (partitions < (uint64_t)world_ranks && partitions < (1ULL << 32)) partitions <<= 1;
        span = partitions * fill;
    }
    if (force_parts) {
        if (force_parts & (force_parts - 1)) { fprintf(stderr, "--partitions 必須是 2 的冪\n"); return 2; }
        partitions = force_parts;
        if (!force_span && !smoke) span = partitions * fill;
    }
    if (force_span) span = force_span;

    int part_bits = 0;
    while ((1ULL << part_bits) < partitions) part_bits++;
    if (part_bits < 1 || part_bits > 32) { fprintf(stderr, "partition 數超出支援範圍\n"); return 2; }

    RunCfg cfg = {0};
    cfg.prefix = prefix;
    cfg.nonce_word = nonce_word; cfg.nonce_shift = nonce_shift;
    cfg.nonce_start = nonce_start;
    cfg.span = span;
    cfg.part_bits = part_bits;
    cfg.partitions = partitions;
    cfg.world_ranks = world_ranks;
    cfg.smoke = smoke;

    double budget_total = rate_min * seconds * (double)world_ranks;
    printf("\n模式 = %s\n", smoke ? "smoke" : "正式");
    printf("partition 數 P = %llu（雜湊前 %d bits）   每桶重掃 S = %.2f G nonce   "
           "每桶留 %.1f M 筆（容量 %.1f M）\n",
           (unsigned long long)partitions, part_bits,
           (double)span / 1e9, (double)fill / 1e6, (double)capacity / 1e6);
    printf("生成階段 %.0f 秒，全叢集預估可生成 %.2f T hash（log2 = %.1f）\n",
           seconds, budget_total / 1e12, log2(budget_total));
    printf("理論期望分數 ≈ log2(每桶筆數) + log2(總生成量) − 1 ≈ %.1f bits\n\n",
           log2((double)fill) + log2(budget_total) - 1.0);
    fflush(stdout);

    signal(SIGINT,  request_stop);
    signal(SIGTERM, request_stop);

    const double run_start = monotonic_seconds();
    cfg.deadline = run_start + seconds;

    // ---- 階段 3：每張卡各自領 partition ----
#ifdef _OPENMP
#pragma omp parallel num_threads(ngpu)
#endif
    {
#ifdef _OPENMP
        int t = omp_get_thread_num();
#else
        int t = 0;
#endif
        DevCtx *c = &ctx[t];
        CUDA_CHECK(cudaSetDevice(c->dev));
        c->capacity = capacity;
        dev_alloc(c);
        run_device(c, &cfg);
    }

    double   elapsed = monotonic_seconds() - run_start;
    uint64_t hashes = 0, parts_total = 0;
    for (int i = 0; i < ngpu; i++) { hashes += ctx[i].hashes; parts_total += ctx[i].parts; }

    uint64_t kept = 0;
    double   gen_s = 0, sort_s = 0, scan_s = 0;
    for (int i = 0; i < ngpu; i++) {
        kept   += ctx[i].kept_total;
        gen_s  += ctx[i].gen_s;
        sort_s += ctx[i].sort_s;
        scan_s += ctx[i].scan_s;
    }
    double node_share = (double)ngpu / (double)world_ranks;
    double g_cluster  = (double)hashes / (node_share > 0 ? node_share : 1.0);

    printf("\n========== 瓶頸診斷 ==========\n");
    printf("本機生成      : %.2f G hash / %.1f 秒 = %.2f GH/s（%d 張卡）\n",
           (double)hashes / 1e9, elapsed,
           (elapsed > 0) ? ((double)hashes / elapsed / 1e9) : 0.0, ngpu);
    printf("GPU 時間佔比  : 生成 %.1f%%  排序 %.1f%%  掃描 %.1f%%\n",
           100.0 * gen_s / (gen_s + sort_s + scan_s + 1e-9),
           100.0 * sort_s / (gen_s + sort_s + scan_s + 1e-9),
           100.0 * scan_s / (gen_s + sort_s + scan_s + 1e-9));
    printf("每桶排序時間  : %.2f 秒 / %.3f G 筆  → %.2f G筆/秒\n",
           sort_s / (parts_total > 0 ? parts_total : 1),
           (double)kept / (parts_total > 0 ? parts_total : 1) / 1e9,
           (sort_s > 0) ? ((double)kept / sort_s / 1e9) : 0.0);
    printf("實際落桶      : 平均 %.3f G 筆（容量 %.3f G，%.1f%%）\n",
           (double)kept / (parts_total > 0 ? parts_total : 1) / 1e9,
           (double)capacity / 1e9,
           100.0 * (double)kept / (parts_total > 0 ? parts_total : 1) / (double)capacity);
    printf("\n分數模型（實測值代入）\n");
    printf("  H = 每桶筆數      = %.3f G      log2(H) = %.2f\n",
           (double)fill / 1e9, log2((double)fill));
    printf("  G = 全叢集生成量  = %.2f T      log2(G) = %.2f\n",
           g_cluster / 1e12, log2(g_cluster));
    printf("  期望分數 = log2(H) + log2(G) − 1 = %.1f bits   實得 %d bits\n",
           log2((double)fill) + log2(g_cluster) - 1.0, g_best.bits);
    printf("\n通往 80 bits 還差多少\n");
    {
        double need   = 81.0 - log2(g_cluster);          // 需要的 log2(H)
        double gap    = need - log2((double)fill);
        printf("  以目前 G，要 80 bits 需要 log2(H) = %.2f（= %.1f G 筆/桶）\n",
               need, pow(2.0, need) / 1e9);
        printf("  目前 log2(H) = %.2f，差 %.2f bits → 需要把單次排序容量放大 %.1f 倍\n",
               log2((double)fill), gap, pow(2.0, gap));
        printf("  單卡 HBM 已用滿；放大 H 的唯一途徑是節點內 8 卡用 NVLink 協同排序\n");
        printf("  （8 卡 = +3.00 bits）。時間翻倍只值 +1.00 bit，不是主要槓桿。\n");
    }
    printf("================================\n\n");
    if (g_best.valid) {
        printf("最佳結果：共同前綴 %d bits\n", g_best.bits);
        printf("nonce_a : %llu\n", (unsigned long long)g_best.a);
        printf("nonce_b : %llu\n", (unsigned long long)g_best.b);
        printf("\n驗證指令：\n");
        printf("  python3 verify_collision.py -p %s -a %llu -b %llu\n",
               prefix, (unsigned long long)g_best.a, (unsigned long long)g_best.b);
        printf("  python3 verify_collision.py --file solution_%d.csv\n", g_best.bits);
    } else {
        fprintf(stderr, "沒有找到合法的不同 nonce 配對\n");
    }
    fflush(stdout);

    for (int i = 0; i < ngpu; i++) {
        DevCtx *c = &ctx[i];
        cudaSetDevice(c->dev);
        cudaFree(c->d_key[0]); cudaFree(c->d_key[1]);
        cudaFree(c->d_val[0]); cudaFree(c->d_val[1]);
        cudaFree(c->d_temp);   cudaFree(c->d_counter); cudaFree(c->d_best);
        cudaFree(c->d_sink);   cudaFree(c->d_pair);    cudaFree(c->d_pair_hash);
    }
    free(ctx);
    return g_best.valid ? 0 : 1;
}
