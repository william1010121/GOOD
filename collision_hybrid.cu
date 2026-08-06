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
//  （與原始範本逐位元組相同，只有下方開放區被重寫）
//
//  === v2 改動摘要（開放區）===
//   1. 排序搬到 GPU（thrust::sort，裝置端）── 移除範本中「複製回主機
//      後用單執行緒 qsort」這個造成 GPU 全程閒置的瓶頸（README 標示
//      為「最重要」的優化點）。
//   2. 支援多張 GPU（OpenMP 驅動，一顆 GPU 一個 host 執行緒）── nonce
//      空間切成不重疊的連續區段，每張卡各自算雜湊＋裝置端排序。
//   3. kernel 直接寫出 AoS 的 Entry（hi,lo,nonce），不再另外開三個
//      SoA 陣列＋host 端逐筆搬移重組，省掉一份多餘的主機記憶體與一輪
//      序列迴圈。
//   4. 主機端不再對「全部資料」做一次 O(N log N) 的 qsort，而是對
//      「已經排序好的 num_gpus 條序列」做 k-way merge（O(N log k)），
//      數學上與整體排序後掃相鄰對完全等價（已用 CPU 版驗證，見
//      verify_algo/test_algorithm.cpp），但更快、也不需要 qsort 那份
//      額外的重排緩衝區。
//
//  這個版本在「只有 1 張 GPU」時會自動退化成單卡流程，邏輯完全一致，
//  所以也可以在單卡開發機上先測，再搬到比賽機器（16 張 H200）上跑。
// =====================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <algorithm>
#include <vector>
#include <queue>
#include <cuda_runtime.h>
#include <omp.h>
#include <mpi.h>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

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


// =====================================================================
// v5：每筆 24 bytes -> 16 bytes，把記憶體天花板抬高
//
//   v4 每筆存 {hi, lo, nonce} = 24 bytes，排序還要一份等大的暫存，
//   所以每張卡需要 N*3 bytes，H200 的 143 GB 把 N 卡在 4.8e10
//   （約 69 bits）。
//
//   v5 只存 {hi, nonce} = 16 bytes（丟掉 lo），每卡需求降為 N*2，
//   上限抬到 7.1e10（約 71 bits）。
//
//   代價：只有 hi 就只能量到 64 bits 的共同前綴。解法是延後計算 ——
//   掃相鄰對時若兩筆的 hi 完全相同（代表 >= 64 bits），才在 kernel 裡
//   當場把兩個 nonce 的 SHA-256 重算一次取得 lo，補足 64 bits 以上的
//   部分。sha256_block 與 put_nonce 本來就是 __device__，可直接呼叫。
//
//   這條分支有多罕見：N 筆樣本中 hi（64 bits）完全相同的配對，期望
//   數量是 N^2 / 2 / 2^64；N = 6e10 時約 97 對。相對於 6e10 個執行緒
//   幾乎不造成分支發散，卻讓我們能精確量到 128 bits。
//
//   （這正是 collision.cu 結尾註解建議的方向之一。）
// =====================================================================
typedef struct { uint64_t hi, nonce; } E16;

struct E16Less {
    __host__ __device__ bool operator()(const E16 &a, const E16 &b) const {
        if (a.hi != b.hi) return a.hi < b.hi;
        return a.nonce < b.nonce;          // 讓相同 hi 的順序是確定的
    }
};

__global__ void hash_kernel16(const uint32_t *base_words, int nonce_word, int nonce_shift,
                              uint64_t start_nonce, uint32_t count, E16 *out) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    uint64_t nonce = start_nonce + idx;
    uint32_t w[16];
#pragma unroll
    for (int i = 0; i < 16; i++) w[i] = base_words[i];
    put_nonce(w, nonce_word, nonce_shift, nonce);
    uint32_t hash[8];
    sha256_block(w, hash);
    out[idx].hi    = ((uint64_t)hash[0] << 32) | hash[1];
    out[idx].nonce = nonce;
}

// 只用 hi 算共同前綴（上限 64）。hi 相同時回傳 64，代表「需要看 lo」。
static int prefix_bits_hi(uint64_t a, uint64_t b) {
    uint64_t x = a ^ b;
    return x ? __builtin_clzll(x) : 64;
}

#define MAX_LAUNCH (1500000000u)
#define PACK_IDX_BITS 40

// ---------------------------------------------------------------------
// 第 1 步：一張 GPU 算一段 nonce 的雜湊，裝置端排序後搬回主機。
// ---------------------------------------------------------------------
static std::vector<E16> run_shard16(int gpu_id, const uint32_t base_words[16],
                                    int nonce_word, int nonce_shift,
                                    uint64_t start_nonce, uint64_t count) {
    CUDA_CHECK(cudaSetDevice(gpu_id));
    uint32_t *d_base = nullptr; E16 *d = nullptr;
    CUDA_CHECK(cudaMalloc(&d_base, 16 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d, count * sizeof(E16)));
    CUDA_CHECK(cudaMemcpy(d_base, base_words, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    uint64_t done = 0;
    while (done < count) {
        uint32_t batch = (uint32_t)std::min<uint64_t>(count - done, MAX_LAUNCH);
        int threads = 256, blocks = (int)((batch + threads - 1) / threads);
        hash_kernel16<<<blocks, threads>>>(d_base, nonce_word, nonce_shift,
                                           start_nonce + done, batch, d + done);
        done += batch;
    }
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<E16> dp(d);
    thrust::sort(thrust::device, dp, dp + count, E16Less());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<E16> h(count);
    CUDA_CHECK(cudaMemcpy(h.data(), d, count * sizeof(E16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_base));
    CUDA_CHECK(cudaFree(d));
    return h;
}

struct Seq { const E16 *p; size_t n; };

// ---------------------------------------------------------------------
// 掃相鄰對。hi 不同 -> clz 直接得到精確 bits（必定 < 64）。
// hi 相同 -> 當場重算兩個 nonce 的 SHA-256 取 lo，補到 128 bits。
// ---------------------------------------------------------------------
__global__ void adj_max16(const E16 *e, uint64_t n, const uint32_t *base_words,
                          int nonce_word, int nonce_shift, unsigned long long *out) {
    extern __shared__ unsigned long long sh[];
    unsigned long long best = 0;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i + 1 < n; i += stride) {
        if (e[i].nonce == e[i + 1].nonce) continue;
        uint64_t x = e[i].hi ^ e[i + 1].hi;
        int b;
        if (x) {
            b = __clzll((long long)x);                 // 常見路徑：< 64 bits
        } else {
            uint32_t w[16], h1[8], h2[8];              // 罕見路徑：hi 全同，回頭算 lo
#pragma unroll
            for (int k = 0; k < 16; k++) w[k] = base_words[k];
            put_nonce(w, nonce_word, nonce_shift, e[i].nonce);
            sha256_block(w, h1);
#pragma unroll
            for (int k = 0; k < 16; k++) w[k] = base_words[k];
            put_nonce(w, nonce_word, nonce_shift, e[i + 1].nonce);
            sha256_block(w, h2);
            uint64_t lo1 = ((uint64_t)h1[2] << 32) | h1[3];
            uint64_t lo2 = ((uint64_t)h2[2] << 32) | h2[3];
            uint64_t y = lo1 ^ lo2;
            b = y ? 64 + __clzll((long long)y) : 128;
        }
        unsigned long long p = ((unsigned long long)b << PACK_IDX_BITS) | (unsigned long long)i;
        if (p > best) best = p;
    }
    sh[threadIdx.x] = best;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && sh[threadIdx.x + s] > sh[threadIdx.x]) sh[threadIdx.x] = sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax(out, sh[0]);
}

struct SliceResult { int bits; uint64_t a, b; bool empty; E16 first, last; };

static SliceResult scan_slice16(int gpu_id, const std::vector<Seq> &seqs,
                                const std::vector<size_t> &lo, const std::vector<size_t> &hi_,
                                const uint32_t base_words[16], int nonce_word, int nonce_shift) {
    CUDA_CHECK(cudaSetDevice(gpu_id));
    uint64_t n = 0;
    for (size_t s = 0; s < seqs.size(); s++) n += (uint64_t)(hi_[s] - lo[s]);
    SliceResult r; r.bits = 0; r.a = 0; r.b = 0; r.empty = (n == 0);
    if (n == 0) return r;

    E16 *d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(E16)));
    uint64_t off = 0;
    for (size_t s = 0; s < seqs.size(); s++) {
        uint64_t len = (uint64_t)(hi_[s] - lo[s]);
        if (!len) continue;
        CUDA_CHECK(cudaMemcpy(d + off, seqs[s].p + lo[s], len * sizeof(E16), cudaMemcpyHostToDevice));
        off += len;
    }
    thrust::device_ptr<E16> dp(d);
    thrust::sort(thrust::device, dp, dp + n, E16Less());

    uint32_t *d_base = nullptr; unsigned long long *d_best = nullptr;
    CUDA_CHECK(cudaMalloc(&d_base, 16 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_base, base_words, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_best, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_best, 0, sizeof(unsigned long long)));
    if (n >= 2) {
        int threads = 256, blocks = 2048;
        adj_max16<<<blocks, threads, threads * sizeof(unsigned long long)>>>(
            d, n, d_base, nonce_word, nonce_shift, d_best);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    unsigned long long packed = 0;
    CUDA_CHECK(cudaMemcpy(&packed, d_best, sizeof(packed), cudaMemcpyDeviceToHost));
    if (packed) {
        uint64_t i = packed & ((1ULL << PACK_IDX_BITS) - 1);
        E16 two[2];
        CUDA_CHECK(cudaMemcpy(two, d + i, 2 * sizeof(E16), cudaMemcpyDeviceToHost));
        r.bits = (int)(packed >> PACK_IDX_BITS);
        r.a = two[0].nonce; r.b = two[1].nonce;
    }
    CUDA_CHECK(cudaMemcpy(&r.first, d, sizeof(E16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&r.last, d + (n - 1), sizeof(E16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_best)); CUDA_CHECK(cudaFree(d_base)); CUDA_CHECK(cudaFree(d));
    return r;
}

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

// =====================================================================
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int wr = 0, ws = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &wr);
    MPI_Comm_size(MPI_COMM_WORLD, &ws);
    int bits_r = 0; while ((1 << bits_r) < ws) bits_r++;
    if ((1 << bits_r) != ws) { if (!wr) fprintf(stderr, "rank 數必須是 2 的冪\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

    const char *prefix = (argc > 1) ? argv[1] : "hipac_demo";
    uint64_t total = (argc > 2) ? strtoull(argv[2], NULL, 10) : 20000000ULL;
    int avail = 0; CUDA_CHECK(cudaGetDeviceCount(&avail));
    if (avail < 1) { fprintf(stderr, "找不到 GPU\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    int G = (argc > 3) ? atoi(argv[3]) : avail;
    if (G < 1) G = 1; if (G > avail) G = avail;
    MPI_Allreduce(MPI_IN_PLACE, &G, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    int n_scan = 1, bits_g = 0; while (n_scan * 2 <= G) { n_scan *= 2; bits_g++; }

    uint64_t pr = total / (uint64_t)ws, rr = total % (uint64_t)ws;
    uint64_t my_start = (uint64_t)wr * pr + ((uint64_t)wr < rr ? (uint64_t)wr : rr);
    uint64_t my_count = pr + ((uint64_t)wr < rr ? 1 : 0);
    if (!my_count) { fprintf(stderr, "total 太小\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    if ((uint64_t)G > my_count) G = (int)my_count;

    uint32_t base_words[16]; int nw, nsft;
    build_base_words(prefix, base_words, &nw, &nsft);

    if (!wr) {
        printf("prefix = \"%s\"    掃描 %llu 個 nonce    %d 節點 x %d GPU\n",
               prefix, (unsigned long long)total, ws, G);
        printf("每筆 %zu bytes（v5）  主機每節點約 %.0f GB   裝置每卡約 %.0f GB\n\n",
               sizeof(E16),
               (double)my_count * sizeof(E16) * (2.0 - 1.0 / ws) / 1e9,
               (double)total / (ws * n_scan) * sizeof(E16) * 2 / 1e9);
        fflush(stdout);
    }

    uint64_t per = my_count / G, rem = my_count % G;
    std::vector<uint64_t> ss(G), sc(G);
    { uint64_t s = my_start; for (int g = 0; g < G; g++) { uint64_t c = per + ((uint64_t)g < rem); ss[g] = s; sc[g] = c; s += c; } }

    double t0 = MPI_Wtime();
    if (!wr) { printf("[1/4] 計算雜湊並在裝置端排序 …\n"); fflush(stdout); }
    std::vector<std::vector<E16> > shards(G);
#pragma omp parallel for num_threads(G) schedule(static, 1)
    for (int g = 0; g < G; g++) shards[g] = run_shard16(g, base_words, nw, nsft, ss[g], sc[g]);
    MPI_Barrier(MPI_COMM_WORLD); double t1 = MPI_Wtime();

    if (!wr) { printf("[2/4] 跨節點重新分配（雜湊+排序 %.0f 秒）…\n", t1 - t0); fflush(stdout); }
    std::vector<std::vector<size_t> > bnd(G, std::vector<size_t>(ws + 1, 0));
    for (int g = 0; g < G; g++) {
        bnd[g][0] = 0; bnd[g][ws] = shards[g].size();
        for (int t = 1; t < ws; t++) {
            uint64_t key = (uint64_t)t << (64 - bits_r);
            const E16 *b = shards[g].data(), *e = b + shards[g].size();
            bnd[g][t] = (size_t)(std::lower_bound(b, e, key,
                [](const E16 &x, uint64_t k) { return x.hi < k; }) - b);
        }
    }
    std::vector<uint64_t> sc2((size_t)ws * G, 0), rc2((size_t)ws * G, 0);
    for (int t = 0; t < ws; t++) for (int g = 0; g < G; g++) sc2[(size_t)t * G + g] = bnd[g][t + 1] - bnd[g][t];
    MPI_Alltoall(sc2.data(), G, MPI_UINT64_T, rc2.data(), G, MPI_UINT64_T, MPI_COMM_WORLD);

    MPI_Datatype ME; MPI_Type_contiguous((int)sizeof(E16), MPI_BYTE, &ME); MPI_Type_commit(&ME);
    const uint64_t CH = 1ULL << 26;
    std::vector<std::vector<std::vector<E16> > > rb(ws);
    std::vector<MPI_Request> rq;
    for (int s = 0; s < ws; s++) { if (s == wr) continue; rb[s].resize(G);
        for (int g = 0; g < G; g++) { uint64_t n = rc2[(size_t)s * G + g]; rb[s][g].resize(n);
            for (uint64_t o = 0; o < n; o += CH) { int c = (int)((n - o < CH) ? (n - o) : CH);
                MPI_Request q; MPI_Irecv(rb[s][g].data() + o, c, ME, s, g, MPI_COMM_WORLD, &q); rq.push_back(q); } } }
    for (int t = 0; t < ws; t++) { if (t == wr) continue;
        for (int g = 0; g < G; g++) { const E16 *b = shards[g].data() + bnd[g][t]; uint64_t n = bnd[g][t + 1] - bnd[g][t];
            for (uint64_t o = 0; o < n; o += CH) { int c = (int)((n - o < CH) ? (n - o) : CH);
                MPI_Request q; MPI_Isend(b + o, c, ME, t, g, MPI_COMM_WORLD, &q); rq.push_back(q); } } }
    if (!rq.empty()) MPI_Waitall((int)rq.size(), rq.data(), MPI_STATUSES_IGNORE);
    MPI_Type_free(&ME); MPI_Barrier(MPI_COMM_WORLD); double t2 = MPI_Wtime();

    if (!wr) { printf("[3/4] %d 張卡各自排序 + 裝置端掃相鄰對（交換 %.0f 秒）…\n", n_scan, t2 - t1); fflush(stdout); }
    std::vector<Seq> seqs;
    for (int g = 0; g < G; g++) seqs.push_back(Seq{shards[g].data() + bnd[g][wr], bnd[g][wr + 1] - bnd[g][wr]});
    for (int s = 0; s < ws; s++) { if (s == wr) continue; for (int g = 0; g < G; g++) seqs.push_back(Seq{rb[s][g].data(), rb[s][g].size()}); }

    int bits_p = bits_r + bits_g;
    std::vector<SliceResult> res(n_scan);
#pragma omp parallel for num_threads(n_scan) schedule(static, 1)
    for (int g = 0; g < n_scan; g++) {
        uint64_t pfx = ((uint64_t)wr << bits_g) | (uint64_t)g;
        uint64_t klo = (bits_p == 0) ? 0ULL : (pfx << (64 - bits_p));
        bool last = (bits_p == 0) || (pfx + 1 == (1ULL << bits_p));
        uint64_t khi = last ? 0ULL : ((pfx + 1) << (64 - bits_p));
        std::vector<size_t> lo(seqs.size()), hi_(seqs.size());
        for (size_t s = 0; s < seqs.size(); s++) {
            const E16 *b = seqs[s].p, *e = b + seqs[s].n;
            auto cmp = [](const E16 &x, uint64_t k) { return x.hi < k; };
            lo[s] = (size_t)(std::lower_bound(b, e, klo, cmp) - b);
            hi_[s] = last ? seqs[s].n : (size_t)(std::lower_bound(b, e, khi, cmp) - b);
        }
        res[g] = scan_slice16(g, seqs, lo, hi_, base_words, nw, nsft);
    }

    int bb = 0; uint64_t ba = 0, bbn = 0;
    for (int g = 0; g < n_scan; g++) if (!res[g].empty && res[g].bits > bb) { bb = res[g].bits; ba = res[g].a; bbn = res[g].b; }
    // 跨卡邊界：兩筆必在不同 bucket，hi 一定不同，clz 給的就是精確值。
    { const SliceResult *pv = NULL;
      for (int g = 0; g < n_scan; g++) { if (res[g].empty) continue;
          if (pv && pv->last.nonce != res[g].first.nonce) {
              int b = prefix_bits_hi(pv->last.hi, res[g].first.hi);
              if (b > bb) { bb = b; ba = pv->last.nonce; bbn = res[g].first.nonce; } }
          pv = &res[g]; } }
    MPI_Barrier(MPI_COMM_WORLD); double t3 = MPI_Wtime();

    struct { int bits; int rank; } loc, glob;
    loc.bits = bb; loc.rank = wr;
    MPI_Allreduce(&loc, &glob, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
    uint64_t ab[2] = { ba, bbn };
    MPI_Bcast(ab, 2, MPI_UINT64_T, glob.rank, MPI_COMM_WORLD);

    if (!wr) {
        printf("[4/4] 完成\n\n----------------------------------------\n");
        printf("最佳結果：共同前綴 %d bits（來自 rank %d）\n", glob.bits, glob.rank);
        printf("nonce_a : %llu\nnonce_b : %llu\n", (unsigned long long)ab[0], (unsigned long long)ab[1]);
        printf("耗時    : %.0f 秒（雜湊+排序 %.0f / 交換 %.0f / GPU掃描 %.0f）\n",
               t3 - t0, t1 - t0, t2 - t1, t3 - t2);
        write_solution(prefix, ab[0], ab[1], glob.bits);
        printf("\n驗證指令：\n  python3 verify_collision.py -p %s -a %llu -b %llu\n",
               prefix, (unsigned long long)ab[0], (unsigned long long)ab[1]);
    }
    MPI_Finalize();
    return 0;
}

// =====================================================================
// 想再往上衝更多 bits、掃更大的 total 時可以考慮的延伸方向（未實作）：
//
//  - 目前每筆固定存 24 bytes（hi+lo+nonce）。若 total 大到連分攤到
//    16 張卡後單卡裝置記憶體仍不夠，可以先只用 hi（8 bytes）配合索引
//    做 radix 排序（thrust::sort_by_key 對純數值鍵會走 radix sort，
//    比目前對整個 struct 做比較排序更快），只有 hi 真的相同時才回頭
//    算 lo，可以把單卡可掃的量再往上推。
//  - 若 total 大到連「跑一輪存全部」都放不下，可以改用 distinguished
//    points 的技巧：只保留雜湊某幾個 bit 為 0 的樣本（機率性地大幅
//    減少要保存的筆數），犧牲一點理論最優性換取能掃過遠大於記憶體
//    上限的搜尋空間。這屬於 exact 解法之外的進階/機率性做法。
// =====================================================================

