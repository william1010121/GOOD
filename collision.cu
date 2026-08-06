// =====================================================================
// HiPAC 隱藏題1 — SHA-256 部分碰撞搜尋「範本程式」
//
//   給定固定字串 prefix，對每個 64-bit 整數 nonce 計算：
//       hash(nonce) = SHA-256( prefix(ASCII) ‖ nonce(8 bytes 大端) )
//
//   目標：找出兩個「不同」的 nonce a、b，讓 hash(a) 與 hash(b)
//         開頭有最多相同的 bits。相同的 bits 越多，分數越高。
//
//  【不可修改區】 —— SHA-256、訊息組法、nonce 擺放位置
// =====================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <cuda_runtime.h>

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
// 計算一批 nonce 的雜湊，取「前 128 bits」當排序鍵。
//
// 為什麼是 128 bits？共同前綴的分數可能超過 64 bits，鍵存太短就量不到。
// 128 bits 遠高於實際可達的範圍，足夠用。
//
// 資料以 SoA（分開的陣列）輸出，GPU 寫入時可以合併存取。
// ---------------------------------------------------------------------
__global__ void hash_kernel(const uint32_t *base_words, int nonce_word, int nonce_shift,
                            uint64_t start_nonce, uint32_t count,
                            uint64_t *out_hi, uint64_t *out_lo, uint64_t *out_nonce) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    uint64_t nonce = start_nonce + idx;

    uint32_t w[16];
#pragma unroll
    for (int i = 0; i < 16; i++) w[i] = base_words[i];
    put_nonce(w, nonce_word, nonce_shift, nonce);

    uint32_t hash[8];
    sha256_block(w, hash);

    out_hi[idx]    = ((uint64_t)hash[0] << 32) | hash[1];   // 前 64 bits
    out_lo[idx]    = ((uint64_t)hash[2] << 32) | hash[3];   // 次 64 bits
    out_nonce[idx] = nonce;
}

// ---------------------------------------------------------------------
// 主機端的排序與比對
// ---------------------------------------------------------------------
typedef struct { uint64_t hi, lo, nonce; } Entry;      // 24 bytes

static int cmp_entry(const void *a, const void *b) {
    const Entry *x = (const Entry *)a, *y = (const Entry *)b;
    if (x->hi != y->hi) return (x->hi < y->hi) ? -1 : 1;
    if (x->lo != y->lo) return (x->lo < y->lo) ? -1 : 1;
    return 0;
}

// 兩個 128-bit 鍵的共同前綴有幾個 bit
static int common_prefix_bits(const Entry *a, const Entry *b) {
    uint64_t x = a->hi ^ b->hi;
    if (x) return __builtin_clzll(x);
    uint64_t y = a->lo ^ b->lo;
    if (y) return 64 + __builtin_clzll(y);
    return 128;                                         // 前 128 bits 完全相同
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
    const char *prefix = (argc > 1) ? argv[1] : "hipac_demo";
    uint64_t total = (argc > 2) ? strtoull(argv[2], NULL, 10) : 20000000ULL;

    // nonce 是 64-bit，搜尋空間實質上無上限；真正的限制是記憶體與時間。
    if (total > 4000000000ULL) {
        fprintf(stderr, "單趟 %llu 筆需要 %.1f GB，請確認記憶體是否足夠\n",
                (unsigned long long)total, total * 24.0 / 1e9);
    }

    uint32_t base_words[16];
    int nonce_word, nonce_shift;
    build_base_words(prefix, base_words, &nonce_word, &nonce_shift);

    // ★ 優化點：這個範本只用「1 張 GPU」。比賽機器有多張。
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("使用 GPU: %s（%d 個 SM）\n", prop.name, prop.multiProcessorCount);
    printf("prefix = \"%s\"    掃描 %llu 個 nonce\n\n",
           prefix, (unsigned long long)total);

    // ★ 優化點：這個範本一次把所有雜湊都存起來，記憶體會成為掃描量的上限，需注意。
    //    每筆 24 bytes，掃 10 億個需要 24 GB。
    size_t n = (size_t)total;
    size_t bytes64 = n * sizeof(uint64_t);
    printf("需要記憶體：GPU %.2f GB，CPU %.2f GB\n\n",
           bytes64 * 3 / 1e9, (bytes64 * 3 + n * sizeof(Entry)) / 1e9);

    uint64_t *d_hi, *d_lo, *d_nonce;
    uint32_t *d_base;
    CUDA_CHECK(cudaMalloc(&d_base,  sizeof(base_words)));
    CUDA_CHECK(cudaMalloc(&d_hi,    bytes64));
    CUDA_CHECK(cudaMalloc(&d_lo,    bytes64));
    CUDA_CHECK(cudaMalloc(&d_nonce, bytes64));
    CUDA_CHECK(cudaMemcpy(d_base, base_words, sizeof(base_words), cudaMemcpyHostToDevice));

    time_t t_start = time(NULL);

    // ---- 第 1 步：GPU 平行計算所有雜湊 ----
    int threads = 256;
    int blocks  = (int)((n + threads - 1) / threads);
    printf("[1/3] GPU 計算 %llu 個雜湊 …\n", (unsigned long long)total);
    hash_kernel<<<blocks, threads>>>(d_base, nonce_word, nonce_shift,
                                     0, (uint32_t)n, d_hi, d_lo, d_nonce);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- 第 2 步：複製回 CPU 並排序 ----
    //   ★ 優化點（最重要）：這一步是瓶頸，整段期間 GPU 閒置。
    printf("[2/3] 複製回主機並排序 …\n");
    uint64_t *h_hi    = (uint64_t *)malloc(bytes64);
    uint64_t *h_lo    = (uint64_t *)malloc(bytes64);
    uint64_t *h_nonce = (uint64_t *)malloc(bytes64);
    Entry    *entries = (Entry *)malloc(n * sizeof(Entry));
    if (!h_hi || !h_lo || !h_nonce || !entries) {
        fprintf(stderr, "主機記憶體不足\n"); exit(1);
    }
    CUDA_CHECK(cudaMemcpy(h_hi,    d_hi,    bytes64, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_lo,    d_lo,    bytes64, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_nonce, d_nonce, bytes64, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; i++) {
        entries[i].hi = h_hi[i];
        entries[i].lo = h_lo[i];
        entries[i].nonce = h_nonce[i];
    }
    qsort(entries, n, sizeof(Entry), cmp_entry);

    // ---- 第 3 步：掃描相鄰配對 ----
    //   排序後，最佳解一定是某一組相鄰對，所以只需掃一次。
    printf("[3/3] 掃描相鄰配對 …\n\n");
    int best_bits = 0;
    uint64_t best_a = 0, best_b = 0;
    for (size_t i = 0; i + 1 < n; i++) {
        if (entries[i].nonce == entries[i+1].nonce) continue;   // 必須是不同 nonce
        int bits = common_prefix_bits(&entries[i], &entries[i+1]);
        if (bits > best_bits) {
            best_bits = bits;
            best_a = entries[i].nonce;
            best_b = entries[i+1].nonce;
        }
    }

    double elapsed = difftime(time(NULL), t_start);
    printf("----------------------------------------\n");
    printf("最佳結果：共同前綴 %d bits\n", best_bits);
    printf("nonce_a : %llu\n", (unsigned long long)best_a);
    printf("nonce_b : %llu\n", (unsigned long long)best_b);
    printf("耗時    : %.0f 秒\n", elapsed);

    write_solution(prefix, best_a, best_b, best_bits);

    printf("\n驗證指令：\n");
    printf("  python3 verify_collision.py -p %s -a %llu -b %llu\n",
           prefix, (unsigned long long)best_a, (unsigned long long)best_b);

    free(entries); free(h_hi); free(h_lo); free(h_nonce);
    cudaFree(d_base); cudaFree(d_hi); cudaFree(d_lo); cudaFree(d_nonce);
    return 0;
}
