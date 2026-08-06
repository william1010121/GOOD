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
#include <signal.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

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
// GPU radix sort 的資料型別：只按前 64 bits 排序，value 保留後 64 bits 與 nonce。
// 排序後相鄰值可直接比較完整的 128-bit hash；同一前綴群組內以 low bits 比較。
typedef struct { uint64_t other, nonce; } SortValue;
typedef struct { int bits, valid; uint64_t nonce_a, nonce_b; } Candidate;

struct CandidateMax {
    __host__ __device__ Candidate operator()(const Candidate &a,
                                             const Candidate &b) const {
        if (a.valid != b.valid) return a.valid ? a : b;
        if (!a.valid || a.bits >= b.bits) return a;
        return b;
    }
};

__global__ void pack_sort_values_kernel(const uint64_t *lo, const uint64_t *nonce,
                                        SortValue *values, uint64_t count) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        values[idx].other = lo[idx];
        values[idx].nonce = nonce[idx];
    }
}

__device__ __forceinline__ int common_prefix_bits_raw(uint64_t hi_a, uint64_t lo_a,
                                                      uint64_t hi_b, uint64_t lo_b) {
    uint64_t x = hi_a ^ hi_b;
    if (x) return __builtin_clzll(x);
    uint64_t y = lo_a ^ lo_b;
    if (y) return 64 + __builtin_clzll(y);
    return 128;
}

__global__ void best_pair_kernel(const uint64_t *hi_sorted,
                                 const SortValue *high_values, uint64_t count,
                                 Candidate *block_best) {
    extern __shared__ Candidate shared[];
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    Candidate best = {0, 0, 0, 0};
    if (idx + 1 < count) {
        // 不排序 low bits：對相同 high-64 群組直接掃描群組內所有 pair，
        // 對不同 high-64 的相鄰項目只需比較這一對。隨機 hash 下群組通常只有 1 筆。
        uint64_t j = idx + 1;
        if (hi_sorted[j] == hi_sorted[idx]) {
            while (j < count && hi_sorted[j] == hi_sorted[idx]) {
                if (high_values[idx].nonce != high_values[j].nonce) {
                    int bits = common_prefix_bits_raw(
                        hi_sorted[idx], high_values[idx].other,
                        hi_sorted[j], high_values[j].other);
                    Candidate candidate = {bits, 1, high_values[idx].nonce,
                                           high_values[j].nonce};
                    best = CandidateMax()(best, candidate);
                }
                ++j;
            }
        } else if (high_values[idx].nonce != high_values[j].nonce) {
            best.bits = common_prefix_bits_raw(
                hi_sorted[idx], high_values[idx].other,
                hi_sorted[j], high_values[j].other);
            best.valid = 1;
            best.nonce_a = high_values[idx].nonce;
            best.nonce_b = high_values[j].nonce;
        }
    }
    shared[threadIdx.x] = best;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            Candidate other = shared[threadIdx.x + stride];
            shared[threadIdx.x] = CandidateMax()(shared[threadIdx.x], other);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) block_best[blockIdx.x] = shared[0];
}

// GPU 已完成排序與相鄰配對 reduction；主機端只處理一筆 candidate。
// ---------------------------------------------------------------------
// 單調時鐘：用於量 CPU 階段與端到端時間，不受系統時間調整影響。
static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static double throughput_mhps(uint64_t count, double seconds) {
    return (seconds > 0.0) ? ((double)count / seconds / 1e6) : 0.0;
}

static volatile sig_atomic_t stop_requested = 0;

static void request_stop(int signal_number) {
    (void)signal_number;
    stop_requested = 1;
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

static uint64_t parse_u64_value(const char *text, const char *name, int allow_zero) {
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (text[0] == '\0' || *end != '\0' || (!allow_zero && value == 0)) {
        fprintf(stderr, "%s 必須是正整數：%s\n", name, text);
        exit(2);
    }
    return (uint64_t)value;
}

static uint64_t parse_positive_u64(const char *text, const char *name) {
    return parse_u64_value(text, name, 0);
}

static void print_usage(const char *program) {
    fprintf(stderr,
            "用法：\n"
            "  %s [--start N] [prefix]              一般模式：持續以 batch 掃描\n"
            "  %s --smoke [--start N] [prefix] [total]  smoke 模式：掃固定 total 後結束\n"
            "環境變數：BATCH_NONCES（一般模式每批數量，預設 100000000）\n",
            program, program);
}

// =====================================================================
int main(int argc, char **argv) {
    int smoke = 0;
    uint64_t nonce_start = 0;
    int argi = 1;
    while (argi < argc && argv[argi][0] == '-') {
        if (strcmp(argv[argi], "--smoke") == 0) {
            smoke = 1;
            argi++;
        } else if (strcmp(argv[argi], "--start") == 0 && argi + 1 < argc) {
            nonce_start = parse_u64_value(argv[argi + 1], "start", 1);
            argi += 2;
        } else {
            print_usage(argv[0]);
            return 2;
        }
    }

    const char *prefix = (argi < argc) ? argv[argi++] : "hipac_demo";
    uint64_t total = 0;
    if (smoke) {
        total = (argi < argc) ? parse_positive_u64(argv[argi++], "total") : 100000000ULL;
    }
    if (argi != argc) {
        print_usage(argv[0]);
        return 2;
    }

    const uint64_t default_batch = 100000000ULL;
    uint64_t batch_size = default_batch;
    const char *batch_text = getenv("BATCH_NONCES");
    if (!smoke && batch_text != NULL) {
        batch_size = parse_positive_u64(batch_text, "BATCH_NONCES");
    }
    if (batch_size > 0xffffffffULL || (smoke && total > 0xffffffffULL)) {
        fprintf(stderr, "單批數量不可超過 4294967295\n");
        return 2;
    }

    // 一般模式不依賴 total：持續掃描固定大小的 batch，直到收到 Ctrl-C。
    // 只有 --smoke 才使用 total 作為固定測試量與總吞吐量的分母。
    uint64_t allocation_n = smoke ? total : batch_size;
    if (smoke && total > 4000000000ULL) {
        fprintf(stderr, "smoke 單趟 %llu 筆需要 %.1f GB，請確認記憶體是否足夠\n",
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
    if (smoke) {
        printf("模式 = smoke，prefix = \"%s\"    掃描 %llu 個 nonce\n\n",
               prefix, (unsigned long long)total);
    } else {
        printf("模式 = continuous，prefix = \"%s\"    每批掃描 %llu 個 nonce\n\n",
               prefix, (unsigned long long)batch_size);
        signal(SIGINT, request_stop);
        signal(SIGTERM, request_stop);
    }
    printf("nonce 起點 = %llu\n", (unsigned long long)nonce_start);

    // ★ 優化點：這個範本一次把所有雜湊都存起來，記憶體會成為掃描量的上限，需注意。
    //    每筆 24 bytes，掃 10 億個需要 24 GB。
    size_t n = (size_t)allocation_n;
    if (n > 0x7fffffffULL) {
        fprintf(stderr, "CUB radix sort 單批不可超過 2147483647 筆\n");
        return 2;
    }
    size_t bytes64 = n * sizeof(uint64_t);
    size_t bytes_value = n * sizeof(SortValue);
    const int threads = 256;
    int pair_blocks = ((int)n + threads - 1) / threads;
    size_t bytes_block_candidates = (size_t)pair_blocks * sizeof(Candidate);

    uint64_t *d_hi, *d_lo, *d_nonce;
    uint64_t *d_hi_sorted;
    SortValue *d_sort_in, *d_sort_out;
    Candidate *d_block_best, *d_best;
    uint32_t *d_base;
    CUDA_CHECK(cudaMalloc(&d_base,  sizeof(base_words)));
    CUDA_CHECK(cudaMalloc(&d_hi,    bytes64));
    CUDA_CHECK(cudaMalloc(&d_lo,    bytes64));
    CUDA_CHECK(cudaMalloc(&d_nonce, bytes64));
    CUDA_CHECK(cudaMalloc(&d_hi_sorted,   bytes64));
    CUDA_CHECK(cudaMalloc(&d_sort_in,   bytes_value));
    CUDA_CHECK(cudaMalloc(&d_sort_out,  bytes_value));
    CUDA_CHECK(cudaMalloc(&d_block_best, bytes_block_candidates));
    CUDA_CHECK(cudaMalloc(&d_best, sizeof(Candidate)));
    CUDA_CHECK(cudaMemcpy(d_base, base_words, sizeof(base_words), cudaMemcpyHostToDevice));

    size_t temp_bytes_sort = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, temp_bytes_sort, d_hi, d_hi_sorted, d_sort_in, d_sort_out,
        (int)n, 0, 64));
    size_t temp_bytes_reduce = 0;
    Candidate initial_candidate = {0, 0, 0, 0};
    CUDA_CHECK(cub::DeviceReduce::Reduce(
        nullptr, temp_bytes_reduce, d_block_best, d_best, pair_blocks,
        CandidateMax(), initial_candidate));
    size_t temp_bytes = (temp_bytes_sort > temp_bytes_reduce) ?
                        temp_bytes_sort : temp_bytes_reduce;
    void *d_temp_storage = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_bytes));
    printf("需要記憶體：GPU %.2f GB（含 CUB sort/reduce temp），CPU %.2f MB\n\n",
           (bytes64 * 4 + bytes_value * 2 +
            bytes_block_candidates + sizeof(Candidate) + temp_bytes) / 1e9,
           sizeof(Candidate) / 1e6);

    // CUDA event 只量 GPU kernel 本身；CPU 單調時鐘量整體時間。
    cudaEvent_t hash_start, hash_stop;
    CUDA_CHECK(cudaEventCreate(&hash_start));
    CUDA_CHECK(cudaEventCreate(&hash_stop));

    Candidate h_best;
    const uint64_t report_every = 10000000ULL;
    const double run_start = monotonic_seconds();
    uint64_t scanned = 0;
    uint64_t next_nonce = nonce_start;
    uint64_t batch_index = 0;
    int best_bits = 0;
    uint64_t best_a = 0, best_b = 0;
    int have_solution = 0;

    while (!stop_requested && (smoke ? scanned < total : 1)) {
        uint64_t batch_count = smoke ? total : batch_size;
        double batch_start = monotonic_seconds();
        double hash_seconds = 0.0;
        uint64_t done = 0;
        uint64_t batch_start_nonce = next_nonce;
        batch_index++;

        printf("[batch %llu] GPU 計算 %llu 個雜湊 …\n",
               (unsigned long long)batch_index,
               (unsigned long long)batch_count);
        while (done < batch_count) {
            uint32_t count = (uint32_t)(((batch_count - done) > report_every) ?
                                        report_every : (batch_count - done));
            int blocks = (count + threads - 1) / threads;

            CUDA_CHECK(cudaEventRecord(hash_start));
            hash_kernel<<<blocks, threads>>>(d_base, nonce_word, nonce_shift,
                                             batch_start_nonce + done, count,
                                             d_hi + done, d_lo + done,
                                             d_nonce + done);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(hash_stop));
            CUDA_CHECK(cudaEventSynchronize(hash_stop));

            float batch_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&batch_ms, hash_start, hash_stop));
            hash_seconds += (double)batch_ms / 1000.0;
            done += count;

            if (smoke) {
                printf("[進度] %llu/%llu (%.1f%%)\n",
                       (unsigned long long)done, (unsigned long long)batch_count,
                       100.0 * (double)done / (double)batch_count);
            } else {
                printf("[進度] 本批 %llu/%llu\n",
                       (unsigned long long)done, (unsigned long long)batch_count);
            }
            fflush(stdout);
        }

        // ---- GPU single 64-bit radix sort + GPU full-key adjacent-pair reduction ----
        printf("[batch %llu] GPU radix sort（64-bit prefix）+ pair reduction …\n",
               (unsigned long long)batch_index);
        int sort_count = (int)batch_count;
        int sort_blocks = (sort_count + threads - 1) / threads;
        pack_sort_values_kernel<<<sort_blocks, threads>>>(d_lo, d_nonce, d_sort_in,
                                                          batch_count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            d_temp_storage, temp_bytes_sort, d_hi, d_hi_sorted, d_sort_in, d_sort_out,
            sort_count, 0, 64));
        best_pair_kernel<<<pair_blocks, threads, threads * sizeof(Candidate)>>>(
            d_hi_sorted, d_sort_out, batch_count, d_block_best);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cub::DeviceReduce::Reduce(
            d_temp_storage, temp_bytes_reduce, d_block_best, d_best, pair_blocks,
            CandidateMax(), initial_candidate));
        CUDA_CHECK(cudaMemcpy(&h_best, d_best, sizeof(Candidate),
                              cudaMemcpyDeviceToHost));

        uint64_t batch_a = h_best.nonce_a, batch_b = h_best.nonce_b;
        int batch_best_bits = h_best.bits;
        if (h_best.valid && batch_best_bits > best_bits) {
            best_bits = batch_best_bits;
            best_a = batch_a;
            best_b = batch_b;
            have_solution = 1;
            write_solution(prefix, best_a, best_b, best_bits);
        }

        scanned += batch_count;
        next_nonce += batch_count;
        double batch_seconds = monotonic_seconds() - batch_start;
        double elapsed_seconds = monotonic_seconds() - run_start;
        printf("目前最多共同前綴 qbit：%d（本批 %d）\n",
               best_bits, batch_best_bits);
        if (smoke) {
            printf("耗時    : %.3f 秒\n", batch_seconds);
            printf("吞吐量  : GPU hash %.2f MH/s，整體 %.2f MH/s\n",
                   throughput_mhps(batch_count, hash_seconds),
                   throughput_mhps(batch_count, batch_seconds));
        } else {
            printf("[batch %llu] 耗時 %.3f 秒，GPU hash %.2f MH/s，整體 %.2f MH/s，累計 %.2f MH/s\n",
                   (unsigned long long)batch_index, batch_seconds,
                   throughput_mhps(batch_count, hash_seconds),
                   throughput_mhps(batch_count, batch_seconds),
                   throughput_mhps(scanned, elapsed_seconds));
        }
        fflush(stdout);

        if (smoke) break;
    }

    if (!smoke && stop_requested) {
        printf("收到停止訊號，已完成 %llu 個 nonce。\n",
               (unsigned long long)scanned);
    }
    if (!have_solution) {
        fprintf(stderr, "沒有找到合法的不同 nonce 配對\n");
    }

    /*
     * 一般模式在每個 batch 完成時已寫入當下最佳解；smoke 模式也同樣
     * 寫入，因此這裡只保留驗證提示，不再用 total 重算吞吐量。
     */
    if (have_solution) {
        printf("\n驗證指令：\n");
        printf("  python3 verify_collision.py -p %s -a %llu -b %llu\n",
               prefix, (unsigned long long)best_a, (unsigned long long)best_b);
    }

    cudaFree(d_temp_storage);
    cudaFree(d_best); cudaFree(d_block_best);
    cudaFree(d_sort_out); cudaFree(d_sort_in);
    cudaFree(d_hi_sorted);
    cudaFree(d_base); cudaFree(d_hi); cudaFree(d_lo); cudaFree(d_nonce);
    cudaEventDestroy(hash_start);
    cudaEventDestroy(hash_stop);
    return have_solution ? 0 : 1;
}
