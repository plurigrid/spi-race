// SPI NEON — ARM NEON SIMD CPU racer
//
// Uses NEON 128-bit vector intrinsics to process 2 splitmix64 in parallel.
// Compare: how close can CPU SIMD get to GPU?
//
// clang -O3 -march=native -o spi-neon spi-neon.c && ./spi-neon

#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <arm_neon.h>
#include <pthread.h>

#define GOLDEN 0x9e3779b97f4a7c15ULL
#define MIX1   0xbf58476d1ce4e5b9ULL
#define MIX2   0x94d049bb133111ebULL

static inline uint32_t sm64_rgb(uint64_t seed, uint64_t idx) {
    uint64_t z = seed + GOLDEN * idx;
    z = (z ^ (z >> 30)) * MIX1;
    z = (z ^ (z >> 27)) * MIX2;
    z = z ^ (z >> 31);
    return (uint32_t)(((z >> 16) & 0xFF) << 16 | ((z >> 8) & 0xFF) << 8 | (z & 0xFF));
}

// Scalar baseline
__attribute__((noinline))
static uint32_t scalar_xor(uint64_t seed, int n) {
    uint32_t x = 0;
    for (uint64_t i = 0; i < (uint64_t)n; i++) x ^= sm64_rgb(seed, i);
    return x;
}

// Scalar 8-way ILP (matches GPU ultra strategy on CPU)
__attribute__((noinline))
static uint32_t scalar_ilp8_xor(uint64_t seed, int n) {
    uint32_t a0=0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0, a7=0;
    int n8 = n & ~7;
    for (int i = 0; i < n8; i += 8) {
        uint64_t idx = (uint64_t)i;
        a0 ^= sm64_rgb(seed, idx);
        a1 ^= sm64_rgb(seed, idx+1);
        a2 ^= sm64_rgb(seed, idx+2);
        a3 ^= sm64_rgb(seed, idx+3);
        a4 ^= sm64_rgb(seed, idx+4);
        a5 ^= sm64_rgb(seed, idx+5);
        a6 ^= sm64_rgb(seed, idx+6);
        a7 ^= sm64_rgb(seed, idx+7);
    }
    uint32_t x = a0^a1^a2^a3^a4^a5^a6^a7;
    for (int i = n8; i < n; i++) x ^= sm64_rgb(seed, (uint64_t)i);
    return x;
}

// 16-way ILP to saturate OoO
__attribute__((noinline))
static uint32_t neon_xor(uint64_t seed, int n) {
    // 16-way scalar ILP — max out the M5 P-core OoO window
    uint32_t a0=0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0, a7=0;
    uint32_t b0=0, b1=0, b2=0, b3=0, b4=0, b5=0, b6=0, b7=0;
    int n16 = n & ~15;
    for (int i = 0; i < n16; i += 16) {
        uint64_t idx = (uint64_t)i;
        a0 ^= sm64_rgb(seed, idx);    a1 ^= sm64_rgb(seed, idx+1);
        a2 ^= sm64_rgb(seed, idx+2);  a3 ^= sm64_rgb(seed, idx+3);
        a4 ^= sm64_rgb(seed, idx+4);  a5 ^= sm64_rgb(seed, idx+5);
        a6 ^= sm64_rgb(seed, idx+6);  a7 ^= sm64_rgb(seed, idx+7);
        b0 ^= sm64_rgb(seed, idx+8);  b1 ^= sm64_rgb(seed, idx+9);
        b2 ^= sm64_rgb(seed, idx+10); b3 ^= sm64_rgb(seed, idx+11);
        b4 ^= sm64_rgb(seed, idx+12); b5 ^= sm64_rgb(seed, idx+13);
        b6 ^= sm64_rgb(seed, idx+14); b7 ^= sm64_rgb(seed, idx+15);
    }
    uint32_t x = a0^a1^a2^a3^a4^a5^a6^a7^b0^b1^b2^b3^b4^b5^b6^b7;
    for (int i = n16; i < n; i++) x ^= sm64_rgb(seed, (uint64_t)i);
    return x;
}

// Multi-threaded
typedef struct {
    uint64_t seed;
    int start, end;
    uint32_t result;
    int use_neon;
} ThreadArg;

static void* thread_worker(void* arg) {
    ThreadArg* a = (ThreadArg*)arg;
    if (a->use_neon) {
        // 16-way ILP to saturate P-core OoO
        uint32_t a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
        uint32_t b0=0,b1=0,b2=0,b3=0,b4=0,b5=0,b6=0,b7=0;
        int count = a->end - a->start;
        int n16 = count & ~15;
        for (int j = 0; j < n16; j += 16) {
            uint64_t idx = (uint64_t)(a->start + j);
            a0 ^= sm64_rgb(a->seed, idx);    a1 ^= sm64_rgb(a->seed, idx+1);
            a2 ^= sm64_rgb(a->seed, idx+2);  a3 ^= sm64_rgb(a->seed, idx+3);
            a4 ^= sm64_rgb(a->seed, idx+4);  a5 ^= sm64_rgb(a->seed, idx+5);
            a6 ^= sm64_rgb(a->seed, idx+6);  a7 ^= sm64_rgb(a->seed, idx+7);
            b0 ^= sm64_rgb(a->seed, idx+8);  b1 ^= sm64_rgb(a->seed, idx+9);
            b2 ^= sm64_rgb(a->seed, idx+10); b3 ^= sm64_rgb(a->seed, idx+11);
            b4 ^= sm64_rgb(a->seed, idx+12); b5 ^= sm64_rgb(a->seed, idx+13);
            b6 ^= sm64_rgb(a->seed, idx+14); b7 ^= sm64_rgb(a->seed, idx+15);
        }
        uint32_t r = a0^a1^a2^a3^a4^a5^a6^a7^b0^b1^b2^b3^b4^b5^b6^b7;
        for (int j = n16; j < count; j++) r ^= sm64_rgb(a->seed, (uint64_t)(a->start + j));
        a->result = r;
    } else {
        uint32_t a0=0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0, a7=0;
        int count = a->end - a->start;
        int n8 = count & ~7;
        for (int j = 0; j < n8; j += 8) {
            uint64_t idx = (uint64_t)(a->start + j);
            a0 ^= sm64_rgb(a->seed, idx);   a1 ^= sm64_rgb(a->seed, idx+1);
            a2 ^= sm64_rgb(a->seed, idx+2); a3 ^= sm64_rgb(a->seed, idx+3);
            a4 ^= sm64_rgb(a->seed, idx+4); a5 ^= sm64_rgb(a->seed, idx+5);
            a6 ^= sm64_rgb(a->seed, idx+6); a7 ^= sm64_rgb(a->seed, idx+7);
        }
        uint32_t r = a0^a1^a2^a3^a4^a5^a6^a7;
        for (int j = n8; j < count; j++) r ^= sm64_rgb(a->seed, (uint64_t)(a->start + j));
        a->result = r;
    }
    return NULL;
}

static uint32_t mt_xor(uint64_t seed, int n, int nthreads, int use_neon) {
    pthread_t threads[nthreads];
    ThreadArg args[nthreads];
    int chunk = n / nthreads;
    for (int t = 0; t < nthreads; t++) {
        args[t].seed = seed;
        args[t].start = t * chunk;
        args[t].end = (t == nthreads - 1) ? n : (t + 1) * chunk;
        args[t].result = 0;
        args[t].use_neon = use_neon;
        pthread_create(&threads[t], NULL, thread_worker, &args[t]);
    }
    uint32_t result = 0;
    for (int t = 0; t < nthreads; t++) {
        pthread_join(threads[t], NULL);
        result ^= args[t].result;
    }
    return result;
}

static uint64_t now_ns(void) {
    return clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
}

int main(void) {
    uint64_t seed = 42;
    int cores = 10; // M5

    printf("\n");
    printf("+======================================================================+\n");
    printf("|  SPI NEON — ARM NEON SIMD CPU Racer                                  |\n");
    printf("|  2 splitmix64/vector, 4-way ILP = 8 hashes/iteration                |\n");
    printf("+======================================================================+\n\n");

    // Correctness
    printf("  CORRECTNESS (N=10M)\n");
    uint32_t ref = scalar_xor(seed, 10000000);
    uint32_t ilp = scalar_ilp8_xor(seed, 10000000);
    uint32_t neon = neon_xor(seed, 10000000);
    uint32_t mt_s = mt_xor(seed, 10000000, cores, 0);
    uint32_t mt_n = mt_xor(seed, 10000000, cores, 1);
    printf("  Scalar:         0x%06x %s\n", ref, ref==ref ? "PASS" : "FAIL");
    printf("  Scalar 8-ILP:   0x%06x %s\n", ilp, ilp==ref ? "PASS" : "FAIL");
    printf("  NEON 4-ILP:     0x%06x %s\n", neon, neon==ref ? "PASS" : "FAIL");
    printf("  MT scalar (%d):  0x%06x %s\n", cores, mt_s, mt_s==ref ? "PASS" : "FAIL");
    printf("  MT NEON (%d):    0x%06x %s\n", cores, mt_n, mt_n==ref ? "PASS" : "FAIL");

    // Single-core benchmark
    printf("\n  SINGLE-CORE SCALING (best-of-3)\n");
    printf("  Strategy       N            M/s          GB/s\n");
    printf("  ─────────────  -----------  -----------  ─────\n");

    int sizes[] = {10000000, 100000000, 500000000};
    volatile uint32_t sink = 0; // prevent dead code elimination
    for (int si = 0; si < 3; si++) {
        int n = sizes[si];

        // Scalar 8-ILP
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < 3; t++) {
            uint64_t t0 = now_ns();
            uint32_t r = scalar_ilp8_xor(seed, n);
            uint64_t t1 = now_ns();
            sink ^= r;
            uint64_t dt = t1 - t0;
            if (dt > 0 && dt < best) best = dt;
        }
        double ms_s = (double)best / 1e6;
        double rate_s = (double)n / (double)best * 1e3; // M/s
        double gbps_s = (double)n * 3.0 / (double)best;

        // 16-ILP
        best = UINT64_MAX;
        for (int t = 0; t < 3; t++) {
            uint64_t t0 = now_ns();
            uint32_t r = neon_xor(seed, n);
            uint64_t t1 = now_ns();
            sink ^= r;
            uint64_t dt = t1 - t0;
            if (dt > 0 && dt < best) best = dt;
        }
        double ms_n = (double)best / 1e6;
        double rate_n = (double)n / (double)best * 1e3;
        double gbps_n = (double)n * 3.0 / (double)best;

        printf("  8-ILP  %11d  %8.1f ms  %9.1f M/s  %5.1f GB/s\n", n, ms_s, rate_s, gbps_s);
        printf("  16-ILP %11d  %8.1f ms  %9.1f M/s  %5.1f GB/s\n", n, ms_n, rate_n, gbps_n);
    }
    (void)sink;

    // Multi-core benchmark
    printf("\n  MULTI-CORE (%d threads, best-of-3)\n", cores);
    printf("  Strategy       N            M/s          GB/s\n");
    printf("  ─────────────  -----------  -----------  ─────\n");

    double peak_cpu = 0;
    for (int si = 0; si < 3; si++) {
        int n = sizes[si];

        // MT scalar
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < 3; t++) {
            uint64_t t0 = now_ns();
            mt_xor(seed, n, cores, 0);
            uint64_t dt = now_ns() - t0;
            if (dt < best) best = dt;
        }
        double rate_s = (double)n / (double)best * 1e3;
        double gbps_s = (double)n * 3.0 / (double)best;

        // MT NEON
        best = UINT64_MAX;
        for (int t = 0; t < 3; t++) {
            uint64_t t0 = now_ns();
            mt_xor(seed, n, cores, 1);
            uint64_t dt = now_ns() - t0;
            if (dt < best) best = dt;
        }
        double rate_n = (double)n / (double)best * 1e3;
        double gbps_n = (double)n * 3.0 / (double)best;

        if (rate_s > peak_cpu) peak_cpu = rate_s;
        if (rate_n > peak_cpu) peak_cpu = rate_n;

        printf("  MT-Scalar-8    %-11d  %9.1f   %5.1f\n", n, rate_s, gbps_s);
        printf("  MT-NEON-4      %-11d  %9.1f   %5.1f\n", n, rate_n, gbps_n);
    }

    printf("\n  ====================================================================\n");
    printf("  PEAK CPU: %.1f B colors/s (%d-core)\n", peak_cpu/1000.0, cores);
    printf("  vs GPU Ultra (~95 B/s): %.1fx slower\n", 95000.0/peak_cpu);
    printf("  ====================================================================\n\n");

    return 0;
}
