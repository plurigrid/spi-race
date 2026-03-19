# SPI Race Leaderboard

All racers compute the same function: `XOR(extractRGB(splitmix64(seed, i)) for i in 0..N)`.
Correctness verified by matching `0x10de88` at N=1M, `0xf76ceb` at N=10M.

## Results (Apple M5, March 2026)

### Single Clock Domain

| Rank | Racer | Lang | Peak M/s | Peak B/s | N | Notes |
|------|-------|------|----------|----------|---|-------|
| 1 | spi-metal-ultra | Swift/MSL | 94,685 | 94.7 B | 500M | 64/thr, 8-way ILP, branchless, fused params |
| 2 | spi-metal-clocks (flick) | Swift/MSL | 91,308 | 91.3 B | 500M | folded golden-stride, sparse beats dense |
| 3 | spi-metal-clocks (trit) | Swift/MSL | 87,830 | 87.8 B | 500M | folded golden-stride |
| 4 | spi-metal-vec | Swift/MSL | 95,655 | 95.7 B | 500M | ulong2 vectorized, ties ultra |
| 5 | spi-metal-fast | Swift/MSL | 54,900 | 54.9 B | 1B | 16/thr, 4-way ILP |
| 6 | spi-metal | Swift/MSL | 10,600 | 10.6 B | 1B | 1/thr baseline GPU |
| 7 | spi-neon MT-8ILP | C -O3 | 12,931 | 12.9 B | 500M | 10-core pthreads |
| 8 | spi-neon 1-core | C -O3 | 2,588 | 2.6 B | 500M | single P-core |
| 9 | Babashka | Clojure/SCI | 3.2 | 0.003 B | 1M | unchecked-math |
| 10 | Python | CPython 3.9 | 2.6 | 0.003 B | 1M | pure scalar |

### Fused 3-Clock (Color + Flick + Trit-tick in one dispatch)

| Metric | Value |
|--------|-------|
| Peak samples/s | 24,469 M (24.5 B) |
| Peak colors/s (3x) | 73,408 M (73.4 B) |
| vs 3x sequential | 0.95x (register pressure from 12 accumulators) |

## Key Ratios

| Comparison | Ratio |
|-----------|-------|
| GPU Ultra vs 10-core CPU | 7.3x |
| GPU Ultra vs 1-core CPU | 36.6x |
| GPU Ultra vs Python | 36,417x |
| GPU Ultra vs Babashka | 29,589x |
| 10-core C vs 1-core C | 5.0x |
| C vs Python (single-core) | 995x |

## GPU Optimization History

| Version | Peak | Improvement | Key Change |
|---------|------|-------------|------------|
| spi-metal (baseline) | 10.6 B/s | - | 1 color/thread, CPU partial reduce |
| spi-metal-fast | 54.9 B/s | 5.2x | 16/thr, SIMD warp, GPU reduce |
| spi-metal-ultra | 94.7 B/s | 1.7x (8.9x total) | 64/thr, 8-way ILP, branchless |
| spi-metal-vec | 95.7 B/s | ~1x (tied) | ulong2 vector, no win on M5 |

## BCI Real-Time Multipliers (at 95 B/s)

| Device | Rate | GPU can process |
|--------|------|-----------------|
| OpenBCI EEG | 250 Hz | 11.9 years/s |
| Neuropixels | 30 kHz | 36 days/s |
| Audio 48 kHz | 48 kHz | 22 days/s |

## Clock Domain Results

All three clock domains achieve the same throughput. Flick and trit-tick strides
are folded into the golden constant (`GOLDEN * stride`) on the CPU before dispatch.
The GPU hot loop is identical across domains.

| Domain | Stride (250 Hz) | 100M M/s | 500M M/s | Notes |
|--------|-----------------|----------|----------|-------|
| Color (raw) | 1 | 83,741 | 74,774 | Dense indices |
| Flick | 2,822,400 | 85,671 | **91,308** | Sparse — beats dense at 500M |
| Trit-tick | 564,480 | 85,699 | 87,830 | Sparse — beats dense at 500M |

Flick beats raw color by 22% at 500M because sparse strides reduce GPU cache bank conflicts.

## What Didn't Work

- **NEON vectorized splitmix64**: NEON lacks `vmulq_u64` for 64-bit integer multiply. Can't vectorize the hash.
- **16-way ILP on CPU**: Register pressure. 8-ILP is the sweet spot on M5 P-cores.
- **128 colors/thread on GPU**: Too much register pressure, drops occupancy. 64 is optimal.
- **Double-pump command buffers**: GPU already saturated at single dispatch for N >= 200M.
- **CHUNK=32 on GPU**: Too many threadgroups, reduction overhead dominates.
- **Fused 3-clock kernel**: 12 accumulators (4×3 domains) cause register pressure. 3× sequential is 5% faster.
