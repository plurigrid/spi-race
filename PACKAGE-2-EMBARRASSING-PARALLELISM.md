# Package 2: Embarrassing Parallelism in SPI Color Generation

## What We Already Have

The SPI (Splitmix Parallel Integrity) pipeline generates deterministic colors:

```
seed, index -> splitmix64(seed + GOLDEN * index) -> extract_rgb -> xor_fold
```

This is the canonical embarrassingly parallel workload:
- **Zero data dependencies** between indices: `color(i)` is independent of `color(j)` for all `i != j`
- **Commutative reduction**: XOR fold is associative + commutative, so any partition/order gives the same fingerprint
- **O(1) memory per worker**: color never needs to exist in memory; generate + reduce in one fused pass
- **No synchronization needed**: workers XOR their local accumulators, one final XOR-fold at the end

## Current Race Results (10-core M-series Apple Silicon, 100M)

| Lang       | Single-thread | Multi-thread     |
|------------|---------------|------------------|
| Zig L2     | 2,902 M/s     | 11,483 M/s (1B)  |
| Julia L3   | 2,775 M/s     | 10,563 M/s       |
| Swift GCD  | 2,551 M/s     | 9,887 M/s        |
| Babashka   | 6 M/s         | 13 M/s           |

Single-threaded ceiling is ~2.9 GHz / (3 multiplies per splitmix64) ~= 3 B/s. We are at 97%.
Multi-threaded ceiling is ~10x that = ~30 B/s. Best achieved: 11.5 B/s = ~38% of theoretical.

## Where the Embarrassing Parallelism Lives

### Layer 0: The Hash Function Itself
`splitmix64(seed, index)` — pure function, no state, no side effects.
Each call is an island. This is why it works on GPU, CPU threads, SIMD, async tasks, etc.
Every index in `[0, N)` can run simultaneously with zero coordination.

### Layer 1: RGB Extraction
`extract_rgb(hash) = ((h>>16)&0xFF)<<16 | ((h>>8)&0xFF)<<8 | (h&0xFF)`
Pure bitwise ops on the hash. Fuses trivially into the hash pipeline.

### Layer 2: XOR Reduction
`accumulator ^= extract_rgb(splitmix64(seed, i))`
XOR is commutative + associative. Any partition of `[0,N)` gives the same answer.
Workers need zero coordination; just XOR partial results at the end.

### Layer 3: GF(3) Trit Balance (the one non-embarrassing part)
`trit(color) = (r + g + b) mod 3 - 1`
Trit sum over N colors requires GF(3) addition, NOT majority vote.
This is still embarrassingly parallel (addition is associative), but the
reduction is modular arithmetic, not plain XOR. One extra add at the end.

## What Is Left on the Table

### 1. NEON/SVE SIMD (untapped ~2-4x single-thread)

Apple M-series has 128-bit NEON with 2x u64 per vector register.
Splitmix64 is 3 multiplies + 3 XOR-shifts.
With 2-wide SIMD, we should get ~5.8 B/s single-threaded.
**Why we don't**: Zig `@Vector(4, u64)` actually produces WORSE code than
scalar on M-series because the compiler can't fuse the
multiply-and-shift pattern into NEON's native MAC instructions
when forced into vector mode. The 8-wide scalar unroll (L2)
actually lets LLVM's scheduler use the 6-wide dispatch naturally.

**Seek**: Can we hand-write NEON intrinsics that beat LLVM's scheduling?
On x86 AVX-512 this would be trivial (8x u64). On ARM, the
gain is smaller because NEON u64 multiply is 2-cycle not 1.

### 2. Metal/GPU Compute (untapped ~10-100x)

Apple M-series has GPU cores with:
- 128 execution units, 1024 threads per threadgroup
- 8-16 GB unified memory (zero-copy)
- Native uint32 multiply (no uint64, but we can split hi/lo)

Gay.jl already has `_ka_colors_kernel!` targeting Metal via KernelAbstractions.jl.
The current bottleneck: splitmix64 uses u64 multiply, and Metal only has u32.
Split-multiply approach (hi32/lo32) adds ~4 ops per multiply = ~12 extra ops.
Still, 128 execution units at 1 GHz each = 128 B/s theoretical.
Even at 10% efficiency = 12 B/s > our best CPU result.

**Seek**: Write a Metal compute shader that does split-u32 splitmix64.
Compare throughput to CPU L6.

### 3. Zig `@prefetch` + Cache Line Alignment (untapped ~10-20%)

The L4 tiled approach does comptime unrolling but doesn't actually prefetch.
On M-series, L1D has 16-cycle miss penalty. Prefetching the next tile's
index computation while current tile multiplies are in flight could hide
this latency. We measured L4 = same as L2 because the inner loop is
compute-bound, not memory-bound. But at 1B scale, TLB pressure on the
thread stacks could cause stalls.

**Seek**: Profile with `perf stat` / Instruments to confirm or deny.

### 4. Async I/O + Color Generation Pipeline (untapped for real workloads)

The benchmark measures raw throughput. Real applications (Terminal rendering,
DuckDB queries, Ghostty tiles) need colors fed into a downstream consumer.
The embarrassing parallelism means we can double-buffer:
- Thread pool A generates colors 0..N/2
- Thread pool B generates colors N/2..N
- Consumer reads from whichever finishes first (both are valid partial results)

This is the Tileable connection: each Ghostty tile is an independent
colorAt(seed, tile_id) call. The tileable_shader.zig already does this
for Game of Life. Generalize to arbitrary tile coloring.

### 5. Cross-Process / Distributed (untapped for cluster scale)

Since `colorAt(seed, i)` is pure with no shared state, you can split
the index range across machines and XOR-fold the fingerprints.
Billion-color verification across 10 machines = 100M per machine = 10ms each.
Network latency dominates, not compute.

**Seek**: zig-syrup already has Syrup serialization. Add a `spi_verify`
message type that distributes index ranges and collects XOR fingerprints.

### 6. The Babashka / JVM / Python Question

These languages are 100-1000x slower for this workload because:
- JVM: no unsigned 64-bit integers, boxing overhead, GC pauses
- CPython: interpreter overhead, GIL for threading
- But: they can FFI into Zig/C for the hot path

**Seek**: Write a minimal C shared library `libspi.{so,dylib}` exposing
`spi_xor_fingerprint(seed, start, count) -> uint64`. Call from bb, Python, Swift.
This is the embarrassingly parallel FFI pattern: host language handles
orchestration, native code handles the tight loop.

## The Fundamental Insight

SPI color generation is embarrassingly parallel at EVERY level:
- Within a single core (ILP: 8 independent accumulators)
- Across cores (thread pool: partition index range)
- Across chips (GPU: 128 EUs, each running the same kernel)
- Across machines (distributed: split index range, XOR-fold fingerprints)

The ONLY coordination point is the final XOR-fold, which is O(num_workers).
For 10 threads, that's 10 XOR ops = ~3 nanoseconds. The overhead of
parallelism is effectively zero relative to the work.

This is what makes it the ideal benchmark: any language/runtime that can't
saturate the hardware on this workload has overhead problems unrelated
to the algorithm. The algorithm itself is maximally parallel.

## Connection to Ensemble Reservoir Computing (Nakamura 2026)

Yuma Nakamura's ERC framework (Kanazawa University) reveals that the SPI
color pipeline and physical reservoir computing share the SAME parallelism
structure, and that ensemble averaging IS the embarrassing parallelism:

### The Isomorphism

| SPI Color Generation                | Ensemble Reservoir Computing         |
|--------------------------------------|---------------------------------------|
| splitmix64(seed, i) = pure hash      | reservoir_state(input, noise_k) = mixed |
| Each index i is independent          | Each replica k is independent         |
| XOR-fold over indices                | Ensemble average over replicas        |
| Commutative + associative reduction  | Mean is commutative + associative     |
| Result depends only on seed          | Result depends only on input history  |
| Noise-free by construction (hash)    | Noise-free by averaging (ERC proof)   |

The key insight from Nakamura: ensemble averaging acts as a **nonlinear
transformation F** that strips out noise, temporal fluctuations, and initial
condition dependencies — leaving only the input-dependent component.

In SPI terms: `splitmix64` IS the transformation F. It's already pure.
ERC needs to CONSTRUCT purity by averaging noisy replicas.
SPI has purity for free because the hash has no state.

### Where ERC Needs What SPI Already Has

Nakamura showed 16 STO replicas achieved 99% CRC accuracy.
Each replica is driven with common input but independent noise.
The averaging step: `X_avg(t) = (1/L) * sum_k X_k(t)` for L=16.

This is embarrassingly parallel: each replica runs independently.
The only coordination is the final average (one add per replica per timestep).

With ReservoirComputing.jl, the ERC pipeline is:

```julia
using ReservoirComputing, Statistics
L = 16
replicas = [ESN(in_size, res_size, out_size;
    init_reservoir=rand_sparse(; radius=1.2, sparsity=6/300))
    for _ in 1:L]

# Drive all L replicas with same input — embarrassingly parallel
states = pmap(replicas) do esn  # or Threads.@threads
    rng = MersenneTwister(rand(UInt32))
    ps, st = setup(rng, esn)
    train!(esn, input_data, target_data, ps, st; return_states=true)
    st.states
end

# Ensemble average — the F transformation
avg_states = mean(states)

# Train single readout on cleaned states
W_out = target_data * avg_states' * inv(avg_states * avg_states' + ridge * I)
```

### Where SPI Could Learn from ERC

ERC shows that ensemble averaging ENHANCES computational capacity beyond
the noise-free baseline (spectral radius > 1 regime). The parallel replicas
don't just cancel noise — they access computational modes that a single
reservoir can't reach.

For SPI: instead of XOR-folding identical hashes (redundant verification),
what if we XOR-fold DIFFERENT hash functions (splitmix64, xxhash, cityhash)
across the same index range? The ensemble would be a multi-hash fingerprint
that's more collision-resistant than any single hash. Same embarrassing
parallelism, but the diversity of hash functions plays the role of
independent noise realizations in ERC.

### Lux-Zig-Julia Triangle

The cross-pollination path:

```
Lux (functional Lisp)
  → LispSyntax.jl (Julia Lisp DSL)
    → ReservoirComputing.jl (ESN/ERC)
      → KernelAbstractions.jl (GPU kernels)
        → Gay.jl SPI kernels (splitmix64)

Lux (functional Lisp)
  → lux-zig backend (program.lux → Zig codegen)
    → zig-syrup (libspi.zig, tileable.zig)
      → notcurses_runtime.zig (terminal rendering)
        → Ghostty tiles (embarrassingly parallel cell coloring)
```

Both paths converge on the same insight: the operation at each spatial
location (tile cell, reservoir state, color index) is independent.
The only question is what reduction you want at the end (XOR, mean, readout).

## Package 2 Concrete Deliverables

1. `libspi.zig` — shared library with C ABI: `spi_xor_fingerprint(seed, start, count)`
2. `spi-metal.swift` — Metal compute shader benchmark
3. `spi-neon.zig` — Hand-written NEON intrinsics vs LLVM auto-vectorization
4. `spi-ffi.bb` / `spi-ffi.py` — FFI callers for the Zig shared lib
5. `spi-distributed.zig` — Syrup-based distributed fingerprint verification
6. `erc-spi.jl` — ERC using Gay.jl splitmix64 as the hash-reservoir bridge
7. Updated race table with all new entrants
