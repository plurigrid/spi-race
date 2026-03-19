# spi-race

**Splitmix Parallel Integrity** — deterministic color generation at hardware bandwidth, tested across every language that matters.

## The Core Invariant

```
color_at(seed, index) = extract_rgb(splitmix64(seed, index))
```

This is a **pure function**: no state, no side effects, no coordination. Every index is independent. Every partition produces the same XOR fingerprint. This makes it the canonical **embarrassingly parallel** workload — any runtime that can't saturate hardware on this has overhead problems unrelated to the algorithm.

## What's Here

### Zig Core

| File | What |
|------|------|
| `libspi.zig` | Shared library (C ABI): `spi_xor_fingerprint`, `spi_color_at`, `spi_trit`, parallel multi-threaded |
| `libspi_color.zig` | Triad-colored variant: comptime seed baked in, XOR involution witness |
| `spi-test.zig` | Correctness + benchmark: 1T vs MT agreement, 1M/10M/100M/1B throughput |
| `spi-flicks.zig` | **Flick time base** (705.6 MHz): frame-aligned color generation, NTSC snow point demo |
| `spi-trit-tick.zig` | **Trit-tick time base** (4 epochs): precomputed BCI device divisor tables, epoch cross-validation race, BCI snow points, cross-modal alignment |
| `vibe_kanban.zig` | CatColab theories as SPI-indexed parallel data structures: Fokker-Planck, DAG gating, theory-typed fingerprints |
| `spi-metal.swift` | **Metal GPU racer**: MSL compute shaders, threadgroup reduction, GPU scaling 1M-1B, BCI tick-space — **10.6 B/s** |
| `spi-metal-fast.swift` | **Optimized GPU**: 16/thr ILP, `simd_xor` warp reduction, 1024 threadgroups, two-pass GPU reduce — **54.9 B/s** |
| `spi-metal-ultra.swift` | **Maximum GPU**: 64/thr, 8-way ILP, branchless hot loop, fused params, N sweep — **94.7 B/s** |
| `spi-metal-vec.swift` | **Strategy shootout**: ulong2 vectorized GPU, scalar-wide GPU comparison — ties ultra at **95.7 B/s** |
| `spi-neon.c` | **CPU racer**: 8/16-ILP, single-core + 10-core pthreads — **12.9 B/s** multi-core |

### Multi-Language Racers

| File | Language | Tricks |
|------|----------|--------|
| `swift-racer.swift` | Swift 6 | `@inline(__always)`, `&+`/`&*`, GCD `concurrentPerform`, `TaskGroup` structured concurrency |
| `clojure-racer.bb` | Babashka/JVM | `unchecked-math`, `pmap`, virtual threads, `loop/recur` |
| `hymlx-racer.py` | Python/MLX | numpy vectorized, MLX Metal GPU (uint32 approx), multiprocessing |
| `spi-ffi.py` | Python ctypes | Zero-overhead FFI into `libspi.dylib` |
| `spi-ffi.bb` | Babashka | Pure Clojure splitmix64, cross-validates against Zig |
| `spi-ffi-ruby.rb` | Ruby | FFI via `fiddle` |
| `spi-ffi-swift.swift` | Swift | `dlopen`/`dlsym` |
| `spi-ffi-guile.scm` | Guile Scheme | `dynamic-link` |
| `spi-ffi-node.mjs` | Node.js | `ffi-napi` |
| `spi-ffi-perl.pl` | Perl | `FFI::Platypus` |
| `spi-ffi-tcl.tcl` | Tcl | `ffidl` |

### Build System

| File | What |
|------|------|
| `build.zig` | Zig build: `libspi.{dylib,so,a}`, `spi-test`, `spi-flicks`, `spi-trit-tick`, `vibe-kanban` |
| `build_triads.py` | Generates GF(3)-balanced triad dylibs: `lib<hex>.dylib` per color triple |

## Race Results (Apple M5, 10-core, March 2026)

| Rank | Backend | Peak M/s | Peak B/s | Notes |
|------|---------|----------|----------|-------|
| **1** | **Metal GPU Ultra** | **94,685** | **94.7 B** | 64/thr, 8-way ILP, branchless, fused params |
| 2 | Metal GPU Vec2 | 95,655 | 95.7 B | ulong2 vectorized (ties ultra) |
| 3 | Metal GPU Fast | 54,900 | 54.9 B | 16/thr, 4-way ILP, SIMD warp reduction |
| 4 | Metal GPU Baseline | 10,600 | 10.6 B | 1/thr, CPU-side partial reduce |
| 5 | C -O3 10-core | 12,931 | 12.9 B | 8-ILP pthreads |
| 6 | Zig L2 10-core | ~11,500 | ~11.5 B | 8-wide unroll |
| 7 | Julia L3 10-core | ~10,500 | ~10.5 B | `@simd` + `Threads.@threads` |
| 8 | Swift GCD 10-core | ~9,900 | ~9.9 B | GCD `concurrentPerform` |
| 9 | C -O3 single-core | 2,588 | 2.6 B | 8-ILP scalar |
| 10 | Babashka | 3.2 | 0.003 B | SCI interpreter, unchecked-math |
| 11 | Python | 2.6 | 0.003 B | CPython 3.9 scalar |
| Python L0 | 0.3 M/s | — | CPython baseline |
| Python FFI | 2,900 M/s | 11,000 M/s | ctypes into `libspi.dylib` — same as native |

### GPU Benchmarks — Ultra (`spi-metal-ultra.swift`)

| N | GPU time | GPU M/s | GB/s (3B/color) | Threadgroups |
|---|----------|---------|-----------------|--------------|
| 100M | 2.0 ms | 49,509 | 148.5 | 1,526 |
| **500M** | **5.3 ms** | **94,685** | **280.8** | **7,630** |
| 1B | 11.7 ms | 85,362 | 256.1 | 15,259 |
| 2B | 24.9 ms | 80,407 | 241.2 | 30,518 |

Peak: **94.7 B colors/s**, 280.8 GB/s bandwidth (M5 Apple Silicon).

GPU vs 10-core CPU: **7.3x speedup** (95 B/s GPU vs 12.9 B/s 10-core C -O3).
GPU vs single-core: **36.6x speedup** (95 B/s GPU vs 2.6 B/s single-core C -O3).

### GPU Optimization History

| Version | Peak B/s | Improvement | Key Change |
|---------|----------|-------------|------------|
| spi-metal (baseline) | 10.6 | 1x | 1 color/thread, CPU partial reduce |
| spi-metal-fast | 54.9 | 5.2x | 16/thr, SIMD warp, GPU reduce |
| spi-metal-ultra | 94.7 | 8.9x | 64/thr, 8-way ILP, branchless |
| spi-metal-vec (ulong2) | 95.7 | ~1x (tied) | Vector types don't help on M5 |

### GPU Benchmarks — Baseline (`spi-metal.swift`)

| N | GPU time | GPU M/s | GB/s (3B/color) | Threadgroups |
|---|----------|---------|-----------------|--------------|
| 100M | 10.7 ms | 9,344 | 28.03 | 390,625 |
| **1B** | **95.2 ms** | **10,507** | **31.52** | **3,906,250** |

BCI long-duration: 1 hour of Neuropixels AP (108M samples) fingerprinted in 10 ms on GPU. All XOR fingerprints match CPU reference at every scale.

## Time Bases

### Flick (Epoch 0) — `spi-flicks.zig`

```
705,600,000 Hz = 2⁶ × 3³ × 5² × 7² × 11 × 13
```

Facebook/Horvath 2018. Every standard audio rate (44.1k, 48k, 96k) and video rate (24, 25, 30, 60, 90, 120, 144 fps) divides exactly. Integer multiply for frame boundaries. The **snow point** demo shows where IEEE 754 f64 accumulated frame timing diverges from integer-exact flick boundaries — producing wrong colors at wrong sample boundaries.

Prior art: Rennes/INA TimeRef (2004) = 50 flicks. The rational-arithmetic critique (Munich) says flicks are unnecessary because `p/q` seconds is exact. Counterargument: integer arithmetic is embarrassingly parallel. Rationals require GCD normalization. Flicks are O(1) AND exact.

### Trit-Tick (Epochs 1-3 + Unbounded) — `spi-trit-tick.zig`

The trit-tick time base extends flicks to cover **all BCI device sample rates** — the rates that flicks cannot handle because they inject new prime factors from four independent mechanisms:

#### Why unusual primes appear in device sample rates

1. **Sensor geometry** — Camera frame rates are `pixel_clock / (rows × cols + blanking)`. The Sony IMX273 (FLIR Blackfly S) produces 206 fps (=2×103), 226 fps (=2×113), etc. The primes are physical constants of the silicon die layout. Change the ADC bit depth and the prime changes.

2. **Biological resonance** — DBS at 130 Hz (prime 13) was chosen by Benabid in 1991 because it suppresses Parkinsonian tremor. 185 Hz (prime 37) targets dystonia in the GPi. These are FDA-approved clinical protocols. The primes are brain constants, not engineering choices.

3. **ADC clock divider chains** — Sigma-delta ADCs decimate from a master oscillator. Delsys Trigno EMG at 1926 Hz (= 2×3²×107) comes from dividing a master clock by a non-integer decimation factor. Prime 107 is an arithmetic remnant of the ADC architecture.

4. **Transport bandwidth ceilings** — USB 3.0 throughput divided by frame size creates a rate ceiling. Each pixel format and resolution produces a different prime factor.

**Consequence**: the universe of device primes is **open**, not closed. No fixed LCM can ever cover all devices.

#### The 4-epoch ladder

| Epoch | Base Hz | Primes | Coverage | Register |
|-------|---------|--------|----------|----------|
| 0 (flick) | 705,600,000 | 6 | 82% of 44 BCI rates | u64 |
| 1 (trit-tick) | 141,120,000 | 4 | 80% | u64 |
| 2 (expanded) | 51.4×10¹⁸ | 9 | 95% (adds 13, 37, 113, 127) | u128 |
| 3 (extreme) | 22.7×10³⁶ | 16 | 95% (Fibonacci/Padovan closure) | u128 |
| ∞ (unbounded) | prime exponent vectors | ∞ | 100% (monzo-style, never overflows) | `[N]i16` |

#### The precomputed divisor table

`spi-trit-tick.zig` precomputes, for all 44 BCI device rates and all 4 epochs: does the rate divide exactly? What's the ticks-per-sample? This tells you **at compile time** which epoch each device needs.

#### The SPI race

For every rate that two epochs both handle exactly, generate colors at sample boundaries in both tick spaces and verify they agree. All 35 testable rates pass, 0 fail.

#### BCI snow points

The BCI equivalent of the NTSC snow point. At 250 Hz (OpenBCI Cyton), f64 accumulated sample timing produces the first wrong color at **sample 1007 (~4 seconds)**. At Neuropixels 30 kHz, **106 million wrong colors per hour**. Integer trit-ticks: zero drift forever.

#### Cross-modal alignment

How often do two devices' sample boundaries coincide?

| Pair | GCD | Alignment rate | Epoch needed |
|------|-----|----------------|--------------|
| EEG 250 + BioSemi 2048 | 2 | 2/sec (500ms) | Epoch 2+ |
| EEG 250 + DBS 130 | 10 | 10/sec (100ms) | Epoch 2+ |
| CD 44100 + DAC 48000 | 300 | 300/sec (3.3ms) | Flick |
| VR 90 + EEG 500 | 10 | 10/sec (100ms) | Flick |

## Experimental setups analyzed

15 real-world multi-device setups from neuroscience literature (Iwama 2024, Kothe/LSL 2025, Both/Syntalos 2025, Kappel 2025), encompassing 44 unique device rates and 127 coprime pairs:

1. EEG + fMRI (simultaneous)
2. EEG + Eye Tracking
3. Neuropixels + Behavior Cameras
4. DBS + LFP Recording
5. TMS + EEG
6. VR + EEG (MoBI)
7. fNIRS + EEG + EMG (Rehab BCI)
8. ECoG + Micro-ECoG + Stimulation
9. Multi-subject Hyperscanning
10. Adaptive/Closed-loop DBS
11. Auditory BCI
12. Full MoBI Setup (canonical hard case)
13. Neuropixels + Optogenetics + Behavior
14. Hi-Res Audio + Video + EEG
15. Haptic BCI + Multi-display

## The Embarrassingly Parallel Property

SPI color generation is embarrassingly parallel at **every level**:

- **Within a core**: 8 independent accumulators exploit ILP
- **Across cores**: partition index range, XOR-fold at end (10 XORs = 3 ns)
- **Across chips**: GPU Metal shader (128 EUs × 1 GHz)
- **Across machines**: split range over network, XOR-fold fingerprints

The XOR reduction is **commutative + associative**: any partition, any order, same answer. This holds in flick space, trit-tick space, and across epoch boundaries.

The connection to Ensemble Reservoir Computing (Nakamura 2026, Kanazawa): ensemble averaging IS the embarrassing parallelism. Independent replicas → commutative reduction → noise-free result. SPI has purity for free (hash has no state); ERC must construct it by averaging.

## GF(3) Triad Teleportation

`build_triads.py` generates one `lib<hex>.dylib` per GF(3)-balanced triad of GitHub Linguist logo colors. The triad seed = XOR of three colors. XOR is involution: `a ^ a = 0`. Loading all triads and XOR-folding their seeds teleports back to origin.

## Build

```sh
zig build              # builds libspi.dylib, libspi.a, spi-test, spi-flicks, spi-trit-tick, vibe-kanban
zig build test         # correctness + benchmark
zig build flicks       # flick time base + NTSC snow point
zig build trit-tick    # BCI device precomputation + races + snow points
zig build vibe-kanban  # CatColab theories at SPI bandwidth
```

## References

- Horvath, C. (2018). "Flicks: A unit of time." Facebook/Oculus. github.com/facebookarchive/Flicks
- Troncy, R. et al. (2004). "CORIMEDIA: Temporal Ontologies for Media." Sorbonne/INA. TimeRef = 50 flicks.
- Iwama, S. et al. (2024). "Two common issues in synchronized multimodal recordings with EEG." Neurosci Res 203:1-7.
- Kothe, C. et al. (2025). "The Lab Streaming Layer for synchronized multimodal recording." Imaging Neurosci 3:IMAG.a.136.
- Both, M. et al. (2025). "Syntalos." Nature Communications.
- Kappel, S. et al. (2025). "Temporal synchronization of multimodal hyperscanning." EMBC 2025.
- Plant, R. & Turner, G. (2009). "Millisecond timing errors in commonly used equipment." Cogn Affect Behav Neurosci 13:598-614.
- Nakamura, Y. et al. (2026). "Ensemble Reservoir Computing." Kanazawa University.
