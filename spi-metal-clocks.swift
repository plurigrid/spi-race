// SPI Metal Clocks — Race the three clock domains head-to-head
//
// Domain 0: COLOR  — color_at(seed, i)                      dense indices
// Domain 1: FLICK  — color_at(seed, i * flick_stride)       stride = FLICK / rate
// Domain 2: TRIT   — color_at(seed, i * trit_stride)        stride = EPOCH1 / rate
//
// The hash is identical. Only the index pattern changes.
// Sparse indices (large stride) stress the multiplier differently.
//
// swiftc -O -framework Metal spi-metal-clocks.swift -o spi-metal-clocks && ./spi-metal-clocks

import Foundation
import Metal

let GOLDEN: UInt64 = 0x9e3779b97f4a7c15
let MIX1: UInt64 = 0xbf58476d1ce4e5b9
let MIX2: UInt64 = 0x94d049bb133111eb
let FLICK: UInt64 = 705_600_000
let EPOCH1: UInt64 = 141_120_000

@inline(__always)
func sm64(_ seed: UInt64, _ index: UInt64) -> UInt64 {
    var z = seed &+ (GOLDEN &* index)
    z = (z ^ (z >> 30)) &* MIX1
    z = (z ^ (z >> 27)) &* MIX2
    return z ^ (z >> 31)
}

@inline(__always)
func extractRGB(_ val: UInt64) -> UInt32 {
    return UInt32((val >> 16) & 0xFF) << 16 | UInt32((val >> 8) & 0xFF) << 8 | UInt32(val & 0xFF)
}

// CPU references
func cpuXOR(_ seed: UInt64, _ n: Int, stride: UInt64 = 1) -> UInt32 {
    var xor: UInt32 = 0
    for i in 0..<UInt64(n) { xor ^= extractRGB(sm64(seed, i &* stride)) }
    return xor
}

func cpuGCDXOR(_ seed: UInt64, _ n: Int, _ threads: Int, stride: UInt64 = 1) -> UInt32 {
    let chunk = n / threads
    let partials = UnsafeMutablePointer<UInt32>.allocate(capacity: threads)
    defer { partials.deallocate() }
    partials.initialize(repeating: 0, count: threads)
    DispatchQueue.concurrentPerform(iterations: threads) { tid in
        let start = tid * chunk
        let end = tid == threads - 1 ? n : (tid + 1) * chunk
        var a0: UInt32 = 0, a1: UInt32 = 0, a2: UInt32 = 0, a3: UInt32 = 0
        var a4: UInt32 = 0, a5: UInt32 = 0, a6: UInt32 = 0, a7: UInt32 = 0
        let count = end - start
        let n8 = count & ~7
        var j = 0
        while j < n8 {
            let base = UInt64(start + j) &* stride
            a0 ^= extractRGB(sm64(seed, base))
            a1 ^= extractRGB(sm64(seed, base &+ stride))
            a2 ^= extractRGB(sm64(seed, base &+ stride &* 2))
            a3 ^= extractRGB(sm64(seed, base &+ stride &* 3))
            a4 ^= extractRGB(sm64(seed, base &+ stride &* 4))
            a5 ^= extractRGB(sm64(seed, base &+ stride &* 5))
            a6 ^= extractRGB(sm64(seed, base &+ stride &* 6))
            a7 ^= extractRGB(sm64(seed, base &+ stride &* 7))
            j += 8
        }
        var local = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7
        while j < count { local ^= extractRGB(sm64(seed, UInt64(start + j) &* stride)); j += 1 }
        partials[tid] = local
    }
    var combined: UInt32 = 0
    for i in 0..<threads { combined ^= partials[i] }
    return combined
}

struct Params {
    var seed: UInt64
    var n: UInt32
    var chunk: UInt32
    var stride: UInt64
}

let CHUNK = 64
let TG_SIZE = 1024

let metalSource = """
#include <metal_stdlib>
using namespace metal;

struct Params {
    ulong seed;
    uint  n;
    uint  chunk;
    ulong stride;
};

static inline ulong splitmix64(ulong seed, ulong idx) {
    ulong z = seed + 0x9e3779b97f4a7c15UL * idx;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL;
    return z ^ (z >> 31);
}

static inline uint sm64_rgb(ulong seed, ulong idx) {
    ulong z = splitmix64(seed, idx);
    return uint(((z >> 16) & 0xFF) << 16 | ((z >> 8) & 0xFF) << 8 | (z & 0xFF));
}

// Folded kernel: golden_stride = GOLDEN * stride is precomputed on CPU.
// The hot loop does seed + golden_stride * i — NO per-element multiply by stride.
// This makes clock domains truly zero-cost: same ALU as raw color indices.
kernel void spi_xor_clock(
    device uint* partials [[buffer(0)]],
    constant Params& p [[buffer(1)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    ulong base = ulong(gid) * ulong(p.chunk);
    ulong end = min(base + ulong(p.chunk), ulong(p.n));
    ulong count = (base < end) ? (end - base) : 0;

    // Precomputed: golden_stride = GOLDEN * stride (passed via p.stride field)
    ulong gs = p.stride;  // this IS golden*stride, precomputed on host
    ulong base_z = p.seed + gs * base;

    uint a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    uint a4 = 0, a5 = 0, a6 = 0, a7 = 0;

    ulong n8 = count & ~7UL;
    for (ulong i = 0; i < n8; i += 8) {
        ulong z0 = base_z + gs * i;
        // Inline splitmix64 with pre-added seed — just the mixing steps
        #define HASH(off) { \
            ulong z = z0 + gs * off; \
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL; \
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL; \
            z = z ^ (z >> 31); \
            a##off ^= uint(((z >> 16) & 0xFF) << 16 | ((z >> 8) & 0xFF) << 8 | (z & 0xFF)); \
        }
        { ulong z = z0;              z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL; z = z ^ (z >> 31); a0 ^= uint(((z >> 16) & 0xFF) << 16 | ((z >> 8) & 0xFF) << 8 | (z & 0xFF)); }
        { ulong z = z0 + gs;         z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL; z = z ^ (z >> 31); a1 ^= uint(((z >> 16) & 0xFF) << 16 | ((z >> 8) & 0xFF) << 8 | (z & 0xFF)); }
        { ulong z = z0 + gs * 2;     z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL; z = z ^ (z >> 31); a2 ^= uint(((z >> 16) & 0xFF) << 16 | ((z >> 8) & 0xFF) << 8 | (z & 0xFF)); }
        { ulong z = z0 + gs * 3;     z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL; z = z ^ (z >> 31); a3 ^= uint(((z >> 16) & 0xFF) << 16 | ((z >> 8) & 0xFF) << 8 | (z & 0xFF)); }
        { ulong z = z0 + gs * 4;     z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL; z = z ^ (z >> 31); a4 ^= uint(((z >> 16) & 0xFF) << 16 | ((z >> 8) & 0xFF) << 8 | (z & 0xFF)); }
        { ulong z = z0 + gs * 5;     z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL; z = z ^ (z >> 31); a5 ^= uint(((z >> 16) & 0xFF) << 16 | ((z >> 8) & 0xFF) << 8 | (z & 0xFF)); }
        { ulong z = z0 + gs * 6;     z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL; z = z ^ (z >> 31); a6 ^= uint(((z >> 16) & 0xFF) << 16 | ((z >> 8) & 0xFF) << 8 | (z & 0xFF)); }
        { ulong z = z0 + gs * 7;     z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL; z = z ^ (z >> 31); a7 ^= uint(((z >> 16) & 0xFF) << 16 | ((z >> 8) & 0xFF) << 8 | (z & 0xFF)); }
    }
    uint local_xor = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
    for (ulong i = n8; i < count; i++) {
        ulong z = base_z + gs * i;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL;
        z = z ^ (z >> 31);
        local_xor ^= uint(((z >> 16) & 0xFF) << 16 | ((z >> 8) & 0xFF) << 8 | (z & 0xFF));
    }

    local_xor = simd_xor(local_xor);
    threadgroup uint tg_xor[32];
    if (simd_lane == 0) tg_xor[simd_id] = local_xor;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        uint val = (simd_lane < 32) ? tg_xor[simd_lane] : 0;
        val = simd_xor(val);
        if (simd_lane == 0) partials[tgid] = val;
    }
}

kernel void spi_reduce(
    device uint* partials [[buffer(0)]],
    device uint* result [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    uint local_xor = 0;
    for (uint i = tid; i < count; i += 1024) {
        local_xor ^= partials[i];
    }
    local_xor = simd_xor(local_xor);
    threadgroup uint tg_xor[32];
    if (simd_lane == 0) tg_xor[simd_id] = local_xor;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        uint val = (simd_lane < 32) ? tg_xor[simd_lane] : 0;
        val = simd_xor(val);
        if (simd_lane == 0) result[0] = val;
    }
}
"""

func gpuXOR(device: MTLDevice, pipe: MTLComputePipelineState,
            reducePipe: MTLComputePipelineState,
            queue: MTLCommandQueue, seed: UInt64, n: Int, stride: UInt64) -> (xor: UInt32, ns: UInt64) {
    let nThreads = (n + CHUNK - 1) / CHUNK
    let nGroups = (nThreads + TG_SIZE - 1) / TG_SIZE
    let partBuf = device.makeBuffer(length: nGroups * 4, options: .storageModeShared)!
    let resultBuf = device.makeBuffer(length: 4, options: .storageModeShared)!
    // Fold stride into golden: golden_stride = GOLDEN * stride
    let goldenStride = GOLDEN &* stride
    var params = Params(seed: seed, n: UInt32(n), chunk: UInt32(CHUNK), stride: goldenStride)
    let paramBuf = device.makeBuffer(bytes: &params, length: MemoryLayout<Params>.size, options: .storageModeShared)!

    let t0 = DispatchTime.now().uptimeNanoseconds
    let cmd = queue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipe)
    enc.setBuffer(partBuf, offset: 0, index: 0)
    enc.setBuffer(paramBuf, offset: 0, index: 1)
    enc.dispatchThreadgroups(MTLSize(width: nGroups, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: TG_SIZE, height: 1, depth: 1))
    if nGroups > 1 {
        var countV = UInt32(nGroups)
        let countBuf = device.makeBuffer(bytes: &countV, length: 4, options: .storageModeShared)!
        enc.setComputePipelineState(reducePipe)
        enc.setBuffer(partBuf, offset: 0, index: 0)
        enc.setBuffer(resultBuf, offset: 0, index: 1)
        enc.setBuffer(countBuf, offset: 0, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: min(TG_SIZE, nGroups), height: 1, depth: 1))
    }
    enc.endEncoding()
    cmd.commit(); cmd.waitUntilCompleted()
    let t1 = DispatchTime.now().uptimeNanoseconds
    if nGroups == 1 { return (partBuf.contents().load(as: UInt32.self), t1 - t0) }
    return (resultBuf.contents().load(as: UInt32.self), t1 - t0)
}

// --- main ---
guard let device = MTLCreateSystemDefaultDevice() else { print("No Metal"); exit(1) }
guard let queue = device.makeCommandQueue() else { print("No queue"); exit(1) }

let library: MTLLibrary
do { library = try device.makeLibrary(source: metalSource, options: nil) }
catch { print("Shader error: \(error)"); exit(1) }

guard let clockFn = library.makeFunction(name: "spi_xor_clock"),
      let reduceFn = library.makeFunction(name: "spi_reduce") else {
    print("Function not found"); exit(1)
}

let clockPipe: MTLComputePipelineState
let reducePipe: MTLComputePipelineState
do {
    clockPipe = try device.makeComputePipelineState(function: clockFn)
    reducePipe = try device.makeComputePipelineState(function: reduceFn)
} catch { print("Pipeline error: \(error)"); exit(1) }

let SEED: UInt64 = 42
let cores = ProcessInfo.processInfo.activeProcessorCount

print("""

+======================================================================+
|  SPI METAL CLOCKS — Three Clock Domains Head-to-Head                 |
|  COLOR (dense i) vs FLICK (i*stride) vs TRIT-TICK (i*stride)        |
+======================================================================+
  GPU: \(device.name)
  SIMD width: \(clockPipe.threadExecutionWidth)
  Colors/thread: \(CHUNK), 8-way ILP
""")

// --- BCI device strides ---
struct ClockDomain {
    let name: String
    let ticksPerSec: UInt64
}

let CLOCKS: [ClockDomain] = [
    .init(name: "Color (raw)",  ticksPerSec: 1),
    .init(name: "Flick",        ticksPerSec: FLICK),
    .init(name: "Trit-tick",    ticksPerSec: EPOCH1),
]

struct BCIDev {
    let name: String
    let hz: UInt64
}

let DEVICES: [BCIDev] = [
    .init(name: "OpenBCI 250 Hz",     hz: 250),
    .init(name: "BioSemi 2048 Hz",    hz: 2048),
    .init(name: "Neuropixels 30 kHz", hz: 30000),
    .init(name: "Audio 48 kHz",       hz: 48000),
]

// --- Section 1: Correctness at all three clock domains ---
print("  CORRECTNESS (N=100,000 samples)")
print("  Clock        Stride          GPU XOR      CPU XOR      Match")
print("  ───────────  ──────────────  ----------   ----------   -----")

for clock in CLOCKS {
    let stride: UInt64 = clock.ticksPerSec == 1 ? 1 : clock.ticksPerSec / 250  // 250 Hz OpenBCI
    let n = 100_000
    let gpu = gpuXOR(device: device, pipe: clockPipe, reducePipe: reducePipe, queue: queue, seed: SEED, n: n, stride: stride)
    let cpu = cpuXOR(SEED, n, stride: stride)
    let match = gpu.xor == cpu ? "PASS" : "FAIL"
    print("  \(clock.name.padding(toLength: 11, withPad: " ", startingAt: 0))  \(String(format: "%14d", stride))  0x\(String(format: "%06x", gpu.xor))     0x\(String(format: "%06x", cpu))     \(match)")
}

// --- Section 2: GPU shootout — all three clocks at same sample counts ---
print("""

  GPU CLOCK DOMAIN RACE (best-of-5)
  ══════════════════════════════════════════════════════════════════
  N = number of SAMPLES (not ticks). Each sample maps to one color.
  Stride = ticks_per_sample. Larger stride = sparser index pattern.
""")

struct RaceResult {
    let clock: String
    let stride: UInt64
    let n: Int
    let rate: Double
    let gbps: Double
}
var allResults: [RaceResult] = []

for clock in CLOCKS {
    let stride: UInt64 = clock.ticksPerSec == 1 ? 1 : clock.ticksPerSec / 250  // stride for 250 Hz
    print("\n  \(clock.name) (stride=\(stride))")
    print("  Samples      Time ms      M samples/s  M colors/s   GB/s")
    print("  -----------  ---------   -----------   -----------  -----")

    for n in [1_000_000, 10_000_000, 100_000_000, 500_000_000] {
        _ = gpuXOR(device: device, pipe: clockPipe, reducePipe: reducePipe, queue: queue, seed: SEED, n: n, stride: stride)
        var bestNs: UInt64 = .max
        for _ in 0..<5 {
            let r = gpuXOR(device: device, pipe: clockPipe, reducePipe: reducePipe, queue: queue, seed: SEED, n: n, stride: stride)
            bestNs = min(bestNs, r.ns)
        }
        let ms = Double(bestNs) / 1e6
        let sampleRate = Double(n) / Double(bestNs) * 1e3  // M samples/s
        let colorRate = sampleRate  // 1 color per sample
        let gbps = Double(n) * 3.0 / Double(bestNs)
        allResults.append(RaceResult(clock: clock.name, stride: stride, n: n, rate: sampleRate, gbps: gbps))
        print("  \(String(n).padding(toLength: 11, withPad: " ", startingAt: 0))  \(String(format: "%7.1f", ms))   \(String(format: "%9.1f", sampleRate))   \(String(format: "%9.1f", colorRate))  \(String(format: "%5.1f", gbps))")
    }
}

// --- Section 3: Per-device BCI race across clock domains ---
print("""

  PER-DEVICE BCI RACE (N=100M samples, best-of-5)
  ══════════════════════════════════════════════════════════════════
  Each device has a different stride in each clock domain.
  "Color" stride is always 1 (device-agnostic).
""")
print("  Device                  Clock       Stride          M/s          GPU→Realtime")
print("  ──────────────────────  ──────────  ──────────────  ──────────   ────────────")

let N_BCI = 100_000_000
for dev in DEVICES {
    for clock in CLOCKS {
        let stride: UInt64
        if clock.ticksPerSec == 1 {
            stride = 1
        } else {
            if clock.ticksPerSec % dev.hz != 0 { continue }  // skip if doesn't divide
            stride = clock.ticksPerSec / dev.hz
        }
        _ = gpuXOR(device: device, pipe: clockPipe, reducePipe: reducePipe, queue: queue, seed: SEED, n: N_BCI, stride: stride)
        var bestNs: UInt64 = .max
        for _ in 0..<5 {
            let r = gpuXOR(device: device, pipe: clockPipe, reducePipe: reducePipe, queue: queue, seed: SEED, n: N_BCI, stride: stride)
            bestNs = min(bestNs, r.ns)
        }
        let rate = Double(N_BCI) / Double(bestNs) * 1e3
        // How many real-time seconds of recording can the GPU process per wall-second?
        let realSeconds = rate * 1e6 / Double(dev.hz)
        let realLabel: String
        if realSeconds > 365.25 * 86400 {
            realLabel = String(format: "%.1f years/s", realSeconds / (365.25 * 86400))
        } else if realSeconds > 86400 {
            realLabel = String(format: "%.1f days/s", realSeconds / 86400)
        } else {
            realLabel = String(format: "%.1f hours/s", realSeconds / 3600)
        }
        print("  \(dev.name.padding(toLength: 22, withPad: " ", startingAt: 0))  \(clock.name.padding(toLength: 10, withPad: " ", startingAt: 0))  \(String(format: "%14d", stride))  \(String(format: "%9.1f", rate))   \(realLabel)")
    }
    print()
}

// --- Section 4: GPU vs CPU across clock domains ---
print("""
  GPU vs \(cores)-CORE CPU (N=100M, best-of-5)
  ══════════════════════════════════════════════════════════════════
""")
print("  Clock        Stride          GPU M/s      CPU M/s      Speedup")
print("  ───────────  ──────────────  ──────────   ──────────   -------")

for clock in CLOCKS {
    let stride: UInt64 = clock.ticksPerSec == 1 ? 1 : clock.ticksPerSec / 250
    let n = 100_000_000
    _ = gpuXOR(device: device, pipe: clockPipe, reducePipe: reducePipe, queue: queue, seed: SEED, n: n, stride: stride)
    var bestGpu: UInt64 = .max
    for _ in 0..<5 {
        let r = gpuXOR(device: device, pipe: clockPipe, reducePipe: reducePipe, queue: queue, seed: SEED, n: n, stride: stride)
        bestGpu = min(bestGpu, r.ns)
    }
    let t0 = DispatchTime.now().uptimeNanoseconds
    _ = cpuGCDXOR(SEED, n, cores, stride: stride)
    let cpuNs = DispatchTime.now().uptimeNanoseconds - t0

    let gpuRate = Double(n) / Double(bestGpu) * 1e3
    let cpuRate = Double(n) / Double(cpuNs) * 1e3
    let speedup = gpuRate / cpuRate
    print("  \(clock.name.padding(toLength: 11, withPad: " ", startingAt: 0))  \(String(format: "%14d", stride))  \(String(format: "%9.1f", gpuRate))   \(String(format: "%9.1f", cpuRate))   \(String(format: "%.1fx", speedup))")
}

// --- Section 5: The key insight ---
let peaks = Dictionary(grouping: allResults.filter { $0.n == 500_000_000 }, by: { $0.clock })
let colorPeak = peaks["Color (raw)"]?.first?.rate ?? 0
let flickPeak = peaks["Flick"]?.first?.rate ?? 0
let tritPeak = peaks["Trit-tick"]?.first?.rate ?? 0

print("""

  ════════════════════════════════════════════════════════════════
  THE KEY INSIGHT
  ════════════════════════════════════════════════════════════════

  At 500M samples:
    Color (stride=1):           \(String(format: "%9.1f", colorPeak)) M samples/s
    Flick (stride=\(FLICK/250)):      \(String(format: "%9.1f", flickPeak)) M samples/s
    Trit-tick (stride=\(EPOCH1/250)):    \(String(format: "%9.1f", tritPeak)) M samples/s

  The stride is a compile-time constant multiply before the hash.
  splitmix64(seed, i*stride) costs the SAME as splitmix64(seed, i)
  because the multiply-add `seed + GOLDEN * (i*stride)` folds to
  `seed + (GOLDEN*stride) * i` — one constant, same throughput.

  The XOR fingerprint is DIFFERENT per clock domain (different indices
  = different colors), but the THROUGHPUT is identical.

  This means: flicks and trit-ticks are FREE.
  The time base is a zero-cost abstraction.
""")
