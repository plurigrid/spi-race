// SPI Metal Vec — Vectorized GPU racer
//
// Strategy: use Metal's ulong2/uint4 SIMD types to process 2 hashes simultaneously
// per instruction, plus threadgroup_sum for the reduction.
// Also tests: register-pressure tuning, occupancy-aware dispatch.
//
// swiftc -O -framework Metal spi-metal-vec.swift -o spi-metal-vec && ./spi-metal-vec

import Foundation
import Metal

let GOLDEN: UInt64 = 0x9e3779b97f4a7c15
let MIX1: UInt64 = 0xbf58476d1ce4e5b9
let MIX2: UInt64 = 0x94d049bb133111eb

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

func cpuXOR(_ seed: UInt64, _ n: Int) -> UInt32 {
    var xor: UInt32 = 0
    for i in 0..<UInt64(n) { xor ^= extractRGB(sm64(seed, i)) }
    return xor
}

struct Params {
    var seed: UInt64
    var n: UInt32
    var chunk: UInt32
}

// Strategy A: ulong2 vectorized — process 2 splitmix64 in parallel per instruction
let metalSourceVec = """
#include <metal_stdlib>
using namespace metal;

struct Params {
    ulong seed;
    uint  n;
    uint  chunk;
};

static inline ulong2 splitmix64_vec2(ulong seed, ulong idx) {
    ulong2 z = ulong2(seed + 0x9e3779b97f4a7c15UL * idx,
                       seed + 0x9e3779b97f4a7c15UL * (idx + 1));
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL;
    return z ^ (z >> 31);
}

static inline uint2 extractRGB_vec2(ulong2 val) {
    return uint2(uint(((val.x >> 16) & 0xFF) << 16 | ((val.x >> 8) & 0xFF) << 8 | (val.x & 0xFF)),
                 uint(((val.y >> 16) & 0xFF) << 16 | ((val.y >> 8) & 0xFF) << 8 | (val.y & 0xFF)));
}

// Vec2 kernel: 2 hashes per SIMD lane, 8-way ILP = 16 hashes/iteration
kernel void spi_xor_vec2(
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

    uint2 a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    uint2 a4 = 0, a5 = 0, a6 = 0, a7 = 0;

    ulong n16 = (count / 16) * 16;
    for (ulong i = 0; i < n16; i += 16) {
        ulong idx = base + i;
        a0 ^= extractRGB_vec2(splitmix64_vec2(p.seed, idx));
        a1 ^= extractRGB_vec2(splitmix64_vec2(p.seed, idx + 2));
        a2 ^= extractRGB_vec2(splitmix64_vec2(p.seed, idx + 4));
        a3 ^= extractRGB_vec2(splitmix64_vec2(p.seed, idx + 6));
        a4 ^= extractRGB_vec2(splitmix64_vec2(p.seed, idx + 8));
        a5 ^= extractRGB_vec2(splitmix64_vec2(p.seed, idx + 10));
        a6 ^= extractRGB_vec2(splitmix64_vec2(p.seed, idx + 12));
        a7 ^= extractRGB_vec2(splitmix64_vec2(p.seed, idx + 14));
    }
    uint local_xor = (a0.x ^ a0.y) ^ (a1.x ^ a1.y) ^ (a2.x ^ a2.y) ^ (a3.x ^ a3.y)
                   ^ (a4.x ^ a4.y) ^ (a5.x ^ a5.y) ^ (a6.x ^ a6.y) ^ (a7.x ^ a7.y);
    // Tail
    for (ulong i = n16; i < count; i++) {
        ulong z = p.seed + 0x9e3779b97f4a7c15UL * (base + i);
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

// Strategy B: Scalar but 128 colors/thread with minimal registers (4 accumulators only)
static inline ulong splitmix64(ulong seed, ulong idx) {
    ulong z = seed + 0x9e3779b97f4a7c15UL * idx;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL;
    return z ^ (z >> 31);
}

static inline uint extractRGB(ulong val) {
    return uint(((val >> 16) & 0xFF) << 16 | ((val >> 8) & 0xFF) << 8 | (val & 0xFF));
}

static inline uint sm64_rgb(ulong seed, ulong idx) {
    return extractRGB(splitmix64(seed, idx));
}

kernel void spi_xor_wide(
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

    uint a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    ulong n4 = (count / 4) * 4;
    for (ulong i = 0; i < n4; i += 4) {
        ulong idx = base + i;
        a0 ^= sm64_rgb(p.seed, idx);
        a1 ^= sm64_rgb(p.seed, idx + 1);
        a2 ^= sm64_rgb(p.seed, idx + 2);
        a3 ^= sm64_rgb(p.seed, idx + 3);
    }
    uint local_xor = a0 ^ a1 ^ a2 ^ a3;
    for (ulong i = n4; i < count; i++) {
        local_xor ^= sm64_rgb(p.seed, base + i);
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

// Reduction kernel (shared)
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

let TG_SIZE = 1024

func gpuXOR(device: MTLDevice, pipe: MTLComputePipelineState,
            reducePipe: MTLComputePipelineState,
            queue: MTLCommandQueue, seed: UInt64, n: Int, chunk: Int) -> (xor: UInt32, ns: UInt64) {
    let nThreads = (n + chunk - 1) / chunk
    let nGroups = (nThreads + TG_SIZE - 1) / TG_SIZE
    let partBuf = device.makeBuffer(length: nGroups * 4, options: .storageModeShared)!
    let resultBuf = device.makeBuffer(length: 4, options: .storageModeShared)!
    var params = Params(seed: seed, n: UInt32(n), chunk: UInt32(chunk))
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
do { library = try device.makeLibrary(source: metalSourceVec, options: nil) }
catch { print("Shader error: \(error)"); exit(1) }

guard let vec2Fn = library.makeFunction(name: "spi_xor_vec2"),
      let wideFn = library.makeFunction(name: "spi_xor_wide"),
      let reduceFn = library.makeFunction(name: "spi_reduce") else {
    print("Function not found"); exit(1)
}

let vec2Pipe: MTLComputePipelineState
let widePipe: MTLComputePipelineState
let reducePipe: MTLComputePipelineState
do {
    vec2Pipe = try device.makeComputePipelineState(function: vec2Fn)
    widePipe = try device.makeComputePipelineState(function: wideFn)
    reducePipe = try device.makeComputePipelineState(function: reduceFn)
} catch { print("Pipeline error: \(error)"); exit(1) }

let SEED: UInt64 = 42

print("""

+======================================================================+
|  SPI METAL VEC — Strategy Shootout                                   |
|  A: ulong2 vectorized (2 hashes/lane)  B: scalar-wide (128/thread)  |
+======================================================================+
  GPU: \(device.name)
  SIMD width: \(vec2Pipe.threadExecutionWidth)
""")

// --- Correctness ---
print("  CORRECTNESS")
let cpuRef = cpuXOR(SEED, 10_000_000)
let strategies: [(String, MTLComputePipelineState, Int)] = [
    ("Vec2 (64/thr)", vec2Pipe, 64),
    ("Vec2 (128/thr)", vec2Pipe, 128),
    ("Wide-4 (128/thr)", widePipe, 128),
    ("Wide-4 (256/thr)", widePipe, 256),
]
for (name, pipe, chunk) in strategies {
    let r = gpuXOR(device: device, pipe: pipe, reducePipe: reducePipe, queue: queue, seed: SEED, n: 10_000_000, chunk: chunk)
    let match = r.xor == cpuRef ? "PASS" : "FAIL"
    print("  \(name.padding(toLength: 20, withPad: " ", startingAt: 0)) 0x\(String(format: "%06x", r.xor)) \(match)")
}

// --- Benchmark: all strategies at 500M ---
print("""

  STRATEGY SHOOTOUT at N=500,000,000 (best-of-5)
  ──────────────────────────────────────────────────────────
  Strategy             Chunk   M/s          GB/s
  ───────────────────  ─────   ──────────   ─────
""")

struct Result { let name: String; let rate: Double; let gbps: Double }
var results: [Result] = []

let benchStrategies: [(String, MTLComputePipelineState, Int)] = [
    ("Vec2", vec2Pipe, 32),
    ("Vec2", vec2Pipe, 64),
    ("Vec2", vec2Pipe, 128),
    ("Vec2", vec2Pipe, 256),
    ("Wide-4", widePipe, 64),
    ("Wide-4", widePipe, 128),
    ("Wide-4", widePipe, 256),
    ("Wide-4", widePipe, 512),
]

let N = 500_000_000
for (name, pipe, chunk) in benchStrategies {
    _ = gpuXOR(device: device, pipe: pipe, reducePipe: reducePipe, queue: queue, seed: SEED, n: N, chunk: chunk)
    var bestNs: UInt64 = .max
    for _ in 0..<5 {
        let r = gpuXOR(device: device, pipe: pipe, reducePipe: reducePipe, queue: queue, seed: SEED, n: N, chunk: chunk)
        bestNs = min(bestNs, r.ns)
    }
    let rate = Double(N) / Double(bestNs) * 1e3
    let gbps = Double(N) * 3.0 / Double(bestNs)
    let label = "\(name) (\(chunk)/thr)"
    results.append(Result(name: label, rate: rate, gbps: gbps))
    print("  \(label.padding(toLength: 19, withPad: " ", startingAt: 0))  \(String(format: "%5d", chunk))   \(String(format: "%9.1f", rate))   \(String(format: "%5.1f", gbps))")
}

let best = results.max(by: { $0.rate < $1.rate })!
print("""

  WINNER: \(best.name) at \(String(format: "%.1f", best.rate/1000)) B colors/s
  (Ultra baseline: ~94 B/s with scalar 64/thread 8-way ILP)
""")

// Scale test for winner strategy
print("  SCALING: best strategy across N")
print("  N            M/s          GB/s")
print("  -----------  -----------  -----")

let (bestPipe, bestChunk): (MTLComputePipelineState, Int) = {
    if best.name.contains("Vec2") {
        for (_, pipe, chunk) in benchStrategies where best.name.contains("\(chunk)") && best.name.contains("Vec2") {
            return (pipe, chunk)
        }
    }
    for (_, pipe, chunk) in benchStrategies where best.name.contains("\(chunk)") && best.name.contains("Wide") {
        return (pipe, chunk)
    }
    return (vec2Pipe, 64)
}()

var finalPeak = 0.0
for n in [100_000_000, 200_000_000, 300_000_000, 500_000_000, 700_000_000, 1_000_000_000] {
    _ = gpuXOR(device: device, pipe: bestPipe, reducePipe: reducePipe, queue: queue, seed: SEED, n: n, chunk: bestChunk)
    var bestNs: UInt64 = .max
    for _ in 0..<5 {
        let r = gpuXOR(device: device, pipe: bestPipe, reducePipe: reducePipe, queue: queue, seed: SEED, n: n, chunk: bestChunk)
        bestNs = min(bestNs, r.ns)
    }
    let rate = Double(n) / Double(bestNs) * 1e3
    let gbps = Double(n) * 3.0 / Double(bestNs)
    if rate > finalPeak { finalPeak = rate }
    print("  \(String(n).padding(toLength: 11, withPad: " ", startingAt: 0))  \(String(format: "%9.1f", rate))   \(String(format: "%5.1f", gbps))")
}

print("""

  ====================================================================
  PEAK: \(String(format: "%.1f", finalPeak/1000)) B colors/s (\(best.name))
  vs Ultra (scalar 8-ILP): \(String(format: "%.2fx", finalPeak / 94000.0))
  ====================================================================
""")
