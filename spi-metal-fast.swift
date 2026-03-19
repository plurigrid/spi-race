// SPI Metal Fast — Optimized GPU Compute Shader Racer
//
// Optimizations over spi-metal.swift:
//   1. Multi-color-per-thread (ILP): each thread XORs 16 colors before reduction
//   2. 1024-thread threadgroups: fewer CPU-side partial XORs
//   3. SIMD-group (warp) reduction: simd_xor before threadgroup tree
//   4. Two-pass reduction for massive N: first pass → partials, second pass → final
//
// swiftc -O -framework Metal spi-metal-fast.swift -o spi-metal-fast && ./spi-metal-fast

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
    let r = UInt32((val >> 16) & 0xFF) << 16
    let g = UInt32((val >> 8) & 0xFF) << 8
    let b = UInt32(val & 0xFF)
    return r | g | b
}

func cpuXOR(_ seed: UInt64, _ n: Int) -> UInt32 {
    var xor: UInt32 = 0
    for i in 0..<UInt64(n) { xor ^= extractRGB(sm64(seed, i)) }
    return xor
}

func cpuGCDXOR(_ seed: UInt64, _ n: Int, _ threads: Int) -> UInt32 {
    let chunk = n / threads
    let partials = UnsafeMutablePointer<UInt32>.allocate(capacity: threads)
    defer { partials.deallocate() }
    partials.initialize(repeating: 0, count: threads)
    DispatchQueue.concurrentPerform(iterations: threads) { tid in
        let start = tid * chunk
        let end = tid == threads - 1 ? n : (tid + 1) * chunk
        var a0: UInt32 = 0, a1: UInt32 = 0, a2: UInt32 = 0, a3: UInt32 = 0
        var b0: UInt32 = 0, b1: UInt32 = 0, b2: UInt32 = 0, b3: UInt32 = 0
        let count = end - start
        let n8 = count & ~7
        var j = 0
        while j < n8 {
            let idx = UInt64(start + j)
            a0 ^= extractRGB(sm64(seed, idx))
            a1 ^= extractRGB(sm64(seed, idx &+ 1))
            a2 ^= extractRGB(sm64(seed, idx &+ 2))
            a3 ^= extractRGB(sm64(seed, idx &+ 3))
            b0 ^= extractRGB(sm64(seed, idx &+ 4))
            b1 ^= extractRGB(sm64(seed, idx &+ 5))
            b2 ^= extractRGB(sm64(seed, idx &+ 6))
            b3 ^= extractRGB(sm64(seed, idx &+ 7))
            j += 8
        }
        var local = a0 ^ a1 ^ a2 ^ a3 ^ b0 ^ b1 ^ b2 ^ b3
        while j < count { local ^= extractRGB(sm64(seed, UInt64(start + j))); j += 1 }
        partials[tid] = local
    }
    var combined: UInt32 = 0
    for i in 0..<threads { combined ^= partials[i] }
    return combined
}

// Each thread processes CHUNK colors, using 4 accumulators for ILP.
// Threadgroup = 1024 threads. SIMD-group reduction first, then threadgroup tree.
let CHUNK = 16
let TG_SIZE = 1024

let metalSource = """
#include <metal_stdlib>
using namespace metal;

static inline ulong splitmix64(ulong seed, ulong idx) {
    ulong z = seed + 0x9e3779b97f4a7c15UL * idx;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL;
    return z ^ (z >> 31);
}

static inline uint extractRGB(ulong val) {
    return uint(((val >> 16) & 0xFF) << 16 | ((val >> 8) & 0xFF) << 8 | (val & 0xFF));
}

// Optimized kernel: each thread processes CHUNK colors with 4 accumulators
kernel void spi_xor_fast(
    device uint* partials [[buffer(0)]],
    constant ulong& seed [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& chunk [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    // Phase 1: each thread XORs CHUNK colors with 4-way ILP
    uint a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    ulong base = ulong(gid) * ulong(chunk);
    uint local_chunk = min(chunk, n - min(uint(base), n));

    for (uint i = 0; i < local_chunk; i += 4) {
        ulong idx = base + ulong(i);
        if (idx < ulong(n))     a0 ^= extractRGB(splitmix64(seed, idx));
        if (idx + 1 < ulong(n)) a1 ^= extractRGB(splitmix64(seed, idx + 1));
        if (idx + 2 < ulong(n)) a2 ^= extractRGB(splitmix64(seed, idx + 2));
        if (idx + 3 < ulong(n)) a3 ^= extractRGB(splitmix64(seed, idx + 3));
    }
    uint local_xor = a0 ^ a1 ^ a2 ^ a3;

    // Phase 2: SIMD-group (warp) reduction — no threadgroup memory needed
    local_xor = simd_xor(local_xor);

    // Phase 3: threadgroup reduction via shared memory (one value per SIMD group)
    threadgroup uint tg_xor[32]; // max 32 SIMD groups per threadgroup (1024/32)
    if (simd_lane == 0) tg_xor[simd_id] = local_xor;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final tree reduction across SIMD groups (only first SIMD group participates)
    if (simd_id == 0) {
        uint val = (simd_lane < 32) ? tg_xor[simd_lane] : 0;
        val = simd_xor(val);
        if (simd_lane == 0) partials[tgid] = val;
    }
}

// Two-pass GPU reduction kernel: reduces partials array to single value
kernel void spi_reduce_partials(
    device uint* partials [[buffer(0)]],
    device uint* result [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    // Each thread accumulates a stride of the partials array
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

// --- GPU dispatch ---

func gpuXORFast(device: MTLDevice, pipe: MTLComputePipelineState,
                reducePipe: MTLComputePipelineState,
                queue: MTLCommandQueue, seed: UInt64, n: Int) -> (xor: UInt32, ns: UInt64) {
    let chunk = CHUNK
    let nThreads = (n + chunk - 1) / chunk
    let nGroups = (nThreads + TG_SIZE - 1) / TG_SIZE
    let partBuf = device.makeBuffer(length: nGroups * 4, options: .storageModeShared)!
    let resultBuf = device.makeBuffer(length: 4, options: .storageModeShared)!
    var s = seed; var nv = UInt32(n); var cv = UInt32(chunk)
    let seedBuf = device.makeBuffer(bytes: &s, length: 8, options: .storageModeShared)!
    let nBuf = device.makeBuffer(bytes: &nv, length: 4, options: .storageModeShared)!
    let chunkBuf = device.makeBuffer(bytes: &cv, length: 4, options: .storageModeShared)!

    let t0 = DispatchTime.now().uptimeNanoseconds

    // Pass 1: compute + reduce to partials
    let cmd = queue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipe)
    enc.setBuffer(partBuf, offset: 0, index: 0)
    enc.setBuffer(seedBuf, offset: 0, index: 1)
    enc.setBuffer(nBuf, offset: 0, index: 2)
    enc.setBuffer(chunkBuf, offset: 0, index: 3)
    enc.dispatchThreadgroups(MTLSize(width: nGroups, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: TG_SIZE, height: 1, depth: 1))

    // Pass 2: reduce partials to single value (if many groups)
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

    if nGroups == 1 {
        return (partBuf.contents().load(as: UInt32.self), t1 - t0)
    }
    return (resultBuf.contents().load(as: UInt32.self), t1 - t0)
}

// --- main ---

guard let device = MTLCreateSystemDefaultDevice() else { print("No Metal"); exit(1) }
guard let queue = device.makeCommandQueue() else { print("No queue"); exit(1) }

let library: MTLLibrary
do { library = try device.makeLibrary(source: metalSource, options: nil) }
catch { print("Shader error: \(error)"); exit(1) }

guard let fastFn = library.makeFunction(name: "spi_xor_fast"),
      let reduceFn = library.makeFunction(name: "spi_reduce_partials") else {
    print("Function not found"); exit(1)
}

let fastPipe: MTLComputePipelineState
let reducePipe: MTLComputePipelineState
do {
    fastPipe = try device.makeComputePipelineState(function: fastFn)
    reducePipe = try device.makeComputePipelineState(function: reduceFn)
} catch { print("Pipeline error: \(error)"); exit(1) }

let SEED: UInt64 = 42
let cores = ProcessInfo.processInfo.activeProcessorCount

print("""

+======================================================================+
|  SPI METAL FAST — Optimized GPU Compute Shader                       |
|  16 colors/thread, SIMD warp reduction, 1024-thread threadgroups     |
+======================================================================+
  GPU: \(device.name)
  Max threadgroup: \(fastPipe.maxTotalThreadsPerThreadgroup)
  SIMD width: \(fastPipe.threadExecutionWidth)
  Colors/thread: \(CHUNK)
  Effective colors/threadgroup: \(CHUNK * TG_SIZE)
""")

// --- Section 1: Correctness at every scale ---
print("  CORRECTNESS CHECK")
print("  N            GPU XOR      CPU XOR      Match")
print("  ----------   ----------   ----------   -----")

for n in [1000, 10_000, 100_000, 1_000_000, 10_000_000] {
    let gpu = gpuXORFast(device: device, pipe: fastPipe, reducePipe: reducePipe, queue: queue, seed: SEED, n: n)
    let cpu = cpuXOR(SEED, n)
    let match = gpu.xor == cpu ? "PASS" : "FAIL (gpu=0x\(String(gpu.xor, radix:16)) cpu=0x\(String(cpu, radix:16)))"
    print("  \(String(n).padding(toLength: 10, withPad: " ", startingAt: 0))   0x\(String(format: "%06x", gpu.xor))     0x\(String(format: "%06x", cpu))     \(match)")
}

// --- Section 2: GPU scaling ---
print("""

  GPU SCALING: OPTIMIZED vs BASELINE
  Baseline: 1 color/thread, 256-threadgroup, CPU-side partial XOR
  Optimized: \(CHUNK) colors/thread, \(TG_SIZE)-threadgroup, SIMD+tree reduction, GPU-side final
""")
print("  N            Time ms      M/s        GB/s       Threadgroups")
print("  -----------  ---------   ---------   --------   ------------")

var peakRate = 0.0
var peakN = 0

for n in [1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000, 500_000_000, 1_000_000_000, 2_000_000_000] {
    // Warmup
    _ = gpuXORFast(device: device, pipe: fastPipe, reducePipe: reducePipe, queue: queue, seed: SEED, n: 1000)
    // Run 3 times, take best
    var bestNs: UInt64 = .max
    var bestXor: UInt32 = 0
    for _ in 0..<3 {
        let r = gpuXORFast(device: device, pipe: fastPipe, reducePipe: reducePipe, queue: queue, seed: SEED, n: n)
        if r.ns < bestNs { bestNs = r.ns; bestXor = r.xor }
    }
    let ms = Double(bestNs) / 1e6
    let rate = Double(n) / Double(bestNs) * 1000
    let gbps = Double(n) * 3.0 / Double(bestNs)
    let nThreads = (n + CHUNK - 1) / CHUNK
    let nGroups = (nThreads + TG_SIZE - 1) / TG_SIZE
    if rate > peakRate { peakRate = rate; peakN = n }
    let label: String
    if n >= 1_000_000_000 { label = "\(n / 1_000_000_000)B" }
    else if n >= 1_000_000 { label = "\(n / 1_000_000)M" }
    else { label = "\(n)" }
    print("  \(label.padding(toLength: 11, withPad: " ", startingAt: 0))  \(String(format: "%8.2f", ms))   \(String(format: "%6.0f M/s", rate))   \(String(format: "%6.2f", gbps))   \(String(format: "%12d", nGroups))")
}

// --- Section 3: GPU vs multi-core CPU ---
print("""

  GPU vs \(cores)-CORE CPU (GCD)
""")
print("  N            GPU ms     GPU M/s    GCD ms     GCD M/s    Speedup   Match")
print("  ----------   --------   --------   --------   --------   -------   -----")

for n in [10_000_000, 100_000_000, 500_000_000, 1_000_000_000] {
    _ = gpuXORFast(device: device, pipe: fastPipe, reducePipe: reducePipe, queue: queue, seed: SEED, n: 1000)
    var bestGpu: (xor: UInt32, ns: UInt64) = (0, .max)
    for _ in 0..<3 {
        let r = gpuXORFast(device: device, pipe: fastPipe, reducePipe: reducePipe, queue: queue, seed: SEED, n: n)
        if r.ns < bestGpu.ns { bestGpu = r }
    }
    _ = cpuGCDXOR(SEED, 1_000, cores)
    let t0 = DispatchTime.now().uptimeNanoseconds
    let cpuRef = cpuGCDXOR(SEED, n, cores)
    let cpuNs = DispatchTime.now().uptimeNanoseconds - t0
    let gpuMs = Double(bestGpu.ns) / 1e6
    let cpuMs = Double(cpuNs) / 1e6
    let gpuRate = Int(Double(n) / Double(bestGpu.ns) * 1000)
    let cpuRate = Int(Double(n) / Double(cpuNs) * 1000)
    let speedup = cpuMs > 0 && gpuMs > 0 ? cpuMs / gpuMs : 0.0
    let match = bestGpu.xor == cpuRef ? "PASS" : "FAIL"
    let label = n >= 1_000_000_000 ? "\(n / 1_000_000_000)B" : "\(n / 1_000_000)M"
    print("  \(label.padding(toLength: 10, withPad: " ", startingAt: 0))   \(String(format: "%7.2f", gpuMs))   \(String(format: "%5d M/s", gpuRate))   \(String(format: "%7.2f", cpuMs))   \(String(format: "%5d M/s", cpuRate))   \(String(format: "%5.2f", speedup))x   \(match)")
}

// --- Section 4: Peak bandwidth ---
print("""

  PEAK BANDWIDTH
""")
let peakLabel = peakN >= 1_000_000_000 ? "\(peakN / 1_000_000_000)B" : "\(peakN / 1_000_000)M"
let peakGBps = peakRate * 3.0 / 1000.0
let peakGbits = peakGBps * 8.0
print("  Peak at \(peakLabel): \(String(format: "%.0f", peakRate)) M colors/s")
print("  Bandwidth:  \(String(format: "%.2f", peakGBps)) GB/s (RGB bytes)")
print("  Bitrate:    \(String(format: "%.1f", peakGbits)) Gbit/s")
print()
print("  Optimizations applied:")
print("    1. \(CHUNK) colors/thread — 4-way ILP in ALU before any reduction")
print("    2. simd_xor — warp-level reduction (no shared memory)")
print("    3. \(TG_SIZE)-thread threadgroups — 4x fewer partials than 256")
print("    4. Two-pass GPU reduction — partials reduced on GPU, not CPU")
print("    5. Best-of-3 timing — eliminates launch jitter")
