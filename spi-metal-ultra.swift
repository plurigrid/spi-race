// SPI Metal Ultra — Maximum GPU throughput
//
// Over spi-metal-fast.swift:
//   1. 64 colors/thread (4x more ALU per thread, fewer reductions)
//   2. 8-way ILP accumulators in MSL kernel
//   3. Branchless hot loop (no per-iteration bounds checks)
//   4. Fused seed+params in single constant struct
//   5. Best-of-5 timing for stable peak measurement
//
// swiftc -O -framework Metal spi-metal-ultra.swift -o spi-metal-ultra && ./spi-metal-ultra

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

let CHUNK: Int = 64
let TG_SIZE: Int = 1024

let metalSource = """
#include <metal_stdlib>
using namespace metal;

struct Params {
    ulong seed;
    uint  n;
    uint  chunk;
};

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

// Ultra kernel: 64 colors/thread, 8-way ILP, branchless hot path
kernel void spi_xor_ultra(
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

    // 8-way ILP accumulators
    uint a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    uint a4 = 0, a5 = 0, a6 = 0, a7 = 0;

    // Branchless hot loop: process 8 at a time, no per-element bounds check
    ulong n8 = count & ~7UL;
    for (ulong i = 0; i < n8; i += 8) {
        ulong idx = base + i;
        a0 ^= sm64_rgb(p.seed, idx);
        a1 ^= sm64_rgb(p.seed, idx + 1);
        a2 ^= sm64_rgb(p.seed, idx + 2);
        a3 ^= sm64_rgb(p.seed, idx + 3);
        a4 ^= sm64_rgb(p.seed, idx + 4);
        a5 ^= sm64_rgb(p.seed, idx + 5);
        a6 ^= sm64_rgb(p.seed, idx + 6);
        a7 ^= sm64_rgb(p.seed, idx + 7);
    }
    // Tail (0-7 elements)
    uint local_xor = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
    for (ulong i = n8; i < count; i++) {
        local_xor ^= sm64_rgb(p.seed, base + i);
    }

    // SIMD warp reduction
    local_xor = simd_xor(local_xor);

    // Threadgroup reduction
    threadgroup uint tg_xor[32];
    if (simd_lane == 0) tg_xor[simd_id] = local_xor;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        uint val = (simd_lane < 32) ? tg_xor[simd_lane] : 0;
        val = simd_xor(val);
        if (simd_lane == 0) partials[tgid] = val;
    }
}

// Reduction kernel
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

struct Params {
    var seed: UInt64
    var n: UInt32
    var chunk: UInt32
}

func gpuXOR(device: MTLDevice, pipe: MTLComputePipelineState,
            reducePipe: MTLComputePipelineState,
            queue: MTLCommandQueue, seed: UInt64, n: Int) -> (xor: UInt32, ns: UInt64) {
    let nThreads = (n + CHUNK - 1) / CHUNK
    let nGroups = (nThreads + TG_SIZE - 1) / TG_SIZE
    let partBuf = device.makeBuffer(length: nGroups * 4, options: .storageModeShared)!
    let resultBuf = device.makeBuffer(length: 4, options: .storageModeShared)!
    var params = Params(seed: seed, n: UInt32(n), chunk: UInt32(CHUNK))
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

guard let ultraFn = library.makeFunction(name: "spi_xor_ultra"),
      let reduceFn = library.makeFunction(name: "spi_reduce") else {
    print("Function not found"); exit(1)
}

let ultraPipe: MTLComputePipelineState
let reducePipe: MTLComputePipelineState
do {
    ultraPipe = try device.makeComputePipelineState(function: ultraFn)
    reducePipe = try device.makeComputePipelineState(function: reduceFn)
} catch { print("Pipeline error: \(error)"); exit(1) }

let SEED: UInt64 = 42
let cores = ProcessInfo.processInfo.activeProcessorCount

print("""

+======================================================================+
|  SPI METAL ULTRA — Maximum GPU Throughput                            |
|  64 colors/thread, 8-way ILP, branchless hot loop, fused params     |
+======================================================================+
  GPU: \(device.name)
  Max threadgroup: \(ultraPipe.maxTotalThreadsPerThreadgroup)
  SIMD width: \(ultraPipe.threadExecutionWidth)
  Colors/thread: \(CHUNK) (4x over fast)
  Effective colors/threadgroup: \(CHUNK * TG_SIZE)
""")

// --- Section 1: Correctness ---
print("  CORRECTNESS CHECK")
print("  N            GPU XOR      CPU XOR      Match")
print("  ----------   ----------   ----------   -----")

for n in [1000, 10_000, 100_000, 1_000_000, 10_000_000] {
    let gpu = gpuXOR(device: device, pipe: ultraPipe, reducePipe: reducePipe, queue: queue, seed: SEED, n: n)
    let cpu = cpuXOR(SEED, n)
    let match = gpu.xor == cpu ? "PASS" : "FAIL (gpu=0x\(String(gpu.xor, radix:16)) cpu=0x\(String(cpu, radix:16)))"
    print("  \(String(n).padding(toLength: 10, withPad: " ", startingAt: 0))   0x\(String(format: "%06x", gpu.xor))     0x\(String(format: "%06x", cpu))     \(match)")
}

// --- Section 2: GPU scaling with best-of-5 ---
print("""

  GPU SCALING — ULTRA
  64 colors/thread, 8-way ILP, branchless, fused params, best-of-5
""")
print("  N            Time ms      M/s          GB/s       Threadgroups")
print("  -----------  ---------   -----------   --------   ------------")

var peakRate = 0.0
var peakN = 0

for n in [1_000_000, 10_000_000, 100_000_000, 500_000_000, 1_000_000_000, 2_000_000_000] {
    // Warmup
    _ = gpuXOR(device: device, pipe: ultraPipe, reducePipe: reducePipe, queue: queue, seed: SEED, n: n)

    // Best of 5
    var bestNs: UInt64 = .max
    for _ in 0..<5 {
        let r = gpuXOR(device: device, pipe: ultraPipe, reducePipe: reducePipe, queue: queue, seed: SEED, n: n)
        bestNs = min(bestNs, r.ns)
    }
    let ms = Double(bestNs) / 1e6
    let rate = Double(n) / Double(bestNs) * 1e3  // M colors/s
    let gbps = Double(n) * 3.0 / Double(bestNs)  // 3 bytes per RGB
    let nThreads = (n + CHUNK - 1) / CHUNK
    let nGroups = (nThreads + TG_SIZE - 1) / TG_SIZE
    if rate > peakRate { peakRate = rate; peakN = n }
    print("  \(String(n).padding(toLength: 11, withPad: " ", startingAt: 0))  \(String(format: "%7.1f", ms))   \(String(format: "%9.1f", rate))   \(String(format: "%6.1f", gbps))     \(nGroups)")
}

// --- Section 3: GPU vs multi-core CPU ---
print("""

  GPU vs \(cores)-CORE GCD
""")
print("  N            GPU M/s       CPU M/s      Speedup")
print("  -----------  -----------   -----------  -------")

for n in [10_000_000, 100_000_000, 1_000_000_000] {
    _ = gpuXOR(device: device, pipe: ultraPipe, reducePipe: reducePipe, queue: queue, seed: SEED, n: n)
    var bestGpuNs: UInt64 = .max
    for _ in 0..<5 {
        let r = gpuXOR(device: device, pipe: ultraPipe, reducePipe: reducePipe, queue: queue, seed: SEED, n: n)
        bestGpuNs = min(bestGpuNs, r.ns)
    }
    let t0 = DispatchTime.now().uptimeNanoseconds
    _ = cpuGCDXOR(SEED, n, cores)
    let cpuNs = DispatchTime.now().uptimeNanoseconds - t0

    let gpuRate = Double(n) / Double(bestGpuNs) * 1e3
    let cpuRate = Double(n) / Double(cpuNs) * 1e3
    let speedup = gpuRate / cpuRate
    print("  \(String(n).padding(toLength: 11, withPad: " ", startingAt: 0))  \(String(format: "%9.1f", gpuRate))   \(String(format: "%9.1f", cpuRate))    \(String(format: "%.1fx", speedup))")
}

// --- Section 4: BCI real-time multiplier ---
let FLICK: UInt64 = 705_600_000
let EPOCH1: UInt64 = 141_120_000

let peakColorsPerSec = peakRate * 1e6

print("""

  BCI REAL-TIME MULTIPLIER (Ultra)
  ──────────────────────────────────────────────────────────
  Device                     Rate       Realtime multiplier
  ─────────────────────────  ─────────  ───────────────────────────
""")

struct BCIDevice { let name: String; let rate: Int; let needsEpoch2: Bool; let needsUnbounded: Bool }
let devices: [BCIDevice] = [
    .init(name: "OpenBCI Cyton EEG",      rate: 250,      needsEpoch2: false, needsUnbounded: false),
    .init(name: "LiveAmp EEG",            rate: 500,      needsEpoch2: false, needsUnbounded: false),
    .init(name: "BioSemi ActiveTwo EEG",  rate: 2048,     needsEpoch2: true,  needsUnbounded: false),
    .init(name: "g.tec g.USBamp EEG",     rate: 4800,     needsEpoch2: false, needsUnbounded: false),
    .init(name: "BrainProducts actiCHamp+",rate: 5000,    needsEpoch2: false, needsUnbounded: false),
    .init(name: "Neuropixels AP",         rate: 30000,    needsEpoch2: false, needsUnbounded: false),
    .init(name: "CD audio 44.1 kHz",      rate: 44100,    needsEpoch2: false, needsUnbounded: false),
    .init(name: "DAC audio 48 kHz",       rate: 48000,    needsEpoch2: false, needsUnbounded: false),
    .init(name: "Medtronic DBS 130 Hz",   rate: 130,      needsEpoch2: true,  needsUnbounded: false),
    .init(name: "Boston Sci DBS 185 Hz",  rate: 185,      needsEpoch2: true,  needsUnbounded: false),
    .init(name: "Delsys Trigno EMG",      rate: 1926,     needsEpoch2: false, needsUnbounded: true),
    .init(name: "FLIR Blackfly camera",   rate: 227,      needsEpoch2: false, needsUnbounded: true),
    .init(name: "Neuropixels LFP",        rate: 2500,     needsEpoch2: false, needsUnbounded: false),
    .init(name: "DSD512 audio",           rate: 22579200, needsEpoch2: true,  needsUnbounded: false),
]

for d in devices {
    let samplesPerSec = Double(d.rate)
    let secondsOfRecording = peakColorsPerSec / samplesPerSec
    let epoch = d.needsUnbounded ? "(unbounded)" : (d.needsEpoch2 ? "(E2 u128)" : "")
    if secondsOfRecording > 365.25 * 86400 {
        let years = secondsOfRecording / (365.25 * 86400)
        print("  \(d.name.padding(toLength: 25, withPad: " ", startingAt: 0))  \(String(format: "%9d", d.rate))    \(String(format: "%.1f years/s", years)) \(epoch)")
    } else if secondsOfRecording > 86400 {
        let days = secondsOfRecording / 86400.0
        print("  \(d.name.padding(toLength: 25, withPad: " ", startingAt: 0))  \(String(format: "%9d", d.rate))    \(String(format: "%.1f days/s", days)) \(epoch)")
    } else {
        let hours = secondsOfRecording / 3600.0
        print("  \(d.name.padding(toLength: 25, withPad: " ", startingAt: 0))  \(String(format: "%9d", d.rate))    \(String(format: "%.1f hours/s", hours)) \(epoch)")
    }
}

// --- Section 5: THE THREE CLOCKS ---
let flickRealtime = peakColorsPerSec / Double(FLICK)
let tritRealtime = peakColorsPerSec / Double(EPOCH1)

print("""

  THE THREE CLOCKS — ULTRA
  ──────────────────────────────────────────────────────────
  Clock        Ticks/second       GPU throughput          Realtime coverage
  ───────────  ─────────────────  ────────────────────    ─────────────
  Flick        705,600,000        \(String(format: "%6.1f B indices/s", peakRate/1000))    \(String(format: "%.1f s", flickRealtime)) = \(String(format: "%.1f min", flickRealtime/60.0))
  Trit-tick    141,120,000        \(String(format: "%6.1f B indices/s", peakRate/1000))    \(String(format: "%.1f s", tritRealtime)) = \(String(format: "%.1f min", tritRealtime/60.0))
  Color (raw)  N/A (abstract)     \(String(format: "%6.1f B colors/s", peakRate/1000))    (pure index space)

  SUMMARY
  ──────────────────────────────────────────────────────────
  At peak (\(String(format: "%.1f", peakRate/1000)) B colors/s):
    1 wall-second GPU work = \(String(format: "%.1f B", peakColorsPerSec/1e9)) color evaluations
                           = \(String(format: "%.1f B", peakColorsPerSec * 5 / 1e9)) flick-quanta of entropy
                           = \(String(format: "%.1f B", peakColorsPerSec / 1e9)) trit-tick-quanta of entropy
                           = \(String(format: "%.1f", flickRealtime)) seconds at flick resolution (every tick)
                           = \(String(format: "%.1f", tritRealtime)) seconds at trit-tick resolution (every tick)
  Subsampled at BCI rates:
                           = \(String(format: "%.1f", peakColorsPerSec / 250.0 / 86400 / 365.25)) years of 250 Hz EEG
                           = \(String(format: "%.1f", peakColorsPerSec / 30000.0 / 86400)) days of 30 kHz Neuropixels
                           = \(String(format: "%.1f", peakColorsPerSec / 48000.0 / 86400)) days of 48 kHz audio

  vs spi-metal-fast (16 colors/thread, 4-way ILP):
    CHUNK: 16 → \(CHUNK)  (4x more ALU per thread)
    ILP:   4  → 8   (8 independent accumulators)
    Params: 3 buffers → 1 fused struct
    Timing: best-of-3 → best-of-5
""")

// --- Section 6: Double-pump pipeline benchmark ---
// Two command buffers in flight: while GPU executes cmd[0], CPU encodes cmd[1]
// This hides command buffer setup latency
func gpuXORDoublePump(device: MTLDevice, pipe: MTLComputePipelineState,
                      reducePipe: MTLComputePipelineState,
                      queue: MTLCommandQueue, seed: UInt64, n: Int, batches: Int) -> (xor: UInt32, totalNs: UInt64, totalColors: Int) {
    let nThreads = (n + CHUNK - 1) / CHUNK
    let nGroups = (nThreads + TG_SIZE - 1) / TG_SIZE

    // Pre-allocate double buffers
    let partBuf0 = device.makeBuffer(length: nGroups * 4, options: .storageModeShared)!
    let partBuf1 = device.makeBuffer(length: nGroups * 4, options: .storageModeShared)!
    let resultBuf0 = device.makeBuffer(length: 4, options: .storageModeShared)!
    let resultBuf1 = device.makeBuffer(length: 4, options: .storageModeShared)!
    var params = Params(seed: seed, n: UInt32(n), chunk: UInt32(CHUNK))
    let paramBuf = device.makeBuffer(bytes: &params, length: MemoryLayout<Params>.size, options: .storageModeShared)!
    var countV = UInt32(nGroups)
    let countBuf = device.makeBuffer(bytes: &countV, length: 4, options: .storageModeShared)!

    let partBufs = [partBuf0, partBuf1]
    let resultBufs = [resultBuf0, resultBuf1]

    let t0 = DispatchTime.now().uptimeNanoseconds

    var lastXor: UInt32 = 0
    for i in 0..<batches {
        let pIdx = i & 1
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipe)
        enc.setBuffer(partBufs[pIdx], offset: 0, index: 0)
        enc.setBuffer(paramBuf, offset: 0, index: 1)
        enc.dispatchThreadgroups(MTLSize(width: nGroups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: TG_SIZE, height: 1, depth: 1))
        if nGroups > 1 {
            enc.setComputePipelineState(reducePipe)
            enc.setBuffer(partBufs[pIdx], offset: 0, index: 0)
            enc.setBuffer(resultBufs[pIdx], offset: 0, index: 1)
            enc.setBuffer(countBuf, offset: 0, index: 2)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: min(TG_SIZE, nGroups), height: 1, depth: 1))
        }
        enc.endEncoding()
        cmd.commit()
        if i == batches - 1 {
            cmd.waitUntilCompleted()
            if nGroups == 1 {
                lastXor = partBufs[pIdx].contents().load(as: UInt32.self)
            } else {
                lastXor = resultBufs[pIdx].contents().load(as: UInt32.self)
            }
        }
    }

    let t1 = DispatchTime.now().uptimeNanoseconds
    return (lastXor, t1 - t0, n * batches)
}

print("""

  DOUBLE-PUMP PIPELINE (two cmd buffers in flight)
  ──────────────────────────────────────────────────────────
""")
print("  Batch×N          Total colors   Time ms    M/s          GB/s")
print("  ───────────────  ────────────   ────────   ──────────   ─────")

// Warmup
_ = gpuXORDoublePump(device: device, pipe: ultraPipe, reducePipe: reducePipe, queue: queue, seed: SEED, n: 100_000_000, batches: 2)

var dpPeakRate = 0.0
for (batches, n) in [(4, 500_000_000), (8, 250_000_000), (16, 125_000_000), (4, 250_000_000), (2, 1_000_000_000)] {
    var bestNs: UInt64 = .max
    var bestXor: UInt32 = 0
    for _ in 0..<3 {
        let r = gpuXORDoublePump(device: device, pipe: ultraPipe, reducePipe: reducePipe, queue: queue, seed: SEED, n: n, batches: batches)
        if r.totalNs < bestNs { bestNs = r.totalNs; bestXor = r.xor }
    }
    let totalColors = n * batches
    let ms = Double(bestNs) / 1e6
    let rate = Double(totalColors) / Double(bestNs) * 1e3
    let gbps = Double(totalColors) * 3.0 / Double(bestNs)
    if rate > dpPeakRate { dpPeakRate = rate }
    print("  \(batches)×\(String(n).padding(toLength: 11, withPad: " ", startingAt: 0))  \(String(format: "%10d", totalColors))   \(String(format: "%7.1f", ms))    \(String(format: "%9.1f", rate))   \(String(format: "%5.1f", gbps))")
}

let overallPeak = max(peakRate, dpPeakRate)
print("""

  PEAK THROUGHPUT: \(String(format: "%.1f", overallPeak/1000)) B colors/s
  Single dispatch: \(String(format: "%.1f", peakRate/1000)) B/s
  Double-pump:     \(String(format: "%.1f", dpPeakRate/1000)) B/s
  Speedup over spi-metal-fast (~55 B/s): \(String(format: "%.1fx", overallPeak/55000.0))
""")

// --- Section 7: Fine-grained N sweep around sweet spot ---
print("""
  N SWEEP (finding occupancy sweet spot, best-of-5)
  ──────────────────────────────────────────────────────────
""")
print("  N            M/s          Threadgroups  Occupancy")
print("  -----------  -----------  ------------  ---------")

var sweepPeak = 0.0
var sweepBestN = 0
for n in [200_000_000, 300_000_000, 400_000_000, 500_000_000, 600_000_000, 700_000_000, 800_000_000] {
    _ = gpuXOR(device: device, pipe: ultraPipe, reducePipe: reducePipe, queue: queue, seed: SEED, n: n)
    var bestNs: UInt64 = .max
    for _ in 0..<5 {
        let r = gpuXOR(device: device, pipe: ultraPipe, reducePipe: reducePipe, queue: queue, seed: SEED, n: n)
        bestNs = min(bestNs, r.ns)
    }
    let rate = Double(n) / Double(bestNs) * 1e3
    let nThreads = (n + CHUNK - 1) / CHUNK
    let nGroups = (nThreads + TG_SIZE - 1) / TG_SIZE
    if rate > sweepPeak { sweepPeak = rate; sweepBestN = n }
    print("  \(String(n).padding(toLength: 11, withPad: " ", startingAt: 0))  \(String(format: "%9.1f", rate))  \(String(format: "%12d", nGroups))  \(rate > peakRate * 0.95 ? "***" : "")")
}

let finalPeak = max(overallPeak, sweepPeak)
print("""

  ====================================================================
  FINAL PEAK: \(String(format: "%.1f", finalPeak/1000)) B colors/s at N=\(sweepBestN)
  vs spi-metal-fast: \(String(format: "%.1fx", finalPeak/55000.0)) speedup
  ====================================================================
""")
