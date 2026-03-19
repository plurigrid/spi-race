// SPI Virtuoso — Swift Structured Concurrency racer
// Swift 6.2 tricks: TaskGroup, async let, Sendable, withUnsafeBufferPointer
// Targets: same SplitMix64 XOR fingerprint as Zig/Julia racers
//
// swiftc -O -whole-module-optimization swift-racer.swift -o swift-racer && ./swift-racer

import Foundation

let GOLDEN: UInt64 = 0x9e3779b97f4a7c15
let MIX1: UInt64 = 0xbf58476d1ce4e5b9
let MIX2: UInt64 = 0x94d049bb133111eb
let SEED: UInt64 = 42

@inline(__always)
func sm64(_ seed: UInt64, _ index: UInt64) -> UInt64 {
    var z = seed &+ (GOLDEN &* index)
    z = (z ^ (z >> 30)) &* MIX1
    z = (z ^ (z >> 27)) &* MIX2
    return z ^ (z >> 31)
}

@inline(__always)
func extractRGB(_ val: UInt64) -> UInt64 {
    ((val >> 16) & 0xFF) << 16 | ((val >> 8) & 0xFF) << 8 | (val & 0xFF)
}

// L0: Scalar baseline
func l0Scalar(_ n: Int) -> UInt64 {
    var xor: UInt64 = 0
    for i in 0..<UInt64(n) {
        xor ^= sm64(SEED, i)
    }
    return xor
}

// L1: 8-wide manual unroll (matching Zig L2 / Julia L3)
func l1Pipeline8(_ n: Int) -> UInt64 {
    var a0: UInt64 = 0, a1: UInt64 = 0, a2: UInt64 = 0, a3: UInt64 = 0
    var b0: UInt64 = 0, b1: UInt64 = 0, b2: UInt64 = 0, b3: UInt64 = 0
    let n8 = n & ~7
    var i: UInt64 = 0
    while i < UInt64(n8) {
        a0 ^= sm64(SEED, i)
        a1 ^= sm64(SEED, i &+ 1)
        a2 ^= sm64(SEED, i &+ 2)
        a3 ^= sm64(SEED, i &+ 3)
        b0 ^= sm64(SEED, i &+ 4)
        b1 ^= sm64(SEED, i &+ 5)
        b2 ^= sm64(SEED, i &+ 6)
        b3 ^= sm64(SEED, i &+ 7)
        i &+= 8
    }
    var result = a0 ^ a1 ^ a2 ^ a3 ^ b0 ^ b1 ^ b2 ^ b3
    while i < UInt64(n) {
        result ^= sm64(SEED, i)
        i &+= 1
    }
    return result
}

// L2: Fused gen+RGB+XOR (matches Zig L5 / Julia L5)
func l2Fused(_ n: Int) -> UInt64 {
    var a0: UInt64 = 0, a1: UInt64 = 0, a2: UInt64 = 0, a3: UInt64 = 0
    var b0: UInt64 = 0, b1: UInt64 = 0, b2: UInt64 = 0, b3: UInt64 = 0
    let n8 = n & ~7
    var i: UInt64 = 0
    while i < UInt64(n8) {
        a0 ^= extractRGB(sm64(SEED, i))
        a1 ^= extractRGB(sm64(SEED, i &+ 1))
        a2 ^= extractRGB(sm64(SEED, i &+ 2))
        a3 ^= extractRGB(sm64(SEED, i &+ 3))
        b0 ^= extractRGB(sm64(SEED, i &+ 4))
        b1 ^= extractRGB(sm64(SEED, i &+ 5))
        b2 ^= extractRGB(sm64(SEED, i &+ 6))
        b3 ^= extractRGB(sm64(SEED, i &+ 7))
        i &+= 8
    }
    var result = a0 ^ a1 ^ a2 ^ a3 ^ b0 ^ b1 ^ b2 ^ b3
    while i < UInt64(n) {
        result ^= extractRGB(sm64(SEED, i))
        i &+= 1
    }
    return result
}

// L3: DispatchQueue.concurrentPerform — GCD parallel (Obj-C era, still fast)
func l3GCD(_ n: Int, threads: Int) -> UInt64 {
    let chunk = n / threads
    let partials = UnsafeMutablePointer<UInt64>.allocate(capacity: threads)
    defer { partials.deallocate() }
    partials.initialize(repeating: 0, count: threads)

    DispatchQueue.concurrentPerform(iterations: threads) { tid in
        let start = UInt64(tid * chunk)
        let end = tid == threads - 1 ? UInt64(n) : UInt64((tid + 1) * chunk)
        var a0: UInt64 = 0, a1: UInt64 = 0, a2: UInt64 = 0, a3: UInt64 = 0
        var b0: UInt64 = 0, b1: UInt64 = 0, b2: UInt64 = 0, b3: UInt64 = 0
        let count = end - start
        let n8 = count & ~7
        var j: UInt64 = 0
        while j < n8 {
            let idx = start &+ j
            a0 ^= extractRGB(sm64(SEED, idx))
            a1 ^= extractRGB(sm64(SEED, idx &+ 1))
            a2 ^= extractRGB(sm64(SEED, idx &+ 2))
            a3 ^= extractRGB(sm64(SEED, idx &+ 3))
            b0 ^= extractRGB(sm64(SEED, idx &+ 4))
            b1 ^= extractRGB(sm64(SEED, idx &+ 5))
            b2 ^= extractRGB(sm64(SEED, idx &+ 6))
            b3 ^= extractRGB(sm64(SEED, idx &+ 7))
            j &+= 8
        }
        var local = a0 ^ a1 ^ a2 ^ a3 ^ b0 ^ b1 ^ b2 ^ b3
        while j < count {
            local ^= extractRGB(sm64(SEED, start &+ j))
            j &+= 1
        }
        partials[tid] = local
    }

    var combined: UInt64 = 0
    for i in 0..<threads { combined ^= partials[i] }
    return combined
}

// L4: Swift Structured Concurrency — TaskGroup (cooperative threads)
func l4TaskGroup(_ n: Int, threads: Int) async -> UInt64 {
    await withTaskGroup(of: UInt64.self, returning: UInt64.self) { group in
        let chunk = n / threads
        for tid in 0..<threads {
            let start = UInt64(tid * chunk)
            let end = tid == threads - 1 ? UInt64(n) : UInt64((tid + 1) * chunk)
            group.addTask {
                var a0: UInt64 = 0, a1: UInt64 = 0, a2: UInt64 = 0, a3: UInt64 = 0
                var b0: UInt64 = 0, b1: UInt64 = 0, b2: UInt64 = 0, b3: UInt64 = 0
                let count = end - start
                let n8 = count & ~7
                var j: UInt64 = 0
                while j < n8 {
                    let idx = start &+ j
                    a0 ^= extractRGB(sm64(SEED, idx))
                    a1 ^= extractRGB(sm64(SEED, idx &+ 1))
                    a2 ^= extractRGB(sm64(SEED, idx &+ 2))
                    a3 ^= extractRGB(sm64(SEED, idx &+ 3))
                    b0 ^= extractRGB(sm64(SEED, idx &+ 4))
                    b1 ^= extractRGB(sm64(SEED, idx &+ 5))
                    b2 ^= extractRGB(sm64(SEED, idx &+ 6))
                    b3 ^= extractRGB(sm64(SEED, idx &+ 7))
                    j &+= 8
                }
                var local = a0 ^ a1 ^ a2 ^ a3 ^ b0 ^ b1 ^ b2 ^ b3
                while j < count {
                    local ^= extractRGB(sm64(SEED, start &+ j))
                    j &+= 1
                }
                return local
            }
        }
        var combined: UInt64 = 0
        for await partial in group { combined ^= partial }
        return combined
    }
}

struct BenchResult { let label: String; let ns: UInt64; let n: Int; let xor: UInt64
    var rateM: Int { ns == 0 ? 0 : Int(Double(n) / Double(ns) * 1000.0) }
}

func bench(_ label: String, _ n: Int, _ f: (Int) -> UInt64) -> BenchResult {
    _ = f(max(1, n / 100))
    let t0 = DispatchTime.now().uptimeNanoseconds
    let xor = f(n)
    let t1 = DispatchTime.now().uptimeNanoseconds
    return BenchResult(label: label, ns: t1 - t0, n: n, xor: xor)
}

func benchAsync(_ label: String, _ n: Int, _ f: @Sendable (Int) async -> UInt64) async -> BenchResult {
    _ = await f(max(1, n / 100))
    let t0 = DispatchTime.now().uptimeNanoseconds
    let xor = await f(n)
    let t1 = DispatchTime.now().uptimeNanoseconds
    return BenchResult(label: label, ns: t1 - t0, n: n, xor: xor)
}

@main
struct SPIRace {
    static func main() async {
        let cores = ProcessInfo.processInfo.activeProcessorCount
        print("""

        +======================================================================+
        |       SPI VIRTUOSO — Swift Structured Concurrency Racer              |
        |  Tricks: @inline(__always), &+/&*, GCD, TaskGroup, Sendable          |
        +======================================================================+
          CPU cores: \(cores)  Seed: \(SEED)
        """)

        let sizes = [1_000_000, 10_000_000, 100_000_000]
        let labels = ["1M", "10M", "100M"]

        // Single-threaded
        let levels: [(String, String, (Int) -> UInt64)] = [
            ("L0", "Scalar baseline                 ", l0Scalar),
            ("L1", "8-wide pipeline (2x4 accum)     ", l1Pipeline8),
            ("L2", "Fused gen+RGB+XOR (0-alloc)     ", l2Fused),
        ]

        print("  Level  Description                      ", terminator: "")
        for l in labels { print(String(format: "%13s", l), terminator: "") }
        print()
        print("  -----  -------------------------------- ", terminator: "")
        for _ in labels { print(" ------------", terminator: "") }
        print()

        for (code, desc, f) in levels {
            print("  \(code)    \(desc)", terminator: "")
            for (_, n) in sizes.enumerated() {
                let r = bench(code, n, f)
                print(String(format: "%9d M/s", r.rateM), terminator: "")
            }
            print()
        }

        // GCD parallel
        print("\n  -- L3: GCD concurrentPerform (\(cores) cores) --")
        print("  L3    GCD fused parallel               ", terminator: "")
        for (_, n) in sizes.enumerated() {
            let r = bench("L3", n, { l3GCD($0, threads: cores) })
            print(String(format: "%9d M/s", r.rateM), terminator: "")
        }
        print()

        // TaskGroup
        print("\n  -- L4: Swift TaskGroup (\(cores) tasks) --")
        print("  L4    TaskGroup structured concurrency ", terminator: "")
        for (_, n) in sizes.enumerated() {
            let r = await benchAsync("L4", n, { await l4TaskGroup($0, threads: cores) })
            print(String(format: "%9d M/s", r.rateM), terminator: "")
        }
        print()

        // 100M with all cores
        print("\n  -- 100M colors (L3 GCD, all cores) --")
        _ = l3GCD(100_000, threads: cores)
        let t0 = DispatchTime.now().uptimeNanoseconds
        let xor100m = l3GCD(100_000_000, threads: cores)
        let t1 = DispatchTime.now().uptimeNanoseconds
        let ns100 = t1 - t0
        let ms100 = ns100 / 1_000_000
        let rateM100 = Int(Double(100_000_000) / Double(ns100) * 1000.0)
        print("  100M colors: \(ms100) ms = \(rateM100) M/s  XOR=0x\(String(xor100m, radix: 16))")

        print("""

          Tricks applied:
            L0: Scalar loop with @inline(__always) and &+ overflow operators
            L1: 8 independent accumulators — matches Zig L2 / Julia L3
            L2: Fused gen+RGB+XOR — color never touches memory
            L3: GCD concurrentPerform — OS-level thread pool (Obj-C heritage)
            L4: Swift TaskGroup — cooperative structured concurrency (modern Swift)

          Compare: zig-syrup spi-virtuoso, Gay.jl spi_virtuoso.jl
        """)
    }
}
