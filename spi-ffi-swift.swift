#!/usr/bin/env swift
// SPI FFI from Swift via dlopen
import Foundation

let libPath = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    .appendingPathComponent("zig-out/lib/libspi.dylib").path
guard let handle = dlopen(libPath, RTLD_NOW) else {
    print("Failed to load \(libPath): \(String(cString: dlerror()))")
    exit(1)
}

typealias ColorAtFn = @convention(c) (UInt64, UInt64) -> UInt32
typealias TritFn = @convention(c) (UInt64, UInt64) -> Int8
typealias FpFn = @convention(c) (UInt64, UInt64, UInt64) -> UInt64
typealias FpParFn = @convention(c) (UInt64, UInt64, UInt32) -> UInt64

let colorAt = unsafeBitCast(dlsym(handle, "spi_color_at"), to: ColorAtFn.self)
let trit = unsafeBitCast(dlsym(handle, "spi_trit"), to: TritFn.self)
let fp = unsafeBitCast(dlsym(handle, "spi_xor_fingerprint"), to: FpFn.self)
let fpPar = unsafeBitCast(dlsym(handle, "spi_xor_fingerprint_parallel"), to: FpParFn.self)

let SEED: UInt64 = 42
let c0 = colorAt(SEED, 0)
let c69 = colorAt(SEED, 69)
let t0 = trit(SEED, 0)
let t69 = trit(SEED, 69)
print("Swift \(ProcessInfo.processInfo.operatingSystemVersionString): dlopen libspi.dylib")
print("  color_at(42,0)=#\(String(c0, radix:16)) color_at(42,69)=#\(String(c69, radix:16))")
print("  trit(42,0)=\(t0) trit(42,69)=\(t69)")

for (label, n): (String, UInt64) in [("1M", 1_000_000), ("100M", 100_000_000), ("1B", 1_000_000_000)] {
    _ = fpPar(SEED, 1000, 0)
    let start = DispatchTime.now().uptimeNanoseconds
    let xor = fpPar(SEED, n, 0)
    let ns = DispatchTime.now().uptimeNanoseconds - start
    let rate = Int(Double(n) / Double(ns) * 1000)
    print("  \(label): \(rate) M/s  xor=0x\(String(xor, radix:16))")
}
dlclose(handle)
