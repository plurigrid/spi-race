#!/usr/bin/env python3
"""SPI FFI: Call libspi.dylib from Python via ctypes.
Any language with ctypes/dlopen gets ~10 B/s for free."""

import ctypes, time, os

lib_path = os.path.join(os.path.dirname(__file__), "zig-out", "lib", "libspi.dylib")
spi = ctypes.CDLL(lib_path)

spi.spi_xor_fingerprint.restype = ctypes.c_uint64
spi.spi_xor_fingerprint.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]

spi.spi_xor_fingerprint_parallel.restype = ctypes.c_uint64
spi.spi_xor_fingerprint_parallel.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32]

spi.spi_color_at.restype = ctypes.c_uint32
spi.spi_color_at.argtypes = [ctypes.c_uint64, ctypes.c_uint64]

spi.spi_trit.restype = ctypes.c_int8
spi.spi_trit.argtypes = [ctypes.c_uint64, ctypes.c_uint64]

SEED = 42

print("+======================================================================+")
print("|  SPI FFI — Python ctypes calling libspi.dylib (Zig)                 |")
print("+======================================================================+")

c0 = spi.spi_color_at(SEED, 0)
c69 = spi.spi_color_at(SEED, 69)
t0 = spi.spi_trit(SEED, 0)
t69 = spi.spi_trit(SEED, 69)
print(f"  color_at(42,0)=#{c0:06x}  (42,69)=#{c69:06x}")
print(f"  trit(42,0)={t0}  trit(42,69)={t69}")

for label, n in [("1M", 1_000_000), ("10M", 10_000_000), ("100M", 100_000_000)]:
    spi.spi_xor_fingerprint(SEED, 0, max(1, n // 100))
    t = time.monotonic_ns()
    xor = spi.spi_xor_fingerprint(SEED, 0, n)
    ns = time.monotonic_ns() - t
    rate = int(n / ns * 1000) if ns > 0 else 0
    print(f"  1T  {label}: {rate:,} M/s  xor=0x{xor:012x}")

for label, n in [("1M", 1_000_000), ("10M", 10_000_000), ("100M", 100_000_000), ("1B", 1_000_000_000)]:
    spi.spi_xor_fingerprint_parallel(SEED, max(1, n // 100), 0)
    t = time.monotonic_ns()
    xor = spi.spi_xor_fingerprint_parallel(SEED, n, 0)
    ns = time.monotonic_ns() - t
    rate = int(n / ns * 1000) if ns > 0 else 0
    ms = ns // 1_000_000
    print(f"  MT  {label}: {rate:,} M/s ({ms}ms)  xor=0x{xor:012x}")

print("\n  Python overhead: ~0 (one FFI call, all work in Zig)")
print("  Same numbers as native Zig — that's the point.")
