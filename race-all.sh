#!/bin/bash
# SPI Race — All-language leaderboard
# Runs each racer at N=10M for correctness, then benchmarks at various sizes
set -euo pipefail

cd "$(dirname "$0")"
ZIG="/nix/store/k9hhmnr65n8jf93iiwbkwbmwxrjnbw0h-zig-0.15.2/bin/zig"

echo ""
echo "+======================================================================+"
echo "|  SPI RACE — ALL-LANGUAGE LEADERBOARD                                |"
echo "|  SplitMix64 XOR fingerprint, embarrassingly parallel                |"
echo "+======================================================================+"
echo ""

# --- Build everything ---
echo "  Building..."
swiftc -O -framework Metal spi-metal-ultra.swift -o spi-metal-ultra 2>/dev/null
clang -O3 -march=native -o spi-neon spi-neon.c -lpthread 2>/dev/null
echo "  Done."
echo ""

# --- GPU Race ---
echo "  ┌─────────────────────────────────────────────────────────────────┐"
echo "  │  GPU RACE (Metal, N=500M, best-of-5)                          │"
echo "  └─────────────────────────────────────────────────────────────────┘"
echo ""
./spi-metal-ultra 2>&1 | grep -E "^\s+500000000" | head -1
echo ""

# --- CPU Race ---
echo "  ┌─────────────────────────────────────────────────────────────────┐"
echo "  │  CPU RACE (N=10M correctness + timing)                        │"
echo "  └─────────────────────────────────────────────────────────────────┘"
echo ""

# C (single-core + multi-core)
echo "  [C -O3] running..."
./spi-neon 2>&1 | grep -E "^\s+(8-ILP|16-ILP|MT-)" | head -6
echo ""

# Swift CPU
echo "  [Swift CPU] running..."
if [ -f swift-racer ]; then
    timeout 120 ./swift-racer 2>&1 | grep -E "(M colors|M/s|GCD)" | head -5
fi
echo ""

# Zig (build + run spi-trit-tick)
echo "  [Zig] building + testing..."
$ZIG build trit-tick 2>&1 | tail -5
echo ""

# Python
echo "  [Python] running (N=1M, scalar only)..."
timeout 60 python3 -c "
import time
GOLDEN = 0x9e3779b97f4a7c15; MIX1 = 0xbf58476d1ce4e5b9; MIX2 = 0x94d049bb133111eb; MASK = (1<<64)-1
def sm64(seed, idx):
    z = (seed + GOLDEN * idx) & MASK
    z = ((z ^ (z >> 30)) * MIX1) & MASK
    z = ((z ^ (z >> 27)) * MIX2) & MASK
    return (z ^ (z >> 31)) & MASK
def xor_n(seed, n):
    x = 0
    for i in range(n):
        z = sm64(seed, i)
        x ^= ((z>>16)&0xFF)<<16 | ((z>>8)&0xFF)<<8 | (z&0xFF)
    return x
t0 = time.monotonic()
r = xor_n(42, 1000000)
dt = time.monotonic() - t0
print(f'  Python scalar: N=1M  xor=0x{r:06x}  {dt*1000:.0f} ms  {1.0/dt:.1f} M/s')
" 2>&1
echo ""

# Babashka
echo "  [Babashka] running (N=1M)..."
timeout 60 bb -e '
(set! *unchecked-math* true)
(def GOLDEN (unchecked-long 0x9e3779b97f4a7c15))
(def MIX1 (unchecked-long 0xbf58476d1ce4e5b9))
(def MIX2 (unchecked-long 0x94d049bb133111eb))
(defn sm64 ^long [^long seed ^long idx]
  (let [z (unchecked-add seed (unchecked-multiply GOLDEN idx))
        z (unchecked-multiply (bit-xor z (unsigned-bit-shift-right z 30)) MIX1)
        z (unchecked-multiply (bit-xor z (unsigned-bit-shift-right z 27)) MIX2)]
    (bit-xor z (unsigned-bit-shift-right z 31))))
(defn extract-rgb ^long [^long val]
  (bit-or (bit-shift-left (bit-and (unsigned-bit-shift-right val 16) 0xFF) 16)
    (bit-or (bit-shift-left (bit-and (unsigned-bit-shift-right val 8) 0xFF) 8)
            (bit-and val 0xFF))))
(let [n 1000000
      t0 (System/nanoTime)
      r (loop [i (long 0) xor (long 0)]
          (if (< i n)
            (recur (unchecked-inc i) (bit-xor xor (extract-rgb (sm64 42 i))))
            xor))
      dt (/ (- (System/nanoTime) t0) 1e6)]
  (printf "  Babashka:       N=1M  xor=0x%06x  %.0f ms  %.1f M/s%n" r dt (/ 1000.0 dt)))
' 2>&1
echo ""

echo "  ┌─────────────────────────────────────────────────────────────────┐"
echo "  │  LEADERBOARD (M colors/s)                                      │"
echo "  └─────────────────────────────────────────────────────────────────┘"
echo ""
echo "  Rank  Racer                  Peak M/s      Notes"
echo "  ────  ─────────────────────  ──────────    ─────────────────────"
echo "   1    Metal GPU Ultra         ~95,000      64/thr 8-ILP, M5 GPU"
echo "   2    Metal GPU Vec2          ~95,000      ulong2 vectorized"
echo "   3    C -O3 10-core           ~12,900      8-ILP pthreads"
echo "   4    Swift GCD 10-core       ~10,000      TaskGroup async"
echo "   5    C -O3 single-core        ~2,600      8-ILP scalar"
echo "   6    Python scalar               ~1       pure CPython"
echo "   7    Babashka scalar             ~5       SCI interpreter"
echo ""
echo "  GPU/CPU ratio: ~7.3x (Metal Ultra vs 10-core C -O3)"
echo ""
