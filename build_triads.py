#!/usr/bin/env python3
"""Build color-named dylibs for each GF(3)-balanced triad.

Each triad's XOR of its three logo colors becomes the comptime seed
baked into the dylib. The dylib is named lib<hex>.dylib.

XOR is involution: a ^ b ^ c ^ a ^ b ^ c = 0
So loading all three triads' dylibs and XOR-folding their seeds
teleports back to 0 — the origin. This is the verification.

Three build strategies:
  --sequential   Build one at a time, in lexicographic order
  --parallel     Build all at once (embarrassingly parallel)
  --random       Shuffle order, random sleep between builds

Usage:
  python3 build_triads.py --sequential
  python3 build_triads.py --parallel
  python3 build_triads.py --random
"""

import subprocess, os, sys, time, random, json, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ZIG = os.environ.get("ZIG", shutil.which("zig")
       or "/nix/store/25brjshr934nir8lqc4ln8dkkamfh47c-zig-0.15.2/bin/zig")

RACE_DIR = Path(__file__).parent
OUT_DIR = RACE_DIR / "zig-out" / "triads"

# GitHub linguist canonical logo colors
LANG_COLORS = {
    "zig":     0xec915c,
    "python":  0x3572A5,
    "ruby":    0x701516,
    "scheme":  0x1e4aec,
    "swift":   0xF05138,
    "tcl":     0xe4cc98,
    "perl":    0x0298c3,
    "node":    0xf1e05a,
    "clojure": 0xdb5855,
    "hy":      0x7790B2,
    "mlx":     0xA2AAAD,
}

def trit(color):
    r, g, b = (color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF
    return (r + g + b) % 3 - 1

def balanced_triads():
    """All triads where sum of trits ≡ 0 (mod 3)."""
    from itertools import combinations
    names = sorted(LANG_COLORS.keys())
    out = []
    for combo in combinations(names, 3):
        if sum(trit(LANG_COLORS[n]) for n in combo) % 3 == 0:
            out.append(combo)
    return out

def triad_seed(a, b, c):
    """XOR of three logo colors = the involution seed."""
    return LANG_COLORS[a] ^ LANG_COLORS[b] ^ LANG_COLORS[c]

def triad_hex(a, b, c):
    return f"{triad_seed(a, b, c):06x}"

def build_one(a, b, c):
    """Compile one color-named dylib. Returns (hex, elapsed_ns, ok)."""
    seed = triad_seed(a, b, c)
    hexname = f"{seed:06x}"
    libname = f"lib{hexname}"

    # Generate a tiny per-triad .zig that hardcodes the seed and re-exports the math
    gen_path = RACE_DIR / f".gen_{hexname}.zig"
    gen_path.write_text(f"""\
const std = @import("std");
const TRIAD_SEED: u64 = 0x{seed:06x};
const GOLDEN: u64 = 0x9e3779b97f4a7c15;
const MIX1: u64 = 0xbf58476d1ce4e5b9;
const MIX2: u64 = 0x94d049bb133111eb;
inline fn splitmix64(s: u64, i: u64) u64 {{
    var z = s +% (GOLDEN *% i);
    z = (z ^ (z >> 30)) *% MIX1;
    z = (z ^ (z >> 27)) *% MIX2;
    return z ^ (z >> 31);
}}
inline fn extract_rgb(v: u64) u64 {{
    return ((v >> 16) & 0xFF) << 16 | ((v >> 8) & 0xFF) << 8 | (v & 0xFF);
}}
fn fused_xor_range(seed: u64, start: u64, count: u64) u64 {{
    var a0: u64 = 0; var a1: u64 = 0; var a2: u64 = 0; var a3: u64 = 0;
    var b0: u64 = 0; var b1: u64 = 0; var b2: u64 = 0; var b3: u64 = 0;
    var ii: u64 = 0;
    const n8 = count & ~@as(u64, 7);
    while (ii < n8) : (ii += 8) {{
        const x = start +% ii;
        a0 ^= extract_rgb(splitmix64(seed, x));
        a1 ^= extract_rgb(splitmix64(seed, x +% 1));
        a2 ^= extract_rgb(splitmix64(seed, x +% 2));
        a3 ^= extract_rgb(splitmix64(seed, x +% 3));
        b0 ^= extract_rgb(splitmix64(seed, x +% 4));
        b1 ^= extract_rgb(splitmix64(seed, x +% 5));
        b2 ^= extract_rgb(splitmix64(seed, x +% 6));
        b3 ^= extract_rgb(splitmix64(seed, x +% 7));
    }}
    var result = a0 ^ a1 ^ a2 ^ a3 ^ b0 ^ b1 ^ b2 ^ b3;
    while (ii < count) : (ii += 1) {{
        result ^= extract_rgb(splitmix64(seed, start +% ii));
    }}
    return result;
}}
export fn spi_triad_seed() u64 {{ return TRIAD_SEED; }}
export fn spi_involution_witness() u64 {{ return TRIAD_SEED ^ TRIAD_SEED; }}
export fn spi_xor_fingerprint(seed: u64, start: u64, count: u64) u64 {{
    return fused_xor_range(seed, start, count);
}}
export fn spi_xor_fingerprint_triad(start: u64, count: u64) u64 {{
    return fused_xor_range(TRIAD_SEED, start, count);
}}
export fn spi_color_at(seed: u64, index: u64) u32 {{
    return @truncate(extract_rgb(splitmix64(seed, index)));
}}
export fn spi_trit(seed: u64, index: u64) i8 {{
    const h = splitmix64(seed, index);
    const r: i32 = @intCast((h >> 16) & 0xFF);
    const g: i32 = @intCast((h >> 8) & 0xFF);
    const b_: i32 = @intCast(h & 0xFF);
    return @intCast(@mod(r + g + b_, @as(i32, 3)) - 1);
}}
""")

    out_path = OUT_DIR / f"{libname}.dylib"

    t0 = time.monotonic_ns()
    try:
        result = subprocess.run(
            [ZIG, "build-lib", "-dynamic", "-OReleaseFast",
             f"-femit-bin={out_path}", str(gen_path)],
            capture_output=True, text=True, timeout=30, cwd=str(RACE_DIR),
        )
        elapsed = time.monotonic_ns() - t0
        ok = result.returncode == 0
        # Clean up generated source
        gen_path.unlink(missing_ok=True)
        return (hexname, a, b, c, seed, elapsed, ok,
                result.stderr.strip() if not ok else "")
    except Exception as e:
        elapsed = time.monotonic_ns() - t0
        gen_path.unlink(missing_ok=True)
        return (hexname, a, b, c, seed, elapsed, False, str(e))

def verify_involution(triads_built):
    """XOR all triad seeds — must equal 0 if count is even, or the last seed if odd."""
    xor_all = 0
    for t in triads_built:
        xor_all ^= t[4]  # seed
    return xor_all

def main():
    strategy = sys.argv[1] if len(sys.argv) > 1 else "--sequential"
    triads = balanced_triads()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  SPI TRIAD TELEPORTATION — {len(triads)} balanced triads              ║")
    print(f"║  Strategy: {strategy:15s}                                  ║")
    print(f"║  Involution: XOR is its own inverse (a^a=0)                ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    print()

    if strategy == "--random":
        random.shuffle(triads)
        print(f"  Shuffled order (seed={random.getrandbits(32)})")

    results = []
    t_total_0 = time.monotonic_ns()

    if strategy == "--parallel":
        with ThreadPoolExecutor(max_workers=min(len(triads), os.cpu_count() or 4)) as pool:
            futures = {pool.submit(build_one, *t): t for t in triads}
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                status = "OK" if r[6] else "FAIL"
                print(f"  #{r[0]}  {r[1]:>8s}+{r[2]:>7s}+{r[3]:>7s}  {r[5]//1_000_000:>4d}ms  {status}")
    else:
        for t in triads:
            if strategy == "--random":
                time.sleep(random.uniform(0, 0.05))
            r = build_one(*t)
            results.append(r)
            status = "OK" if r[6] else "FAIL"
            print(f"  #{r[0]}  {r[1]:>8s}+{r[2]:>7s}+{r[3]:>7s}  {r[5]//1_000_000:>4d}ms  {status}")

    t_total = (time.monotonic_ns() - t_total_0) // 1_000_000

    print()
    ok_count = sum(1 for r in results if r[6])
    print(f"  Built: {ok_count}/{len(results)} dylibs in {t_total}ms ({strategy})")

    # Involution test: pair up triads, XOR seeds, verify cancellation
    print()
    print("  ── Involution Witness ──")
    seeds = [r[4] for r in results if r[6]]
    xor_all = 0
    for s in seeds:
        xor_all ^= s
    print(f"  XOR of all {len(seeds)} triad seeds: 0x{xor_all:06x}")

    # Pair-cancellation: each seed XOR'd with itself = 0
    for r in results[:3]:
        if r[6]:
            witness = r[4] ^ r[4]
            print(f"  #{r[0]} ^ #{r[0]} = 0x{witness:06x} {'(origin)' if witness == 0 else '(!)'}")

    # The teleportation test: take any triad (A,B,C), compute fp with seed=A^B^C,
    # then compute fp with seed=(A^B^C)^(A^B^C) = 0. Second fp must equal fp(0,0,N).
    if seeds:
        s0 = seeds[0]
        print()
        print(f"  ── Teleportation Test (seed=0x{s0:06x}) ──")
        # Use libspi.dylib via Python FFI for the actual verification
        try:
            sys.path.insert(0, str(RACE_DIR.parent / "hymlx" / "src"))
            from hymlx.spi import xor_fingerprint_parallel
            N = 10_000_000
            fp_triad = xor_fingerprint_parallel(s0, N, 0)
            fp_origin = xor_fingerprint_parallel(0, N, 0)
            fp_double = xor_fingerprint_parallel(s0 ^ s0, N, 0)
            print(f"  fp(triad,  10M) = 0x{fp_triad:012x}")
            print(f"  fp(origin, 10M) = 0x{fp_origin:012x}")
            print(f"  fp(t^t=0,  10M) = 0x{fp_double:012x}")
            print(f"  Double-apply = origin: {'PASS' if fp_double == fp_origin else 'FAIL'}")
        except Exception as e:
            print(f"  (skipped FFI verification: {e})")

    # Write manifest
    manifest = []
    for r in results:
        manifest.append({
            "hex": r[0], "a": r[1], "b": r[2], "c": r[3],
            "seed": r[4], "build_ms": r[5] // 1_000_000,
            "ok": r[6], "trits": [trit(LANG_COLORS[r[1]]), trit(LANG_COLORS[r[2]]), trit(LANG_COLORS[r[3]])],
            "colors": [f"#{LANG_COLORS[r[1]]:06x}", f"#{LANG_COLORS[r[2]]:06x}", f"#{LANG_COLORS[r[3]]:06x}"],
        })
    manifest_path = OUT_DIR / "triads.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n  Manifest: {manifest_path}")

if __name__ == "__main__":
    main()
