const std = @import("std");

extern fn spi_xor_fingerprint(seed: u64, start: u64, count: u64) u64;
extern fn spi_xor_fingerprint_parallel(seed: u64, n: u64, n_threads: u32) u64;
extern fn spi_color_at(seed: u64, index: u64) u32;
extern fn spi_trit(seed: u64, index: u64) i8;

fn writeAll(buf: []const u8) void {
    _ = std.posix.write(std.posix.STDOUT_FILENO, buf) catch {};
}

fn printBuf(comptime fmt: []const u8, args: anytype) void {
    var buf: [512]u8 = undefined;
    const slice = std.fmt.bufPrint(&buf, fmt, args) catch &buf;
    writeAll(slice);
}

pub fn main() !void {
    const SEED: u64 = 42;

    writeAll(
        \\
        \\+======================================================================+
        \\|  libspi — Embarrassingly Parallel SPI via C ABI                      |
        \\|  Any language with FFI can now hit ~3 B/s single, ~11 B/s parallel   |
        \\+======================================================================+
        \\
    );

    // Correctness: single vs parallel must agree
    const fp_1t = spi_xor_fingerprint(SEED, 0, 1_000_000);
    const fp_mt = spi_xor_fingerprint_parallel(SEED, 1_000_000, 0);
    printBuf("  Correctness 1M: 1T=0x{x:0>12} MT=0x{x:0>12} {s}\n", .{
        fp_1t, fp_mt, if (fp_1t == fp_mt) "PASS" else "FAIL",
    });

    // Single color
    const c0 = spi_color_at(SEED, 0);
    const c1 = spi_color_at(SEED, 1);
    const c69 = spi_color_at(SEED, 69);
    printBuf("  color_at(42,0)=#{x:0>6}  (42,1)=#{x:0>6}  (42,69)=#{x:0>6}\n", .{ c0, c1, c69 });

    // Trits
    const t0 = spi_trit(SEED, 0);
    const t1 = spi_trit(SEED, 1);
    const t69 = spi_trit(SEED, 69);
    printBuf("  trit(42,0)={}  (42,1)={}  (42,69)={}\n", .{ t0, t1, t69 });

    // Benchmark: 100M single-threaded
    const sizes = [_]u64{ 1_000_000, 10_000_000, 100_000_000 };
    const labels = [_][]const u8{ "1M", "10M", "100M" };

    writeAll("\n  Single-threaded:\n");
    for (sizes, 0..) |n, si| {
        _ = spi_xor_fingerprint(SEED, 0, n / 100 + 1);
        const st0 = std.time.nanoTimestamp();
        const xor = spi_xor_fingerprint(SEED, 0, n);
        const st1 = std.time.nanoTimestamp();
        const ns = st1 - st0;
        const rate: i128 = if (ns > 0) @divFloor(@as(i128, n) * 1000, ns) else 0;
        printBuf("    {s}: {} M/s  xor=0x{x:0>12}\n", .{ labels[si], rate, xor });
    }

    const cpu_count = std.Thread.getCpuCount() catch 4;
    printBuf("\n  Multi-threaded ({} cores):\n", .{cpu_count});
    for (sizes, 0..) |n, si| {
        _ = spi_xor_fingerprint_parallel(SEED, n / 100 + 1, 0);
        const mt0 = std.time.nanoTimestamp();
        const xor = spi_xor_fingerprint_parallel(SEED, n, 0);
        const mt1 = std.time.nanoTimestamp();
        const ns = mt1 - mt0;
        const rate: i128 = if (ns > 0) @divFloor(@as(i128, n) * 1000, ns) else 0;
        printBuf("    {s}: {} M/s  xor=0x{x:0>12}\n", .{ labels[si], rate, xor });
    }

    writeAll("\n  1 BILLION (all cores):\n");
    const billion: u64 = 1_000_000_000;
    _ = spi_xor_fingerprint_parallel(SEED, 1_000_000, 0);
    const bt0 = std.time.nanoTimestamp();
    const xor_b = spi_xor_fingerprint_parallel(SEED, billion, 0);
    const bt1 = std.time.nanoTimestamp();
    const ns_b = bt1 - bt0;
    const rate_b: i128 = if (ns_b > 0) @divFloor(@as(i128, billion) * 1000, ns_b) else 0;
    const ms_b = @divFloor(ns_b, 1_000_000);
    printBuf("    1B: {} ms = {} M/s  xor=0x{x:0>12}\n", .{ ms_b, rate_b, xor_b });

    writeAll(
        \\
        \\  FFI usage (any language):
        \\    dlopen("libspi.dylib") or ctypes.CDLL("./libspi.dylib")
        \\    uint64_t spi_xor_fingerprint(uint64_t seed, uint64_t start, uint64_t count)
        \\    uint64_t spi_xor_fingerprint_parallel(uint64_t seed, uint64_t n, uint32_t threads)
        \\    uint32_t spi_color_at(uint64_t seed, uint64_t index)
        \\    int8_t   spi_trit(uint64_t seed, uint64_t index)
        \\
    );
}
