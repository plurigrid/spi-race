const std = @import("std");

const GOLDEN: u64 = 0x9e3779b97f4a7c15;
const MIX1: u64 = 0xbf58476d1ce4e5b9;
const MIX2: u64 = 0x94d049bb133111eb;

inline fn splitmix64(seed: u64, index: u64) u64 {
    var z = seed +% (GOLDEN *% index);
    z = (z ^ (z >> 30)) *% MIX1;
    z = (z ^ (z >> 27)) *% MIX2;
    return z ^ (z >> 31);
}

inline fn extract_rgb(val: u64) u64 {
    return ((val >> 16) & 0xFF) << 16 | ((val >> 8) & 0xFF) << 8 | (val & 0xFF);
}

fn fused_xor_range(seed: u64, start: u64, count: u64) u64 {
    var a0: u64 = 0;
    var a1: u64 = 0;
    var a2: u64 = 0;
    var a3: u64 = 0;
    var b0: u64 = 0;
    var b1: u64 = 0;
    var b2: u64 = 0;
    var b3: u64 = 0;
    var i: u64 = 0;
    const n8 = count & ~@as(u64, 7);

    while (i < n8) : (i += 8) {
        const x = start +% i;
        a0 ^= extract_rgb(splitmix64(seed, x));
        a1 ^= extract_rgb(splitmix64(seed, x +% 1));
        a2 ^= extract_rgb(splitmix64(seed, x +% 2));
        a3 ^= extract_rgb(splitmix64(seed, x +% 3));
        b0 ^= extract_rgb(splitmix64(seed, x +% 4));
        b1 ^= extract_rgb(splitmix64(seed, x +% 5));
        b2 ^= extract_rgb(splitmix64(seed, x +% 6));
        b3 ^= extract_rgb(splitmix64(seed, x +% 7));
    }

    var result = a0 ^ a1 ^ a2 ^ a3 ^ b0 ^ b1 ^ b2 ^ b3;
    while (i < count) : (i += 1) {
        result ^= extract_rgb(splitmix64(seed, start +% i));
    }
    return result;
}

const WorkerCtx = struct {
    seed: u64,
    start: u64,
    count: u64,
    result: u64 = 0,
};

fn threadEntry(ctx: *WorkerCtx) void {
    ctx.result = fused_xor_range(ctx.seed, ctx.start, ctx.count);
}

// C ABI: single-threaded fused XOR fingerprint over [start, start+count)
export fn spi_xor_fingerprint(seed: u64, start: u64, count: u64) u64 {
    return fused_xor_range(seed, start, count);
}

// C ABI: multi-threaded fused XOR fingerprint over [0, n)
export fn spi_xor_fingerprint_parallel(seed: u64, n: u64, n_threads_req: u32) u64 {
    const cpu_count = std.Thread.getCpuCount() catch 4;
    const n_threads: usize = if (n_threads_req == 0) cpu_count else @min(@as(usize, n_threads_req), cpu_count);

    if (n_threads <= 1) return fused_xor_range(seed, 0, n);

    const alloc = std.heap.page_allocator;
    var contexts = alloc.alloc(WorkerCtx, n_threads) catch return fused_xor_range(seed, 0, n);
    defer alloc.free(contexts);
    var handles = alloc.alloc(std.Thread, n_threads) catch return fused_xor_range(seed, 0, n);
    defer alloc.free(handles);

    const chunk = n / n_threads;
    const remainder = n % n_threads;

    for (0..n_threads) |tid| {
        const s = chunk * tid + @min(tid, remainder);
        const c = chunk + @as(u64, if (tid < remainder) 1 else 0);
        contexts[tid] = .{ .seed = seed, .start = s, .count = c };
        handles[tid] = std.Thread.spawn(.{}, threadEntry, .{&contexts[tid]}) catch {
            contexts[tid].result = fused_xor_range(seed, s, c);
            continue;
        };
    }

    for (handles) |h| h.join();

    var combined: u64 = 0;
    for (contexts) |c| combined ^= c.result;
    return combined;
}

// C ABI: generate single color (R,G,B as packed u32: 0x00RRGGBB)
export fn spi_color_at(seed: u64, index: u64) u32 {
    return @truncate(extract_rgb(splitmix64(seed, index)));
}

// C ABI: GF(3) trit for a single color
export fn spi_trit(seed: u64, index: u64) i8 {
    const h = splitmix64(seed, index);
    const r: i32 = @intCast((h >> 16) & 0xFF);
    const g: i32 = @intCast((h >> 8) & 0xFF);
    const b: i32 = @intCast(h & 0xFF);
    const sum3 = @mod(r + g + b, @as(i32, 3));
    return @intCast(sum3 - 1);
}

// C ABI: GF(3) trit sum over [start, start+count), returned mod 3 centered to {-1,0,1}
export fn spi_trit_sum(seed: u64, start: u64, count: u64) i32 {
    var sum: i32 = 0;
    for (0..count) |i| {
        sum = @mod(sum + @as(i32, spi_trit(seed, start +% @as(u64, @intCast(i)))), 3);
    }
    return sum;
}
