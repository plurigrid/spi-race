const std = @import("std");

// SPI x Trit-Tick: precomputed divisor tables for all BCI device rates
// across 4 epochs, tested via SPI color-at-sample-boundary invariance.
//
// The contract: color_at(seed, sample_k * ticks_per_sample) must be
// identical regardless of which epoch you compute it in, provided
// both epochs divide the rate exactly.
//
// When a rate does NOT divide an epoch, the tick-per-sample is fractional
// and you get the WRONG sample boundary — and therefore the WRONG COLOR.
// This is the BCI equivalent of the "snow point" from spi-flicks.zig.

// --- SPI core (same as libspi.zig) ---
const GOLDEN: u64 = 0x9e3779b97f4a7c15;
const MIX1: u64 = 0xbf58476d1ce4e5b9;
const MIX2: u64 = 0x94d049bb133111eb;

inline fn splitmix64(seed: u64, index: u64) u64 {
    var z = seed +% (GOLDEN *% index);
    z = (z ^ (z >> 30)) *% MIX1;
    z = (z ^ (z >> 27)) *% MIX2;
    return z ^ (z >> 31);
}

inline fn extract_rgb(val: u64) u32 {
    return @truncate(((val >> 16) & 0xFF) << 16 | ((val >> 8) & 0xFF) << 8 | (val & 0xFF));
}

// --- Epoch definitions ---
const EPOCH1: u64 = 141_120_000;
const FLICK: u64 = 705_600_000;
const EPOCH2: u128 = 51_433_932_566_016_000_000;
const EPOCH3: u128 = 22_699_189_348_598_680_245_940_103_331_840_000_000;

// --- All 43 device rates from sync analysis ---
const DeviceRate = struct {
    hz: u64,
    name: []const u8,
    setup: []const u8,
};

const ALL_RATES = [_]DeviceRate{
    // BCI/Neural recording
    .{ .hz = 250,       .name = "OpenBCI Cyton EEG",       .setup = "rehab BCI" },
    .{ .hz = 500,       .name = "mBrainTrain/LiveAmp EEG", .setup = "VR+EEG / audio" },
    .{ .hz = 2000,      .name = "EyeLink 1000+",           .setup = "eye tracking" },
    .{ .hz = 2048,      .name = "BioSemi ActiveTwo EEG",   .setup = "MoBI / ECoG" },
    .{ .hz = 2500,      .name = "Neuropixels LFP",         .setup = "Neuropixels" },
    .{ .hz = 4800,      .name = "g.tec g.USBamp EEG",      .setup = "auditory BCI" },
    .{ .hz = 5000,      .name = "BrainProducts actiCHamp", .setup = "EEG+fMRI / TMS" },
    .{ .hz = 30000,     .name = "Neuropixels AP",           .setup = "Neuropixels" },
    // DBS / stimulation (biological constants)
    .{ .hz = 130,       .name = "Medtronic DBS (PD)",       .setup = "DBS" },
    .{ .hz = 185,       .name = "Boston Sci DBS (dystonia)", .setup = "DBS" },
    .{ .hz = 300,       .name = "Blackrock cortical stim",  .setup = "ECoG" },
    // EMG (sigma-delta remnant)
    .{ .hz = 1926,      .name = "Delsys Trigno EMG",        .setup = "rehab / MoBI" },
    // Cameras (sensor geometry primes)
    .{ .hz = 227,       .name = "FLIR Blackfly (prime!)",   .setup = "Neuropixels" },
    .{ .hz = 200,       .name = "Pupil Labs / DeepLabCut",  .setup = "various" },
    .{ .hz = 240,       .name = "Xsens MVN mocap",          .setup = "MoBI" },
    .{ .hz = 360,       .name = "OptiTrack mocap",          .setup = "MoBI" },
    .{ .hz = 1200,      .name = "Tobii Pro Spectrum",       .setup = "eye tracking" },
    // Physiology
    .{ .hz = 7,         .name = "NIRx fNIRS",               .setup = "rehab BCI" },
    .{ .hz = 10,        .name = "fNIRS / BCI decoder",      .setup = "various" },
    .{ .hz = 100,       .name = "fMRI physio / accel",      .setup = "various" },
    .{ .hz = 148,       .name = "goniometer",               .setup = "rehab BCI" },
    .{ .hz = 400,       .name = "Siemens PULS/RESP",        .setup = "EEG+fMRI" },
    .{ .hz = 1000,      .name = "force plate / Harp",       .setup = "various" },
    .{ .hz = 10000,     .name = "SyncBox TTL / wheel",      .setup = "various" },
    // Audio
    .{ .hz = 44100,     .name = "CD audio",                 .setup = "hyperscanning" },
    .{ .hz = 48000,     .name = "DAC audio",                .setup = "auditory BCI" },
    .{ .hz = 96000,     .name = "hi-res audio",             .setup = "audio+EEG" },
    .{ .hz = 100000,    .name = "photodiode",               .setup = "audio+EEG" },
    .{ .hz = 11289600,  .name = "DSD256 audio",             .setup = "hi-res audio" },
    .{ .hz = 22579200,  .name = "DSD512 audio",             .setup = "hi-res audio" },
    // Video / display
    .{ .hz = 24,        .name = "cinema",                   .setup = "various" },
    .{ .hz = 24000,     .name = "NTSC base",                .setup = "hi-res audio" },
    .{ .hz = 30,        .name = "video / miniscope",        .setup = "various" },
    .{ .hz = 50,        .name = "Unity physics / stim",     .setup = "VR+EEG" },
    .{ .hz = 60,        .name = "display / neuronavigation",.setup = "TMS / various" },
    .{ .hz = 90,        .name = "Quest Pro VR",             .setup = "VR+EEG" },
    .{ .hz = 120,       .name = "Bonsai video",             .setup = "Neuropixels" },
    .{ .hz = 144,       .name = "gaming monitor",           .setup = "haptic BCI" },
    // Misc
    .{ .hz = 1,         .name = "TMS single pulse",         .setup = "TMS+EEG" },
    .{ .hz = 5,         .name = "BCI decoder / classifier", .setup = "various" },
    .{ .hz = 20,        .name = "BCI decoder output",       .setup = "haptic BCI" },
    .{ .hz = 32000,     .name = "Harp timestamp board",     .setup = "Neuropixels" },
    .{ .hz = 40,        .name = "ASSR modulation",          .setup = "auditory BCI" },
    .{ .hz = 20000,     .name = "optogenetic laser",        .setup = "Neuropixels" },
};

// --- Precomputed divisor table ---
// For each rate and each epoch: does the epoch divide exactly?
// If yes: ticks_per_sample is an exact integer.
// If no:  ticks_per_sample is fractional => sample boundaries drift.
const EpochCoverage = struct {
    rate: u64,
    flick_exact: bool,
    epoch1_exact: bool,
    epoch2_exact: bool,
    epoch3_exact: bool,
    flick_tps: u64,   // ticks per sample (0 if inexact)
    epoch1_tps: u64,
};

fn precompute(rate: u64) EpochCoverage {
    return .{
        .rate = rate,
        .flick_exact = FLICK % rate == 0,
        .epoch1_exact = EPOCH1 % rate == 0,
        .epoch2_exact = EPOCH2 % @as(u128, rate) == 0,
        .epoch3_exact = EPOCH3 % @as(u128, rate) == 0,
        .flick_tps = if (FLICK % rate == 0) FLICK / rate else 0,
        .epoch1_tps = if (EPOCH1 % rate == 0) EPOCH1 / rate else 0,
    };
}

// --- Epoch upgrade lossless verification ---
// Verifies the retroactive cascade: epoch N+1 tick = epoch N tick × exact factor.
// If this holds, ALL prior timestamps remain valid at higher resolution.
const FACTOR_FLICK_TO_E2: u128 = EPOCH2 / @as(u128, FLICK);
const FACTOR_E2_TO_E3: u128 = EPOCH3 / EPOCH2;

fn race_epoch_upgrade(rate: u64, n_samples: u64) struct {
    flick_ok: bool,
    e2_ok: bool,
    e3_ok: bool,
    flick_e2_lossless: bool,
    e2_e3_lossless: bool,
    min_epoch: u8, // 0=flick, 2=E2, 3=E3, 4=unbounded
} {
    const f_ok = FLICK % rate == 0;
    const e2_ok = EPOCH2 % @as(u128, rate) == 0;
    const e3_ok = EPOCH3 % @as(u128, rate) == 0;
    var f_e2 = true;
    var e2_e3 = true;
    // Cap samples to avoid u128 overflow: k * (EPOCH/rate) <= EPOCH when k <= rate
    const safe_n = @min(n_samples, rate);

    if (f_ok and e2_ok) {
        const tps_f = @as(u128, FLICK / rate);
        const tps_2 = EPOCH2 / @as(u128, rate);
        var k: u64 = 0;
        while (k < safe_n) : (k += 1) {
            if (@as(u128, k) * tps_2 != @as(u128, k) * tps_f * FACTOR_FLICK_TO_E2) {
                f_e2 = false;
                break;
            }
        }
    }
    if (e2_ok and e3_ok) {
        const tps_2 = EPOCH2 / @as(u128, rate);
        const tps_3 = EPOCH3 / @as(u128, rate);
        var k: u64 = 0;
        while (k < safe_n) : (k += 1) {
            if (@as(u128, k) * tps_3 != @as(u128, k) * tps_2 * FACTOR_E2_TO_E3) {
                e2_e3 = false;
                break;
            }
        }
    }
    return .{
        .flick_ok = f_ok,
        .e2_ok = e2_ok,
        .e3_ok = e3_ok,
        .flick_e2_lossless = f_e2,
        .e2_e3_lossless = e2_e3,
        .min_epoch = if (f_ok) 0 else if (e2_ok) 2 else if (e3_ok) 3 else 4,
    };
}

fn min_epoch_for_rates(rates: []const u64) u8 {
    var need: u8 = 0;
    for (rates) |rate| {
        if (EPOCH3 % @as(u128, rate) != 0) return 4;
        if (EPOCH2 % @as(u128, rate) != 0) { need = 3; continue; }
        if (FLICK % rate != 0) { if (need < 2) need = 2; }
    }
    return need;
}

// --- The SPI race: generate colors at sample boundaries across epochs ---
// If two epochs both handle a rate exactly, the color at sample k
// must be derivable from either epoch's tick space.
// The "race" is: compute the same sample boundary two ways, verify same color.

fn race_one_rate(seed: u64, rate: u64, n_samples: u64) struct { ok: bool, mismatches: u64 } {
    const cov = precompute(rate);
    if (!cov.flick_exact or !cov.epoch1_exact) {
        // Can't race if one epoch doesn't handle it
        return .{ .ok = true, .mismatches = 0 };
    }

    var mismatches: u64 = 0;
    var k: u64 = 0;
    while (k < n_samples) : (k += 1) {
        // Flick-space index for sample k
        const flick_idx = k * cov.flick_tps;
        // Epoch1-space index for sample k, then convert to flick space
        // epoch1_tick * 5 = flick (since 1 trit-tick = 5 flicks)
        const epoch1_idx = k * cov.epoch1_tps * 5;

        const c_flick = extract_rgb(splitmix64(seed, flick_idx));
        const c_epoch1 = extract_rgb(splitmix64(seed, epoch1_idx));

        if (c_flick != c_epoch1) mismatches += 1;
    }
    return .{ .ok = mismatches == 0, .mismatches = mismatches };
}

// --- Snow point for BCI: when does f64 accumulated sample time
//     diverge from the integer tick boundary? ---
fn bci_snow_point(seed: u64, rate: u64, duration_s: u64) struct {
    first_drift_sample: u64,
    total_drifted: u64,
    max_delta_ticks: u64,
    wrong_colors: u64,
} {
    const sample_sec: f64 = 1.0 / @as(f64, @floatFromInt(rate));
    var accum: f64 = 0.0;
    var first_drift: u64 = 0;
    var total_drifted: u64 = 0;
    var max_delta: u64 = 0;
    var wrong_colors: u64 = 0;
    const n_samples = duration_s * rate;

    var k: u64 = 0;
    while (k < n_samples) : (k += 1) {
        accum += sample_sec;
        const float_tick: u64 = @intFromFloat(accum * @as(f64, @floatFromInt(FLICK)));
        const int_tick: u64 = (k + 1) * (FLICK / rate);

        if (FLICK % rate != 0) break; // can't test inexact rates this way

        if (float_tick != int_tick) {
            total_drifted += 1;
            const delta = if (float_tick > int_tick) float_tick - int_tick else int_tick - float_tick;
            if (delta > max_delta) max_delta = delta;
            if (first_drift == 0) first_drift = k + 1;

            // Check if drift causes wrong color
            const c_float = extract_rgb(splitmix64(seed, float_tick));
            const c_integer = extract_rgb(splitmix64(seed, int_tick));
            if (c_float != c_integer) wrong_colors += 1;
        }
    }
    return .{
        .first_drift_sample = first_drift,
        .total_drifted = total_drifted,
        .max_delta_ticks = max_delta,
        .wrong_colors = wrong_colors,
    };
}

fn writeAll(buf: []const u8) void {
    _ = std.posix.write(std.posix.STDOUT_FILENO, buf) catch {};
}

fn p(comptime fmt: []const u8, args: anytype) void {
    var buf: [512]u8 = undefined;
    const slice = std.fmt.bufPrint(&buf, fmt, args) catch &buf;
    writeAll(slice);
}

pub fn main() !void {
    const SEED: u64 = 1069; // monaduck

    writeAll(
        \\
        \\+======================================================================+
        \\|  SPI x Trit-Tick — Precomputed Divisor Tables & BCI Device Races     |
        \\|  43 device rates × 4 epochs, sample-boundary color invariance        |
        \\+======================================================================+
        \\
    );

    // ================================================================
    // SECTION 1: Precomputed divisor table
    // ================================================================
    writeAll("  PRECOMPUTED DIVISOR TABLE\n");
    writeAll("  Rate Hz       Device                    flick   E1    E2    E3   tps(flick)     tps(E1)\n");
    writeAll("  -----------   ------------------------  -----  -----  ----  ----  ----------  ----------\n");

    var n_flick_exact: u32 = 0;
    var n_e1_exact: u32 = 0;
    var n_e2_exact: u32 = 0;
    var n_e3_exact: u32 = 0;

    for (ALL_RATES) |rate| {
        const cov = precompute(rate.hz);
        if (cov.flick_exact) n_flick_exact += 1;
        if (cov.epoch1_exact) n_e1_exact += 1;
        if (cov.epoch2_exact) n_e2_exact += 1;
        if (cov.epoch3_exact) n_e3_exact += 1;

        p("  {: >10}   {s: <24}  {s: >5}  {s: >5}  {s: >4}  {s: >4}  {: >10}  {: >10}\n", .{
            rate.hz,
            rate.name,
            if (cov.flick_exact) "  OK " else " FAIL",
            if (cov.epoch1_exact) "  OK " else " FAIL",
            if (cov.epoch2_exact) " OK " else "FAIL",
            if (cov.epoch3_exact) " OK " else "FAIL",
            cov.flick_tps,
            cov.epoch1_tps,
        });
    }

    p("\n  Coverage: flick={}/{} E1={}/{} E2={}/{} E3={}/{}\n", .{
        n_flick_exact, ALL_RATES.len,
        n_e1_exact,    ALL_RATES.len,
        n_e2_exact,    ALL_RATES.len,
        n_e3_exact,    ALL_RATES.len,
    });

    // ================================================================
    // SECTION 2: SPI Race — epoch cross-validation
    // ================================================================
    writeAll(
        \\
        \\  SPI RACE: EPOCH CROSS-VALIDATION
        \\  For rates that both flick and epoch 1 handle exactly,
        \\  color at sample boundary must match across both tick spaces.
        \\
    );

    var race_pass: u32 = 0;
    var race_skip: u32 = 0;
    var race_fail: u32 = 0;

    for (ALL_RATES) |rate| {
        const cov = precompute(rate.hz);
        if (!cov.flick_exact or !cov.epoch1_exact) {
            race_skip += 1;
            continue;
        }
        const result = race_one_rate(SEED, rate.hz, @min(rate.hz, 10000));
        if (result.ok) {
            race_pass += 1;
        } else {
            race_fail += 1;
            p("  FAIL: {} Hz ({s}) — {} mismatches\n", .{ rate.hz, rate.name, result.mismatches });
        }
    }
    p("  Race results: {} PASS, {} SKIP (epoch mismatch), {} FAIL\n", .{
        race_pass, race_skip, race_fail,
    });

    // ================================================================
    // SECTION 2b: EPOCH UPGRADE — LOSSLESS RETROACTIVE CASCADE
    // For each rate: verify flick_tick × FACTOR = epoch2_tick (exact),
    // and epoch2_tick × FACTOR = epoch3_tick (exact).
    // This is the core invariant: every prior timestamp remains valid
    // at higher resolution with zero migration cost.
    // ================================================================
    writeAll(
        \\
        \\  EPOCH UPGRADE: LOSSLESS RETROACTIVE CASCADE
        \\  flick_tick x 72,893,895,360 = epoch2_tick (exact for shared rates)
        \\  epoch2_tick x FACTOR = epoch3_tick (exact for shared rates)
        \\
    );

    var upgrade_pass: u32 = 0;
    var upgrade_fail: u32 = 0;
    var n_flick_ok: u32 = 0;
    var n_need_e2: u32 = 0;
    var n_need_e3: u32 = 0;
    var n_need_ub: u32 = 0;

    writeAll("  Rate Hz       Device                    Min     F->E2   E2->E3\n");
    writeAll("  -----------   ------------------------  ------  ------  ------\n");

    for (ALL_RATES) |rate| {
        const r = race_epoch_upgrade(rate.hz, 1000);
        const me: []const u8 = switch (r.min_epoch) {
            0 => "flick ",
            2 => "E2    ",
            3 => "E3    ",
            else => "unbnd ",
        };
        const f_e2: []const u8 = if (!r.flick_ok or !r.e2_ok) " n/a " else if (r.flick_e2_lossless) " PASS" else " FAIL";
        const e23: []const u8 = if (!r.e2_ok or !r.e3_ok) " n/a " else if (r.e2_e3_lossless) " PASS" else " FAIL";

        p("  {: >10}   {s: <24}  {s}  {s}   {s}\n", .{
            rate.hz, rate.name, me, f_e2, e23,
        });

        const any_fail = (r.flick_ok and r.e2_ok and !r.flick_e2_lossless) or
            (r.e2_ok and r.e3_ok and !r.e2_e3_lossless);
        if (any_fail) upgrade_fail += 1 else upgrade_pass += 1;
        switch (r.min_epoch) {
            0 => n_flick_ok += 1,
            2 => n_need_e2 += 1,
            3 => n_need_e3 += 1,
            else => n_need_ub += 1,
        }
    }

    p("\n  Upgrade race: {} PASS, {} FAIL\n", .{ upgrade_pass, upgrade_fail });
    p("  Min epoch: flick={} E2={} E3={} unbounded={}\n", .{
        n_flick_ok, n_need_e2, n_need_e3, n_need_ub,
    });

    // ================================================================
    // SECTION 2c: EXPERIMENTAL SETUP EPOCH REQUIREMENTS
    // For each of the 15 real experimental setups from the sync analysis,
    // find the minimum epoch that handles ALL device rates simultaneously.
    // ================================================================
    writeAll(
        \\
        \\  EXPERIMENTAL SETUP: MINIMUM EPOCH PER REAL-WORLD RIG
        \\  15 setups from sync analysis — which epoch handles ALL rates?
        \\
    );

    const setups = [_]struct { name: []const u8, rates: []const u64 }{
        .{ .name = "Rehab BCI",            .rates = &[_]u64{ 250, 7, 148, 100, 1000 } },
        .{ .name = "VR + EEG",             .rates = &[_]u64{ 500, 90, 50, 2000, 1000 } },
        .{ .name = "Neuropixels + FLIR",    .rates = &[_]u64{ 30000, 2500, 227, 200, 32000, 1000 } },
        .{ .name = "DBS + LFP + EEG",      .rates = &[_]u64{ 250, 130, 185, 4800, 100 } },
        .{ .name = "EEG + fMRI",            .rates = &[_]u64{ 5000, 400, 100, 1000 } },
        .{ .name = "Auditory BCI",          .rates = &[_]u64{ 4800, 48000, 40, 500 } },
        .{ .name = "Audio + EEG",           .rates = &[_]u64{ 96000, 100000, 500 } },
        .{ .name = "Hyperscanning",         .rates = &[_]u64{ 2048, 44100, 60 } },
        .{ .name = "TMS + EEG",             .rates = &[_]u64{ 5000, 1, 1000 } },
        .{ .name = "MoBI (hardest fixed)",  .rates = &[_]u64{ 2048, 240, 360, 1926, 2000, 1000 } },
        .{ .name = "ECoG + stimulation",    .rates = &[_]u64{ 300, 2048, 2000 } },
        .{ .name = "Neuropixels + Bonsai",  .rates = &[_]u64{ 30000, 120, 32000, 20000 } },
        .{ .name = "VR + EEG + eye",        .rates = &[_]u64{ 500, 90, 2000, 1200 } },
        .{ .name = "Hi-res audio",          .rates = &[_]u64{ 11289600, 22579200, 24000 } },
        .{ .name = "Haptic BCI",            .rates = &[_]u64{ 20, 144, 500, 1000 } },
    };

    var se_flick: u32 = 0;
    var se_e2: u32 = 0;
    var se_e3: u32 = 0;
    var se_ub: u32 = 0;

    for (setups) |s| {
        const me = min_epoch_for_rates(s.rates);
        const tag: []const u8 = switch (me) {
            0 => "flick (u64) ",
            2 => "E2 (u128)   ",
            3 => "E3 (u128)   ",
            else => "UNBOUNDED   ",
        };
        switch (me) {
            0 => se_flick += 1,
            2 => se_e2 += 1,
            3 => se_e3 += 1,
            else => se_ub += 1,
        }
        // Find the rate that forces this epoch
        var bottleneck: u64 = 0;
        if (me > 0) {
            for (s.rates) |rate| {
                const rm: u8 = if (FLICK % rate == 0) 0
                    else if (EPOCH2 % @as(u128, rate) == 0) 2
                    else if (EPOCH3 % @as(u128, rate) == 0) 3
                    else 4;
                if (rm == me) {
                    bottleneck = rate;
                    break;
                }
            }
        }
        if (bottleneck > 0) {
            p("  {s: <24} -> {s} (bottleneck: {} Hz)\n", .{ s.name, tag, bottleneck });
        } else {
            p("  {s: <24} -> {s}\n", .{ s.name, tag });
        }
    }

    p("\n  Setup distribution: flick={} E2={} E3={} unbounded={}\n", .{
        se_flick, se_e2, se_e3, se_ub,
    });

    // ================================================================
    // SECTION 3: BCI Snow Points — where f64 drift causes wrong colors
    // ================================================================
    writeAll(
        \\
        \\  BCI SNOW POINTS (f64 drift → wrong color, 1 hour recording)
        \\  The BCI equivalent of the NTSC snow point from spi-flicks.zig.
        \\
    );

    const snow_rates = [_]struct { hz: u64, name: []const u8 }{
        .{ .hz = 250,   .name = "OpenBCI Cyton (250 Hz)  " },
        .{ .hz = 500,   .name = "LiveAmp EEG (500 Hz)    " },
        .{ .hz = 2048,  .name = "BioSemi (2048 Hz)       " },
        .{ .hz = 5000,  .name = "actiCHamp+ (5000 Hz)    " },
        .{ .hz = 30000, .name = "Neuropixels AP (30 kHz) " },
        .{ .hz = 44100, .name = "CD audio (44.1 kHz)     " },
        .{ .hz = 48000, .name = "DAC audio (48 kHz)      " },
    };

    writeAll("  Device                       1st drift    Total drift   Max delta     Wrong colors\n");
    writeAll("  ---------------------------  ----------  ------------  -----------  ------------\n");

    for (snow_rates) |sr| {
        if (FLICK % sr.hz != 0) continue;
        const sp = bci_snow_point(SEED, sr.hz, 3600);
        p("  {s}  {: >10}  {: >12}  {: >8} flk  {: >12}\n", .{
            sr.name,
            sp.first_drift_sample,
            sp.total_drifted,
            sp.max_delta_ticks,
            sp.wrong_colors,
        });
    }

    // ================================================================
    // SECTION 4: Cross-modal alignment race
    // Two devices recording simultaneously: do their sample boundaries
    // ever land on the same tick? How often?
    // ================================================================
    writeAll(
        \\
        \\  CROSS-MODAL ALIGNMENT RACE
        \\  For device pairs sharing a common epoch, compute how often
        \\  their sample boundaries coincide (= alignment events).
        \\
    );

    const pairs = [_]struct { a: u64, b: u64, na: []const u8, nb: []const u8 }{
        .{ .a = 250,   .b = 2048,  .na = "Cyton EEG 250",   .nb = "BioSemi 2048" },
        .{ .a = 250,   .b = 44100, .na = "Cyton EEG 250",   .nb = "CD audio 44100" },
        .{ .a = 2048,  .b = 30000, .na = "BioSemi 2048",    .nb = "Neuropixels 30k" },
        .{ .a = 44100, .b = 48000, .na = "CD audio 44100",  .nb = "DAC audio 48000" },
        .{ .a = 250,   .b = 130,   .na = "Cyton EEG 250",   .nb = "DBS 130 Hz" },
        .{ .a = 90,    .b = 500,   .na = "Quest Pro 90fps",  .nb = "EEG 500 Hz" },
    };

    for (pairs) |pair| {
        // GCD = alignment rate, LCM = combined tick space
        const g = std.math.gcd(pair.a, pair.b);
        const l = pair.a / g * pair.b;
        const aligns_per_sec = g;
        const align_period_ms = 1000.0 / @as(f64, @floatFromInt(g));

        // Check if both divide flick
        const a_ok = FLICK % pair.a == 0;
        const b_ok = FLICK % pair.b == 0;

        p("  {s: <20} x {s: <20} GCD={: >6} LCM={: >12} align={: >5}/s ({d:.2}ms) flick:{s}\n", .{
            pair.na, pair.nb,
            aligns_per_sec, l,
            aligns_per_sec, align_period_ms,
            if (a_ok and b_ok) "both OK" else "NEEDS EPOCH 2+",
        });
    }

    // ================================================================
    // SECTION 5: Embarrassingly parallel partition across BCI samples
    // Same XOR-fingerprint property as spi-flicks, but in tick space.
    // ================================================================
    writeAll(
        \\
        \\  EMBARRASSINGLY PARALLEL: 1 second of Neuropixels at 30 kHz
        \\  Partition 30000 samples across workers, XOR must agree.
        \\
    );

    const NP_TPS: u64 = FLICK / 30000; // ticks per sample in flick space
    var full_xor: u64 = 0;
    for (0..30000) |k| {
        full_xor ^= extract_rgb(splitmix64(SEED, @as(u64, k) * NP_TPS));
    }
    p("  Full (30000 samples):  0x{x:0>6}\n", .{full_xor});

    // 6-way split (5000 samples each)
    var parts: [6]u64 = .{ 0, 0, 0, 0, 0, 0 };
    for (0..30000) |k| {
        parts[k / 5000] ^= extract_rgb(splitmix64(SEED, @as(u64, k) * NP_TPS));
    }
    var combined: u64 = 0;
    for (parts) |pp| combined ^= pp;
    p("  6-split (5000 each):   0x{x:0>6} {s}\n", .{
        combined, if (combined == full_xor) "PASS" else "FAIL",
    });

    // 30-way split (1000 samples each)
    var parts30: [30]u64 = [_]u64{0} ** 30;
    for (0..30000) |k| {
        parts30[k / 1000] ^= extract_rgb(splitmix64(SEED, @as(u64, k) * NP_TPS));
    }
    var combined30: u64 = 0;
    for (parts30) |pp| combined30 ^= pp;
    p("  30-split (1000 each):  0x{x:0>6} {s}\n", .{
        combined30, if (combined30 == full_xor) "PASS" else "FAIL",
    });

    // ================================================================
    // SECTION 6: Unbounded epoch — registerRate for holdout primes
    // Demonstrates the monzo-style prime extension for 227 and 1926
    // ================================================================
    writeAll(
        \\
        \\  UNBOUNDED EPOCH: REGISTERING HOLDOUT PRIMES
        \\  Rates 227 (FLIR, prime) and 1926 (Delsys, 107 prime) break
        \\  all fixed epochs. The unbounded epoch absorbs them at zero cost.
        \\
    );

    // Simulate the unbounded prime basis registration
    // (Can't import trit_tick.zig directly — different project — so inline the logic)
    const epoch3_primes = [16]u16{ 2, 3, 5, 7, 11, 13, 17, 19, 29, 37, 43, 89, 113, 127, 151, 233 };
    var basis_len: u8 = 16;
    var basis: [32]u16 = [_]u16{0} ** 32;
    for (epoch3_primes, 0..) |pr, i| basis[i] = pr;

    const holdouts = [_]struct { rate: u64, name: []const u8 }{
        .{ .rate = 227,  .name = "FLIR Blackfly 227 Hz (prime 227)" },
        .{ .rate = 1926, .name = "Delsys Trigno 1926 Hz (prime 107)" },
        .{ .rate = 508,  .name = "4D/BTi MEG 508 Hz (no new prime)" },
        .{ .rate = 1597, .name = "future Fibonacci prime 1597" },
    };

    for (holdouts) |h| {
        var r = h.rate;
        var new_primes: u8 = 0;
        var d: u64 = 2;
        while (d * d <= r) : (d += 1) {
            while (r % d == 0) {
                r /= d;
                var found = false;
                for (basis[0..basis_len]) |bp| {
                    if (bp == @as(u16, @intCast(d))) { found = true; break; }
                }
                if (!found and basis_len < 32) {
                    basis[basis_len] = @intCast(d);
                    basis_len += 1;
                    new_primes += 1;
                }
            }
        }
        if (r > 1) {
            var found = false;
            for (basis[0..basis_len]) |bp| {
                if (bp == @as(u16, @intCast(r))) { found = true; break; }
            }
            if (!found and basis_len < 32) {
                basis[basis_len] = @intCast(r);
                basis_len += 1;
                new_primes += 1;
            }
        }
        p("  registerRate({: >5}) — {s}: +{} new primes, basis now {}\n", .{
            h.rate, h.name, new_primes, basis_len,
        });
    }

    // ================================================================
    // SECTION 7: The 4-mechanism summary
    // ================================================================
    // ================================================================
    // SECTION 8: Clock Domain Stride Verification
    // Proves golden-stride folding is valid for all E1 rates.
    // See spi-metal-clocks.swift for GPU proof (flick 91.3 B/s > dense 74.8 B/s).
    // ================================================================
    writeAll(
        \\
        \\  CLOCK DOMAIN STRIDES — folded golden-stride verification
        \\  GOLDEN * (TICKS/rate) precomputed → same ALU as dense color indices.
        \\  GPU result: flick stride BEATS dense by 22% at 500M samples.
        \\
    );

    const clock_rates = [_]struct { hz: u64, name: []const u8 }{
        .{ .hz = 250,   .name = "OpenBCI 250 Hz  " },
        .{ .hz = 500,   .name = "LiveAmp 500 Hz  " },
        .{ .hz = 2500,  .name = "NPX LFP 2500 Hz" },
        .{ .hz = 5000,  .name = "actiCHamp 5 kHz " },
        .{ .hz = 30000, .name = "NPX AP 30 kHz   " },
        .{ .hz = 48000, .name = "Audio 48 kHz    " },
    };

    p("  {s: <18} {s: >12} {s: >12} {s: >6}  {s}\n", .{
        "Device", "Flick stride", "Trit stride", "Ratio", "Fold OK",
    });
    p("  {s: <18} {s: >12} {s: >12} {s: >6}  {s}\n", .{
        "──────────────────", "────────────", "────────────", "──────", "───────",
    });

    var clock_pass: u32 = 0;
    var clock_total: u32 = 0;
    for (clock_rates) |cr| {
        clock_total += 1;
        const flick_stride = FLICK / cr.hz;
        const trit_stride = EPOCH1 / cr.hz;
        const ratio_ok = (flick_stride == trit_stride * 5);

        // Verify golden-stride fold: GOLDEN*(TICKS/rate)*k == GOLDEN*(k*TICKS/rate) for k=0..7
        const gs_flick = GOLDEN *% flick_stride;
        const gs_trit = GOLDEN *% trit_stride;
        var fold_ok = true;
        const seed: u64 = 42;
        var k: u64 = 0;
        while (k < 8) : (k += 1) {
            const direct_f = seed +% GOLDEN *% (flick_stride * k);
            const folded_f = seed +% gs_flick *% k;
            const direct_t = seed +% GOLDEN *% (trit_stride * k);
            const folded_t = seed +% gs_trit *% k;
            if (direct_f != folded_f or direct_t != folded_t) fold_ok = false;
        }
        if (ratio_ok and fold_ok) clock_pass += 1;
        const status = if (ratio_ok and fold_ok) "PASS" else "FAIL";
        p("  {s: <18} {d: >12} {d: >12} {s: >6}  {s}\n", .{
            cr.name, flick_stride, trit_stride,
            if (ratio_ok) "5:1" else "???",
            status,
        });
    }
    p("\n  Clock domain strides: {}/{} PASS\n", .{ clock_pass, clock_total });

    // ================================================================
    // SECTION 9: The 4-mechanism summary
    // ================================================================
    writeAll(
        \\
        \\  WHY UNUSUAL PRIMES APPEAR IN DEVICE SAMPLE RATES
        \\  ================================================
        \\  1. SENSOR GEOMETRY: pixel_clock / (rows x cols + blanking)
        \\     → primes from silicon die layout (103, 113, 227)
        \\  2. BIOLOGICAL RESONANCE: DBS/ASSR chosen for brain effect
        \\     → primes from neuroscience (13 from 130 Hz, 37 from 185 Hz)
        \\  3. ADC CLOCK DIVIDERS: sigma-delta decimation chain remnants
        \\     → primes from arithmetic (107 from Delsys 1926 Hz)
        \\  4. TRANSPORT BANDWIDTH: USB throughput / frame size ceiling
        \\     → primes that change per configuration
        \\
        \\  The universe of device primes is OPEN, not closed.
        \\  The only correct time base can absorb new primes at zero cost.
        \\
        \\  EPOCH LADDER:
        \\    Epoch 1 (u64):  fast path, 80% of devices
        \\    Epoch 2 (u128): +DBS, +EEG 2048, +DSD audio = 95%
        \\    Epoch 3 (u128): +Fibonacci/Padovan closure
        \\    Unbounded:      monzo-style prime vectors, 100% forever
        \\
    );
}
