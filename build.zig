const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Shared library: libspi.dylib / libspi.so
    const lib = b.addLibrary(.{
        .name = "spi",
        .root_module = b.createModule(.{
            .root_source_file = b.path("libspi.zig"),
            .target = target,
            .optimize = optimize,
        }),
        .linkage = .dynamic,
    });
    b.installArtifact(lib);

    // Static library: libspi.a
    const static_lib = b.addLibrary(.{
        .name = "spi_static",
        .root_module = b.createModule(.{
            .root_source_file = b.path("libspi.zig"),
            .target = target,
            .optimize = optimize,
        }),
        .linkage = .static,
    });
    b.installArtifact(static_lib);

    // Test runner — links against the static lib
    const test_exe = b.addExecutable(.{
        .name = "spi-test",
        .root_module = b.createModule(.{
            .root_source_file = b.path("spi-test.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_exe.linkLibrary(static_lib);
    b.installArtifact(test_exe);
    const run_test = b.addRunArtifact(test_exe);
    const test_step = b.step("test", "Run SPI tests and benchmark");
    test_step.dependOn(&run_test.step);

    // Flicks benchmark
    const flicks_exe = b.addExecutable(.{
        .name = "spi-flicks",
        .root_module = b.createModule(.{
            .root_source_file = b.path("spi-flicks.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(flicks_exe);
    const run_flicks = b.addRunArtifact(flicks_exe);
    const flicks_step = b.step("flicks", "Run SPI x Flicks invariance test");
    flicks_step.dependOn(&run_flicks.step);

    // Vibe-kanban: CatColab theories at SPI color bandwidth
    // Trit-tick precomputation + BCI device race
    const tt_exe = b.addExecutable(.{
        .name = "spi-trit-tick",
        .root_module = b.createModule(.{
            .root_source_file = b.path("spi-trit-tick.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(tt_exe);
    const run_tt = b.addRunArtifact(tt_exe);
    const tt_step = b.step("trit-tick", "Run SPI x Trit-Tick precomputation and BCI device races");
    tt_step.dependOn(&run_tt.step);

    const vk_lib = b.addLibrary(.{
        .name = "vibe_kanban",
        .root_module = b.createModule(.{
            .root_source_file = b.path("vibe_kanban.zig"),
            .target = target,
            .optimize = optimize,
        }),
        .linkage = .dynamic,
    });
    b.installArtifact(vk_lib);

    const vk_static = b.addLibrary(.{
        .name = "vibe_kanban_static",
        .root_module = b.createModule(.{
            .root_source_file = b.path("vibe_kanban.zig"),
            .target = target,
            .optimize = optimize,
        }),
        .linkage = .static,
    });
    b.installArtifact(vk_static);

    const vk_exe = b.addExecutable(.{
        .name = "vibe-kanban",
        .root_module = b.createModule(.{
            .root_source_file = b.path("vibe_kanban.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(vk_exe);
    const run_vk = b.addRunArtifact(vk_exe);
    const vk_step = b.step("vibe-kanban", "Run vibe-kanban CatColab x SPI demo");
    vk_step.dependOn(&run_vk.step);
}
