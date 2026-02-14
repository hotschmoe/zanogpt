const std = @import("std");
const Backend = @import("backend.zig").Backend;
const cpu = @import("cpu.zig");
const intel_npu = @import("intel_npu.zig");
const intel_gpu = @import("intel_gpu.zig");
const log = std.log.scoped(.detect);

const DetectedHardware = struct {
    intel_npu: bool,
    intel_gpu: bool,
    amd_npu: bool,
    qualcomm_npu: bool,
};

fn probeLib(name: [:0]const u8) bool {
    var dll = std.DynLib.open(name) catch return false;
    dll.close();
    return true;
}

fn detectHardware() DetectedHardware {
    // DLL-only probes: calling zeInit() during detection is process-global and
    // poisons later backend init (e.g. zeInit(0) prevents VPU driver loading).
    // Each backend's init() handles its own zeInit with the correct flags.
    const has_l0 = probeLib("ze_loader.dll");
    return .{
        .intel_npu = has_l0,
        .intel_gpu = has_l0,
        .amd_npu = probeLib("xrt_coreutil.dll"),
        .qualcomm_npu = probeLib("QnnHtp.dll") or probeLib("libcdsprpc.dll"),
    };
}

const AccelChoice = enum { npu, gpu, cpu };

fn promptAccelerator(hw: *const DetectedHardware) AccelChoice {
    var options: [3]AccelChoice = undefined;
    var count: u8 = 0;
    if (hw.intel_npu) {
        options[count] = .npu;
        count += 1;
    }
    if (hw.intel_gpu) {
        options[count] = .gpu;
        count += 1;
    }
    options[count] = .cpu;
    count += 1;

    if (count == 1) return .cpu;

    const stdout = std.fs.File.stdout();
    _ = stdout.write("\nAvailable accelerators:\n") catch return .cpu;
    for (options[0..count], 1..) |opt, i| {
        var line_buf: [64]u8 = undefined;
        const line = switch (opt) {
            .npu => std.fmt.bufPrint(&line_buf, "  [{d}] Intel NPU\n", .{i}) catch continue,
            .gpu => std.fmt.bufPrint(&line_buf, "  [{d}] Intel GPU\n", .{i}) catch continue,
            .cpu => std.fmt.bufPrint(&line_buf, "  [{d}] CPU only\n", .{i}) catch continue,
        };
        _ = stdout.write(line) catch {};
    }

    _ = stdout.write("Select [1]: ") catch return .cpu;

    var buf: [16]u8 = undefined;
    const n = std.fs.File.stdin().read(&buf) catch return .cpu;
    if (n == 0 or buf[0] == '\r' or buf[0] == '\n' or buf[0] == ' ') return options[0];

    const sel = std.fmt.parseInt(u8, buf[0..1], 10) catch return options[0];
    if (sel >= 1 and sel <= count) return options[sel - 1];
    return options[0];
}

pub fn selectBackend() Backend {
    const hw = detectHardware();

    if (hw.amd_npu) {
        log.info("AMD NPU detected (not yet supported, using CPU)", .{});
    }
    if (hw.qualcomm_npu) {
        log.info("Qualcomm NPU detected (not yet supported, using CPU)", .{});
    }

    const choice = promptAccelerator(&hw);
    switch (choice) {
        .npu => {
            intel_npu.init() catch |err| {
                log.err("failed to initialize Intel NPU: {s}, falling back to CPU", .{@errorName(err)});
                return cpu.backend();
            };
            return intel_npu.backend();
        },
        .gpu => {
            intel_gpu.init() catch |err| {
                log.err("failed to initialize Intel GPU: {s}, falling back to CPU", .{@errorName(err)});
                return cpu.backend();
            };
            return intel_gpu.backend();
        },
        .cpu => return cpu.backend(),
    }
}
