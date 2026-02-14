const std = @import("std");
const Backend = @import("backend.zig").Backend;
const cpu = @import("cpu.zig");
const intel_npu = @import("intel_npu.zig");
const log = std.log.scoped(.detect);

const DetectedHardware = struct {
    intel_npu: bool,
    amd_npu: bool,
    qualcomm_npu: bool,
};

fn probeLib(name: [:0]const u8) bool {
    var dll = std.DynLib.open(name) catch return false;
    dll.close();
    return true;
}

fn detectHardware() DetectedHardware {
    return .{
        .intel_npu = probeLib("ze_loader.dll"),
        .amd_npu = probeLib("xrt_coreutil.dll"),
        .qualcomm_npu = probeLib("QnnHtp.dll") or probeLib("libcdsprpc.dll"),
    };
}

fn promptIntelNpu() bool {
    const stdout_file = std.fs.File.stdout();
    const stdin_file = std.fs.File.stdin();
    _ = stdout_file.write("Intel NPU detected. Use for acceleration? [y/n]: ") catch return false;
    var buf: [16]u8 = undefined;
    const n = stdin_file.read(&buf) catch return false;
    if (n == 0) return false;
    return buf[0] == 'y' or buf[0] == 'Y';
}

pub fn selectBackend() Backend {
    const hw = detectHardware();

    if (hw.amd_npu) {
        log.info("AMD NPU detected (not yet supported, using CPU)", .{});
    }
    if (hw.qualcomm_npu) {
        log.info("Qualcomm NPU detected (not yet supported, using CPU)", .{});
    }

    if (hw.intel_npu) {
        if (promptIntelNpu()) {
            intel_npu.init() catch |err| {
                log.err("failed to initialize Intel NPU: {s}, falling back to CPU", .{@errorName(err)});
                return cpu.backend();
            };
            return intel_npu.backend();
        }
        log.info("Intel NPU declined, using CPU", .{});
    }

    return cpu.backend();
}
