const std = @import("std");
const ze = @import("ze.zig");
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
    gpu_name: [ze.ZE_MAX_DEVICE_NAME]u8 = .{0} ** ze.ZE_MAX_DEVICE_NAME,
    npu_name: [ze.ZE_MAX_DEVICE_NAME]u8 = .{0} ** ze.ZE_MAX_DEVICE_NAME,
};

fn probeLib(name: [:0]const u8) bool {
    var dll = std.DynLib.open(name) catch return false;
    dll.close();
    return true;
}

fn detectHardware() DetectedHardware {
    var hw = DetectedHardware{
        .intel_npu = false,
        .intel_gpu = false,
        .amd_npu = probeLib("xrt_coreutil.dll"),
        .qualcomm_npu = probeLib("QnnHtp.dll") or probeLib("libcdsprpc.dll"),
    };

    // Try Level-Zero device enumeration for Intel GPU/NPU
    var dll = std.DynLib.open("ze_loader.dll") catch return hw;
    defer dll.close();

    const zeInit = dll.lookup(ze.pfnInit, "zeInit") orelse return hw;
    const zeDriverGet = dll.lookup(ze.pfnDriverGet, "zeDriverGet") orelse return hw;
    const zeDeviceGet = dll.lookup(ze.pfnDeviceGet, "zeDeviceGet") orelse return hw;
    const zeDeviceGetProperties = dll.lookup(ze.pfnDeviceGetProperties, "zeDeviceGetProperties") orelse return hw;

    if (zeInit(0) != .SUCCESS) return hw;

    var driver_count: u32 = 0;
    if (zeDriverGet(&driver_count, null) != .SUCCESS or driver_count == 0) return hw;

    var driver_buf: [8]ze.ze_driver_handle_t = undefined;
    var fetch_count: u32 = @min(driver_count, 8);
    if (zeDriverGet(&fetch_count, &driver_buf) != .SUCCESS) return hw;

    for (driver_buf[0..fetch_count]) |drv| {
        var dev_count: u32 = 0;
        if (zeDeviceGet(drv, &dev_count, null) != .SUCCESS or dev_count == 0) continue;

        var dev_buf: [16]ze.ze_device_handle_t = undefined;
        var dev_fetch: u32 = @min(dev_count, 16);
        if (zeDeviceGet(drv, &dev_fetch, &dev_buf) != .SUCCESS) continue;

        for (dev_buf[0..dev_fetch]) |dev| {
            var props: ze.ze_device_properties_t = .{};
            if (zeDeviceGetProperties(dev, &props) != .SUCCESS) continue;

            if (props.type == .GPU and !hw.intel_gpu) {
                hw.intel_gpu = true;
                hw.gpu_name = props.name;
            }
            if (props.type == .VPU and !hw.intel_npu) {
                hw.intel_npu = true;
                hw.npu_name = props.name;
            }
        }
    }

    return hw;
}

const AccelChoice = enum { npu, gpu, cpu };

fn promptAccelerator(hw: *const DetectedHardware) AccelChoice {
    const stdout = std.fs.File.stdout();
    const stdin = std.fs.File.stdin();

    var option_count: u8 = 0;
    var options: [3]AccelChoice = undefined;

    // Build option list
    if (hw.intel_npu) {
        option_count += 1;
        options[option_count - 1] = .npu;
    }
    if (hw.intel_gpu) {
        option_count += 1;
        options[option_count - 1] = .gpu;
    }
    option_count += 1;
    options[option_count - 1] = .cpu;

    // If only CPU available, skip prompt
    if (option_count == 1) return .cpu;

    _ = stdout.write("\nAvailable accelerators:\n") catch return .cpu;
    for (0..option_count) |i| {
        var line_buf: [128]u8 = undefined;
        const line = switch (options[i]) {
            .npu => std.fmt.bufPrint(&line_buf, "  [{d}] Intel NPU  — {s}\n", .{
                i + 1,
                std.mem.sliceTo(&hw.npu_name, 0),
            }) catch continue,
            .gpu => std.fmt.bufPrint(&line_buf, "  [{d}] Intel GPU  — {s}\n", .{
                i + 1,
                std.mem.sliceTo(&hw.gpu_name, 0),
            }) catch continue,
            .cpu => std.fmt.bufPrint(&line_buf, "  [{d}] CPU only\n", .{i + 1}) catch continue,
        };
        _ = stdout.write(line) catch {};
    }

    _ = stdout.write("Select [1]: ") catch return .cpu;

    var buf: [16]u8 = undefined;
    const n = stdin.read(&buf) catch return .cpu;
    if (n == 0) return options[0]; // default to first

    // Strip whitespace/newline
    const sel_byte = buf[0];
    if (sel_byte == '\r' or sel_byte == '\n' or sel_byte == ' ') return options[0]; // default

    const sel_num = std.fmt.parseInt(u8, buf[0..1], 10) catch return options[0];
    if (sel_num >= 1 and sel_num <= option_count) {
        return options[sel_num - 1];
    }
    return options[0]; // default to first option
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
