const std = @import("std");
const ze = @import("ze.zig");
const log = std.log.scoped(.intel_npu);

// ── Module-level state ──

var lib: ?std.DynLib = null;
var dispatch: ?ze.Dispatch = null;
var driver: ?ze.ze_driver_handle_t = null;
var device: ?ze.ze_device_handle_t = null;
var context: ?ze.ze_context_handle_t = null;
var queue: ?ze.ze_command_queue_handle_t = null;

// ── Lifecycle ──

pub fn init() !void {
    // 1. Load ze_loader.dll at runtime
    lib = std.DynLib.open("ze_loader.dll") catch |err| {
        log.err("failed to load ze_loader.dll: {s}", .{@errorName(err)});
        return err;
    };

    // 2. Resolve function pointers
    dispatch = ze.Dispatch.load(&lib.?) catch |err| {
        log.err("failed to resolve Level-Zero symbols: {s}", .{@errorName(err)});
        return err;
    };
    const d = dispatch.?;

    // 3. Initialize Level-Zero (VPU/NPU devices only)
    try ze.check(d.zeInit(ze.ZE_INIT_FLAG_VPU_ONLY));
    log.info("Level-Zero initialized (VPU-only mode)", .{});

    // 4. Enumerate drivers
    var driver_count: u32 = 0;
    try ze.check(d.zeDriverGet(&driver_count, null));
    if (driver_count == 0) {
        log.err("no Level-Zero drivers found", .{});
        return error.InvalidArgument;
    }
    log.info("found {d} Level-Zero driver(s)", .{driver_count});

    // Allocate on stack for small counts (typically 1-2 drivers)
    var driver_buf: [8]ze.ze_driver_handle_t = undefined;
    const clamped_count = @min(driver_count, 8);
    try ze.check(d.zeDriverGet(&driver_count, &driver_buf));

    // 5. Find a VPU device across all drivers
    var found_driver: ?ze.ze_driver_handle_t = null;
    var found_device: ?ze.ze_device_handle_t = null;
    var found_props: ze.ze_device_properties_t = .{};

    for (driver_buf[0..clamped_count]) |drv| {
        var dev_count: u32 = 0;
        try ze.check(d.zeDeviceGet(drv, &dev_count, null));
        if (dev_count == 0) continue;

        var dev_buf: [16]ze.ze_device_handle_t = undefined;
        const clamped_dev = @min(dev_count, 16);
        try ze.check(d.zeDeviceGet(drv, &dev_count, &dev_buf));

        for (dev_buf[0..clamped_dev]) |dev| {
            var props: ze.ze_device_properties_t = .{};
            try ze.check(d.zeDeviceGetProperties(dev, &props));
            if (props.type == .VPU) {
                found_driver = drv;
                found_device = dev;
                found_props = props;
                break;
            }
        }
        if (found_device != null) break;
    }

    if (found_device == null) {
        log.err("no VPU/NPU device found among {d} driver(s)", .{clamped_count});
        return error.InvalidArgument;
    }

    driver = found_driver;
    device = found_device;

    // Log device name (null-terminated C string in the name field)
    const name_slice = std.mem.sliceTo(&found_props.name, 0);
    log.info("NPU device: {s}", .{name_slice});

    // 7. Create context
    const ctx_desc: ze.ze_context_desc_t = .{};
    var ctx: ze.ze_context_handle_t = undefined;
    try ze.check(d.zeContextCreate(driver.?, &ctx_desc, &ctx));
    context = ctx;
    log.info("Level-Zero context created", .{});

    // 8. Create synchronous command queue
    const cq_desc: ze.ze_command_queue_desc_t = .{
        .mode = .SYNCHRONOUS,
        .priority = .NORMAL,
    };
    var cq: ze.ze_command_queue_handle_t = undefined;
    try ze.check(d.zeCommandQueueCreate(context.?, device.?, &cq_desc, &cq));
    queue = cq;
    log.info("Level-Zero command queue created (synchronous)", .{});

    log.info("Intel NPU backend initialized successfully", .{});
}

pub fn deinit() void {
    const d = dispatch orelse return;

    if (queue) |q| {
        ze.check(d.zeCommandQueueDestroy(q)) catch {};
        queue = null;
    }
    if (context) |ctx| {
        ze.check(d.zeContextDestroy(ctx)) catch {};
        context = null;
    }

    driver = null;
    device = null;
    dispatch = null;

    if (lib) |*l| {
        l.close();
        lib = null;
    }

    log.info("Intel NPU backend shut down", .{});
}

// ── Backend contract (forward ops) — Phase 4 work ──

pub fn matmul_fwd(_: []const f32, _: []const f32, _: []f32, _: usize, _: usize) void {
    @panic("Intel NPU matmul not yet implemented (Phase 4)");
}

pub fn softmax_fwd(_: []const f32, _: []f32, _: usize) void {
    @panic("Intel NPU softmax not yet implemented (Phase 4)");
}

pub fn rmsnorm_fwd(_: []const f32, _: []f32, _: usize, _: *f32) void {
    @panic("Intel NPU rmsnorm not yet implemented (Phase 4)");
}

pub fn relu_fwd(_: []const f32, _: []f32, _: usize) void {
    @panic("Intel NPU relu not yet implemented (Phase 4)");
}

// ── Graph cache skeleton ──

pub const ShapeKey = struct {
    op: enum { matmul, softmax, rmsnorm, relu },
    dims: [4]u32,

    pub fn hash(self: ShapeKey) u64 {
        var h = std.hash.Wyhash.init(0);
        h.update(std.mem.asBytes(&self.op));
        h.update(std.mem.asBytes(&self.dims));
        return h.final();
    }

    pub fn eql(a: ShapeKey, b: ShapeKey) bool {
        return std.meta.eql(a, b);
    }
};

pub const GraphHandle = usize;

pub var graph_cache: ?std.HashMap(ShapeKey, GraphHandle, struct {
    pub fn hash(_: @This(), key: ShapeKey) u64 {
        return key.hash();
    }
    pub fn eql(_: @This(), a: ShapeKey, b: ShapeKey) bool {
        return a.eql(b);
    }
}, std.hash_map.default_max_load_percentage) = null;
