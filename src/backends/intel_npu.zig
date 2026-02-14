const std = @import("std");
const ze = @import("ze.zig");
const ov_ir = @import("ov_ir.zig");
const cpu = @import("cpu.zig");
const log = std.log.scoped(.intel_npu);

var lib: ?std.DynLib = null;
var dispatch: ?ze.Dispatch = null;
var driver: ?ze.ze_driver_handle_t = null;
var device: ?ze.ze_device_handle_t = null;
var context: ?ze.ze_context_handle_t = null;
var queue: ?ze.ze_command_queue_handle_t = null;
var graph_ddi: ?*const ze.ze_graph_dditable_t = null;
var cmd_list: ?ze.ze_command_list_handle_t = null;
var fence: ?ze.ze_fence_handle_t = null;
var compiler_ver: ze.ze_graph_compiler_version_info_t = .{};

pub const ShapeKey = struct {
    pub const Op = enum { matmul, softmax, relu };
    op: Op,
    dims: [4]u32,
};

const SharedBuf = struct {
    ptr: *anyopaque,
    size: usize,

    fn asF32Slice(self: SharedBuf) []f32 {
        const n = self.size / @sizeOf(f32);
        const typed: [*]f32 = @ptrCast(@alignCast(self.ptr));
        return typed[0..n];
    }
};

const CachedGraph = struct {
    graph: ze.ze_graph_handle_t,
    num_args: u32,
    arg_bufs: [4]SharedBuf,
};

const CacheEntry = struct {
    key: ShapeKey = .{ .op = .relu, .dims = .{ 0, 0, 0, 0 } },
    graph: CachedGraph = .{ .graph = undefined, .num_args = 0, .arg_bufs = undefined },
    occupied: bool = false,
};

const MAX_CACHED = 32;
var cached_graphs: [MAX_CACHED]CacheEntry = .{.{}} ** MAX_CACHED;
var num_cached: usize = 0;

pub fn init() !void {
    lib = std.DynLib.open("ze_loader.dll") catch |err| {
        log.err("failed to load ze_loader.dll: {s}", .{@errorName(err)});
        return err;
    };

    dispatch = ze.Dispatch.load(&lib.?) catch |err| {
        log.err("failed to resolve Level-Zero symbols: {s}", .{@errorName(err)});
        return err;
    };
    const d = dispatch.?;

    try ze.check(d.zeInit(ze.ZE_INIT_FLAG_VPU_ONLY));
    log.info("Level-Zero initialized (VPU-only mode)", .{});

    // Enumerate drivers
    var driver_count: u32 = 0;
    try ze.check(d.zeDriverGet(&driver_count, null));
    if (driver_count == 0) {
        log.err("no Level-Zero drivers found", .{});
        return error.InvalidArgument;
    }
    log.info("found {d} Level-Zero driver(s)", .{driver_count});

    var driver_buf: [8]ze.ze_driver_handle_t = undefined;
    const clamped_count = @min(driver_count, 8);
    try ze.check(d.zeDriverGet(&driver_count, &driver_buf));

    // Find a VPU/NPU device across all drivers
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
    log.info("NPU device: {s}", .{std.mem.sliceTo(&found_props.name, 0)});

    const ctx_desc: ze.ze_context_desc_t = .{};
    var ctx: ze.ze_context_handle_t = undefined;
    try ze.check(d.zeContextCreate(driver.?, &ctx_desc, &ctx));
    context = ctx;

    const cq_desc: ze.ze_command_queue_desc_t = .{
        .mode = .SYNCHRONOUS,
        .priority = .NORMAL,
    };
    var cq: ze.ze_command_queue_handle_t = undefined;
    try ze.check(d.zeCommandQueueCreate(context.?, device.?, &cq_desc, &cq));
    queue = cq;

    graph_ddi = try acquireGraphDDI(d, driver.?);

    // Query compiler version
    var dev_graph_props: ze.ze_device_graph_properties_t = .{};
    try ze.check(graph_ddi.?.pfnDeviceGetGraphProperties(device.?, &dev_graph_props));
    compiler_ver = dev_graph_props.compilerVersion;
    log.info("NPU compiler version: {d}.{d}, max opset: {d}, formats: 0x{x}, ext ver: 0x{x}", .{
        compiler_ver.major,
        compiler_ver.minor,
        dev_graph_props.maxOVOpsetVersionSupported,
        dev_graph_props.graphFormatsSupported,
        dev_graph_props.graphExtensionVersion,
    });
    if (graph_ddi.?.pfnCreate2 != null) {
        log.info("pfnCreate2 (v1.5) available", .{});
    } else {
        log.info("pfnCreate2 (v1.5) NOT available, using pfnCreate (v1.0)", .{});
    }

    const cl_desc: ze.ze_command_list_desc_t = .{};
    var cl: ze.ze_command_list_handle_t = undefined;
    try ze.check(d.zeCommandListCreate(context.?, device.?, &cl_desc, &cl));
    cmd_list = cl;

    const fence_desc: ze.ze_fence_desc_t = .{};
    var f: ze.ze_fence_handle_t = undefined;
    try ze.check(d.zeFenceCreate(queue.?, &fence_desc, &f));
    fence = f;

    try smokeTest();
    log.info("Intel NPU backend initialized successfully", .{});
}

/// Acquire the graph extension DDI table using OpenVINO's two-tier mechanism:
/// 1. Enumerate driver extensions to find ZE_extension_graph name + version
/// 2. Try ZE_extension_driver_npu pfnGetExtension (versioned acquisition)
/// 3. Fallback: zeDriverGetExtensionFunctionAddress with the extension name
fn acquireGraphDDI(d: ze.Dispatch, drv: ze.ze_driver_handle_t) !*const ze.ze_graph_dditable_t {
    var ext_count: u32 = 0;
    try ze.check(d.zeDriverGetExtensionProperties(drv, &ext_count, null));

    var ext_buf: [32]ze.ze_driver_extension_properties_t = undefined;
    var fetch_count: u32 = @min(ext_count, 32);
    try ze.check(d.zeDriverGetExtensionProperties(drv, &fetch_count, &ext_buf));

    const target_version: u32 = (1 << 16) | 5; // ZE_GRAPH_EXT_VERSION_1_5
    var graph_ext_version: u32 = 0;
    var graph_ext_idx: ?usize = null;
    const graph_prefix = "ZE_extension_graph";
    var has_npu_driver_ext = false;

    for (ext_buf[0..fetch_count], 0..) |*ext, i| {
        const name = std.mem.sliceTo(&ext.name, 0);
        if (std.mem.eql(u8, name, "ZE_extension_driver_npu")) {
            has_npu_driver_ext = true;
        }
        if (!std.mem.startsWith(u8, name, graph_prefix)) continue;

        if (ext.version >= target_version) {
            graph_ext_version = target_version;
            graph_ext_idx = i;
            break;
        }
        if (ext.version > graph_ext_version) {
            graph_ext_version = ext.version;
            graph_ext_idx = i;
        }
    }

    if (graph_ext_idx == null) {
        log.err("no ZE_extension_graph extension found", .{});
        return error.UnsupportedFeature;
    }

    const graph_ext_name: [*:0]const u8 = @ptrCast(&ext_buf[graph_ext_idx.?].name);

    // Try versioned path via ZE_extension_driver_npu
    if (has_npu_driver_ext) {
        var npu_ddi_ptr: ?*anyopaque = null;
        if (d.zeDriverGetExtensionFunctionAddress(drv, "ZE_extension_driver_npu", &npu_ddi_ptr) == .SUCCESS) {
            if (npu_ddi_ptr) |ptr| {
                const npu_ddi: *const ze.ze_driver_npu_dditable_ext_t = @ptrCast(@alignCast(ptr));
                var result_ptr: ?*anyopaque = null;
                var ext_req = ze.ze_driver_extension_npu_ext_t{
                    .name = graph_ext_name,
                    .version = graph_ext_version,
                    .ppFunctionAddress = &result_ptr,
                };
                if (npu_ddi.pfnGetExtension(drv, &ext_req) == .SUCCESS) {
                    if (result_ptr) |rp| return @ptrCast(@alignCast(rp));
                }
            }
        }
    }

    // Fallback: direct function address lookup
    var ddi_ptr: ?*anyopaque = null;
    try ze.check(d.zeDriverGetExtensionFunctionAddress(drv, graph_ext_name, &ddi_ptr));
    return @ptrCast(@alignCast(ddi_ptr.?));
}

fn smokeTest() !void {
    log.info("running NPU smoke test (ReLU [4])...", .{});
    const blob = ov_ir.reluBlob(4, compiler_ver);
    const graph = try compileGraph(&blob, ov_ir.relu_build_flags);
    _ = graph_ddi.?.pfnDestroy(graph);
    log.info("smoke test passed", .{});
}

/// Compile a graph from an IR blob using pfnCreate2 (v1.5) with pfnCreate (v1.0) fallback.
fn compileGraph(blob: *const ov_ir.BlobBuf, build_flags: [*:0]const u8) !ze.ze_graph_handle_t {
    const ddi = graph_ddi.?;
    var graph: ze.ze_graph_handle_t = undefined;

    if (ddi.pfnCreate2) |create2| {
        const desc: ze.ze_graph_desc_2_t = .{
            .format = .NGRAPH_LITE,
            .inputSize = blob.len,
            .pInput = &blob.data,
            .pBuildFlags = build_flags,
        };
        try ze.check(create2(context.?, device.?, &desc, &graph));
    } else {
        const desc: ze.ze_graph_desc_t = .{
            .format = .NGRAPH_LITE,
            .inputSize = blob.len,
            .pInput = &blob.data,
            .pBuildFlags = build_flags,
        };
        try ze.check(ddi.pfnCreate(context.?, device.?, &desc, &graph));
    }
    return graph;
}

pub fn deinit() void {
    const d = dispatch orelse return;
    const ddi = graph_ddi;

    for (&cached_graphs) |*entry| {
        if (!entry.occupied) continue;
        if (ddi) |g| ze.check(g.pfnDestroy(entry.graph.graph)) catch {};
        for (0..entry.graph.num_args) |i| {
            ze.check(d.zeMemFree(context.?, entry.graph.arg_bufs[i].ptr)) catch {};
        }
        entry.occupied = false;
    }
    num_cached = 0;

    if (fence) |f| {
        ze.check(d.zeFenceDestroy(f)) catch {};
        fence = null;
    }
    if (cmd_list) |cl| {
        ze.check(d.zeCommandListDestroy(cl)) catch {};
        cmd_list = null;
    }
    graph_ddi = null;

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

fn getOrCompileGraph(key: ShapeKey, arg_sizes: []const usize, num_args_expected: u32, build_flags: [*:0]const u8) !*CachedGraph {
    for (&cached_graphs) |*entry| {
        if (entry.occupied and std.meta.eql(entry.key, key)) {
            return &entry.graph;
        }
    }

    if (num_cached >= MAX_CACHED) {
        log.err("graph cache full ({d} entries)", .{MAX_CACHED});
        return error.Unknown;
    }

    const d = dispatch.?;
    const ddi = graph_ddi.?;

    const blob = switch (key.op) {
        .matmul => ov_ir.matmulBlob(key.dims[0], key.dims[1], compiler_ver),
        .softmax => ov_ir.softmaxBlob(key.dims[0], compiler_ver),
        .relu => ov_ir.reluBlob(key.dims[0], compiler_ver),
    };

    log.info("compiling graph: op={s} dims=[{d},{d},{d},{d}]", .{
        @tagName(key.op), key.dims[0], key.dims[1], key.dims[2], key.dims[3],
    });

    const graph = try compileGraph(&blob, build_flags);

    var props: ze.ze_graph_properties_t = .{};
    try ze.check(ddi.pfnGetProperties(graph, &props));
    if (props.numGraphArgs != num_args_expected) {
        log.warn("graph has {d} args, expected {d}", .{ props.numGraphArgs, num_args_expected });
    }

    // Allocate shared memory for each argument and bind to graph
    var arg_bufs: [4]SharedBuf = undefined;
    const dev_mem_desc: ze.ze_device_mem_alloc_desc_t = .{};
    const host_mem_desc: ze.ze_host_mem_alloc_desc_t = .{};

    for (0..num_args_expected) |i| {
        var ptr: ?*anyopaque = null;
        try ze.check(d.zeMemAllocShared(
            context.?,
            &dev_mem_desc,
            &host_mem_desc,
            arg_sizes[i],
            64,
            device.?,
            &ptr,
        ));
        arg_bufs[i] = .{ .ptr = ptr.?, .size = arg_sizes[i] };
        try ze.check(ddi.pfnSetArgumentValue(graph, @intCast(i), ptr.?));
    }

    // Initialize graph (one-time submit + sync)
    const cl = cmd_list.?;
    try ze.check(ddi.pfnAppendGraphInitialize(cl, graph, null, 0, null));
    try submitAndSync(cl);

    const entry = &cached_graphs[num_cached];
    entry.* = .{
        .key = key,
        .graph = .{
            .graph = graph,
            .num_args = num_args_expected,
            .arg_bufs = arg_bufs,
        },
        .occupied = true,
    };
    num_cached += 1;

    return &entry.graph;
}

fn executeGraph(cached: *const CachedGraph) void {
    const ddi = graph_ddi.?;
    const cl = cmd_list.?;
    ze.check(ddi.pfnAppendGraphExecute(cl, cached.graph, null, null, 0, null)) catch
        @panic("pfnAppendGraphExecute failed");
    submitAndSync(cl) catch @panic("graph execution submit/sync failed");
}

fn submitAndSync(cl: ze.ze_command_list_handle_t) !void {
    const d = dispatch.?;
    try ze.check(d.zeCommandListClose(cl));
    var cl_tmp = cl;
    try ze.check(d.zeCommandQueueExecuteCommandLists(queue.?, 1, &cl_tmp, fence.?));
    try ze.check(d.zeFenceHostSynchronize(fence.?, ze.MAX_U64));
    try ze.check(d.zeFenceReset(fence.?));
    try ze.check(d.zeCommandListReset(cl));
}

pub fn matmul_fwd(W: []const f32, x: []const f32, out: []f32, M: usize, K: usize) void {
    const cached = getOrCompileGraph(
        .{ .op = .matmul, .dims = .{ @intCast(M), @intCast(K), 0, 0 } },
        &.{ M * K * @sizeOf(f32), K * @sizeOf(f32), M * @sizeOf(f32) },
        3,
        ov_ir.matmul_build_flags,
    ) catch @panic("NPU matmul graph compilation failed");

    @memcpy(cached.arg_bufs[0].asF32Slice()[0 .. M * K], W[0 .. M * K]);
    @memcpy(cached.arg_bufs[1].asF32Slice()[0..K], x[0..K]);
    executeGraph(cached);
    @memcpy(out[0..M], cached.arg_bufs[2].asF32Slice()[0..M]);
}

pub fn softmax_fwd(input: []const f32, output: []f32, n: usize) void {
    runUnaryGraph(.softmax, input, output, n, ov_ir.softmax_build_flags);
}

pub fn relu_fwd(input: []const f32, output: []f32, n: usize) void {
    runUnaryGraph(.relu, input, output, n, ov_ir.relu_build_flags);
}

fn runUnaryGraph(op: ShapeKey.Op, input: []const f32, output: []f32, n: usize, build_flags: [*:0]const u8) void {
    const size = n * @sizeOf(f32);
    const cached = getOrCompileGraph(
        .{ .op = op, .dims = .{ @intCast(n), 0, 0, 0 } },
        &.{ size, size },
        2,
        build_flags,
    ) catch @panic("NPU unary graph compilation failed");

    @memcpy(cached.arg_bufs[0].asF32Slice()[0..n], input[0..n]);
    executeGraph(cached);
    @memcpy(output[0..n], cached.arg_bufs[1].asF32Slice()[0..n]);
}

/// CPU fallback: rmsnorm needs scale_out for backward pass and is too small for NPU dispatch.
pub fn rmsnorm_fwd(input: []const f32, output: []f32, n: usize, scale_out: *f32) void {
    cpu.rmsnorm_fwd(input, output, n, scale_out);
}
