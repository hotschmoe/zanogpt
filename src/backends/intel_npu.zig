const std = @import("std");
const ze = @import("ze.zig");
const ov_ir = @import("ov_ir.zig");
const cpu = @import("cpu.zig");
const log = std.log.scoped(.intel_npu);

// ── Module-level state ──

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

// ── Graph cache ──

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
    key: ShapeKey,
    graph: CachedGraph,
    occupied: bool,
};

const MAX_CACHED = 32;
var cached_graphs: [MAX_CACHED]CacheEntry = .{empty_cache_entry} ** MAX_CACHED;
var num_cached: usize = 0;

const empty_cache_entry = CacheEntry{
    .key = .{ .op = .relu, .dims = .{ 0, 0, 0, 0 } },
    .graph = .{ .graph = undefined, .num_args = 0, .arg_bufs = undefined },
    .occupied = false,
};

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

    // 9. Get graph extension DDI table
    var ddi_ptr: ?*anyopaque = null;
    try ze.check(d.zeDriverGetExtensionFunctionAddress(driver.?, "ZE_extension_graph", &ddi_ptr));
    graph_ddi = @ptrCast(@alignCast(ddi_ptr.?));
    log.info("graph extension DDI table acquired", .{});

    // 10. Query compiler version from device graph properties
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
    if (graph_ddi.?.pfnCreate2) |_| {
        log.info("pfnCreate2 (v1.5) available", .{});
    } else {
        log.info("pfnCreate2 (v1.5) NOT available, using pfnCreate (v1.0)", .{});
    }

    // 11. Create reusable command list
    const cl_desc: ze.ze_command_list_desc_t = .{};
    var cl: ze.ze_command_list_handle_t = undefined;
    try ze.check(d.zeCommandListCreate(context.?, device.?, &cl_desc, &cl));
    cmd_list = cl;
    log.info("reusable command list created", .{});

    // 12. Create reusable fence
    const fence_desc: ze.ze_fence_desc_t = .{};
    var f: ze.ze_fence_handle_t = undefined;
    try ze.check(d.zeFenceCreate(queue.?, &fence_desc, &f));
    fence = f;
    log.info("reusable fence created", .{});

    // 13. Smoke test: compile + execute a tiny ReLU graph
    try smokeTest();

    log.info("Intel NPU backend initialized successfully", .{});
}

fn smokeTest() !void {
    log.info("running NPU smoke test (ReLU [4])...", .{});

    // Compile a ReLU graph for 4 elements
    const cached = try getOrCompileGraph(
        .{ .op = .relu, .dims = .{ 4, 0, 0, 0 } },
        &.{ 4 * @sizeOf(f32), 4 * @sizeOf(f32) },
        2,
        ov_ir.unary_build_flags,
    );

    const test_input = [_]f32{ -1.0, 0.0, 2.0, -3.0 };
    @memcpy(cached.arg_bufs[0].asF32Slice()[0..4], &test_input);

    executeGraph(cached);

    // Verify output: [0, 0, 2, 0]
    const out_buf = cached.arg_bufs[1].asF32Slice();
    const expected = [_]f32{ 0.0, 0.0, 2.0, 0.0 };
    for (0..4) |i| {
        if (@abs(out_buf[i] - expected[i]) > 1e-5) {
            log.err("smoke test failed at index {d}: expected {d}, got {d}", .{ i, expected[i], out_buf[i] });
            return error.Unknown;
        }
    }

    log.info("NPU smoke test passed", .{});
}

pub fn deinit() void {
    const d = dispatch orelse return;
    const ddi = graph_ddi;

    // 1. Destroy cached graphs + free shared memory
    for (&cached_graphs) |*entry| {
        if (!entry.occupied) continue;
        if (ddi) |g| {
            ze.check(g.pfnDestroy(entry.graph.graph)) catch {};
        }
        for (0..entry.graph.num_args) |i| {
            ze.check(d.zeMemFree(context.?, entry.graph.arg_bufs[i].ptr)) catch {};
        }
        entry.occupied = false;
    }
    num_cached = 0;

    // 2. Destroy fence + command list
    if (fence) |f| {
        ze.check(d.zeFenceDestroy(f)) catch {};
        fence = null;
    }
    if (cmd_list) |cl| {
        ze.check(d.zeCommandListDestroy(cl)) catch {};
        cmd_list = null;
    }

    graph_ddi = null;

    // 3. Destroy queue, context, and unload library
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

// ── Graph compilation and execution ──

fn getOrCompileGraph(key: ShapeKey, arg_sizes: []const usize, num_args_expected: u32, build_flags: [*:0]const u8) !*CachedGraph {
    // Linear scan for cache hit
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

    // Generate blob
    var blob = switch (key.op) {
        .matmul => ov_ir.matmulBlob(key.dims[0], key.dims[1]),
        .softmax => ov_ir.softmaxBlob(key.dims[0]),
        .relu => ov_ir.reluBlob(key.dims[0]),
    };

    // Compile graph — try pfnCreate2 (v1.5) first, fall back to pfnCreate (v1.0)
    log.info("compiling graph: op={s} dims=[{d},{d},{d},{d}] blob_len={d}", .{
        @tagName(key.op), key.dims[0], key.dims[1], key.dims[2], key.dims[3], blob.len,
    });

    var graph: ze.ze_graph_handle_t = undefined;

    if (ddi.pfnCreate2) |create2| {
        const desc2: ze.ze_graph_desc_2_t = .{
            .format = .NGRAPH_LITE,
            .inputSize = blob.len,
            .pInput = &blob.data,
            .pBuildFlags = build_flags,
            .flags = 0,
        };
        const result2 = create2(context.?, device.?, &desc2, &graph);
        if (result2 != .SUCCESS) {
            log.warn("pfnCreate2 failed (0x{x}), trying pfnCreate", .{@intFromEnum(result2)});
        } else {
            log.info("pfnCreate2 succeeded", .{});
        }
        if (result2 == .SUCCESS) {} else {
            // Fall through to pfnCreate
            const desc: ze.ze_graph_desc_t = .{
                .format = .NGRAPH_LITE,
                .inputSize = blob.len,
                .pInput = &blob.data,
                .pBuildFlags = build_flags,
                .compilerVersion = compiler_ver,
            };
            const result1 = ddi.pfnCreate(context.?, device.?, &desc, &graph);
            if (result1 != .SUCCESS) {
                log.err("pfnCreate also failed (0x{x})", .{@intFromEnum(result1)});
                try ze.check(result1);
            }
        }
    } else {
        const desc: ze.ze_graph_desc_t = .{
            .format = .NGRAPH_LITE,
            .inputSize = blob.len,
            .pInput = &blob.data,
            .pBuildFlags = build_flags,
            .compilerVersion = compiler_ver,
        };
        try ze.check(ddi.pfnCreate(context.?, device.?, &desc, &graph));
    }
    log.info("compiled graph: op={s} dims=[{d},{d},{d},{d}]", .{
        @tagName(key.op), key.dims[0], key.dims[1], key.dims[2], key.dims[3],
    });

    // Query number of arguments
    var props: ze.ze_graph_properties_t = .{};
    try ze.check(ddi.pfnGetProperties(graph, &props));

    if (props.numGraphArgs != num_args_expected) {
        log.warn("graph has {d} args, expected {d}", .{ props.numGraphArgs, num_args_expected });
    }

    // Allocate shared memory for each argument and bind
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
            64, // alignment
            device.?,
            &ptr,
        ));
        arg_bufs[i] = .{ .ptr = ptr.?, .size = arg_sizes[i] };

        // Bind to graph argument
        try ze.check(ddi.pfnSetArgumentValue(graph, @intCast(i), ptr.?));
    }

    // Initialize graph (one-time submit + sync)
    const cl = cmd_list.?;
    try ze.check(ddi.pfnAppendGraphInitialize(cl, graph, null, 0, null));
    try submitAndSync(cl);

    // Store in cache
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

/// Close the command list, submit it to the queue, wait on the fence, then reset both.
fn submitAndSync(cl: ze.ze_command_list_handle_t) !void {
    const d = dispatch.?;
    try ze.check(d.zeCommandListClose(cl));
    var cl_tmp = cl;
    try ze.check(d.zeCommandQueueExecuteCommandLists(queue.?, 1, &cl_tmp, fence.?));
    try ze.check(d.zeFenceHostSynchronize(fence.?, ze.MAX_U64));
    try ze.check(d.zeFenceReset(fence.?));
    try ze.check(d.zeCommandListReset(cl));
}

// ── Backend contract (forward ops) ──

pub fn matmul_fwd(W: []const f32, x: []const f32, out: []f32, M: usize, K: usize) void {
    const cached = getOrCompileGraph(
        .{ .op = .matmul, .dims = .{ @intCast(M), @intCast(K), 0, 0 } },
        &.{ M * K * @sizeOf(f32), K * @sizeOf(f32), M * @sizeOf(f32) },
        3,
        ov_ir.matmul_build_flags,
    ) catch @panic("NPU matmul graph compilation failed");

    // Copy W → shared mem (arg 0)
    const w_buf = cached.arg_bufs[0].asF32Slice();
    @memcpy(w_buf[0 .. M * K], W[0 .. M * K]);

    // Copy x → shared mem (arg 1)
    const x_buf = cached.arg_bufs[1].asF32Slice();
    @memcpy(x_buf[0..K], x[0..K]);

    executeGraph(cached);

    // Copy result ← shared mem (arg 2)
    const out_buf = cached.arg_bufs[2].asF32Slice();
    @memcpy(out[0..M], out_buf[0..M]);
}

pub fn softmax_fwd(input: []const f32, output: []f32, n: usize) void {
    runUnaryGraph(.softmax, input, output, n);
}

pub fn relu_fwd(input: []const f32, output: []f32, n: usize) void {
    runUnaryGraph(.relu, input, output, n);
}

/// Shared implementation for unary NPU ops (2-arg graph: input + output).
fn runUnaryGraph(op: ShapeKey.Op, input: []const f32, output: []f32, n: usize) void {
    const size = n * @sizeOf(f32);
    const cached = getOrCompileGraph(
        .{ .op = op, .dims = .{ @intCast(n), 0, 0, 0 } },
        &.{ size, size },
        2,
        ov_ir.unary_build_flags,
    ) catch @panic("NPU unary graph compilation failed");

    @memcpy(cached.arg_bufs[0].asF32Slice()[0..n], input[0..n]);
    executeGraph(cached);
    @memcpy(output[0..n], cached.arg_bufs[1].asF32Slice()[0..n]);
}

pub fn rmsnorm_fwd(input: []const f32, output: []f32, n: usize, scale_out: *f32) void {
    // CPU fallback — rmsnorm needs scale_out for backward pass, and 16-element
    // tensors make NPU dispatch overhead dominate.
    cpu.rmsnorm_fwd(input, output, n, scale_out);
}
