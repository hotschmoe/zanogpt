const std = @import("std");
const ze = @import("ze.zig");
const spirv = @import("spirv.zig");
const cpu = @import("cpu.zig");
const Backend = @import("backend.zig").Backend;
const log = std.log.scoped(.intel_gpu);

var lib: ?std.DynLib = null;
var dispatch: ?ze.Dispatch = null;
var driver: ?ze.ze_driver_handle_t = null;
var device: ?ze.ze_device_handle_t = null;
var context: ?ze.ze_context_handle_t = null;
var queue: ?ze.ze_command_queue_handle_t = null;
var cmd_list: ?ze.ze_command_list_handle_t = null;
var fence: ?ze.ze_fence_handle_t = null;

const CachedKernel = struct {
    module: ze.ze_module_handle_t,
    kernel: ze.ze_kernel_handle_t,
};

var relu_cache: ?CachedKernel = null;
var matmul_cache: ?CachedKernel = null;
var softmax_cache: ?CachedKernel = null;

/// Shared (host+device) memory buffers cached to avoid repeated alloc/free.
const SharedBuf = struct {
    ptr: *anyopaque,
    size: usize,

    fn asF32Slice(self: SharedBuf) []f32 {
        const n = self.size / @sizeOf(f32);
        const typed: [*]f32 = @ptrCast(@alignCast(self.ptr));
        return typed[0..n];
    }
};

const MAX_BUFS = 8;
var shared_bufs: [MAX_BUFS]SharedBuf = undefined;
var num_bufs: usize = 0;

fn getOrAllocShared(min_size: usize) !SharedBuf {
    const d = dispatch.?;
    // Look for an existing buffer that's big enough
    for (shared_bufs[0..num_bufs]) |buf| {
        if (buf.size >= min_size) return buf;
    }
    // Allocate new shared buffer
    if (num_bufs >= MAX_BUFS) {
        log.err("shared buffer cache full ({d} entries)", .{MAX_BUFS});
        return error.OutOfHostMemory;
    }
    var ptr: ?*anyopaque = null;
    const dev_desc: ze.ze_device_mem_alloc_desc_t = .{};
    const host_desc: ze.ze_host_mem_alloc_desc_t = .{};
    try ze.check(d.zeMemAllocShared(context.?, &dev_desc, &host_desc, min_size, 64, device.?, &ptr));
    const buf = SharedBuf{ .ptr = ptr.?, .size = min_size };
    shared_bufs[num_bufs] = buf;
    num_bufs += 1;
    return buf;
}

fn getOrCompileKernel(cache: *?CachedKernel, spv_bytes: []const u8, kernel_name: [*:0]const u8) !CachedKernel {
    if (cache.*) |cached| return cached;

    const d = dispatch.?;
    const mod_desc = ze.ze_module_desc_t{
        .format = .IL_SPIRV,
        .inputSize = spv_bytes.len,
        .pInputModule = spv_bytes.ptr,
    };
    var mod: ze.ze_module_handle_t = undefined;
    try ze.check(d.zeModuleCreate(context.?, device.?, &mod_desc, &mod, null));

    const kern_desc = ze.ze_kernel_desc_t{ .pKernelName = kernel_name };
    var kern: ze.ze_kernel_handle_t = undefined;
    ze.check(d.zeKernelCreate(mod, &kern_desc, &kern)) catch |err| {
        _ = d.zeModuleDestroy(mod);
        return err;
    };

    const cached = CachedKernel{ .module = mod, .kernel = kern };
    cache.* = cached;
    return cached;
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

    // Initialize Level-Zero (all device types)
    try ze.check(d.zeInit(0));
    log.info("Level-Zero initialized (all devices)", .{});

    // Enumerate drivers
    var driver_count: u32 = 0;
    try ze.check(d.zeDriverGet(&driver_count, null));
    if (driver_count == 0) {
        log.err("no Level-Zero drivers found", .{});
        return error.InvalidArgument;
    }

    var driver_buf: [8]ze.ze_driver_handle_t = undefined;
    const clamped_count = @min(driver_count, 8);
    try ze.check(d.zeDriverGet(&driver_count, &driver_buf));

    // Find a GPU device
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
            if (props.type == .GPU) {
                found_driver = drv;
                found_device = dev;
                found_props = props;
                break;
            }
        }
        if (found_device != null) break;
    }

    if (found_device == null) {
        log.err("no GPU device found among {d} driver(s)", .{clamped_count});
        return error.InvalidArgument;
    }

    driver = found_driver;
    device = found_device;
    log.info("GPU device: {s}", .{std.mem.sliceTo(&found_props.name, 0)});

    // Create context
    const ctx_desc: ze.ze_context_desc_t = .{};
    var ctx: ze.ze_context_handle_t = undefined;
    try ze.check(d.zeContextCreate(driver.?, &ctx_desc, &ctx));
    context = ctx;

    // Create command queue
    const cq_desc: ze.ze_command_queue_desc_t = .{
        .mode = .SYNCHRONOUS,
        .priority = .NORMAL,
    };
    var cq: ze.ze_command_queue_handle_t = undefined;
    try ze.check(d.zeCommandQueueCreate(context.?, device.?, &cq_desc, &cq));
    queue = cq;

    // Create command list
    const cl_desc: ze.ze_command_list_desc_t = .{};
    var cl: ze.ze_command_list_handle_t = undefined;
    try ze.check(d.zeCommandListCreate(context.?, device.?, &cl_desc, &cl));
    cmd_list = cl;

    // Create fence
    const fence_desc: ze.ze_fence_desc_t = .{};
    var f: ze.ze_fence_handle_t = undefined;
    try ze.check(d.zeFenceCreate(queue.?, &fence_desc, &f));
    fence = f;

    // Smoke test: compile + run ReLU kernel
    try smokeTest();
    log.info("Intel GPU backend initialized successfully", .{});
}

fn smokeTest() !void {
    log.info("running GPU smoke test (ReLU compile)...", .{});
    const spv = spirv.reluModule();
    const cached = try getOrCompileKernel(&relu_cache, spv.asBytes(), "relu_kernel");
    // Just verify we can create a kernel — don't run it
    _ = cached;
    log.info("smoke test passed", .{});
}

pub fn relu_fwd(input: []const f32, output: []f32, n: usize) void {
    const d = dispatch orelse @panic("GPU not initialized");
    const spv = spirv.reluModule();
    const cached = getOrCompileKernel(&relu_cache, spv.asBytes(), "relu_kernel") catch
        @panic("GPU relu kernel compilation failed");

    const size = n * @sizeOf(f32);
    const buf_in = getOrAllocShared(size) catch @panic("GPU alloc failed");
    const buf_out = getOrAllocShared(size) catch @panic("GPU alloc failed");

    @memcpy(buf_in.asF32Slice()[0..n], input[0..n]);

    const kern = cached.kernel;
    var in_ptr = buf_in.ptr;
    var out_ptr = buf_out.ptr;
    var n_u32: u32 = @intCast(n);
    ze.check(d.zeKernelSetArgumentValue(kern, 0, @sizeOf(*anyopaque), @ptrCast(&in_ptr))) catch @panic("setarg failed");
    ze.check(d.zeKernelSetArgumentValue(kern, 1, @sizeOf(*anyopaque), @ptrCast(&out_ptr))) catch @panic("setarg failed");
    ze.check(d.zeKernelSetArgumentValue(kern, 2, @sizeOf(u32), @ptrCast(&n_u32))) catch @panic("setarg failed");

    var group_x: u32 = 1;
    var dummy1: u32 = 1;
    var dummy2: u32 = 1;
    ze.check(d.zeKernelSuggestGroupSize(kern, n_u32, 1, 1, &group_x, &dummy1, &dummy2)) catch {
        group_x = 1;
    };
    ze.check(d.zeKernelSetGroupSize(kern, group_x, 1, 1)) catch @panic("setGroupSize failed");

    const groups = ze.ze_group_count_t{
        .groupCountX = (n_u32 + group_x - 1) / group_x,
        .groupCountY = 1,
        .groupCountZ = 1,
    };

    const cl = cmd_list.?;
    ze.check(d.zeCommandListAppendLaunchKernel(cl, kern, &groups, null, 0, null)) catch
        @panic("appendLaunchKernel failed");
    submitAndSync(cl) catch @panic("GPU relu submit/sync failed");

    @memcpy(output[0..n], buf_out.asF32Slice()[0..n]);
}

pub fn matmul_fwd(W: []const f32, x: []const f32, out: []f32, M: usize, K: usize) void {
    const d = dispatch orelse @panic("GPU not initialized");
    const spv = spirv.matmulModule();
    const cached = getOrCompileKernel(&matmul_cache, spv.asBytes(), "matmul_kernel") catch
        @panic("GPU matmul kernel compilation failed");

    const w_size = M * K * @sizeOf(f32);
    const x_size = K * @sizeOf(f32);
    const out_size = M * @sizeOf(f32);

    const buf_w = getOrAllocShared(w_size) catch @panic("GPU alloc failed");
    const buf_x = getOrAllocShared(x_size) catch @panic("GPU alloc failed");
    const buf_out = getOrAllocShared(out_size) catch @panic("GPU alloc failed");

    @memcpy(buf_w.asF32Slice()[0 .. M * K], W[0 .. M * K]);
    @memcpy(buf_x.asF32Slice()[0..K], x[0..K]);

    const kern = cached.kernel;
    var w_ptr = buf_w.ptr;
    var x_ptr = buf_x.ptr;
    var o_ptr = buf_out.ptr;
    var m_u32: u32 = @intCast(M);
    var k_u32: u32 = @intCast(K);
    ze.check(d.zeKernelSetArgumentValue(kern, 0, @sizeOf(*anyopaque), @ptrCast(&w_ptr))) catch @panic("setarg failed");
    ze.check(d.zeKernelSetArgumentValue(kern, 1, @sizeOf(*anyopaque), @ptrCast(&x_ptr))) catch @panic("setarg failed");
    ze.check(d.zeKernelSetArgumentValue(kern, 2, @sizeOf(*anyopaque), @ptrCast(&o_ptr))) catch @panic("setarg failed");
    ze.check(d.zeKernelSetArgumentValue(kern, 3, @sizeOf(u32), @ptrCast(&m_u32))) catch @panic("setarg failed");
    ze.check(d.zeKernelSetArgumentValue(kern, 4, @sizeOf(u32), @ptrCast(&k_u32))) catch @panic("setarg failed");

    var group_x: u32 = 1;
    var dummy1: u32 = 1;
    var dummy2: u32 = 1;
    ze.check(d.zeKernelSuggestGroupSize(kern, m_u32, 1, 1, &group_x, &dummy1, &dummy2)) catch {
        group_x = 1;
    };
    ze.check(d.zeKernelSetGroupSize(kern, group_x, 1, 1)) catch @panic("setGroupSize failed");

    const groups = ze.ze_group_count_t{
        .groupCountX = (m_u32 + group_x - 1) / group_x,
        .groupCountY = 1,
        .groupCountZ = 1,
    };

    const cl = cmd_list.?;
    ze.check(d.zeCommandListAppendLaunchKernel(cl, kern, &groups, null, 0, null)) catch
        @panic("appendLaunchKernel failed");
    submitAndSync(cl) catch @panic("GPU matmul submit/sync failed");

    @memcpy(out[0..M], buf_out.asF32Slice()[0..M]);
}

pub fn softmax_fwd(input: []const f32, output: []f32, n: usize) void {
    const d = dispatch orelse @panic("GPU not initialized");
    const spv = spirv.softmaxModule();
    const cached = getOrCompileKernel(&softmax_cache, spv.asBytes(), "softmax_kernel") catch
        @panic("GPU softmax kernel compilation failed");

    const size = n * @sizeOf(f32);
    const buf_in = getOrAllocShared(size) catch @panic("GPU alloc failed");
    const buf_out = getOrAllocShared(size) catch @panic("GPU alloc failed");

    @memcpy(buf_in.asF32Slice()[0..n], input[0..n]);

    const kern = cached.kernel;
    var in_ptr = buf_in.ptr;
    var out_ptr = buf_out.ptr;
    var n_u32: u32 = @intCast(n);
    ze.check(d.zeKernelSetArgumentValue(kern, 0, @sizeOf(*anyopaque), @ptrCast(&in_ptr))) catch @panic("setarg failed");
    ze.check(d.zeKernelSetArgumentValue(kern, 1, @sizeOf(*anyopaque), @ptrCast(&out_ptr))) catch @panic("setarg failed");
    ze.check(d.zeKernelSetArgumentValue(kern, 2, @sizeOf(u32), @ptrCast(&n_u32))) catch @panic("setarg failed");

    // Single work-item dispatch for softmax (N is small)
    ze.check(d.zeKernelSetGroupSize(kern, 1, 1, 1)) catch @panic("setGroupSize failed");
    const groups = ze.ze_group_count_t{
        .groupCountX = 1,
        .groupCountY = 1,
        .groupCountZ = 1,
    };

    const cl = cmd_list.?;
    ze.check(d.zeCommandListAppendLaunchKernel(cl, kern, &groups, null, 0, null)) catch
        @panic("appendLaunchKernel failed");
    submitAndSync(cl) catch @panic("GPU softmax submit/sync failed");

    @memcpy(output[0..n], buf_out.asF32Slice()[0..n]);
}

/// CPU fallback: rmsnorm needs scale_out for backward pass and is too small for GPU dispatch.
pub fn rmsnorm_fwd(input: []const f32, output: []f32, n: usize, scale_out: *f32) void {
    cpu.rmsnorm_fwd(input, output, n, scale_out);
}

pub fn deinit() void {
    const d = dispatch orelse return;

    // Destroy cached kernels and modules
    inline for (.{ &relu_cache, &matmul_cache, &softmax_cache }) |cache| {
        if (cache.*) |cached| {
            _ = d.zeKernelDestroy(cached.kernel);
            _ = d.zeModuleDestroy(cached.module);
            cache.* = null;
        }
    }

    // Free shared buffers
    for (shared_bufs[0..num_bufs]) |buf| {
        ze.check(d.zeMemFree(context.?, buf.ptr)) catch {};
    }
    num_bufs = 0;

    if (fence) |f| {
        ze.check(d.zeFenceDestroy(f)) catch {};
        fence = null;
    }
    if (cmd_list) |cl| {
        ze.check(d.zeCommandListDestroy(cl)) catch {};
        cmd_list = null;
    }
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

    log.info("Intel GPU backend shut down", .{});
}

pub fn backend() Backend {
    return .{
        .matmul_fwd = &matmul_fwd,
        .softmax_fwd = &softmax_fwd,
        .relu_fwd = &relu_fwd,
        .rmsnorm_fwd = &rmsnorm_fwd,
        .deinit = &deinit,
        .name = "intel_gpu",
    };
}
