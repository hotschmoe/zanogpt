const std = @import("std");

// Level-Zero NPU headers — uncomment when SDK is available on Windows:
// const ze = @cImport({
//     @cInclude("ze_api.h");
//     @cInclude("ze_graph_ext.h");
// });

// --- Backend contract (forward ops) ---

pub fn matmul_fwd(_: []const f32, _: []const f32, _: []f32, _: usize, _: usize) void {
    @panic("Intel NPU backend not yet implemented");
}

pub fn softmax_fwd(_: []const f32, _: []f32, _: usize) void {
    @panic("Intel NPU backend not yet implemented");
}

pub fn rmsnorm_fwd(_: []const f32, _: []f32, _: usize, _: *f32) void {
    @panic("Intel NPU backend not yet implemented");
}

pub fn relu_fwd(_: []const f32, _: []f32, _: usize) void {
    @panic("Intel NPU backend not yet implemented");
}

// --- Graph cache skeleton ---

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

// Compiled graph handle — will hold ze_graph_handle_t once Level-Zero is linked.
pub const GraphHandle = usize;

pub var graph_cache: ?std.HashMap(ShapeKey, GraphHandle, struct {
    pub fn hash(_: @This(), key: ShapeKey) u64 {
        return key.hash();
    }
    pub fn eql(_: @This(), a: ShapeKey, b: ShapeKey) bool {
        return a.eql(b);
    }
}, std.hash_map.default_max_load_percentage) = null;

// --- NPU lifecycle ---

pub fn init() void {
    @panic("Intel NPU init not yet implemented");
}

pub fn deinit() void {
    @panic("Intel NPU deinit not yet implemented");
}
