pub const Backend = struct {
    matmul_fwd: *const fn ([]const f32, []const f32, []f32, usize, usize) void,
    softmax_fwd: *const fn ([]const f32, []f32, usize) void,
    relu_fwd: *const fn ([]const f32, []f32, usize) void,
    rmsnorm_fwd: *const fn ([]const f32, []f32, usize, *f32) void,
    deinit: *const fn () void,
    name: []const u8,
};
