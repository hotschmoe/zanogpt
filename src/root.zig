const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;
const build_options = @import("build_options");

pub const backend = switch (build_options.backend) {
    .cpu => @import("backends/cpu.zig"),
    .intel_npu => @import("backends/intel_npu.zig"),
};

pub const Tensor = struct {
    data: []f32,
    grad: ?[]f32,
    shape: [MAX_DIMS]usize,
    ndim: u8,
    requires_grad: bool,
    allocator: Allocator,

    pub const MAX_DIMS = 4;

    pub fn init(allocator: Allocator, shape: []const usize, requires_grad: bool) !*Tensor {
        const t = try allocator.create(Tensor);
        var s: [MAX_DIMS]usize = .{ 0, 0, 0, 0 };
        var n: usize = 1;
        for (0..shape.len) |i| {
            s[i] = shape[i];
            n *= shape[i];
        }
        t.* = .{
            .data = try allocator.alloc(f32, n),
            .grad = null,
            .shape = s,
            .ndim = @intCast(shape.len),
            .requires_grad = requires_grad,
            .allocator = allocator,
        };
        @memset(t.data, 0);
        return t;
    }

    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.data);
        if (self.grad) |g| self.allocator.free(g);
        self.allocator.destroy(self);
    }

    pub fn numel(self: *const Tensor) usize {
        var n: usize = 1;
        for (0..self.ndim) |i| n *= self.shape[i];
        return n;
    }

    pub fn ensureGrad(self: *Tensor) !void {
        if (self.grad == null) {
            self.grad = try self.allocator.alloc(f32, self.numel());
            @memset(self.grad.?, 0);
        }
    }

    pub fn zeroGrad(self: *Tensor) void {
        if (self.grad) |g| @memset(g, 0);
    }

    pub fn fillRandom(self: *Tensor, rng: *std.Random.Xoshiro256, std_dev: f32) void {
        for (self.data) |*d| {
            const r1 = rng.random().float(f64);
            const r2 = rng.random().float(f64);
            const z: f32 = @floatCast(@sqrt(-2.0 * @log(r1)) * @cos(2.0 * math.pi * r2));
            d.* = z * std_dev;
        }
    }
};


pub const OpKind = enum {
    leaf,
    embedding_lookup,
    add,
    matmul,
    rmsnorm,
    relu,
    mul_scalar,
    softmax,
    nll_loss,
    attention,
};

pub const TapeEntry = struct {
    op: OpKind,
    output: usize,
    inputs: [3]usize = .{ 0, 0, 0 },
    n_inputs: u8 = 0,
    saved_f32: f32 = 0,
    saved_usize: usize = 0,
    saved_usize2: usize = 0,
    extra: ?[]const usize = null,
};

pub const Tape = struct {
    entries: std.ArrayList(TapeEntry),
    tensors: std.ArrayList(*Tensor),
    arena: Allocator,

    pub fn init(arena: Allocator) Tape {
        return .{
            .entries = .empty,
            .tensors = .empty,
            .arena = arena,
        };
    }

    pub fn register(self: *Tape, t: *Tensor) !usize {
        const idx = self.tensors.items.len;
        try self.tensors.append(self.arena, t);
        try self.entries.append(self.arena, .{
            .op = .leaf,
            .output = idx,
        });
        return idx;
    }

    pub fn get(self: *const Tape, idx: usize) *Tensor {
        return self.tensors.items[idx];
    }

    fn newTensor(self: *Tape, shape: []const usize) !usize {
        const t = try Tensor.init(self.arena, shape, false);
        const idx = self.tensors.items.len;
        try self.tensors.append(self.arena, t);
        return idx;
    }

    fn record(self: *Tape, entry: TapeEntry) !void {
        try self.entries.append(self.arena, entry);
    }

    pub fn embeddingLookup(self: *Tape, weight_idx: usize, token_id: usize) !usize {
        const W = self.get(weight_idx);
        const K = W.shape[1];
        const out_idx = try self.newTensor(&.{K});
        const out = self.get(out_idx);
        @memcpy(out.data[0..K], W.data[token_id * K ..][0..K]);

        try self.record(.{
            .op = .embedding_lookup,
            .output = out_idx,
            .inputs = .{ weight_idx, 0, 0 },
            .n_inputs = 1,
            .saved_usize = token_id,
        });
        return out_idx;
    }

    pub fn addOp(self: *Tape, a_idx: usize, b_idx: usize) !usize {
        const a = self.get(a_idx);
        const n = a.numel();
        const out_idx = try self.newTensor(a.shape[0..a.ndim]);
        const out = self.get(out_idx);
        for (0..n) |i| out.data[i] = a.data[i] + self.get(b_idx).data[i];

        try self.record(.{
            .op = .add,
            .output = out_idx,
            .inputs = .{ a_idx, b_idx, 0 },
            .n_inputs = 2,
        });
        return out_idx;
    }

    pub fn matmulOp(self: *Tape, w_idx: usize, x_idx: usize) !usize {
        const W = self.get(w_idx);
        const x = self.get(x_idx);
        const M = W.shape[0];
        const K = W.shape[1];
        const out_idx = try self.newTensor(&.{M});
        const out = self.get(out_idx);
        backend.matmul_fwd(W.data, x.data, out.data, M, K);

        try self.record(.{
            .op = .matmul,
            .output = out_idx,
            .inputs = .{ w_idx, x_idx, 0 },
            .n_inputs = 2,
        });
        return out_idx;
    }

    pub fn rmsnormOp(self: *Tape, a_idx: usize) !usize {
        const a = self.get(a_idx);
        const n = a.numel();
        const out_idx = try self.newTensor(a.shape[0..a.ndim]);
        const out = self.get(out_idx);
        var scale: f32 = 0;
        backend.rmsnorm_fwd(a.data, out.data, n, &scale);

        try self.record(.{
            .op = .rmsnorm,
            .output = out_idx,
            .inputs = .{ a_idx, 0, 0 },
            .n_inputs = 1,
            .saved_f32 = scale,
        });
        return out_idx;
    }

    pub fn reluOp(self: *Tape, a_idx: usize) !usize {
        const a = self.get(a_idx);
        const n = a.numel();
        const out_idx = try self.newTensor(a.shape[0..a.ndim]);
        const out = self.get(out_idx);
        backend.relu_fwd(a.data, out.data, n);

        try self.record(.{
            .op = .relu,
            .output = out_idx,
            .inputs = .{ a_idx, 0, 0 },
            .n_inputs = 1,
        });
        return out_idx;
    }

    pub fn mulScalarOp(self: *Tape, a_idx: usize, s: f32) !usize {
        const a = self.get(a_idx);
        const n = a.numel();
        const out_idx = try self.newTensor(a.shape[0..a.ndim]);
        const out = self.get(out_idx);
        for (0..n) |i| out.data[i] = a.data[i] * s;

        try self.record(.{
            .op = .mul_scalar,
            .output = out_idx,
            .inputs = .{ a_idx, 0, 0 },
            .n_inputs = 1,
            .saved_f32 = s,
        });
        return out_idx;
    }

    pub fn softmaxOp(self: *Tape, a_idx: usize) !usize {
        const a = self.get(a_idx);
        const n = a.numel();
        const out_idx = try self.newTensor(a.shape[0..a.ndim]);
        const out = self.get(out_idx);
        backend.softmax_fwd(a.data, out.data, n);

        try self.record(.{
            .op = .softmax,
            .output = out_idx,
            .inputs = .{ a_idx, 0, 0 },
            .n_inputs = 1,
        });
        return out_idx;
    }

    pub fn nllLoss(self: *Tape, probs_idx: usize, target_id: usize) !usize {
        const probs = self.get(probs_idx);
        const out_idx = try self.newTensor(&.{1});
        const out = self.get(out_idx);
        out.data[0] = -@log(probs.data[target_id]);

        try self.record(.{
            .op = .nll_loss,
            .output = out_idx,
            .inputs = .{ probs_idx, 0, 0 },
            .n_inputs = 1,
            .saved_usize = target_id,
        });
        return out_idx;
    }

    pub fn attentionOp(
        self: *Tape,
        q_idx: usize,
        k_cache: []const usize,
        v_cache: []const usize,
        n_head: usize,
        head_dim: usize,
    ) !usize {
        const q = self.get(q_idx);
        const n_embd = q.numel();
        const n_cached = k_cache.len;
        const out_idx = try self.newTensor(&.{n_embd});
        const out = self.get(out_idx);
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        const scores = try self.arena.alloc(f32, n_cached);
        const weights = try self.arena.alloc(f32, n_cached);

        for (0..n_head) |h| {
            const hs = h * head_dim;

            for (0..n_cached) |t| {
                const k_t = self.get(k_cache[t]);
                var dot: f32 = 0;
                for (0..head_dim) |d| {
                    dot += q.data[hs + d] * k_t.data[hs + d];
                }
                scores[t] = dot * scale;
            }
            backend.softmax_fwd(scores, weights, n_cached);
            for (0..head_dim) |d| {
                var sum: f32 = 0;
                for (0..n_cached) |t| {
                    const v_t = self.get(v_cache[t]);
                    sum += weights[t] * v_t.data[hs + d];
                }
                out.data[hs + d] = sum;
            }
        }

        const extra = try self.arena.alloc(usize, n_cached * 2);
        @memcpy(extra[0..n_cached], k_cache);
        @memcpy(extra[n_cached..], v_cache);

        try self.record(.{
            .op = .attention,
            .output = out_idx,
            .inputs = .{ q_idx, 0, 0 },
            .n_inputs = 1,
            .saved_f32 = scale,
            .saved_usize = n_head,
            .saved_usize2 = head_dim,
            .extra = extra,
        });
        return out_idx;
    }

    pub fn backward(self: *Tape, loss_idx: usize) !void {
        for (self.tensors.items) |t| try t.ensureGrad();
        self.get(loss_idx).grad.?[0] = 1.0;

        var i: usize = self.entries.items.len;
        while (i > 0) {
            i -= 1;
            const entry = self.entries.items[i];
            const d_out = self.get(entry.output).grad.?;

            switch (entry.op) {
                .leaf => {},

                .embedding_lookup => {
                    const W = self.get(entry.inputs[0]);
                    const K = W.shape[1];
                    const idx = entry.saved_usize;
                    const w_grad = W.grad.?;
                    for (0..K) |j| w_grad[idx * K + j] += d_out[j];
                },

                .add => {
                    const a_grad = self.get(entry.inputs[0]).grad.?;
                    const b_grad = self.get(entry.inputs[1]).grad.?;
                    const n = self.get(entry.output).numel();
                    for (0..n) |j| {
                        a_grad[j] += d_out[j];
                        b_grad[j] += d_out[j];
                    }
                },

                .matmul => {
                    const W = self.get(entry.inputs[0]);
                    const x = self.get(entry.inputs[1]);
                    const w_grad = W.grad.?;
                    const x_grad = x.grad.?;
                    const M = W.shape[0];
                    const K = W.shape[1];
                    for (0..M) |row| {
                        for (0..K) |j| {
                            w_grad[row * K + j] += d_out[row] * x.data[j];
                        }
                    }
                    for (0..K) |j| {
                        var sum: f32 = 0;
                        for (0..M) |row| sum += W.data[row * K + j] * d_out[row];
                        x_grad[j] += sum;
                    }
                },

                .rmsnorm => {
                    const a = self.get(entry.inputs[0]);
                    const a_grad = a.grad.?;
                    const scale = entry.saved_f32;
                    const n = a.numel();
                    const n_f: f32 = @floatFromInt(n);
                    var dot_gx: f32 = 0;
                    for (0..n) |j| dot_gx += d_out[j] * a.data[j];
                    const coeff = scale * scale * scale * dot_gx / n_f;
                    for (0..n) |j| {
                        a_grad[j] += scale * d_out[j] - coeff * a.data[j];
                    }
                },

                .relu => {
                    const a = self.get(entry.inputs[0]);
                    const a_grad = a.grad.?;
                    const n = a.numel();
                    for (0..n) |j| {
                        if (a.data[j] > 0) a_grad[j] += d_out[j];
                    }
                },

                .mul_scalar => {
                    const a_grad = self.get(entry.inputs[0]).grad.?;
                    const n = self.get(entry.output).numel();
                    const s = entry.saved_f32;
                    for (0..n) |j| a_grad[j] += d_out[j] * s;
                },

                .softmax => {
                    const out_t = self.get(entry.output);
                    const a_grad = self.get(entry.inputs[0]).grad.?;
                    const n = out_t.numel();
                    var dot_go: f32 = 0;
                    for (0..n) |j| dot_go += d_out[j] * out_t.data[j];
                    for (0..n) |j| {
                        a_grad[j] += out_t.data[j] * (d_out[j] - dot_go);
                    }
                },

                .nll_loss => {
                    const probs = self.get(entry.inputs[0]);
                    const probs_grad = probs.grad.?;
                    const target = entry.saved_usize;
                    probs_grad[target] += -d_out[0] / probs.data[target];
                },

                .attention => try self.backwardAttention(entry, d_out),
            }
        }
    }

    fn backwardAttention(self: *Tape, entry: TapeEntry, d_out: []f32) !void {
        const q = self.get(entry.inputs[0]);
        const q_grad = q.grad.?;
        const scale = entry.saved_f32;
        const n_head = entry.saved_usize;
        const head_dim = entry.saved_usize2;
        const extra = entry.extra.?;
        const n_cached = extra.len / 2;
        const k_cache = extra[0..n_cached];
        const v_cache = extra[n_cached..];

        const scores = try self.arena.alloc(f32, n_cached);
        const weights = try self.arena.alloc(f32, n_cached);
        const d_weights = try self.arena.alloc(f32, n_cached);

        for (0..n_head) |h| {
            const hs = h * head_dim;

            for (0..n_cached) |t| {
                const k_t = self.get(k_cache[t]);
                var dot: f32 = 0;
                for (0..head_dim) |d| dot += q.data[hs + d] * k_t.data[hs + d];
                scores[t] = dot * scale;
            }
            backend.softmax_fwd(scores, weights, n_cached);

            @memset(d_weights, 0);
            for (0..head_dim) |d| {
                for (0..n_cached) |t| {
                    const v_t = self.get(v_cache[t]);
                    d_weights[t] += d_out[hs + d] * v_t.data[hs + d];
                    v_t.grad.?[hs + d] += d_out[hs + d] * weights[t];
                }
            }

            var dot_wdw: f32 = 0;
            for (0..n_cached) |t| dot_wdw += d_weights[t] * weights[t];

            for (0..n_cached) |t| {
                const d_score = weights[t] * (d_weights[t] - dot_wdw);
                const d_raw = d_score * scale;
                const k_t = self.get(k_cache[t]);
                const k_grad = k_t.grad.?;
                for (0..head_dim) |d| {
                    q_grad[hs + d] += d_raw * k_t.data[hs + d];
                    k_grad[hs + d] += d_raw * q.data[hs + d];
                }
            }
        }
    }
};

test "Tensor creation and numel" {
    const alloc = std.testing.allocator;
    const t = try Tensor.init(alloc, &.{ 3, 4 }, false);
    defer t.deinit();
    try std.testing.expectEqual(@as(usize, 12), t.numel());
    try std.testing.expectEqual(@as(u8, 2), t.ndim);
    try std.testing.expectEqual(@as(usize, 3), t.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), t.shape[1]);
}

test "CPU matmul correctness" {
    // W = [[1,2],[3,4],[5,6]], x = [1,1] → out = [3, 7, 11]
    const W = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const x = [_]f32{ 1, 1 };
    var out: [3]f32 = undefined;
    backend.matmul_fwd(&W, &x, &out, 3, 2);
    try std.testing.expectApproxEqAbs(@as(f32, 3), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 7), out[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 11), out[2], 1e-5);
}

test "CPU softmax sums to 1" {
    const input = [_]f32{ 1, 2, 3 };
    var output: [3]f32 = undefined;
    backend.softmax_fwd(&input, &output, 3);
    var sum: f32 = 0;
    for (&output) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
    try std.testing.expect(output[2] > output[1]);
    try std.testing.expect(output[1] > output[0]);
}

test "CPU rmsnorm unit scale" {
    const input = [_]f32{ 1, 2, 3 };
    var output: [3]f32 = undefined;
    var scale: f32 = undefined;
    backend.rmsnorm_fwd(&input, &output, 3, &scale);
    var rms_sq: f32 = 0;
    for (&output) |v| rms_sq += v * v;
    rms_sq /= 3.0;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), @sqrt(rms_sq), 1e-3);
}

test "Tape: matmul gradient check (finite differences)" {
    const alloc = std.testing.allocator;
    const eps: f32 = 1e-3;

    // W = [[1,2],[3,4]], x = [1, -0.5]
    // loss = nll_loss(softmax(W @ x), target=0)
    const W = try Tensor.init(alloc, &.{ 2, 2 }, true);
    defer W.deinit();
    W.data[0] = 1;
    W.data[1] = 2;
    W.data[2] = 3;
    W.data[3] = 4;

    const x = try Tensor.init(alloc, &.{2}, true);
    defer x.deinit();
    x.data[0] = 1;
    x.data[1] = -0.5;

    var arena_impl = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_impl.deinit();

    // Compute analytical gradients
    {
        const arena = arena_impl.allocator();
        var tape = Tape.init(arena);
        const wi = try tape.register(W);
        const xi = try tape.register(x);
        const logits = try tape.matmulOp(wi, xi);
        const probs = try tape.softmaxOp(logits);
        const loss = try tape.nllLoss(probs, 0);
        try tape.backward(loss);
    }

    // Verify each W gradient element with finite differences
    for (0..4) |idx| {
        const orig = W.data[idx];

        W.data[idx] = orig + eps;
        _ = arena_impl.reset(.retain_capacity);
        const loss_plus = blk: {
            const arena = arena_impl.allocator();
            var tape = Tape.init(arena);
            const wi = try tape.register(W);
            const xi = try tape.register(x);
            const logits = try tape.matmulOp(wi, xi);
            const probs = try tape.softmaxOp(logits);
            const loss = try tape.nllLoss(probs, 0);
            break :blk tape.get(loss).data[0];
        };

        W.data[idx] = orig - eps;
        _ = arena_impl.reset(.retain_capacity);
        const loss_minus = blk: {
            const arena = arena_impl.allocator();
            var tape = Tape.init(arena);
            const wi = try tape.register(W);
            const xi = try tape.register(x);
            const logits = try tape.matmulOp(wi, xi);
            const probs = try tape.softmaxOp(logits);
            const loss = try tape.nllLoss(probs, 0);
            break :blk tape.get(loss).data[0];
        };

        W.data[idx] = orig;
        const numerical = (loss_plus - loss_minus) / (2.0 * eps);
        try std.testing.expectApproxEqAbs(numerical, W.grad.?[idx], 1e-2);
    }
}

test "Tape: add and mul_scalar forward" {
    const alloc = std.testing.allocator;
    var arena_impl = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_impl.deinit();
    const arena = arena_impl.allocator();

    const a = try Tensor.init(alloc, &.{3}, true);
    defer a.deinit();
    a.data[0] = 1;
    a.data[1] = 2;
    a.data[2] = 3;

    const b = try Tensor.init(alloc, &.{3}, true);
    defer b.deinit();
    b.data[0] = 4;
    b.data[1] = 5;
    b.data[2] = 6;

    var tape = Tape.init(arena);
    const ai = try tape.register(a);
    const bi = try tape.register(b);
    const sum_idx = try tape.addOp(ai, bi);
    const scaled = try tape.mulScalarOp(sum_idx, 2.0);

    // (a + b) * 2 = [10, 14, 18]
    const out = tape.get(scaled);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), out.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 14.0), out.data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 18.0), out.data[2], 1e-5);
}

test "Tape: nll_loss forward + backward" {
    const alloc = std.testing.allocator;
    var arena_impl = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_impl.deinit();
    const arena = arena_impl.allocator();

    // probs = [0.2, 0.3, 0.5], target = 1
    // loss = -log(0.3) ≈ 1.2039
    const probs = try Tensor.init(alloc, &.{3}, true);
    defer probs.deinit();
    probs.data[0] = 0.2;
    probs.data[1] = 0.3;
    probs.data[2] = 0.5;

    var tape = Tape.init(arena);
    const pi = try tape.register(probs);
    const loss_idx = try tape.nllLoss(pi, 1);

    try std.testing.expectApproxEqAbs(-@log(@as(f32, 0.3)), tape.get(loss_idx).data[0], 1e-5);

    try tape.backward(loss_idx);

    // d_probs[1] = -1/0.3 ≈ -3.333
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), probs.grad.?[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0 / 0.3), probs.grad.?[1], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), probs.grad.?[2], 1e-5);
}

test "Tape: full gradient check with finite differences" {
    const alloc = std.testing.allocator;

    // Simple model: loss = -log(softmax(W @ x)[target])
    const W = try Tensor.init(alloc, &.{ 3, 2 }, true);
    defer W.deinit();
    W.data[0] = 0.1;
    W.data[1] = 0.2;
    W.data[2] = 0.3;
    W.data[3] = 0.4;
    W.data[4] = 0.5;
    W.data[5] = 0.6;

    const x = try Tensor.init(alloc, &.{2}, true);
    defer x.deinit();
    x.data[0] = 1.0;
    x.data[1] = -0.5;

    const target: usize = 1;
    const eps: f32 = 1e-3;

    // Compute analytical gradients
    var arena_impl = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_impl.deinit();
    {
        const arena = arena_impl.allocator();
        var tape = Tape.init(arena);
        const wi = try tape.register(W);
        const xi = try tape.register(x);
        const logits = try tape.matmulOp(wi, xi);
        const probs = try tape.softmaxOp(logits);
        const loss = try tape.nllLoss(probs, target);
        try tape.backward(loss);
    }

    // Check each W gradient with finite differences
    for (0..6) |idx| {
        const orig = W.data[idx];
        W.data[idx] = orig + eps;
        _ = arena_impl.reset(.retain_capacity);
        const loss_plus = blk: {
            const arena = arena_impl.allocator();
            var tape = Tape.init(arena);
            const wi = try tape.register(W);
            const xi = try tape.register(x);
            const logits = try tape.matmulOp(wi, xi);
            const probs = try tape.softmaxOp(logits);
            const loss = try tape.nllLoss(probs, target);
            break :blk tape.get(loss).data[0];
        };
        W.data[idx] = orig - eps;
        _ = arena_impl.reset(.retain_capacity);
        const loss_minus = blk: {
            const arena = arena_impl.allocator();
            var tape = Tape.init(arena);
            const wi = try tape.register(W);
            const xi = try tape.register(x);
            const logits = try tape.matmulOp(wi, xi);
            const probs = try tape.softmaxOp(logits);
            const loss = try tape.nllLoss(probs, target);
            break :blk tape.get(loss).data[0];
        };
        W.data[idx] = orig;

        const numerical = (loss_plus - loss_minus) / (2.0 * eps);
        const analytical = W.grad.?[idx];
        try std.testing.expectApproxEqAbs(numerical, analytical, 1e-2);
    }
}
