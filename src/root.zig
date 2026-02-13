const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;

// ============================================================================
// Value: Autograd scalar node
// ============================================================================

pub const Value = struct {
    data: f64,
    grad: f64 = 0,
    n_children: u2 = 0,
    children: [2]*Value = undefined,
    local_grads: [2]f64 = undefined,

    pub fn create(alloc: Allocator, data: f64) !*Value {
        const v = try alloc.create(Value);
        v.* = .{ .data = data };
        return v;
    }

    // --- Binary ops ---

    pub fn add(self: *Value, other: *Value, alloc: Allocator) !*Value {
        const out = try alloc.create(Value);
        out.* = .{
            .data = self.data + other.data,
            .n_children = 2,
            .children = .{ self, other },
            .local_grads = .{ 1.0, 1.0 },
        };
        return out;
    }

    pub fn mul(self: *Value, other: *Value, alloc: Allocator) !*Value {
        const out = try alloc.create(Value);
        out.* = .{
            .data = self.data * other.data,
            .n_children = 2,
            .children = .{ self, other },
            .local_grads = .{ other.data, self.data },
        };
        return out;
    }

    pub fn sub(self: *Value, other: *Value, alloc: Allocator) !*Value {
        const out = try alloc.create(Value);
        out.* = .{
            .data = self.data - other.data,
            .n_children = 2,
            .children = .{ self, other },
            .local_grads = .{ 1.0, -1.0 },
        };
        return out;
    }

    // --- Scalar ops ---

    pub fn addScalar(self: *Value, s: f64, alloc: Allocator) !*Value {
        const out = try alloc.create(Value);
        out.* = .{
            .data = self.data + s,
            .n_children = 1,
            .children = .{ self, undefined },
            .local_grads = .{ 1.0, undefined },
        };
        return out;
    }

    pub fn mulScalar(self: *Value, s: f64, alloc: Allocator) !*Value {
        const out = try alloc.create(Value);
        out.* = .{
            .data = self.data * s,
            .n_children = 1,
            .children = .{ self, undefined },
            .local_grads = .{ s, undefined },
        };
        return out;
    }

    // --- Unary ops ---

    pub fn pow(self: *Value, exponent: f64, alloc: Allocator) !*Value {
        const out = try alloc.create(Value);
        out.* = .{
            .data = math.pow(f64, self.data, exponent),
            .n_children = 1,
            .children = .{ self, undefined },
            .local_grads = .{ exponent * math.pow(f64, self.data, exponent - 1.0), undefined },
        };
        return out;
    }

    pub fn log(self: *Value, alloc: Allocator) !*Value {
        const out = try alloc.create(Value);
        out.* = .{
            .data = @log(self.data),
            .n_children = 1,
            .children = .{ self, undefined },
            .local_grads = .{ 1.0 / self.data, undefined },
        };
        return out;
    }

    pub fn exp(self: *Value, alloc: Allocator) !*Value {
        const e = @exp(self.data);
        const out = try alloc.create(Value);
        out.* = .{
            .data = e,
            .n_children = 1,
            .children = .{ self, undefined },
            .local_grads = .{ e, undefined },
        };
        return out;
    }

    pub fn relu(self: *Value, alloc: Allocator) !*Value {
        const out = try alloc.create(Value);
        out.* = .{
            .data = @max(0.0, self.data),
            .n_children = 1,
            .children = .{ self, undefined },
            .local_grads = .{ if (self.data > 0) 1.0 else 0.0, undefined },
        };
        return out;
    }

    pub fn negate(self: *Value, alloc: Allocator) !*Value {
        return self.mulScalar(-1.0, alloc);
    }

    // --- Backward pass ---

    pub fn backward(self: *Value, alloc: Allocator) !void {
        // Iterative topological sort using DFS
        var topo: std.ArrayList(*Value) = .empty;
        defer topo.deinit(alloc);

        // Use a simple ArrayList-based visited set via pointer comparison
        var visited_list: std.ArrayList(*Value) = .empty;
        defer visited_list.deinit(alloc);

        const Frame = struct { node: *Value, child_idx: u2 };
        var stack: std.ArrayList(Frame) = .empty;
        defer stack.deinit(alloc);

        try stack.append(alloc, .{ .node = self, .child_idx = 0 });
        try visited_list.append(alloc, self);

        while (stack.items.len > 0) {
            const top = &stack.items[stack.items.len - 1];
            if (top.child_idx < top.node.n_children) {
                const child = top.node.children[top.child_idx];
                top.child_idx += 1;
                const already_visited = for (visited_list.items) |v| {
                    if (v == child) break true;
                } else false;
                if (!already_visited) {
                    try visited_list.append(alloc, child);
                    try stack.append(alloc, .{ .node = child, .child_idx = 0 });
                }
            } else {
                try topo.append(alloc, top.node);
                _ = stack.pop();
            }
        }

        // Reverse pass
        self.grad = 1.0;
        var i: usize = topo.items.len;
        while (i > 0) {
            i -= 1;
            const v = topo.items[i];
            for (0..v.n_children) |ci| {
                v.children[ci].grad += v.local_grads[ci] * v.grad;
            }
        }
    }
};

// ============================================================================
// Helper functions: linear, softmax, rmsnorm
// ============================================================================

/// Matrix-vector multiply: result[i] = sum_j(w[i][j] * x[j])
pub fn linear(x: []*Value, w: []const []*Value, alloc: Allocator) ![]*Value {
    const nout = w.len;
    const result = try alloc.alloc(*Value, nout);
    for (0..nout) |i| {
        var acc = try x[0].mul(w[i][0], alloc);
        for (1..x.len) |j| {
            const prod = try x[j].mul(w[i][j], alloc);
            acc = try acc.add(prod, alloc);
        }
        result[i] = acc;
    }
    return result;
}

/// Numerically stable softmax
pub fn softmax(logits: []*Value, alloc: Allocator) ![]*Value {
    // Find max for numerical stability
    var max_val: f64 = logits[0].data;
    for (logits[1..]) |v| {
        if (v.data > max_val) max_val = v.data;
    }
    // Compute exp(x - max)
    const n = logits.len;
    const exps = try alloc.alloc(*Value, n);
    for (0..n) |i| {
        const shifted = try logits[i].addScalar(-max_val, alloc);
        exps[i] = try shifted.exp(alloc);
    }
    // Sum
    var total = exps[0];
    for (1..n) |i| {
        total = try total.add(exps[i], alloc);
    }
    // Divide
    const inv_total = try total.pow(-1.0, alloc);
    const result = try alloc.alloc(*Value, n);
    for (0..n) |i| {
        result[i] = try exps[i].mul(inv_total, alloc);
    }
    return result;
}

/// RMS normalization
pub fn rmsnorm(x: []*Value, alloc: Allocator) ![]*Value {
    const n = x.len;
    const n_f: f64 = @floatFromInt(n);
    // mean square
    var ms = try x[0].mul(x[0], alloc);
    for (1..n) |i| {
        const sq = try x[i].mul(x[i], alloc);
        ms = try ms.add(sq, alloc);
    }
    ms = try ms.mulScalar(1.0 / n_f, alloc);
    // (ms + eps)^-0.5
    const ms_eps = try ms.addScalar(1e-5, alloc);
    const scale = try ms_eps.pow(-0.5, alloc);
    // scale each element
    const result = try alloc.alloc(*Value, n);
    for (0..n) |i| {
        result[i] = try x[i].mul(scale, alloc);
    }
    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "Value forward pass" {
    const alloc = std.testing.allocator;
    const a = try Value.create(alloc, 2.0);
    defer alloc.destroy(a);
    const b = try Value.create(alloc, 3.0);
    defer alloc.destroy(b);

    var arena_impl = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_impl.deinit();
    const arena = arena_impl.allocator();

    const c = try a.add(b, arena);
    try std.testing.expectApproxEqAbs(5.0, c.data, 1e-10);

    const d = try a.mul(b, arena);
    try std.testing.expectApproxEqAbs(6.0, d.data, 1e-10);

    const e = try a.sub(b, arena);
    try std.testing.expectApproxEqAbs(-1.0, e.data, 1e-10);

    const f = try a.pow(3.0, arena);
    try std.testing.expectApproxEqAbs(8.0, f.data, 1e-10);

    const g = try a.relu(arena);
    try std.testing.expectApproxEqAbs(2.0, g.data, 1e-10);
}

test "Value backward - add and mul" {
    const alloc = std.testing.allocator;
    const a = try Value.create(alloc, 2.0);
    defer alloc.destroy(a);
    const b = try Value.create(alloc, 3.0);
    defer alloc.destroy(b);

    var arena_impl = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_impl.deinit();
    const arena = arena_impl.allocator();

    // loss = a * b + a
    const ab = try a.mul(b, arena);
    const loss = try ab.add(a, arena);
    try loss.backward(alloc);

    // d(loss)/da = b + 1 = 4
    try std.testing.expectApproxEqAbs(4.0, a.grad, 1e-10);
    // d(loss)/db = a = 2
    try std.testing.expectApproxEqAbs(2.0, b.grad, 1e-10);
}

test "Value backward - more complex" {
    const alloc = std.testing.allocator;
    const x = try Value.create(alloc, 3.0);
    defer alloc.destroy(x);

    var arena_impl = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_impl.deinit();
    const arena = arena_impl.allocator();

    // loss = relu(x^2 - 5)
    const x2 = try x.pow(2.0, arena);
    const shifted = try x2.addScalar(-5.0, arena);
    const loss = try shifted.relu(arena);
    try loss.backward(alloc);

    // x^2 - 5 = 4 > 0, so relu passes through
    // d(loss)/dx = 2*x = 6
    try std.testing.expectApproxEqAbs(6.0, x.grad, 1e-10);
}

test "softmax sums to 1" {
    const alloc = std.testing.allocator;

    var arena_impl = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_impl.deinit();
    const arena = arena_impl.allocator();

    var logits: [3]*Value = undefined;
    logits[0] = try Value.create(alloc, 1.0);
    logits[1] = try Value.create(alloc, 2.0);
    logits[2] = try Value.create(alloc, 3.0);
    defer for (&logits) |v| alloc.destroy(v);

    const probs = try softmax(&logits, arena);
    var sum: f64 = 0;
    for (probs) |p| sum += p.data;
    try std.testing.expectApproxEqAbs(1.0, sum, 1e-10);

    // Largest logit should have largest probability
    try std.testing.expect(probs[2].data > probs[1].data);
    try std.testing.expect(probs[1].data > probs[0].data);
}

test "rmsnorm output scale" {
    const alloc = std.testing.allocator;

    var arena_impl = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_impl.deinit();
    const arena = arena_impl.allocator();

    var x: [3]*Value = undefined;
    x[0] = try Value.create(alloc, 1.0);
    x[1] = try Value.create(alloc, 2.0);
    x[2] = try Value.create(alloc, 3.0);
    defer for (&x) |v| alloc.destroy(v);

    const normed = try rmsnorm(&x, arena);
    // RMS of [1,2,3] = sqrt((1+4+9)/3) = sqrt(14/3) ≈ 2.16
    // Each element divided by RMS should give roughly unit-scale output
    var rms_sq: f64 = 0;
    for (normed) |v| rms_sq += v.data * v.data;
    rms_sq /= 3.0;
    // After rmsnorm, the RMS should be approximately 1
    try std.testing.expectApproxEqAbs(1.0, @sqrt(rms_sq), 1e-3);
}
