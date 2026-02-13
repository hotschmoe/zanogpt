const std = @import("std");
const zanogpt = @import("zanogpt");
const Value = zanogpt.Value;

// ============================================================================
// Hyperparameters (matching microgpt.py)
// ============================================================================
const n_embd: usize = 16;
const n_head: usize = 4;
const n_layer: usize = 1;
const block_size: usize = 16;
const head_dim: usize = n_embd / n_head; // 4

// ============================================================================
// Tokenizer
// ============================================================================
const uchars = "abcdefghijklmnopqrstuvwxyz";
const BOS: usize = uchars.len; // 26
const vocab_size: usize = uchars.len + 1; // 27

fn charToToken(ch: u8) usize {
    return ch - 'a';
}

fn tokenToChar(token: usize) u8 {
    return @intCast(token + 'a');
}

// ============================================================================
// PRNG
// ============================================================================
fn gaussRandom(rng: *std.Random.Xoshiro256, std_dev: f64) f64 {
    // Box-Muller transform
    const r1 = rng.random().float(f64);
    const r2 = rng.random().float(f64);
    const z = @sqrt(-2.0 * @log(r1)) * @cos(2.0 * std.math.pi * r2);
    return z * std_dev;
}

// ============================================================================
// Weight matrices: stored as slices of *Value pointers
// ============================================================================
fn initMatrix(
    gpa: std.mem.Allocator,
    rng: *std.Random.Xoshiro256,
    nout: usize,
    nin: usize,
    std_dev: f64,
) ![][]*Value {
    const rows = try gpa.alloc([]*Value, nout);
    for (0..nout) |i| {
        rows[i] = try gpa.alloc(*Value, nin);
        for (0..nin) |j| {
            rows[i][j] = try Value.create(gpa, gaussRandom(rng, std_dev));
        }
    }
    return rows;
}

// ============================================================================
// State dict: all model weight matrices
// ============================================================================
const LayerWeights = struct {
    attn_wq: [][]*Value,
    attn_wk: [][]*Value,
    attn_wv: [][]*Value,
    attn_wo: [][]*Value,
    mlp_fc1: [][]*Value,
    mlp_fc2: [][]*Value,
};

const StateDict = struct {
    wte: [][]*Value, // [vocab_size][n_embd]
    wpe: [][]*Value, // [block_size][n_embd]
    lm_head: [][]*Value, // [vocab_size][n_embd]
    layers: [n_layer]LayerWeights,
};

fn initStateDict(gpa: std.mem.Allocator, rng: *std.Random.Xoshiro256) !StateDict {
    var sd: StateDict = undefined;
    sd.wte = try initMatrix(gpa, rng, vocab_size, n_embd, 0.08);
    sd.wpe = try initMatrix(gpa, rng, block_size, n_embd, 0.08);
    sd.lm_head = try initMatrix(gpa, rng, vocab_size, n_embd, 0.08);
    for (0..n_layer) |i| {
        sd.layers[i] = .{
            .attn_wq = try initMatrix(gpa, rng, n_embd, n_embd, 0.08),
            .attn_wk = try initMatrix(gpa, rng, n_embd, n_embd, 0.08),
            .attn_wv = try initMatrix(gpa, rng, n_embd, n_embd, 0.08),
            .attn_wo = try initMatrix(gpa, rng, n_embd, n_embd, 0.08),
            .mlp_fc1 = try initMatrix(gpa, rng, 4 * n_embd, n_embd, 0.08),
            .mlp_fc2 = try initMatrix(gpa, rng, n_embd, 4 * n_embd, 0.08),
        };
    }
    return sd;
}

fn appendMatrixParams(list: *std.ArrayList(*Value), alloc: std.mem.Allocator, mat: [][]*Value) !void {
    for (mat) |row| {
        for (row) |v| {
            try list.append(alloc, v);
        }
    }
}

fn flattenParams(gpa: std.mem.Allocator, sd: *const StateDict) ![]*Value {
    var list: std.ArrayList(*Value) = .empty;
    try appendMatrixParams(&list, gpa, sd.wte);
    try appendMatrixParams(&list, gpa, sd.wpe);
    try appendMatrixParams(&list, gpa, sd.lm_head);
    for (&sd.layers) |*layer| {
        try appendMatrixParams(&list, gpa, layer.attn_wq);
        try appendMatrixParams(&list, gpa, layer.attn_wk);
        try appendMatrixParams(&list, gpa, layer.attn_wv);
        try appendMatrixParams(&list, gpa, layer.attn_wo);
        try appendMatrixParams(&list, gpa, layer.mlp_fc1);
        try appendMatrixParams(&list, gpa, layer.mlp_fc2);
    }
    return list.toOwnedSlice(gpa);
}

// ============================================================================
// KV Cache
// ============================================================================
const KVCache = struct {
    keys: [n_layer]std.ArrayList([]*Value),
    values: [n_layer]std.ArrayList([]*Value),

    fn init() KVCache {
        var kv: KVCache = undefined;
        for (&kv.keys, &kv.values) |*k, *v| {
            k.* = .empty;
            v.* = .empty;
        }
        return kv;
    }
};

// ============================================================================
// GPT forward pass
// ============================================================================
fn gpt(
    token_id: usize,
    pos_id: usize,
    kv: *KVCache,
    sd: *const StateDict,
    arena: std.mem.Allocator,
) ![]*Value {
    // Token + position embedding
    var x = try arena.alloc(*Value, n_embd);
    for (0..n_embd) |j| {
        x[j] = try sd.wte[token_id][j].add(sd.wpe[pos_id][j], arena);
    }
    x = try zanogpt.rmsnorm(x, arena);

    for (0..n_layer) |li| {
        const layer = &sd.layers[li];

        // 1) Multi-head attention
        const x_residual = x;
        x = try zanogpt.rmsnorm(x, arena);
        const q = try zanogpt.linear(x, layer.attn_wq, arena);
        const k = try zanogpt.linear(x, layer.attn_wk, arena);
        const v = try zanogpt.linear(x, layer.attn_wv, arena);
        try kv.keys[li].append(arena, k);
        try kv.values[li].append(arena, v);

        var x_attn = try arena.alloc(*Value, n_embd);
        const scale: f64 = comptime 1.0 / @sqrt(@as(f64, @floatFromInt(head_dim)));
        for (0..n_head) |h| {
            const hs = h * head_dim;
            const q_h = q[hs .. hs + head_dim];
            const n_cached = kv.keys[li].items.len;

            // Compute attention logits
            const attn_logits = try arena.alloc(*Value, n_cached);
            for (0..n_cached) |t| {
                const k_t = kv.keys[li].items[t];
                // dot product q_h . k_t_h
                var dot = try q_h[0].mul(k_t[hs], arena);
                for (1..head_dim) |jj| {
                    const prod = try q_h[jj].mul(k_t[hs + jj], arena);
                    dot = try dot.add(prod, arena);
                }
                attn_logits[t] = try dot.mulScalar(scale, arena);
            }
            const attn_weights = try zanogpt.softmax(attn_logits, arena);

            // Weighted sum of values
            for (0..head_dim) |jj| {
                // head_out[jj] = sum_t attn_weights[t] * v_t[hs+jj]
                var acc = try attn_weights[0].mul(kv.values[li].items[0][hs + jj], arena);
                for (1..n_cached) |t| {
                    const prod = try attn_weights[t].mul(kv.values[li].items[t][hs + jj], arena);
                    acc = try acc.add(prod, arena);
                }
                x_attn[hs + jj] = acc;
            }
        }
        x = try zanogpt.linear(x_attn, layer.attn_wo, arena);
        // Residual connection
        for (0..n_embd) |j| {
            x[j] = try x[j].add(x_residual[j], arena);
        }

        // 2) MLP block
        const x_residual2 = x;
        x = try zanogpt.rmsnorm(x, arena);
        var hidden = try zanogpt.linear(x, layer.mlp_fc1, arena);
        for (0..hidden.len) |j| {
            hidden[j] = try hidden[j].relu(arena);
        }
        x = try zanogpt.linear(hidden, layer.mlp_fc2, arena);
        // Residual connection
        for (0..n_embd) |j| {
            x[j] = try x[j].add(x_residual2[j], arena);
        }
    }

    return zanogpt.linear(x, sd.lm_head, arena);
}

// ============================================================================
// Data loading
// ============================================================================
fn loadDocs(gpa: std.mem.Allocator, rng: *std.Random.Xoshiro256) !struct { docs: [][]const u8, backing: []u8 } {
    // Read file at runtime
    const file = try std.fs.cwd().openFile("data/names.txt", .{});
    defer file.close();
    const backing = try file.readToEndAlloc(gpa, 1024 * 1024);

    var docs: std.ArrayList([]const u8) = .empty;
    var iter = std.mem.splitScalar(u8, backing, '\n');
    while (iter.next()) |line| {
        const trimmed = std.mem.trim(u8, line, &[_]u8{ ' ', '\r', '\t' });
        if (trimmed.len > 0) {
            try docs.append(gpa, trimmed);
        }
    }
    // Shuffle with Fisher-Yates
    const items = docs.items;
    var i: usize = items.len - 1;
    while (i > 0) : (i -= 1) {
        const j = rng.random().intRangeAtMost(usize, 0, i);
        const tmp = items[i];
        items[i] = items[j];
        items[j] = tmp;
    }
    return .{ .docs = try docs.toOwnedSlice(gpa), .backing = backing };
}

// ============================================================================
// Main: training + inference
// ============================================================================
pub fn main() !void {
    var gpa_impl: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa_impl.deinit();
    const gpa = gpa_impl.allocator();

    // Use buffered writer for stdout
    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buf);
    const stdout = &stdout_writer.interface;

    // PRNG
    var rng = std.Random.Xoshiro256.init(42);

    // Load data
    const loaded = try loadDocs(gpa, &rng);
    const docs = loaded.docs;
    defer gpa.free(docs);
    defer gpa.free(loaded.backing);
    try stdout.print("num docs: {d}\n", .{docs.len});

    try stdout.print("vocab size: {d}\n", .{vocab_size});

    // Initialize model
    var sd = try initStateDict(gpa, &rng);
    const params = try flattenParams(gpa, &sd);
    defer gpa.free(params);
    try stdout.print("num params: {d}\n", .{params.len});
    try stdout.flush();

    // Adam buffers
    const adam_m = try gpa.alloc(f64, params.len);
    defer gpa.free(adam_m);
    @memset(adam_m, 0.0);
    const adam_v = try gpa.alloc(f64, params.len);
    defer gpa.free(adam_v);
    @memset(adam_v, 0.0);
    const learning_rate: f64 = 0.01;
    const beta1: f64 = 0.85;
    const beta2: f64 = 0.99;
    const eps_adam: f64 = 1e-8;

    // Arena for intermediate computation graph values
    var arena_impl = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_impl.deinit();

    // Training loop
    const num_steps: usize = 1000;
    for (0..num_steps) |step| {
        const arena = arena_impl.allocator();

        // Pick document
        const doc = docs[step % docs.len];

        // Tokenize: BOS + chars + BOS
        const tokens = try arena.alloc(usize, doc.len + 2);
        tokens[0] = BOS;
        for (0..doc.len) |i| {
            tokens[i + 1] = charToToken(doc[i]);
        }
        tokens[doc.len + 1] = BOS;

        const n = @min(block_size, tokens.len - 1);

        // Forward pass: accumulate losses
        var kv = KVCache.init();
        var losses = try arena.alloc(*Value, n);
        for (0..n) |pos_id| {
            const token_id = tokens[pos_id];
            const target_id = tokens[pos_id + 1];
            const logits = try gpt(token_id, pos_id, &kv, &sd, arena);
            const probs = try zanogpt.softmax(logits, arena);
            const neg_log = try probs[target_id].log(arena);
            losses[pos_id] = try neg_log.mulScalar(-1.0, arena);
        }

        // Average loss
        var total_loss = losses[0];
        for (1..n) |i| {
            total_loss = try total_loss.add(losses[i], arena);
        }
        const loss = try total_loss.mulScalar(1.0 / @as(f64, @floatFromInt(n)), arena);

        // Backward
        try loss.backward(gpa);

        // Adam update
        const lr_t = learning_rate * (1.0 - @as(f64, @floatFromInt(step)) / @as(f64, @floatFromInt(num_steps)));
        const step_f: f64 = @floatFromInt(step + 1);
        for (0..params.len) |i| {
            const g = params[i].grad;
            adam_m[i] = beta1 * adam_m[i] + (1.0 - beta1) * g;
            adam_v[i] = beta2 * adam_v[i] + (1.0 - beta2) * g * g;
            const m_hat = adam_m[i] / (1.0 - std.math.pow(f64, beta1, step_f));
            const v_hat = adam_v[i] / (1.0 - std.math.pow(f64, beta2, step_f));
            params[i].data -= lr_t * m_hat / (@sqrt(v_hat) + eps_adam);
            params[i].grad = 0;
        }

        try stdout.print("step {d:4} / {d:4} | loss {d:.4}\n", .{ step + 1, num_steps, loss.data });
        if ((step + 1) % 100 == 0) try stdout.flush();

        // Reset arena for next step (free all intermediate Values)
        _ = arena_impl.reset(.retain_capacity);
    }

    try stdout.flush();

    // ========================================================================
    // Inference
    // ========================================================================
    const temperature: f64 = 0.5;
    try stdout.print("\n--- inference (new, hallucinated names) ---\n", .{});
    try stdout.flush();

    for (0..20) |sample_idx| {
        _ = arena_impl.reset(.retain_capacity);
        const arena = arena_impl.allocator();
        var kv = KVCache.init();
        var token_id: usize = BOS;
        var name_buf: [block_size]u8 = undefined;
        var name_len: usize = 0;

        for (0..block_size) |pos_id| {
            const logits = try gpt(token_id, pos_id, &kv, &sd, arena);

            // Apply temperature
            const scaled = try arena.alloc(*Value, logits.len);
            for (0..logits.len) |j| {
                scaled[j] = try logits[j].mulScalar(1.0 / temperature, arena);
            }
            const probs = try zanogpt.softmax(scaled, arena);

            // Weighted random sampling
            const weights = try arena.alloc(f64, vocab_size);
            for (0..vocab_size) |j| {
                weights[j] = probs[j].data;
            }
            token_id = weightedSample(&rng, weights);
            if (token_id == BOS) break;
            if (name_len < block_size) {
                name_buf[name_len] = tokenToChar(token_id);
                name_len += 1;
            }
        }
        try stdout.print("sample {d:2}: {s}\n", .{ sample_idx + 1, name_buf[0..name_len] });
    }
    try stdout.flush();
}

fn weightedSample(rng: *std.Random.Xoshiro256, weights: []const f64) usize {
    var total: f64 = 0;
    for (weights) |w| total += w;
    var r = rng.random().float(f64) * total;
    for (weights, 0..) |w, i| {
        r -= w;
        if (r <= 0) return i;
    }
    return weights.len - 1;
}
