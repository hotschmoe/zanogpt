const std = @import("std");
const zanogpt = @import("zanogpt");
const Tensor = zanogpt.Tensor;
const Tape = zanogpt.Tape;

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
// Weight initialization
// ============================================================================
fn initWeight(gpa: std.mem.Allocator, rng: *std.Random.Xoshiro256, shape: []const usize, std_dev: f32) !*Tensor {
    const t = try Tensor.init(gpa, shape, true);
    t.fillRandom(rng, std_dev);
    return t;
}

// ============================================================================
// State dict: all model weight tensors
// ============================================================================
const LayerWeights = struct {
    attn_wq: *Tensor, // [n_embd, n_embd]
    attn_wk: *Tensor,
    attn_wv: *Tensor,
    attn_wo: *Tensor,
    mlp_fc1: *Tensor, // [4*n_embd, n_embd]
    mlp_fc2: *Tensor, // [n_embd, 4*n_embd]
};

const StateDict = struct {
    wte: *Tensor, // [vocab_size, n_embd]
    wpe: *Tensor, // [block_size, n_embd]
    lm_head: *Tensor, // [vocab_size, n_embd]
    layers: [n_layer]LayerWeights,
};

fn initStateDict(gpa: std.mem.Allocator, rng: *std.Random.Xoshiro256) !StateDict {
    var sd: StateDict = undefined;
    sd.wte = try initWeight(gpa, rng, &.{ vocab_size, n_embd }, 0.08);
    sd.wpe = try initWeight(gpa, rng, &.{ block_size, n_embd }, 0.08);
    sd.lm_head = try initWeight(gpa, rng, &.{ vocab_size, n_embd }, 0.08);
    for (0..n_layer) |i| {
        sd.layers[i] = .{
            .attn_wq = try initWeight(gpa, rng, &.{ n_embd, n_embd }, 0.08),
            .attn_wk = try initWeight(gpa, rng, &.{ n_embd, n_embd }, 0.08),
            .attn_wv = try initWeight(gpa, rng, &.{ n_embd, n_embd }, 0.08),
            .attn_wo = try initWeight(gpa, rng, &.{ n_embd, n_embd }, 0.08),
            .mlp_fc1 = try initWeight(gpa, rng, &.{ 4 * n_embd, n_embd }, 0.08),
            .mlp_fc2 = try initWeight(gpa, rng, &.{ n_embd, 4 * n_embd }, 0.08),
        };
    }
    return sd;
}

fn deinitStateDict(sd: *StateDict) void {
    sd.wte.deinit();
    sd.wpe.deinit();
    sd.lm_head.deinit();
    for (&sd.layers) |*layer| {
        layer.attn_wq.deinit();
        layer.attn_wk.deinit();
        layer.attn_wv.deinit();
        layer.attn_wo.deinit();
        layer.mlp_fc1.deinit();
        layer.mlp_fc2.deinit();
    }
}

// Registered tape indices for all weights
const RegisteredLayerWeights = struct {
    attn_wq: usize,
    attn_wk: usize,
    attn_wv: usize,
    attn_wo: usize,
    mlp_fc1: usize,
    mlp_fc2: usize,
};

const RegisteredWeights = struct {
    wte: usize,
    wpe: usize,
    lm_head: usize,
    layers: [n_layer]RegisteredLayerWeights,
};

fn registerWeights(tape: *Tape, sd: *const StateDict) !RegisteredWeights {
    var reg: RegisteredWeights = undefined;
    reg.wte = try tape.register(sd.wte);
    reg.wpe = try tape.register(sd.wpe);
    reg.lm_head = try tape.register(sd.lm_head);
    for (0..n_layer) |i| {
        reg.layers[i] = .{
            .attn_wq = try tape.register(sd.layers[i].attn_wq),
            .attn_wk = try tape.register(sd.layers[i].attn_wk),
            .attn_wv = try tape.register(sd.layers[i].attn_wv),
            .attn_wo = try tape.register(sd.layers[i].attn_wo),
            .mlp_fc1 = try tape.register(sd.layers[i].mlp_fc1),
            .mlp_fc2 = try tape.register(sd.layers[i].mlp_fc2),
        };
    }
    return reg;
}

const param_count = 3 + n_layer * 6;

fn flattenParams(sd: *const StateDict) [param_count]*Tensor {
    var params: [param_count]*Tensor = undefined;
    var idx: usize = 0;
    params[idx] = sd.wte;
    idx += 1;
    params[idx] = sd.wpe;
    idx += 1;
    params[idx] = sd.lm_head;
    idx += 1;
    for (&sd.layers) |*layer| {
        params[idx] = layer.attn_wq;
        idx += 1;
        params[idx] = layer.attn_wk;
        idx += 1;
        params[idx] = layer.attn_wv;
        idx += 1;
        params[idx] = layer.attn_wo;
        idx += 1;
        params[idx] = layer.mlp_fc1;
        idx += 1;
        params[idx] = layer.mlp_fc2;
        idx += 1;
    }
    return params;
}

// ============================================================================
// KV Cache (stores tape indices)
// ============================================================================
const KVCache = struct {
    keys: [n_layer]std.ArrayList(usize),
    values: [n_layer]std.ArrayList(usize),

    fn init_cache() KVCache {
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
    tape: *Tape,
    token_id: usize,
    pos_id: usize,
    kv: *KVCache,
    reg: *const RegisteredWeights,
    arena: std.mem.Allocator,
) !usize {
    // Token + position embedding
    const tok_emb = try tape.embeddingLookup(reg.wte, token_id);
    const pos_emb = try tape.embeddingLookup(reg.wpe, pos_id);
    var x = try tape.addOp(tok_emb, pos_emb);
    x = try tape.rmsnormOp(x);

    for (0..n_layer) |li| {
        const layer = &reg.layers[li];

        // 1) Multi-head attention
        const x_residual = x;
        x = try tape.rmsnormOp(x);
        const q = try tape.matmulOp(layer.attn_wq, x);
        const k = try tape.matmulOp(layer.attn_wk, x);
        const v = try tape.matmulOp(layer.attn_wv, x);

        try kv.keys[li].append(arena, k);
        try kv.values[li].append(arena, v);

        const attn_out = try tape.attentionOp(
            q,
            kv.keys[li].items,
            kv.values[li].items,
            n_head,
            head_dim,
        );

        x = try tape.matmulOp(layer.attn_wo, attn_out);
        x = try tape.addOp(x, x_residual);

        // 2) MLP block
        const x_residual2 = x;
        x = try tape.rmsnormOp(x);
        var hidden = try tape.matmulOp(layer.mlp_fc1, x);
        hidden = try tape.reluOp(hidden);
        x = try tape.matmulOp(layer.mlp_fc2, hidden);
        x = try tape.addOp(x, x_residual2);
    }

    return tape.matmulOp(reg.lm_head, x);
}

// ============================================================================
// Data loading
// ============================================================================
fn loadDocs(gpa: std.mem.Allocator, rng: *std.Random.Xoshiro256) !struct { docs: [][]const u8, backing: []u8 } {
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

    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buf);
    const stdout = &stdout_writer.interface;

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
    defer deinitStateDict(&sd);
    var params = flattenParams(&sd);

    // Count total scalar parameters
    var total_params: usize = 0;
    for (&params) |p| total_params += p.numel();
    try stdout.print("num params: {d}\n", .{total_params});
    try stdout.flush();

    // Adam buffers (one entry per scalar parameter element)
    const adam_m = try gpa.alloc(f32, total_params);
    defer gpa.free(adam_m);
    @memset(adam_m, 0);
    const adam_v = try gpa.alloc(f32, total_params);
    defer gpa.free(adam_v);
    @memset(adam_v, 0);
    const learning_rate: f32 = 0.01;
    const beta1: f32 = 0.85;
    const beta2: f32 = 0.99;
    const eps_adam: f32 = 1e-8;

    // Ensure all params have grad buffers (persistent, on GPA)
    for (&params) |p| try p.ensureGrad();

    // Arena for tape and intermediate tensors
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

        // Zero param gradients
        for (&params) |p| p.zeroGrad();

        // Create tape for this step
        var tape = Tape.init(arena);
        const reg = try registerWeights(&tape, &sd);

        // Forward pass: accumulate loss across positions
        var kv = KVCache.init_cache();
        var total_loss_idx: ?usize = null;

        for (0..n) |pos_id| {
            const token_id = tokens[pos_id];
            const target_id = tokens[pos_id + 1];
            const logits = try gpt(&tape, token_id, pos_id, &kv, &reg, arena);
            const probs_idx = try tape.softmaxOp(logits);
            const loss_pos = try tape.nllLoss(probs_idx, target_id);

            if (total_loss_idx) |tl| {
                total_loss_idx = try tape.addOp(tl, loss_pos);
            } else {
                total_loss_idx = loss_pos;
            }
        }

        // Average loss
        const avg_loss_idx = try tape.mulScalarOp(total_loss_idx.?, 1.0 / @as(f32, @floatFromInt(n)));

        // Backward
        try tape.backward(avg_loss_idx);

        const loss_val = tape.get(avg_loss_idx).data[0];

        // Adam update
        const lr_t = learning_rate * (1.0 - @as(f32, @floatFromInt(step)) / @as(f32, @floatFromInt(num_steps)));
        const step_f: f32 = @floatFromInt(step + 1);
        var adam_idx: usize = 0;
        for (&params) |p| {
            const g = p.grad.?;
            for (0..p.numel()) |j| {
                adam_m[adam_idx] = beta1 * adam_m[adam_idx] + (1.0 - beta1) * g[j];
                adam_v[adam_idx] = beta2 * adam_v[adam_idx] + (1.0 - beta2) * g[j] * g[j];
                const m_hat = adam_m[adam_idx] / (1.0 - std.math.pow(f32, beta1, step_f));
                const v_hat = adam_v[adam_idx] / (1.0 - std.math.pow(f32, beta2, step_f));
                p.data[j] -= lr_t * m_hat / (@sqrt(v_hat) + eps_adam);
                adam_idx += 1;
            }
        }

        try stdout.print("step {d:4} / {d:4} | loss {d:.4}\n", .{ step + 1, num_steps, loss_val });
        if ((step + 1) % 100 == 0) try stdout.flush();

        // Reset arena (frees tape + all intermediate tensors)
        _ = arena_impl.reset(.retain_capacity);
    }

    try stdout.flush();

    // ========================================================================
    // Inference
    // ========================================================================
    const temperature: f32 = 0.5;
    try stdout.print("\n--- inference (new, hallucinated names) ---\n", .{});
    try stdout.flush();

    for (0..20) |sample_idx| {
        _ = arena_impl.reset(.retain_capacity);
        const arena = arena_impl.allocator();
        var tape = Tape.init(arena);
        const reg = try registerWeights(&tape, &sd);
        var kv = KVCache.init_cache();
        var token_id: usize = BOS;
        var name_buf: [block_size]u8 = undefined;
        var name_len: usize = 0;

        for (0..block_size) |pos_id| {
            const logits = try gpt(&tape, token_id, pos_id, &kv, &reg, arena);

            // Apply temperature
            const scaled = try tape.mulScalarOp(logits, 1.0 / temperature);
            const probs_idx = try tape.softmaxOp(scaled);
            const probs = tape.get(probs_idx);

            // Weighted random sampling
            token_id = weightedSample(&rng, probs.data[0..vocab_size]);
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

fn weightedSample(rng: *std.Random.Xoshiro256, weights: []const f32) usize {
    var total: f32 = 0;
    for (weights) |w| total += w;
    var r = rng.random().float(f32) * total;
    for (weights, 0..) |w, i| {
        r -= w;
        if (r <= 0) return i;
    }
    return weights.len - 1;
}
