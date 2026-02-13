pub fn init() !void {}
pub fn deinit() void {}

pub fn matmul_fwd(W: []const f32, x: []const f32, out: []f32, M: usize, K: usize) void {
    const VEC = 8;
    for (0..M) |i| {
        const row = W[i * K ..][0..K];
        var j: usize = 0;
        var vec_acc: @Vector(VEC, f32) = @splat(0);
        while (j + VEC <= K) : (j += VEC) {
            const a: @Vector(VEC, f32) = row[j..][0..VEC].*;
            const b: @Vector(VEC, f32) = x[j..][0..VEC].*;
            vec_acc += a * b;
        }
        var sum: f32 = @reduce(.Add, vec_acc);
        while (j < K) : (j += 1) sum += row[j] * x[j];
        out[i] = sum;
    }
}

pub fn softmax_fwd(input: []const f32, output: []f32, n: usize) void {
    var max_val: f32 = input[0];
    for (input[1..n]) |v| if (v > max_val) {
        max_val = v;
    };
    var sum: f32 = 0;
    for (0..n) |i| {
        output[i] = @exp(input[i] - max_val);
        sum += output[i];
    }
    const inv = 1.0 / sum;
    for (0..n) |i| output[i] *= inv;
}

pub fn rmsnorm_fwd(input: []const f32, output: []f32, n: usize, scale_out: *f32) void {
    var ms: f32 = 0;
    for (0..n) |i| ms += input[i] * input[i];
    ms /= @as(f32, @floatFromInt(n));
    const scale = 1.0 / @sqrt(ms + 1e-5);
    scale_out.* = scale;
    for (0..n) |i| output[i] = input[i] * scale;
}

pub fn relu_fwd(input: []const f32, output: []f32, n: usize) void {
    for (0..n) |i| output[i] = @max(0, input[i]);
}
