const std = @import("std");

/// Stack-allocated buffer for an OpenVINO IR NGRAPH_LITE blob.
pub const BlobBuf = struct {
    data: [4096]u8 = undefined,
    len: usize = 0,

    pub fn slice(self: *const BlobBuf) []const u8 {
        return self.data[0..self.len];
    }
};

/// Generate an NGRAPH_LITE blob for MatMul(W[M,K] @ x[K]) -> out[M].
/// W is input 0 (shape [M,K]), x is input 1 (shape [K]), result is shape [M].
pub fn matmulBlob(M: usize, K: usize) BlobBuf {
    return makeBlob(matmul_xml_template, .{ M, K, M, K, K, K, M, K, K, M, M });
}

/// Generate an NGRAPH_LITE blob for SoftMax over N elements.
pub fn softmaxBlob(N: usize) BlobBuf {
    return makeBlob(softmax_xml_template, .{ N, N, N, N, N });
}

/// Generate an NGRAPH_LITE blob for ReLU over N elements.
pub fn reluBlob(N: usize) BlobBuf {
    return makeBlob(relu_xml_template, .{ N, N, N, N, N });
}

/// Format an XML template with args into a BlobBuf with an 8-byte LE length header.
fn makeBlob(comptime template: []const u8, args: anytype) BlobBuf {
    var buf = BlobBuf{};
    const xml = std.fmt.bufPrint(buf.data[8..], template, args) catch
        @panic("IR XML exceeded BlobBuf capacity");
    const xml_len: u64 = @intCast(xml.len);
    @memcpy(buf.data[0..8], std.mem.asBytes(&xml_len));
    buf.len = 8 + xml.len;
    return buf;
}

// ── IR XML Templates (OpenVINO IR v11) ──

const matmul_xml_template =
    \\<?xml version="1.0"?>
    \\<net name="matmul" version="11">
    \\<layers>
    \\<layer id="0" name="W" type="Parameter" version="opset1">
    \\<data shape="{d},{d}" element_type="f32"/>
    \\<output><port id="0" precision="FP32"><dim>{d}</dim><dim>{d}</dim></port></output>
    \\</layer>
    \\<layer id="1" name="x" type="Parameter" version="opset1">
    \\<data shape="{d}" element_type="f32"/>
    \\<output><port id="0" precision="FP32"><dim>{d}</dim></port></output>
    \\</layer>
    \\<layer id="2" name="mm" type="MatMul" version="opset1">
    \\<data transpose_a="false" transpose_b="false"/>
    \\<input><port id="0"><dim>{d}</dim><dim>{d}</dim></port><port id="1"><dim>{d}</dim></port></input>
    \\<output><port id="2" precision="FP32"><dim>{d}</dim></port></output>
    \\</layer>
    \\<layer id="3" name="result" type="Result" version="opset1">
    \\<input><port id="0"><dim>{d}</dim></port></input>
    \\</layer>
    \\</layers>
    \\<edges>
    \\<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
    \\<edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
    \\<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
    \\</edges>
    \\</net>
;

const softmax_xml_template =
    \\<?xml version="1.0"?>
    \\<net name="softmax" version="11">
    \\<layers>
    \\<layer id="0" name="input" type="Parameter" version="opset1">
    \\<data shape="{d}" element_type="f32"/>
    \\<output><port id="0" precision="FP32"><dim>{d}</dim></port></output>
    \\</layer>
    \\<layer id="1" name="sm" type="SoftMax" version="opset1">
    \\<data axis="0"/>
    \\<input><port id="0"><dim>{d}</dim></port></input>
    \\<output><port id="1" precision="FP32"><dim>{d}</dim></port></output>
    \\</layer>
    \\<layer id="2" name="result" type="Result" version="opset1">
    \\<input><port id="0"><dim>{d}</dim></port></input>
    \\</layer>
    \\</layers>
    \\<edges>
    \\<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    \\<edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
    \\</edges>
    \\</net>
;

const relu_xml_template =
    \\<?xml version="1.0"?>
    \\<net name="relu" version="11">
    \\<layers>
    \\<layer id="0" name="input" type="Parameter" version="opset1">
    \\<data shape="{d}" element_type="f32"/>
    \\<output><port id="0" precision="FP32"><dim>{d}</dim></port></output>
    \\</layer>
    \\<layer id="1" name="r" type="ReLU" version="opset1">
    \\<input><port id="0"><dim>{d}</dim></port></input>
    \\<output><port id="1" precision="FP32"><dim>{d}</dim></port></output>
    \\</layer>
    \\<layer id="2" name="result" type="Result" version="opset1">
    \\<input><port id="0"><dim>{d}</dim></port></input>
    \\</layer>
    \\</layers>
    \\<edges>
    \\<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    \\<edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
    \\</edges>
    \\</net>
;

// ── Tests ──

fn expectValidBlob(blob: *const BlobBuf, expected_op: []const u8) !void {
    try std.testing.expect(blob.len > 8);
    const xml_len = std.mem.readInt(u64, blob.data[0..8], .little);
    try std.testing.expectEqual(blob.len - 8, xml_len);
    const xml = blob.data[8..blob.len];
    try std.testing.expect(std.mem.startsWith(u8, xml, "<?xml"));
    try std.testing.expect(std.mem.indexOf(u8, xml, expected_op) != null);
}

test "matmulBlob produces valid blob" {
    const blob = matmulBlob(3, 2);
    try expectValidBlob(&blob, "MatMul");
}

test "softmaxBlob produces valid blob" {
    const blob = softmaxBlob(16);
    try expectValidBlob(&blob, "SoftMax");
}

test "reluBlob produces valid blob" {
    const blob = reluBlob(4);
    try expectValidBlob(&blob, "ReLU");
}
