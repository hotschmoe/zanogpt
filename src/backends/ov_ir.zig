const std = @import("std");
const ze = @import("ze.zig");

/// Stack-allocated buffer for an OpenVINO IR NGRAPH_LITE blob.
pub const BlobBuf = struct {
    data: [4096]u8 = undefined,
    len: usize = 0,

    pub fn slice(self: *const BlobBuf) []const u8 {
        return self.data[0..self.len];
    }
};

/// VCL header size: compiler_version (4) + numberOfInputData (4) = 8 bytes.
const VCL_HDR_SIZE = 8;

/// Generate an NGRAPH_LITE blob for MatMul(W[M,K] @ x[K]) -> out[M].
/// W is input 0 (shape [M,K]), x is input 1 (shape [K]), result is shape [M].
pub fn matmulBlob(M: usize, K: usize, compiler_ver: ze.ze_graph_compiler_version_info_t) BlobBuf {
    return makeBlob(matmul_xml_template, .{ M, K, M, K, K, K, M, K, K, M, M }, compiler_ver);
}

/// Generate an NGRAPH_LITE blob for SoftMax over N elements.
pub fn softmaxBlob(N: usize, compiler_ver: ze.ze_graph_compiler_version_info_t) BlobBuf {
    return makeBlob(softmax_xml_template, .{ N, N, N, N, N }, compiler_ver);
}

/// Generate an NGRAPH_LITE blob for ReLU over N elements.
pub fn reluBlob(N: usize, compiler_ver: ze.ze_graph_compiler_version_info_t) BlobBuf {
    return makeBlob(relu_xml_template, .{ N, N, N, N, N }, compiler_ver);
}

/// Build flags for the NPU compiler (input/output precisions and layouts).
/// Output names MUST match the producing op's layer name (NOT the "result" sink).
/// The `--config ` suffix is required by the NPU driver's VCL compiler.
pub const relu_build_flags: [*:0]const u8 =
    "--inputs_precisions=\"input:FP32\" --inputs_layouts=\"input:C\" " ++
    "--outputs_precisions=\"r:FP32\" --outputs_layouts=\"r:C\" --config ";

pub const softmax_build_flags: [*:0]const u8 =
    "--inputs_precisions=\"input:FP32\" --inputs_layouts=\"input:C\" " ++
    "--outputs_precisions=\"sm:FP32\" --outputs_layouts=\"sm:C\" --config ";

pub const matmul_build_flags: [*:0]const u8 =
    "--inputs_precisions=\"W:FP32 x:FP32\" --inputs_layouts=\"W:NC x:C\" " ++
    "--outputs_precisions=\"mm:FP32\" --outputs_layouts=\"mm:C\" --config ";

/// Format an XML template with args into a BlobBuf.
/// VCL blob format (NGRAPH_LITE):
///   [u16 major][u16 minor]        — compiler version (4 bytes)
///   [u32 numberOfInputData = 2]   — always 2 (xml + weights) (4 bytes)
///   [u64 xmlSize]                 — XML byte count (8 bytes)
///   [xmlSize bytes]               — XML data
///   [u64 weightsSize]             — weights byte count (8 bytes)
///   [weightsSize bytes]           — weights data
fn makeBlob(comptime template: []const u8, args: anytype, compiler_ver: ze.ze_graph_compiler_version_info_t) BlobBuf {
    var buf = BlobBuf{};

    // 1. Write VCL header: compiler version + numberOfInputData
    @memcpy(buf.data[0..2], std.mem.asBytes(&compiler_ver.major));
    @memcpy(buf.data[2..4], std.mem.asBytes(&compiler_ver.minor));
    const num_inputs: u32 = 2; // always 2: xml + weights
    @memcpy(buf.data[4..8], std.mem.asBytes(&num_inputs));

    // 2. Format XML into buffer (after VCL header + xmlSize field)
    const xml_data_offset = VCL_HDR_SIZE + 8; // 8 for VCL header + 8 for xmlSize
    const xml = std.fmt.bufPrint(buf.data[xml_data_offset..], template, args) catch
        @panic("IR XML exceeded BlobBuf capacity");
    const xml_len: u64 = @intCast(xml.len);

    // 3. Write xmlSize just before the XML data
    @memcpy(buf.data[VCL_HDR_SIZE..][0..8], std.mem.asBytes(&xml_len));

    // 4. Append weightsSize = 0 (no weights for parameter-only ops)
    const weights_hdr_offset = xml_data_offset + xml.len;
    const zero_weights: u64 = 0;
    @memcpy(buf.data[weights_hdr_offset..][0..8], std.mem.asBytes(&zero_weights));

    buf.len = weights_hdr_offset + 8;
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
    \\<layer id="1" name="sm" type="Softmax" version="opset1">
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
    \\<layer id="1" name="r" type="Relu" version="opset1">
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

const test_compiler_ver = ze.ze_graph_compiler_version_info_t{ .major = 6, .minor = 3 };

fn expectValidBlob(blob: *const BlobBuf, expected_op: []const u8) !void {
    // VCL blob: [4 compiler_ver][4 numInputData=2][u64 xml_len][xml][u64 weights_len][weights]
    try std.testing.expect(blob.len > VCL_HDR_SIZE + 16);
    // Check VCL header
    const num_inputs = std.mem.readInt(u32, blob.data[4..8], .little);
    try std.testing.expectEqual(@as(u32, 2), num_inputs);
    // Check XML
    const xml_len = std.mem.readInt(u64, blob.data[VCL_HDR_SIZE..][0..8], .little);
    const xml = blob.data[VCL_HDR_SIZE + 8 ..][0..xml_len];
    try std.testing.expect(std.mem.startsWith(u8, xml, "<?xml"));
    try std.testing.expect(std.mem.indexOf(u8, xml, expected_op) != null);
    // Weights section follows XML
    const weights_len = std.mem.readInt(u64, blob.data[VCL_HDR_SIZE + 8 + xml_len ..][0..8], .little);
    try std.testing.expectEqual(@as(u64, 0), weights_len);
    try std.testing.expectEqual(VCL_HDR_SIZE + 8 + xml_len + 8, blob.len);
}

test "matmulBlob produces valid blob" {
    const blob = matmulBlob(3, 2, test_compiler_ver);
    try expectValidBlob(&blob, "MatMul");
}

test "softmaxBlob produces valid blob" {
    const blob = softmaxBlob(16, test_compiler_ver);
    try expectValidBlob(&blob, "Softmax");
}

test "reluBlob produces valid blob" {
    const blob = reluBlob(4, test_compiler_ver);
    try expectValidBlob(&blob, "Relu");
}
