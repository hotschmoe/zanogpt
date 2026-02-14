const std = @import("std");
const ze = @import("ze.zig");

pub const BlobBuf = struct {
    data: [4096]u8 = undefined,
    len: usize = 0,

    pub fn slice(self: *const BlobBuf) []const u8 {
        return self.data[0..self.len];
    }
};

const VCL_HDR_SIZE = 8; // compiler_version (4) + numberOfInputData (4)

/// MatMul(W[M,K] @ x[K]) -> out[M].
pub fn matmulBlob(M: usize, K: usize, compiler_ver: ze.ze_graph_compiler_version_info_t) BlobBuf {
    return makeBlob(matmul_xml_template, .{ M, K, M, K, K, K, M, K, K, M, M }, compiler_ver);
}

pub fn softmaxBlob(N: usize, compiler_ver: ze.ze_graph_compiler_version_info_t) BlobBuf {
    return makeBlob(softmax_xml_template, .{ N, N, N, N, N }, compiler_ver);
}

pub fn reluBlob(N: usize, compiler_ver: ze.ze_graph_compiler_version_info_t) BlobBuf {
    return makeBlob(relu_xml_template, .{ N, N, N, N, N }, compiler_ver);
}

/// NPU compiler build flags. Output names must match the producing op's layer name
/// (not the "result" sink). The trailing `--config ` is required by the VCL compiler.
pub const relu_build_flags: [*:0]const u8 =
    "--inputs_precisions=\"input:FP32\" --inputs_layouts=\"input:C\" " ++
    "--outputs_precisions=\"r:FP32\" --outputs_layouts=\"r:C\" --config ";

pub const softmax_build_flags: [*:0]const u8 =
    "--inputs_precisions=\"input:FP32\" --inputs_layouts=\"input:C\" " ++
    "--outputs_precisions=\"sm:FP32\" --outputs_layouts=\"sm:C\" --config ";

pub const matmul_build_flags: [*:0]const u8 =
    "--inputs_precisions=\"W:FP32 x:FP32\" --inputs_layouts=\"W:NC x:C\" " ++
    "--outputs_precisions=\"mm:FP32\" --outputs_layouts=\"mm:C\" --config ";

/// Build a VCL NGRAPH_LITE blob: [compiler_ver(4)][numInputs(4)][xmlSize(8)][xml][weightsSize(8)].
fn makeBlob(comptime template: []const u8, args: anytype, compiler_ver: ze.ze_graph_compiler_version_info_t) BlobBuf {
    var buf = BlobBuf{};

    // VCL header: compiler version + numberOfInputData (always 2: xml + weights)
    @memcpy(buf.data[0..2], std.mem.asBytes(&compiler_ver.major));
    @memcpy(buf.data[2..4], std.mem.asBytes(&compiler_ver.minor));
    const num_inputs: u32 = 2;
    @memcpy(buf.data[4..8], std.mem.asBytes(&num_inputs));

    // Format XML after the header + xmlSize field
    const xml_start = VCL_HDR_SIZE + 8;
    const xml = std.fmt.bufPrint(buf.data[xml_start..], template, args) catch
        @panic("IR XML exceeded BlobBuf capacity");
    const xml_len: u64 = @intCast(xml.len);

    // Write xmlSize field
    @memcpy(buf.data[VCL_HDR_SIZE..][0..8], std.mem.asBytes(&xml_len));

    // Append weightsSize = 0
    const weights_offset = xml_start + xml.len;
    const zero: u64 = 0;
    @memcpy(buf.data[weights_offset..][0..8], std.mem.asBytes(&zero));

    buf.len = weights_offset + 8;
    return buf;
}

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

const test_compiler_ver = ze.ze_graph_compiler_version_info_t{ .major = 6, .minor = 3 };

fn expectValidBlob(blob: *const BlobBuf, expected_op: []const u8) !void {
    try std.testing.expect(blob.len > VCL_HDR_SIZE + 16);
    const num_inputs = std.mem.readInt(u32, blob.data[4..8], .little);
    try std.testing.expectEqual(@as(u32, 2), num_inputs);

    const xml_len = std.mem.readInt(u64, blob.data[VCL_HDR_SIZE..][0..8], .little);
    const xml = blob.data[VCL_HDR_SIZE + 8 ..][0..xml_len];
    try std.testing.expect(std.mem.startsWith(u8, xml, "<?xml"));
    try std.testing.expect(std.mem.indexOf(u8, xml, expected_op) != null);

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
