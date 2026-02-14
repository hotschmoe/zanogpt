const std = @import("std");

/// Stack-allocated buffer for a SPIR-V module (binary stream of 32-bit words).
pub const SpvModule = struct {
    words: [8192]u32 = undefined,
    len: usize = 0,

    pub fn asBytes(self: *const SpvModule) []const u8 {
        const word_slice = self.words[0..self.len];
        return std.mem.sliceAsBytes(word_slice);
    }

    fn emit(self: *SpvModule, opcode: u16, operands: []const u32) void {
        const wc: u16 = @intCast(1 + operands.len);
        self.words[self.len] = @as(u32, wc) << 16 | @as(u32, opcode);
        self.len += 1;
        for (operands) |op| {
            self.words[self.len] = op;
            self.len += 1;
        }
    }

    /// Emit an instruction whose last logical operand is a NUL-terminated string.
    /// `pre` are the operands before the string.
    fn emitString(self: *SpvModule, opcode: u16, pre: []const u32, s: []const u8) void {
        const str_words = (s.len + 4) / 4; // includes NUL terminator
        const wc: u16 = @intCast(1 + pre.len + str_words);
        self.words[self.len] = @as(u32, wc) << 16 | @as(u32, opcode);
        self.len += 1;
        for (pre) |op| {
            self.words[self.len] = op;
            self.len += 1;
        }
        var i: usize = 0;
        while (i < str_words) : (i += 1) {
            var word: u32 = 0;
            for (0..4) |b| {
                const idx = i * 4 + b;
                if (idx < s.len) {
                    word |= @as(u32, s[idx]) << @intCast(b * 8);
                }
            }
            self.words[self.len] = word;
            self.len += 1;
        }
    }

    /// Emit OpEntryPoint with interface variables listed after the name string.
    /// Format: OpEntryPoint <exec_model> <func_id> "name" <interface_var>...
    fn emitEntryPoint(self: *SpvModule, exec_model: u32, func_id: u32, name: []const u8, interface: []const u32) void {
        const str_words = (name.len + 4) / 4;
        const wc: u16 = @intCast(1 + 2 + str_words + interface.len);
        self.words[self.len] = @as(u32, wc) << 16 | @as(u32, OpEntryPoint);
        self.len += 1;
        self.words[self.len] = exec_model;
        self.len += 1;
        self.words[self.len] = func_id;
        self.len += 1;
        // Pack string bytes into words
        var i: usize = 0;
        while (i < str_words) : (i += 1) {
            var word: u32 = 0;
            for (0..4) |b| {
                const idx = i * 4 + b;
                if (idx < name.len) {
                    word |= @as(u32, name[idx]) << @intCast(b * 8);
                }
            }
            self.words[self.len] = word;
            self.len += 1;
        }
        // Interface variables
        for (interface) |v| {
            self.words[self.len] = v;
            self.len += 1;
        }
    }

    fn emitHeader(self: *SpvModule) void {
        self.words[0] = MAGIC;
        self.words[1] = 0x00010000; // SPIR-V 1.0
        self.words[2] = 0; // generator
        self.words[3] = 0; // bound (patched later)
        self.words[4] = 0; // reserved
        self.len = 5;
    }

    fn patchBound(self: *SpvModule, bound: u32) void {
        self.words[3] = bound;
    }

    /// Emit the common preamble shared by all kernel modules:
    /// header, capabilities, OpenCL.std import, and memory model.
    /// Returns the ext_id (always 1) used for OpExtInst calls.
    fn emitPreamble(self: *SpvModule) u32 {
        const ext_id: u32 = 1;
        self.emitHeader();
        self.emit(OpCapability, &.{Capability_Addresses});
        self.emit(OpCapability, &.{Capability_Kernel});
        self.emit(OpCapability, &.{Capability_Int64});
        self.emitString(OpExtInstImport, &.{ext_id}, "OpenCL.std");
        self.emit(OpMemoryModel, &.{ AddressModel_Physical64, MemoryModel_OpenCL });
        return ext_id;
    }
};

// ── SPIR-V magic and opcodes ──

const MAGIC: u32 = 0x07230203;

const OpCapability = 17;
const OpExtInstImport = 11;
const OpMemoryModel = 14;
const OpEntryPoint = 15;
const OpDecorate = 71;
const OpTypeVoid = 19;
const OpTypeBool = 20;
const OpTypeInt = 21;
const OpTypeFloat = 22;
const OpTypePointer = 32;
const OpTypeFunction = 33;
const OpTypeVector = 23;
const OpConstant = 43;
const OpVariable = 59;
const OpFunction = 54;
const OpFunctionEnd = 56;
const OpLabel = 248;
const OpLoad = 61;
const OpStore = 62;
const OpReturn = 253;
const OpAccessChain = 65;
const OpFunctionParameter = 55;
const OpIAdd = 128;
const OpFAdd = 129;
const OpFSub = 131;
const OpIMul = 132;
const OpFMul = 133;
const OpFDiv = 136;
const OpFOrdGreaterThan = 186;
const OpSelect = 169;
const OpBranch = 249;
const OpBranchConditional = 250;
const OpLoopMerge = 246;
const OpSelectionMerge = 247;
const OpPhi = 245;
const OpULessThan = 176;
const OpExtInst = 12;
const OpCompositeExtract = 81;

// Decoration / BuiltIn
const BuiltIn = 11;
const BuiltIn_GlobalInvocationId = 28;

// Storage classes
const StorageClass_Function = 7;
const StorageClass_CrossWorkgroup = 5;
const StorageClass_Input = 1;

// Function control
const FunctionControl_None = 0;

// Capabilities
const Capability_Addresses = 4;
const Capability_Kernel = 6;
const Capability_Int64 = 11;

// Address model / Memory model / Execution model
const AddressModel_Physical64 = 2;
const MemoryModel_OpenCL = 2;
const ExecutionModel_Kernel = 6;

// OpenCL extended instruction set IDs
const OpenCL_Exp = 19;
const OpenCL_Fmax = 27;

/// Generate a SPIR-V module for: kernel void relu_kernel(global float* in, global float* out, uint N)
pub fn reluModule() SpvModule {
    var m = SpvModule{};
    _ = m.emitPreamble();

    // Type IDs
    const void_t: u32 = 2;
    const uint_t: u32 = 3;
    const float_t: u32 = 4;
    const ptr_float_cw: u32 = 5;
    const func_t: u32 = 6;
    const uint3_t: u32 = 7;
    const ptr_uint3_in: u32 = 8;
    const ptr_uint_func: u32 = 9;
    const bool_t: u32 = 10;

    const gid_var: u32 = 11;
    const func_id: u32 = 12;
    const param_in: u32 = 13;
    const param_out: u32 = 14;
    const param_N: u32 = 15;

    const const_0u: u32 = 16;
    const const_0f: u32 = 17;

    const label_entry: u32 = 18;
    const label_then: u32 = 19;
    const label_merge: u32 = 20;
    const gid_vec: u32 = 21;
    const gid_x: u32 = 22;
    const cmp: u32 = 23;
    const in_ptr: u32 = 24;
    const in_val: u32 = 25;
    const sel: u32 = 26;
    const out_ptr: u32 = 27;
    const cmp_gt: u32 = 28;

    m.emitEntryPoint(ExecutionModel_Kernel, func_id, "relu_kernel", &.{gid_var});

    // Annotations (must precede type/variable declarations per SPIR-V layout)
    m.emit(OpDecorate, &.{ gid_var, BuiltIn, BuiltIn_GlobalInvocationId });

    // Types
    m.emit(OpTypeVoid, &.{void_t});
    m.emit(OpTypeInt, &.{ uint_t, 32, 0 });
    m.emit(OpTypeFloat, &.{ float_t, 32 });
    m.emit(OpTypeBool, &.{bool_t});
    m.emit(OpTypePointer, &.{ ptr_float_cw, StorageClass_CrossWorkgroup, float_t });
    m.emit(OpTypeVector, &.{ uint3_t, uint_t, 3 });
    m.emit(OpTypePointer, &.{ ptr_uint3_in, StorageClass_Input, uint3_t });
    m.emit(OpTypePointer, &.{ ptr_uint_func, StorageClass_Function, uint_t });
    m.emit(OpTypeFunction, &.{ func_t, void_t, ptr_float_cw, ptr_float_cw, uint_t });

    // Constants
    m.emit(OpConstant, &.{ uint_t, const_0u, 0 });
    m.emit(OpConstant, &.{ float_t, const_0f, @bitCast(@as(f32, 0.0)) });

    // Global variables
    m.emit(OpVariable, &.{ ptr_uint3_in, gid_var, StorageClass_Input });

    // Function definition
    m.emit(OpFunction, &.{ void_t, func_id, FunctionControl_None, func_t });
    m.emit(OpFunctionParameter, &.{ ptr_float_cw, param_in });
    m.emit(OpFunctionParameter, &.{ ptr_float_cw, param_out });
    m.emit(OpFunctionParameter, &.{ uint_t, param_N });

    m.emit(OpLabel, &.{label_entry});
    m.emit(OpLoad, &.{ uint3_t, gid_vec, gid_var });
    m.emit(OpCompositeExtract, &.{ uint_t, gid_x, gid_vec, 0 });

    // if (gid < N)
    m.emit(OpULessThan, &.{ bool_t, cmp, gid_x, param_N });
    m.emit(OpSelectionMerge, &.{ label_merge, 0 });
    m.emit(OpBranchConditional, &.{ cmp, label_then, label_merge });

    m.emit(OpLabel, &.{label_then});
    m.emit(OpAccessChain, &.{ ptr_float_cw, in_ptr, param_in, gid_x });
    m.emit(OpLoad, &.{ float_t, in_val, in_ptr });
    m.emit(OpFOrdGreaterThan, &.{ bool_t, cmp_gt, in_val, const_0f });
    m.emit(OpSelect, &.{ float_t, sel, cmp_gt, in_val, const_0f });
    m.emit(OpAccessChain, &.{ ptr_float_cw, out_ptr, param_out, gid_x });
    m.emit(OpStore, &.{ out_ptr, sel });
    m.emit(OpBranch, &.{label_merge});

    m.emit(OpLabel, &.{label_merge});
    m.emit(OpReturn, &.{});
    m.emit(OpFunctionEnd, &.{});

    m.patchBound(29);
    return m;
}

/// Generate a SPIR-V module for:
/// kernel void matmul_kernel(global float* W, global float* x, global float* out, uint M, uint K)
/// One work-item per output row.
pub fn matmulModule() SpvModule {
    var m = SpvModule{};
    _ = m.emitPreamble();

    // Type IDs
    const void_t: u32 = 2;
    const uint_t: u32 = 3;
    const float_t: u32 = 4;
    const ptr_float_cw: u32 = 5;
    const func_t: u32 = 6;
    const uint3_t: u32 = 7;
    const ptr_uint3_in: u32 = 8;
    const bool_t: u32 = 9;
    const ptr_float_func: u32 = 10;

    const gid_var: u32 = 11;
    const func_id: u32 = 12;
    const param_W: u32 = 13;
    const param_x: u32 = 14;
    const param_out: u32 = 15;
    const param_M: u32 = 16;
    const param_K: u32 = 17;

    const const_0u: u32 = 18;
    const const_0f: u32 = 19;
    const const_1u: u32 = 20;

    // Labels
    const label_entry: u32 = 21;
    const label_bounds_ok: u32 = 22;
    const label_exit: u32 = 23;
    const label_loop_hdr: u32 = 24;
    const label_loop_body: u32 = 25;
    const label_loop_end: u32 = 26;

    // SSA
    const gid_vec: u32 = 27;
    const gid_row: u32 = 28;
    const cmp_row: u32 = 29;
    const row_offset: u32 = 30;
    const phi_j: u32 = 31;
    const phi_sum: u32 = 32;
    const cmp_j: u32 = 33;
    const w_idx: u32 = 34;
    const w_ptr: u32 = 35;
    const w_val: u32 = 36;
    const x_ptr: u32 = 37;
    const x_val: u32 = 38;
    const prod: u32 = 39;
    const new_sum: u32 = 40;
    const new_j: u32 = 41;
    const out_ptr: u32 = 42;

    const bound: u32 = 43;

    m.emitEntryPoint(ExecutionModel_Kernel, func_id, "matmul_kernel", &.{gid_var});

    // Annotations
    m.emit(OpDecorate, &.{ gid_var, BuiltIn, BuiltIn_GlobalInvocationId });

    // Types
    m.emit(OpTypeVoid, &.{void_t});
    m.emit(OpTypeInt, &.{ uint_t, 32, 0 });
    m.emit(OpTypeFloat, &.{ float_t, 32 });
    m.emit(OpTypeBool, &.{bool_t});
    m.emit(OpTypePointer, &.{ ptr_float_cw, StorageClass_CrossWorkgroup, float_t });
    m.emit(OpTypeVector, &.{ uint3_t, uint_t, 3 });
    m.emit(OpTypePointer, &.{ ptr_uint3_in, StorageClass_Input, uint3_t });
    m.emit(OpTypePointer, &.{ ptr_float_func, StorageClass_Function, float_t });
    m.emit(OpTypeFunction, &.{ func_t, void_t, ptr_float_cw, ptr_float_cw, ptr_float_cw, uint_t, uint_t });

    // Constants
    m.emit(OpConstant, &.{ uint_t, const_0u, 0 });
    m.emit(OpConstant, &.{ float_t, const_0f, @bitCast(@as(f32, 0.0)) });
    m.emit(OpConstant, &.{ uint_t, const_1u, 1 });

    // Global variables
    m.emit(OpVariable, &.{ ptr_uint3_in, gid_var, StorageClass_Input });

    // Function
    m.emit(OpFunction, &.{ void_t, func_id, FunctionControl_None, func_t });
    m.emit(OpFunctionParameter, &.{ ptr_float_cw, param_W });
    m.emit(OpFunctionParameter, &.{ ptr_float_cw, param_x });
    m.emit(OpFunctionParameter, &.{ ptr_float_cw, param_out });
    m.emit(OpFunctionParameter, &.{ uint_t, param_M });
    m.emit(OpFunctionParameter, &.{ uint_t, param_K });

    m.emit(OpLabel, &.{label_entry});
    m.emit(OpLoad, &.{ uint3_t, gid_vec, gid_var });
    m.emit(OpCompositeExtract, &.{ uint_t, gid_row, gid_vec, 0 });
    m.emit(OpULessThan, &.{ bool_t, cmp_row, gid_row, param_M });
    m.emit(OpSelectionMerge, &.{ label_exit, 0 });
    m.emit(OpBranchConditional, &.{ cmp_row, label_bounds_ok, label_exit });

    m.emit(OpLabel, &.{label_bounds_ok});
    m.emit(OpIMul, &.{ uint_t, row_offset, gid_row, param_K });
    m.emit(OpBranch, &.{label_loop_hdr});

    // Loop header
    m.emit(OpLabel, &.{label_loop_hdr});
    m.emit(OpPhi, &.{ uint_t, phi_j, const_0u, label_bounds_ok, new_j, label_loop_body });
    m.emit(OpPhi, &.{ float_t, phi_sum, const_0f, label_bounds_ok, new_sum, label_loop_body });
    m.emit(OpULessThan, &.{ bool_t, cmp_j, phi_j, param_K });
    m.emit(OpLoopMerge, &.{ label_loop_end, label_loop_body, 0 });
    m.emit(OpBranchConditional, &.{ cmp_j, label_loop_body, label_loop_end });

    // Loop body
    m.emit(OpLabel, &.{label_loop_body});
    m.emit(OpIAdd, &.{ uint_t, w_idx, row_offset, phi_j });
    m.emit(OpAccessChain, &.{ ptr_float_cw, w_ptr, param_W, w_idx });
    m.emit(OpLoad, &.{ float_t, w_val, w_ptr });
    m.emit(OpAccessChain, &.{ ptr_float_cw, x_ptr, param_x, phi_j });
    m.emit(OpLoad, &.{ float_t, x_val, x_ptr });
    m.emit(OpFMul, &.{ float_t, prod, w_val, x_val });
    m.emit(OpFAdd, &.{ float_t, new_sum, phi_sum, prod });
    m.emit(OpIAdd, &.{ uint_t, new_j, phi_j, const_1u });
    m.emit(OpBranch, &.{label_loop_hdr});

    // Loop end -- store result
    m.emit(OpLabel, &.{label_loop_end});
    m.emit(OpAccessChain, &.{ ptr_float_cw, out_ptr, param_out, gid_row });
    m.emit(OpStore, &.{ out_ptr, phi_sum });
    m.emit(OpBranch, &.{label_exit});

    m.emit(OpLabel, &.{label_exit});
    m.emit(OpReturn, &.{});
    m.emit(OpFunctionEnd, &.{});

    m.patchBound(bound);
    return m;
}

/// Generate a SPIR-V module for softmax (single work-item):
/// kernel void softmax_kernel(global float* input, global float* output, uint N)
pub fn softmaxModule() SpvModule {
    var m = SpvModule{};
    const ext_id = m.emitPreamble();

    // Types
    const void_t: u32 = 2;
    const uint_t: u32 = 3;
    const float_t: u32 = 4;
    const ptr_float_cw: u32 = 5;
    const func_t: u32 = 6;
    const uint3_t: u32 = 7;
    const ptr_uint3_in: u32 = 8;
    const bool_t: u32 = 9;

    const gid_var: u32 = 10;
    const func_id: u32 = 11;
    const param_in: u32 = 12;
    const param_out: u32 = 13;
    const param_N: u32 = 14;

    const const_0u: u32 = 15;
    const const_0f: u32 = 16;
    const const_1u: u32 = 17;

    // Labels
    const label_entry: u32 = 18;

    // Pass 1: find max
    const label_max_hdr: u32 = 19;
    const label_max_body: u32 = 20;
    const label_max_end: u32 = 21;

    // Pass 2: compute exp and sum
    const label_exp_hdr: u32 = 22;
    const label_exp_body: u32 = 23;
    const label_exp_end: u32 = 24;

    // Pass 3: normalize
    const label_norm_hdr: u32 = 25;
    const label_norm_body: u32 = 26;
    const label_norm_end: u32 = 27;

    // SSA -- pass 1 (max)
    const in0_ptr: u32 = 28;
    const in0_val: u32 = 29;
    const phi_max_i: u32 = 30;
    const phi_max_v: u32 = 31;
    const cmp_max_i: u32 = 32;
    const max_ptr: u32 = 33;
    const max_load: u32 = 34;
    const max_new_val: u32 = 35;
    const max_new_i: u32 = 36;

    // SSA -- pass 2 (exp + sum)
    const phi_exp_i: u32 = 37;
    const phi_exp_sum: u32 = 38;
    const cmp_exp_i: u32 = 39;
    const exp_in_ptr: u32 = 40;
    const exp_in_val: u32 = 41;
    const exp_diff: u32 = 42;
    const exp_val: u32 = 43;
    const exp_out_ptr: u32 = 44;
    const exp_new_sum: u32 = 45;
    const exp_new_i: u32 = 46;

    // SSA -- pass 3 (normalize)
    const phi_norm_i: u32 = 47;
    const cmp_norm_i: u32 = 48;
    const norm_out_ptr: u32 = 49;
    const norm_load: u32 = 50;
    const norm_div: u32 = 51;
    const norm_new_i: u32 = 52;

    const bound: u32 = 53;

    m.emitEntryPoint(ExecutionModel_Kernel, func_id, "softmax_kernel", &.{gid_var});

    // Annotations
    m.emit(OpDecorate, &.{ gid_var, BuiltIn, BuiltIn_GlobalInvocationId });

    // Types
    m.emit(OpTypeVoid, &.{void_t});
    m.emit(OpTypeInt, &.{ uint_t, 32, 0 });
    m.emit(OpTypeFloat, &.{ float_t, 32 });
    m.emit(OpTypeBool, &.{bool_t});
    m.emit(OpTypePointer, &.{ ptr_float_cw, StorageClass_CrossWorkgroup, float_t });
    m.emit(OpTypeVector, &.{ uint3_t, uint_t, 3 });
    m.emit(OpTypePointer, &.{ ptr_uint3_in, StorageClass_Input, uint3_t });
    m.emit(OpTypeFunction, &.{ func_t, void_t, ptr_float_cw, ptr_float_cw, uint_t });

    // Constants
    m.emit(OpConstant, &.{ uint_t, const_0u, 0 });
    m.emit(OpConstant, &.{ float_t, const_0f, @bitCast(@as(f32, 0.0)) });
    m.emit(OpConstant, &.{ uint_t, const_1u, 1 });

    // Global variables
    m.emit(OpVariable, &.{ ptr_uint3_in, gid_var, StorageClass_Input });

    // Function
    m.emit(OpFunction, &.{ void_t, func_id, FunctionControl_None, func_t });
    m.emit(OpFunctionParameter, &.{ ptr_float_cw, param_in });
    m.emit(OpFunctionParameter, &.{ ptr_float_cw, param_out });
    m.emit(OpFunctionParameter, &.{ uint_t, param_N });

    m.emit(OpLabel, &.{label_entry});
    m.emit(OpAccessChain, &.{ ptr_float_cw, in0_ptr, param_in, const_0u });
    m.emit(OpLoad, &.{ float_t, in0_val, in0_ptr });
    m.emit(OpBranch, &.{label_max_hdr});

    // Pass 1: find max
    m.emit(OpLabel, &.{label_max_hdr});
    m.emit(OpPhi, &.{ uint_t, phi_max_i, const_1u, label_entry, max_new_i, label_max_body });
    m.emit(OpPhi, &.{ float_t, phi_max_v, in0_val, label_entry, max_new_val, label_max_body });
    m.emit(OpULessThan, &.{ bool_t, cmp_max_i, phi_max_i, param_N });
    m.emit(OpLoopMerge, &.{ label_max_end, label_max_body, 0 });
    m.emit(OpBranchConditional, &.{ cmp_max_i, label_max_body, label_max_end });

    m.emit(OpLabel, &.{label_max_body});
    m.emit(OpAccessChain, &.{ ptr_float_cw, max_ptr, param_in, phi_max_i });
    m.emit(OpLoad, &.{ float_t, max_load, max_ptr });
    m.emit(OpExtInst, &.{ float_t, max_new_val, ext_id, OpenCL_Fmax, phi_max_v, max_load });
    m.emit(OpIAdd, &.{ uint_t, max_new_i, phi_max_i, const_1u });
    m.emit(OpBranch, &.{label_max_hdr});

    m.emit(OpLabel, &.{label_max_end});

    // Pass 2: exp(x - max) + sum
    m.emit(OpBranch, &.{label_exp_hdr});

    m.emit(OpLabel, &.{label_exp_hdr});
    m.emit(OpPhi, &.{ uint_t, phi_exp_i, const_0u, label_max_end, exp_new_i, label_exp_body });
    m.emit(OpPhi, &.{ float_t, phi_exp_sum, const_0f, label_max_end, exp_new_sum, label_exp_body });
    m.emit(OpULessThan, &.{ bool_t, cmp_exp_i, phi_exp_i, param_N });
    m.emit(OpLoopMerge, &.{ label_exp_end, label_exp_body, 0 });
    m.emit(OpBranchConditional, &.{ cmp_exp_i, label_exp_body, label_exp_end });

    m.emit(OpLabel, &.{label_exp_body});
    m.emit(OpAccessChain, &.{ ptr_float_cw, exp_in_ptr, param_in, phi_exp_i });
    m.emit(OpLoad, &.{ float_t, exp_in_val, exp_in_ptr });
    m.emit(OpFSub, &.{ float_t, exp_diff, exp_in_val, phi_max_v });
    m.emit(OpExtInst, &.{ float_t, exp_val, ext_id, OpenCL_Exp, exp_diff });
    m.emit(OpAccessChain, &.{ ptr_float_cw, exp_out_ptr, param_out, phi_exp_i });
    m.emit(OpStore, &.{ exp_out_ptr, exp_val });
    m.emit(OpFAdd, &.{ float_t, exp_new_sum, phi_exp_sum, exp_val });
    m.emit(OpIAdd, &.{ uint_t, exp_new_i, phi_exp_i, const_1u });
    m.emit(OpBranch, &.{label_exp_hdr});

    m.emit(OpLabel, &.{label_exp_end});

    // Pass 3: normalize
    m.emit(OpBranch, &.{label_norm_hdr});

    m.emit(OpLabel, &.{label_norm_hdr});
    m.emit(OpPhi, &.{ uint_t, phi_norm_i, const_0u, label_exp_end, norm_new_i, label_norm_body });
    m.emit(OpULessThan, &.{ bool_t, cmp_norm_i, phi_norm_i, param_N });
    m.emit(OpLoopMerge, &.{ label_norm_end, label_norm_body, 0 });
    m.emit(OpBranchConditional, &.{ cmp_norm_i, label_norm_body, label_norm_end });

    m.emit(OpLabel, &.{label_norm_body});
    m.emit(OpAccessChain, &.{ ptr_float_cw, norm_out_ptr, param_out, phi_norm_i });
    m.emit(OpLoad, &.{ float_t, norm_load, norm_out_ptr });
    m.emit(OpFDiv, &.{ float_t, norm_div, norm_load, phi_exp_sum });
    m.emit(OpStore, &.{ norm_out_ptr, norm_div });
    m.emit(OpIAdd, &.{ uint_t, norm_new_i, phi_norm_i, const_1u });
    m.emit(OpBranch, &.{label_norm_hdr});

    m.emit(OpLabel, &.{label_norm_end});
    m.emit(OpReturn, &.{});
    m.emit(OpFunctionEnd, &.{});

    m.patchBound(bound);
    return m;
}

// ── Tests ──

test "reluModule produces valid SPIR-V" {
    const m = reluModule();
    try std.testing.expect(m.len > 5);
    try std.testing.expectEqual(MAGIC, m.words[0]);
    try std.testing.expect(m.words[3] > 0);
    try std.testing.expect(m.words[3] < 200);
}

test "matmulModule produces valid SPIR-V" {
    const m = matmulModule();
    try std.testing.expect(m.len > 5);
    try std.testing.expectEqual(MAGIC, m.words[0]);
    try std.testing.expect(m.words[3] > 0);
    try std.testing.expect(m.words[3] < 200);
}

test "softmaxModule produces valid SPIR-V" {
    const m = softmaxModule();
    try std.testing.expect(m.len > 5);
    try std.testing.expectEqual(MAGIC, m.words[0]);
    try std.testing.expect(m.words[3] > 0);
    try std.testing.expect(m.words[3] < 200);
}
