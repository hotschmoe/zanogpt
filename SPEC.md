# ZanoGPT NPU Migration Spec

## Current State

ZanoGPT implements a scalar autograd engine in Zig: every `Value` is a single `f64` with graph
metadata. A 16-dim matmul creates ~4000 heap-allocated `Value` nodes. This cannot map onto any
hardware accelerator. The migration replaces scalar autograd with tensor-level operations that
can dispatch to CPU, NPU, or GPU backends.

## Target Architecture

```
                   ┌──────────────────────────────────┐
                   │         zanogpt (model)          │
                   │  transformer, training loop, etc │
                   └──────────────┬───────────────────┘
                                  │ Tensor ops API
                   ┌──────────────▼───────────────────┐
                   │       Tensor Autograd Engine      │
                   │  tape-based, records ops on       │
                   │  Tensors, backward() computes     │
                   │  gradients at tensor level        │
                   └──────────────┬───────────────────┘
                                  │ Backend dispatch
              ┌───────────┬───────┴────────┬──────────────┐
              ▼           ▼                ▼              ▼
         ┌────────┐ ┌──────────┐  ┌──────────────┐ ┌──────────┐
         │  CPU   │ │Intel NPU │  │  AMD XDNA    │ │ Qualcomm │
         │ (SIMD) │ │(LevelZero│  │  (XRT/DRM)   │ │  (QNN)   │
         └────────┘ │ + Graph) │  └──────────────┘ └──────────┘
                    └──────────┘
                          ▲
                     (future: Vulkan compute backend)
```

## Phase 1: Tensor Type + CPU Backend

**Goal:** Replace scalar `Value` with `Tensor`. All model code uses Tensor ops. CPU-only.

### Tensor struct

```zig
pub const Tensor = struct {
    data: []f32,               // contiguous storage (f32 for NPU compat, not f64)
    shape: [MAX_DIMS]usize,    // e.g. [16, 16, 0, 0] for a 16x16 matrix
    ndim: u8,                  // number of dimensions (1-4)
    strides: [MAX_DIMS]usize,  // for views / transposes without copying
    requires_grad: bool,
    grad: ?*Tensor,            // lazily allocated on backward
    _tape_idx: ?usize,         // index into autograd tape (null = leaf / detached)
    allocator: Allocator,

    pub const MAX_DIMS = 4;    // scalars, vectors, matrices, batched matrices
};
```

**Key design decisions:**
- **f32 not f64.** All three NPUs are optimized for f32/f16/int8. f64 has no hardware path.
  The Python reference uses f64 by default; we accept the precision tradeoff.
- **Contiguous row-major by default.** Strides allow transposed views without copies for
  attention's K^T. `Tensor.transpose()` just swaps shape/strides.
- **Static max dims = 4.** Avoids heap-allocating shape arrays. 4 dims covers everything
  in a transformer (batch, seq, heads, dim).

### Autograd tape

Instead of building a graph of Value pointers, use a tape (list of recorded operations):

```zig
pub const TapeEntry = struct {
    op: OpKind,                      // matmul, add, relu, softmax, rmsnorm, log, ...
    inputs: [2]?usize,               // tape indices of input tensors
    output: usize,                   // tape index of output tensor
    // saved tensors for backward (op-specific)
    saved: [2]?*const Tensor,        // e.g. softmax saves its output for backward
};

pub const OpKind = enum {
    matmul,
    add,
    mul_scalar,
    relu,
    softmax,
    rmsnorm,
    log,
    exp,
    transpose,
    embedding_lookup,
};
```

Each forward op appends a `TapeEntry`. Backward walks the tape in reverse, calling the
op-specific gradient function. This is how PyTorch works internally.

### Tensor-level backward implementations needed

| Op | Forward | Backward (dL/dA, dL/dB) |
|----|---------|--------------------------|
| C = matmul(A, B) | C[i,j] = sum_k A[i,k]*B[k,j] | dA = dC @ B^T, dB = A^T @ dC |
| C = A + B | elementwise | dA = dC, dB = dC |
| C = softmax(A) | standard | dC_i * (C_i - C_i * sum_j(dC_j * C_j)) |
| C = rmsnorm(A) | normalize by RMS | chain through scale and mean-square |
| C = relu(A) | max(0, x) | dA = dC * (A > 0) |
| C = log(A) | ln(x) | dA = dC / A |
| C = embedding(idx, W) | W[idx, :] | scatter dC into dW at idx |

### CPU backend ops

Implement each op as a plain Zig function over `[]f32` slices:

```zig
pub const cpu = struct {
    pub fn matmul(a: []const f32, b: []const f32, out: []f32, M: usize, K: usize, N: usize) void {
        // Triple loop with optional @Vector SIMD for inner dot product
    }
    pub fn softmax(input: []const f32, output: []f32, n: usize) void { ... }
    pub fn relu(input: []const f32, output: []f32, n: usize) void { ... }
    pub fn rmsnorm(input: []const f32, output: []f32, n: usize) void { ... }
};
```

SIMD via `@Vector(8, f32)` for the inner loop of matmul. This alone should be 100-500x faster
than scalar autograd for the current model size.

### Model rewrite

The forward pass becomes ~30 lines instead of the current pointer-chasing maze:

```zig
// pseudocode
fn gpt(token: usize, pos: usize, sd: *StateDict, tape: *Tape) !Tensor {
    var x = tape.embedding_lookup(sd.wte, token).add(tape.embedding_lookup(sd.wpe, pos));
    x = tape.rmsnorm(x);
    // attention
    const q = tape.matmul(x, sd.wq);
    const k = tape.matmul(x, sd.wk);
    const v = tape.matmul(x, sd.wv);
    // ... attention score, softmax, weighted sum ...
    return tape.matmul(x, sd.lm_head);
}
```

### What gets deleted

- `Value` struct and all methods (root.zig, ~180 lines)
- `initMatrix`/`freeMatrix`/`flattenParams` (main.zig, ~60 lines)
- All `try val.add(other, alloc)` plumbing in the forward pass


## Phase 2: Backend Abstraction

**Goal:** Define a comptime interface that backends implement. Allow switching backends at
build time or runtime.

```zig
pub const Backend = struct {
    // Core tensor ops
    matmul: *const fn (a: *const Tensor, b: *const Tensor, out: *Tensor) void,
    add: *const fn (a: *const Tensor, b: *const Tensor, out: *Tensor) void,
    softmax: *const fn (input: *const Tensor, out: *Tensor) void,
    relu: *const fn (input: *const Tensor, out: *Tensor) void,
    rmsnorm: *const fn (input: *const Tensor, out: *Tensor) void,

    // Memory management
    alloc_tensor: *const fn (shape: []const usize, allocator: Allocator) *Tensor,
    free_tensor: *const fn (t: *Tensor) void,
    sync: *const fn () void,  // wait for async ops to complete

    // Lifecycle
    init: *const fn (allocator: Allocator) Backend,
    deinit: *const fn (self: *Backend) void,
};
```

The CPU backend fills this table with direct function pointers. NPU backends fill it with
functions that build/submit device command buffers.

**Comptime vs runtime dispatch:** Use comptime selection by default (zero overhead), with
runtime vtable as an option for testing multiple backends in one binary:

```zig
// build.zig option: -Dbackend=cpu|intel_npu|amd_xdna|qnn
const backend = switch (build_options.backend) {
    .cpu => @import("backends/cpu.zig").backend,
    .intel_npu => @import("backends/intel_npu.zig").backend,
    // ...
};
```


## Phase 3: Intel NPU Backend (Level-Zero + Graph Extensions)

### Hardware model

Intel NPU (Meteor Lake / Lunar Lake / Arrow Lake) is a **graph-execution accelerator**:
- 2 Neural Compute Engine (NCE) tiles, each with 512 MAC Processing Engines
- ~9.5 TOPS INT8, ~50 GFLOPS FP32 (via SHAVE DSP cores)
- 2 MB software-managed SRAM per NCE
- Operates on compiled graph blobs — no individual kernel dispatch
- Accessed as a PCIe device through the Intel NPU Linux driver

### C API surface (Zig @cImport)

```zig
const ze = @cImport({
    @cInclude("level_zero/ze_api.h");
    // NPU-specific extensions from intel/level-zero-npu-extensions:
    @cInclude("level_zero/ze_graph_ext.h");           // graph create/execute — THE main NPU API
    @cInclude("level_zero/ze_graph_profiling_ext.h");  // profiling pools and queries
    @cInclude("level_zero/ze_driver_npu_ext.h");       // NPU driver extension discovery
    @cInclude("level_zero/ze_command_queue_npu_ext.h"); // turbo mode, workload type
    @cInclude("level_zero/ze_intel_npu_uuid.h");       // device UUID for enumeration
});
```

**Package:** `level-zero-devel` (headers at `/usr/include/level_zero/`).
**NPU extensions:** from [intel/level-zero-npu-extensions](https://github.com/intel/level-zero-npu-extensions).
NPU extension functions are accessed via **dispatch tables** (DDI tables), not direct symbols.
Obtained through `zeDriverGetExtensionFunctionAddress`.

### Initialization flow

```
zeInit(ZE_INIT_FLAG_VPU_ONLY)
  → zeDriverGet(&driverCount, &drivers)
    → zeDeviceGet(driver, &deviceCount, &devices)
      → zeContextCreate(driver, &contextDesc, &context)
        → zeCommandQueueCreate(context, device, &queueDesc, &queue)
```

### Programming model

The Intel NPU does **not** support SPIR-V compute kernels. You must:

1. **Build a computation graph** using the graph extension API (`ze_graph_ext`):
   - Compile an OpenVINO IR / ONNX model into an NPU blob
   - Or use `intel-npu-acceleration-library` NNFactory to build graphs programmatically
2. **Create a graph handle:** `pfnCreate(context, device, &graphDesc, &graphHandle)`
3. **Set input/output arguments:** `pfnSetArgumentValue(graphHandle, argIndex, data_ptr)`
4. **Append to command list:** `pfnAppendGraphExecute(cmdList, graphHandle, ...)`
5. **Submit and synchronize:** `zeCommandQueueExecuteCommandLists`, `zeCommandQueueSynchronize`

### Memory model

```
zeMemAllocShared(context, &deviceDesc, &hostDesc, size, alignment, device, &ptr)
```

Shared memory is accessible by both CPU and NPU. For input/output tensors, allocate shared
memory, write f32 data from CPU, execute graph, read results back from the same pointer.

### Strategy for ZanoGPT

Two viable approaches:

**Option A: Compile the whole transformer as one graph blob.**
- Export the forward pass as ONNX → compile to NPU blob → execute per token
- Pros: NPU handles everything, maximum hardware utilization
- Cons: No per-op flexibility, graph is static, harder to debug
- Training: NPU blob is inference-only. Backward pass stays on CPU.

**Option B: Build individual ops as small graphs programmatically.**
- Construct OpenVINO IR blobs in memory describing single ops (matmul, softmax, etc.)
- Submit via `pfnCreate2` with `ZE_GRAPH_FORMAT_NGRAPH_LITE`, cache native binary
  via `pfnGetNativeBinary`, reload with `ZE_GRAPH_FORMAT_NATIVE` to skip recompilation
- Pros: Flexible, can mix NPU and CPU ops, easier to debug
- Cons: Graph compilation overhead per op (mitigated by caching), less cross-op optimization
- Note: `intel-npu-acceleration-library` (NNFactory) is **archived/EOL** — we build
  the IR blobs ourselves in Zig, which is actually more aligned with the project goals

**Recommendation: Start with Option B** (individual op graphs with native binary caching),
then optimize to Option A once everything works.

### Graph compilation and caching flow

```
1. Build OpenVINO IR blob in memory (XML+BIN describing the op graph)
2. pfnCreate2(context, device, &desc, &graph)   // ZE_GRAPH_FORMAT_NGRAPH_LITE
3. pfnSetArgumentValue(graph, i, sharedMemPtr)   // bind I/O buffers
4. pfnAppendGraphInitialize(cmdList, graph, ...)  // one-time init
5. pfnAppendGraphExecute(cmdList, graph, ...)     // execute
6. zeCommandQueueExecuteCommandLists(queue, ...)  // submit
7. zeFenceHostSynchronize(fence, ...)             // wait

// Cache for reuse:
8. pfnGetNativeBinary(graph, &size, blob)         // extract compiled blob
9. Save to disk, reload later with ZE_GRAPH_FORMAT_NATIVE (skip recompilation)
```

### Build integration

```zig
// build.zig
if (backend == .intel_npu) {
    exe.linkSystemLibrary("ze_loader");
    exe.addIncludePath(.{ .cwd_relative = "/usr/include" });
    exe.addIncludePath(.{ .cwd_relative = "vendor/level-zero-npu-extensions/include" });
}
```


## Phase 4: AMD XDNA Backend (XRT + DRM)

### Hardware model

AMD XDNA (Ryzen AI 7000/8000/9000) is an **AI Engine tile array**:
- Based on Xilinx/AMD Versal AI Engine architecture (AIE2)
- Array of programmable tiles connected by a network-on-chip
- Each tile has vector/MAC units + local data memory
- DMA engines for moving data between tiles and DDR
- Programmed via compiled instruction sequences + xclbin containers

### Two API layers

**Layer 1: XRT C API (recommended starting point):**
```zig
const xrt = @cImport({
    @cInclude("xrt/xrt_device.h");
    @cInclude("xrt/xrt_bo.h");
    @cInclude("xrt/xrt_kernel.h");
    @cInclude("xrt/xrt_xclbin.h");
});
```

Key functions:
```
xrtDeviceOpen(index) → xrtDeviceHandle
xrtDeviceLoadXclbinFile(device, "model.xclbin") → int
xrtPLKernelOpen(device, uuid, "MLIR_AIE") → xrtKernelHandle
xrtBOAlloc(device, size, flags, group) → xrtBufferHandle
xrtBOMap(buffer) → void*
xrtBOSync(buffer, direction, size, offset) → int
xrtKernelRun(kernel, opcode, bo_instr, instr_count, bo_in, bo_out) → xrtRunHandle
xrtRunWait(run) → ert_cmd_state
xrtBOFree(buffer)
xrtDeviceClose(device)
```

**Layer 2: Raw DRM ioctls (maximum control, for later):**

Device node: `/dev/accel/accel0` (DRM accel subsystem, merged Linux 6.14).

```zig
const amdxdna = @cImport({
    @cInclude("drm/amdxdna_accel.h");
    @cInclude("xf86drm.h");  // drmIoctl() helper
});
```

DRM IOCTLs:
| Ioctl | Purpose |
|-------|---------|
| `DRM_IOCTL_AMDXDNA_CREATE_HWCTX` | Allocate AIE tile columns, get hardware context |
| `DRM_IOCTL_AMDXDNA_DESTROY_HWCTX` | Release context, reclaim tiles |
| `DRM_IOCTL_AMDXDNA_CONFIG_HWCTX` | Configure CU overlay on context |
| `DRM_IOCTL_AMDXDNA_CREATE_BO` | Allocate GEM buffer object (SHMEM, DEV, CMD) |
| `DRM_IOCTL_AMDXDNA_GET_BO_INFO` | Get mmap offset and device address |
| `DRM_IOCTL_AMDXDNA_SYNC_BO` | Flush CPU cache to/from device |
| `DRM_IOCTL_AMDXDNA_EXEC_CMD` | Submit command for execution |
| `DRM_IOCTL_AMDXDNA_GET_INFO` | Query AIE status, firmware version |

Synchronization via DRM syncobj (returned from CREATE_HWCTX).

### Programming model

The IRON/MLIR-AIE toolchain produces **two artifacts**:
1. **`final.xclbin`** — Static NPU configuration: stream switch routing + ELF binaries for
   compute tiles. Loaded once at context creation.
2. **`insts.txt`** — Instruction buffer (ctrlcode opcodes for the NPU's ERT firmware):
   DMA block writes, register writes, runtime reconfiguration. Submitted per execution.

Host-side flow (XRT C API):
```
1. xrtDeviceOpen(0)                                    // open NPU
2. xrtDeviceLoadXclbinFile(device, "final.xclbin")     // program overlay
3. xrtPLKernelOpen(device, uuid, "MLIR_AIE")           // get kernel handle
4. xrtBOAlloc(device, size, flags, group)               // allocate BOs for instr/in/out
5. xrtBOMap(bo) → memcpy data into mapped ptr           // fill input data
6. xrtBOSync(bo, XCL_BO_SYNC_BO_TO_DEVICE, ...)        // sync to device
7. xrtKernelRun(kernel, opcode=3, bo_instr, count,      // execute
                 bo_input, bo_output)
8. xrtRunWait(run)                                      // wait for completion
9. xrtBOSync(bo_out, XCL_BO_SYNC_BO_FROM_DEVICE, ...)  // read results
```

### Strategy for ZanoGPT

**The challenge:** AMD XDNA requires pre-compiled AIE kernels in xclbin format. You can't
dynamically build computation graphs at runtime like Intel NPU.

**Approach:**
1. Use MLIR-AIE to compile matmul/softmax/relu/rmsnorm kernels for common sizes
   (16x16, 64x64, etc.) → produces xclbin files
2. At ZanoGPT startup, load the appropriate xclbin
3. For each tensor op, select the right pre-compiled kernel, allocate BOs, execute
4. For shapes that don't have pre-compiled kernels, fall back to CPU

**This is the most "bare metal" of the three NPU targets** — you're essentially programming
an FPGA-like tile array. Great for learning, but more setup work.

### Build integration

```zig
if (backend == .amd_xdna) {
    exe.linkSystemLibrary("xrt_coreutil");
    exe.addIncludePath(.{ .cwd_relative = "/opt/xilinx/xrt/include" });
    exe.addLibraryPath(.{ .cwd_relative = "/opt/xilinx/xrt/lib" });
}
```


## Phase 5: Qualcomm QNN Backend

### Hardware model

Qualcomm Hexagon NPU (Snapdragon X Elite/Plus):
- Hexagon Tensor Processor (HTP) — dedicated matrix/vector accelerator
- Hexagon Vector eXtensions (HVX) — SIMD vector unit
- Supports INT8, INT16, FP16 natively; FP32 via upconversion
- Graph-based execution model

### C API surface (Zig @cImport)

QNN uses a **dynamic loading + function pointer table** pattern:

```zig
const qnn = @cImport({
    @cInclude("QNN/QnnInterface.h");
    @cInclude("QNN/QnnBackend.h");
    @cInclude("QNN/QnnContext.h");
    @cInclude("QNN/QnnGraph.h");
    @cInclude("QNN/QnnTensor.h");
    @cInclude("QNN/QnnOpDef.h");
});
```

### Initialization flow

```
dlopen("libQnnHtp.so")
  → dlsym("QnnInterface_getProviders")
    → getProviders(&providerList, &numProviders)
      → providerList[0].QNN_INTERFACE_VER_NAME  // get function table
        → interface.backendCreate(logHandle, &backendConfigs, &backend)
          → interface.contextCreate(backend, NULL, &contextConfigs, &context)
            → interface.graphCreate(context, "forward_pass", &graphConfigs, &graph)
```

### Programming model (graph-build API)

QNN is the most ergonomic NPU API for this project — you build graphs by adding nodes:

```
// Add a matmul node
interface.graphAddNode(graph, QnnNode{
    .name = "attn_wq",
    .packageName = "qti.aisw",
    .typeName = "MatMul",        // QNN_OP_MAT_MUL
    .params = &params,
    .numOfParams = numParams,
    .inputTensors = &inputs,
    .numOfInputs = 2,
    .outputTensors = &outputs,
    .numOfOutputs = 1,
})

// After adding all nodes:
interface.graphFinalize(graph, NULL, NULL)

// Execute:
interface.graphExecute(graph, inputs, numInputs, outputs, numOutputs, NULL, NULL)
```

### Built-in ops available in QNN

From `QnnOpDef.h` (package: `QNN_OP_PACKAGE_NAME_QTI_AISW`) — transformer ops available:
- `QNN_OP_MAT_MUL` — matrix multiplication (with `TRANSPOSE_IN0`/`TRANSPOSE_IN1` params)
- `QNN_OP_FULLY_CONNECTED` — matmul variant (used when 2nd input is rank-2 initializer)
- `QNN_OP_SOFTMAX` — softmax (with `AXIS` param)
- `QNN_OP_RELU` — ReLU activation
- `QNN_OP_LAYER_NORM` / RMS norm via element-wise ops
- `QNN_OP_ELEMENT_WISE_ADD` — residual connections
- `QNN_OP_GATHER` — embedding lookup
- `QNN_OP_LOG_SOFTMAX`, `QNN_OP_REDUCE_SUM`, etc.

Tensor types: `QNN_TENSOR_TYPE_NATIVE` (internal), `QNN_TENSOR_TYPE_APP_READ` (output),
`QNN_TENSOR_TYPE_STATIC` (weights/initializers).

### Strategy for ZanoGPT

**QNN maps most naturally onto ZanoGPT's needs:**
1. Build the entire forward pass as a QNN graph (one graph per sequence length)
2. Or build per-layer graphs for more flexibility
3. Graph finalization compiles and optimizes for HTP
4. Execute returns tensor results

**For training:** Build a separate backward-pass graph, or compute gradients on CPU
and only use QNN for the forward pass (common approach for NPU training).

### Build integration

```zig
if (backend == .qnn) {
    // QNN uses runtime dlopen, so no link-time dependency
    // Just need headers at compile time
    exe.addIncludePath(.{ .cwd_relative = "vendor/qnn-sdk/include" });
}
```


## Phase 6 (Future): Vulkan Compute Backend

Not specced in detail since this is a separate project. Key notes:

- Use existing Vulkan Zig bindings (from the other project)
- Compute shaders for matmul, softmax, etc.
- VkBuffer for tensor storage, vkCmdDispatch for execution
- Most flexible backend — works on any Vulkan-capable GPU
- Would slot into the same `Backend` interface


## Implementation Order

```
Phase 1: Tensor + CPU       ← foundation, everything depends on this
  │
  ├─ Phase 2: Backend API   ← thin abstraction layer
  │    │
  │    ├─ Phase 3: Intel NPU    ← most mature Linux driver
  │    ├─ Phase 4: AMD XDNA     ← most interesting low-level model
  │    ├─ Phase 5: Qualcomm QNN ← most ergonomic API
  │    └─ Phase 6: Vulkan       ← future, separate project synergy
  │
  └─ Tests at every phase:
       - Numerical correctness vs Python reference (compare loss curves)
       - Gradient checking (finite differences vs autograd)
       - Backend equivalence (CPU output ≈ NPU output within f32 tolerance)
```

## File Structure (Target)

```
src/
  root.zig             → Tensor, autograd tape, ops API (public library)
  main.zig             → model definition, training loop, inference
  backends/
    cpu.zig            → SIMD f32 implementations
    intel_npu.zig      → Level-Zero + graph extension bindings
    amd_xdna.zig       → XRT C API bindings
    qnn.zig            → QNN function pointer table bindings
vendor/
  level-zero-npu-extensions/ → Intel NPU extension headers (git submodule)
  qnn-sdk/                   → QNN SDK headers (manual install, gitignored)
data/
  names.txt            → training data
  kernels/             → pre-compiled xclbin files for AMD XDNA
reference/
  microgpt.py          → Python reference implementation
```

## Open Questions

1. **f16 support?** All NPUs benefit from f16. Could add f16 tensor dtype later, but f32 first
   for correctness parity with the Python reference.

2. **Batched training?** Current implementation trains one document at a time. NPUs benefit
   from batched execution. Batch dim support in Tensor would help but adds complexity.

3. **Training on NPU vs inference-only?** Forward pass on NPU + backward on CPU is the
   pragmatic starting point. Full NPU training requires compiling backward graphs, which
   is feasible on QNN and Intel NPU but harder on AMD XDNA.

4. **Graph caching?** Intel NPU and QNN both compile graphs — this is slow. Cache compiled
   graphs for repeated execution (same shapes = same compiled graph).
