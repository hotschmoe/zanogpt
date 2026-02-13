# Intel NPU Backend Roadmap

Target: Intel Ultra 9 275HX (Arrow Lake) with integrated NPU, Windows.

## Status

- [x] Phase 2: Backend abstraction + stubs (this machine, Linux)
- [ ] Phase 3: Level-Zero init/deinit on Windows
- [ ] Phase 4: Single-op NPU execution (matmul)
- [ ] Phase 5: Graph compilation + caching
- [ ] Phase 6: Full forward pass on NPU

## Implementation Phases

### Phase 3 — Level-Zero Lifecycle (Windows)
- Install Level-Zero SDK + Intel NPU driver
- Implement `init()`: discover NPU device via `zeInit` / `zeDriverGet` / `zeDeviceGet`
- Implement `deinit()`: tear down context + command queue
- Validate with a simple device info print

### Phase 4 — Single-Op Execution
- Implement `matmul_fwd` using Level-Zero graph extension
- Allocate device memory, copy input, execute, copy output
- Benchmark vs CPU SIMD baseline

### Phase 5 — Graph Compilation + Caching
- Use `ShapeKey` to cache compiled graphs by op type + dimensions
- Avoid recompilation when shapes repeat across tokens
- Implement `ze_graph_ext` graph creation + profiling

### Phase 6 — Full Forward Pass
- Implement remaining ops: `softmax_fwd`, `rmsnorm_fwd`, `relu_fwd`
- Fuse multi-op subgraphs where profitable
- End-to-end inference on NPU

## Windows Setup (Intel Ultra 9 275HX)

### 1. Intel NPU Driver
```
Download from: https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html
Verify: Device Manager → Neural Processing Unit → Intel(R) AI Boost
```

### 2. Level-Zero SDK
```powershell
# Option A: oneAPI installer (includes Level-Zero)
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html

# Option B: standalone Level-Zero from GitHub
git clone https://github.com/oneapi-src/level-zero.git
cd level-zero && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=C:/level-zero
cmake --build . --config Release --target install
```

### 3. NPU Graph Extensions
```powershell
cd vendor/level-zero-npu-extensions
git submodule add https://github.com/intel/linux-npu-driver.git
# Headers needed: ze_graph_ext.h, ze_graph_profiling_ext.h
```

### 4. Build with NPU backend
```powershell
zig build -Dbackend=intel_npu
```

## Graph Caching Design

The NPU compiles computation graphs ahead of execution. Recompiling per-token is too slow, so we cache compiled graphs keyed by `(op_type, shape)`:

```
ShapeKey { .op = .matmul, .dims = { M, K, 0, 0 } }  →  compiled ze_graph_handle_t
```

For ZanoGPT's fixed hyperparameters (n_embd=16, n_head=4), there are ~6 unique shapes, so the cache stays small. The attention softmax dimension grows with sequence length (1..block_size), producing up to `block_size` cached softmax graphs.
