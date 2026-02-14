# GPU Loss Investigation Plan

## Problem

GPU backend produces significantly higher loss values than CPU:
- **CPU**: step 1 loss ~3.40, step 10 loss ~2.97 (converging)
- **GPU**: step 1 loss ~37.16, step 10 loss ~20.24 (10x higher, slow convergence)

The GPU backend initializes successfully and runs all operations (matmul, relu, softmax, rmsnorm), but the numerical results diverge from the CPU baseline.

## Hypothesis Priority

### 1. Shared buffer aliasing (MOST LIKELY)
`getOrAllocShared()` returns the first buffer >= requested size. If two kernel arguments map to the same shared buffer (e.g., input and output both need 64 bytes and the cache returns the same buffer), the kernel would read corrupted data.

**Test**: Add logging to `getOrAllocShared()` to print pointer addresses and sizes. Verify no two arguments in a single kernel launch share a buffer.

**Fix**: Either allocate distinct buffers per argument, or change the lookup to only reuse buffers not currently in use.

### 2. Matmul kernel index arithmetic
The matmul kernel computes `W[row * K + j]` using u32 arithmetic. If `row * K` overflows u32 (unlikely at current model size but worth checking) or if the index computation is subtly wrong, the result would be garbage.

**Test**: Run a small matmul (e.g., 4x4 * 4x1) on GPU and CPU, compare results element-by-element.

### 3. Softmax numerical stability
The softmax kernel uses single work-item dispatch. The `exp(x - max)` and normalization passes should be numerically stable, but verify:
- The `fmax` extended instruction works correctly on the GPU
- The `exp` extended instruction produces matching results
- The division in the normalization pass doesn't produce NaN/Inf for edge cases

**Test**: Run softmax on a known input vector on both CPU and GPU, compare results.

### 4. ReLU kernel correctness
Simplest kernel -- unlikely to be wrong, but easy to verify.

**Test**: Run relu on `[-1, 0, 1, 2, -3]` on both CPU and GPU, compare.

### 5. Memory copy direction or size
The pattern in each `*_fwd` function is:
1. Copy input from host slice to shared buffer
2. Launch kernel
3. Copy output from shared buffer to host slice

If the `@memcpy` sizes or slices are wrong, the GPU would operate on partially-initialized memory.

**Test**: After the GPU writes output, print the first few values and compare with CPU.

### 6. Kernel argument order
`launchKernel` sets arguments by index. If the SPIR-V parameter order doesn't match the argument order in the `launchKernel` call, the kernel would read the wrong data.

**Test**: Verify SPIR-V `OpFunctionParameter` order matches the `KernelArg` array order for each kernel.

## Investigation Steps

### Phase 1: Isolate which operation diverges
1. Add a `--debug-gpu` flag or `ZANOGPT_DEBUG=1` env var
2. After each GPU operation (relu, matmul, softmax), also run the CPU version on the same input
3. Compare results and print the max absolute difference
4. This immediately tells us which kernel(s) are wrong

### Phase 2: Fix the broken kernel(s)
Based on Phase 1 results:
- If matmul is wrong: check index arithmetic, test small matrices
- If softmax is wrong: check extended instruction behavior, test small vectors
- If shared buffer aliasing: redesign buffer allocation

### Phase 3: Regression test
1. Add a test that runs each GPU kernel on known inputs and verifies outputs match CPU
2. Add a test that runs 10 training steps on GPU and verifies loss is within 1% of CPU

## Quick Win: Buffer Aliasing Check

The most likely cause is buffer aliasing in `getOrAllocShared()`. The current implementation:

```zig
fn getOrAllocShared(min_size: usize) !SharedBuf {
    for (shared_bufs[0..num_bufs]) |buf| {
        if (buf.size >= min_size) return buf;  // BUG: returns same buffer for same size!
    }
    // ... allocate new
}
```

If `matmul_fwd` calls `getOrAllocShared(M*K*4)` for W, then `getOrAllocShared(K*4)` for x, and then `getOrAllocShared(M*4)` for out -- the first buffer (size M*K*4) would be returned for ALL THREE since it's >= all requested sizes. The `@memcpy` for x would overwrite part of W, and the output would overwrite more.

**Fix**: Either:
- (A) Allocate per-operation buffers (don't cache/reuse)
- (B) Mark buffers as "in use" and only return unused ones
- (C) Allocate one large buffer and carve out sub-regions with distinct offsets
