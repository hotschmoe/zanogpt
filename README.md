# ZanoGPT

A zero-dependency, Zig-native micro GPT — a from-scratch recreation of [Karpathy's microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) in pure Zig.

The original Python implementation fits the entire GPT training + inference loop in ~200 lines with no libraries. ZanoGPT aims to do the same in Zig, leveraging Zig's strengths: comptime evaluation, manual memory control, and no hidden allocations.

## What It Does

Trains a tiny GPT (transformer) model on a dataset of names, then generates new, never-before-seen names. The architecture follows GPT-2 with minor simplifications (RMSNorm instead of LayerNorm, ReLU instead of GeLU, no biases).

**Core components — all implemented from scratch in Zig:**

- **Autograd engine** — scalar-level automatic differentiation with reverse-mode backprop
- **Transformer** — token/position embeddings, multi-head self-attention, MLP blocks
- **Adam optimizer** — with linear learning rate decay
- **Tokenizer** — character-level tokenization

## Hardware Acceleration

ZanoGPT auto-detects available accelerators at startup and prompts you to choose:

| Backend | Hardware | API |
|---------|----------|-----|
| **Intel NPU** | Intel AI Boost (Core Ultra) | Level-Zero Graph Extension (OpenVINO IR) |
| **Intel GPU** | Intel Arc Graphics | Level-Zero Compute (SPIR-V kernels) |
| **CPU** | Any | Zig SIMD/scalar |

To skip the interactive prompt, set the `ZANOGPT_BACKEND` environment variable:

```bash
ZANOGPT_BACKEND=gpu zig build run    # Force GPU
ZANOGPT_BACKEND=npu zig build run    # Force NPU
ZANOGPT_BACKEND=cpu zig build run    # Force CPU
```

## Project Structure

```
zanogpt/
├── build.zig              # Zig build system
├── build.zig.zon          # Package manifest
├── src/
│   ├── main.zig           # Executable entry point
│   ├── root.zig           # Library module (autograd, transformer, etc.)
│   └── backends/
│       ├── detect.zig     # Hardware detection & backend selection
│       ├── backend.zig    # Backend interface (function pointer table)
│       ├── cpu.zig        # CPU backend
│       ├── intel_npu.zig  # Intel NPU backend (Level-Zero + Graph API)
│       ├── intel_gpu.zig  # Intel GPU backend (Level-Zero + compute)
│       ├── ze.zig         # Level-Zero FFI bindings
│       ├── ov_ir.zig      # OpenVINO IR XML generation (for NPU)
│       └── spirv.zig      # Programmatic SPIR-V generation (for GPU)
├── data/
│   └── names.txt          # 32k names dataset (from Karpathy's makemore)
└── reference/
    └── microgpt.py        # Original Python reference implementation
```

## Dataset

`data/names.txt` contains ~32,000 names sourced from [Karpathy's makemore](https://github.com/karpathy/makemore). This is the same dataset used in the original Python implementation. The model learns character-level patterns from these names and generates plausible new ones.

## Model Hyperparameters

| Parameter       | Value |
|-----------------|-------|
| Embedding dim   | 16    |
| Attention heads | 4     |
| Layers          | 1     |
| Block size      | 16    |
| Vocab size      | unique chars + 1 (BOS) |

## Building & Running

```bash
zig build run          # Build and run (interactive accelerator prompt)
zig build test         # Run all tests
zig build              # Build only (output in zig-out/)
```

Requires Zig **0.15.2+**.

## Why Zig?

- **Zero dependencies** — no libc, no allocator libraries, no BLAS. Just Zig.
- **Comptime** — embedding tables, layer counts, and dimensions are compile-time known, enabling the compiler to unroll and optimize aggressively.
- **Explicit memory** — no GC, no hidden allocations. Every buffer is visible and controlled.
- **Simplicity** — Zig's straightforward semantics map cleanly to the math. No operator overloading magic, no class hierarchies — just structs and functions.

## Reference

- [microGPT by @karpathy](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — the original Python implementation this project is based on
- [makemore](https://github.com/karpathy/makemore) — source of the names dataset
- [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — the architecture this model simplifies
