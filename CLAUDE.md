# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ZanoGPT is a zero-dependency, Zig-native reimplementation of [Karpathy's microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). It trains a tiny GPT-2-style transformer on character-level name data and generates new names. The reference Python implementation lives at `reference/microgpt.py`.

**Key differences from GPT-2:** RMSNorm (not LayerNorm), ReLU (not GeLU), no biases.

## Build & Run

```bash
zig build run          # Build and run the executable
zig build test         # Run all tests (both library and executable modules)
zig build              # Build only (output in zig-out/)
```

Requires Zig **0.15.2+** (see `build.zig.zon`). Zero external dependencies.

## Architecture

Two Zig modules defined in `build.zig`:

- **`src/root.zig`** — Library module exposed as `"zanogpt"`. This is the public API surface; all reusable logic (autograd, transformer, tokenizer, optimizer) belongs here.
- **`src/main.zig`** — Executable entry point. Imports the library via `@import("zanogpt")`. Orchestrates training and inference.

The project is currently scaffolded — the core GPT components (autograd `Value` type, transformer forward pass, Adam optimizer, tokenizer) still need to be implemented in Zig, following the reference Python in `reference/microgpt.py`.

## Model Hyperparameters

| Parameter      | Value |
|----------------|-------|
| Embedding dim  | 16    |
| Attention heads| 4     |
| Layers         | 1     |
| Block size     | 16    |
| Head dim       | 4 (n_embd / n_head) |

## Dataset

`data/names.txt` — ~32,000 names from Karpathy's makemore project. Character-level tokenization: each unique character gets a token ID, plus a BOS (Beginning of Sequence) token.
