---
title: "FlashAttention-2: Code architecture"
parent: "023 FlashAttention-2"
nav_order: 1
---
# Code Architecture — FlashAttention-2 Notebook

## Overview

The notebook benchmarks three attention implementations — naive O(N²) attention, a pure-Python tiled attention using the online softmax trick, and PyTorch's `scaled_dot_product_attention` (which dispatches to FlashAttention-2 on GPU) — across sequence lengths from 512 to 8192. It measures forward-pass wall-clock time and peak GPU memory for each. On a Kaggle T4 GPU (15.6 GB), SDPA is 1.7-1.9x faster than naive attention and uses up to 25.7x less peak memory. Naive attention OOMs at seq=8192 (the 8 GB scores matrix exceeds the T4's capacity), directly demonstrating the O(N²) memory bottleneck. SDPA memory stays flat at ~331 MB across all sequence lengths, confirming O(N) scaling.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants:
- `HEAD_DIM = 64` — dimension per attention head (paper tests 64 and 128)
- `N_HEADS = 16` — number of heads (paper uses 32 for d=64 or 16 for d=128 with hidden dim 2048)
- `HIDDEN_DIM = 1024` — hidden dimension (paper: 2048)
- `SEQ_LENS = [512, 1024, 2048, 4096, 8192]` — sequence lengths to benchmark (paper: up to 16k)
- `TOTAL_TOKENS = 16384` — total tokens per batch (paper: 16k); batch size = TOTAL_TOKENS / seq_len
- `BLOCK_SIZE = 128` — tile size for the Python tiled attention (paper: 64 or 128)
- `EPOCHS = 20`, `LR = 1e-3` — placeholder hyperparameters for the toy training demo
- `DEVICE` — CUDA if available, else CPU

### 2. Naive Attention

**Function: `naive_attention`** — the standard attention implementation: compute S = Q @ K^T, apply softmax row-wise, multiply by V. Materializes the full N-by-N attention matrix in GPU memory, requiring O(N²) memory.

### 3. Tiled Attention (Online Softmax)

**Function: `tiled_attention`** — a pure-Python implementation of the tiling concept from FlashAttention. Divides Q into row blocks of size BLOCK_SIZE and K, V into column blocks. For each Q block, iterates over K/V blocks, computing partial attention scores and using the online softmax trick (running max + running sum of exponentials) to accumulate the correct output without materializing the full N-by-N matrix. Memory is O(N * d) instead of O(N²).

This implementation is for pedagogical clarity — it runs in Python loops, so it is slower than both naive attention (which uses CUDA matmuls) and SDPA (which uses the hand-optimized FlashAttention-2 CUDA kernel). Its purpose is to show that the memory saving comes from the tiling algorithm, not from CUDA tricks.

### 4. Correctness Check

Verifies that all three implementations produce the same output (within floating-point tolerance). Generates random Q, K, V of shape (batch, n_heads, seq_len, head_dim) and compares the outputs pairwise. The max difference should be effectively zero (around 1e-6 or less).

### 5. Benchmark — Forward Pass Speed

Measures wall-clock time for a single forward pass of each attention implementation across all sequence lengths. Uses `torch.cuda.Event` for GPU timing with warmup runs. Reports milliseconds per forward pass.

### 6. Benchmark — Peak Memory

Measures peak GPU memory usage for each attention implementation across all sequence lengths using `torch.cuda.max_memory_allocated()`. Resets the memory counter before each call. Reports memory in MB.

### 7. Speed and Memory Plots

Plots forward-pass time vs sequence length (log-log) and peak memory vs sequence length (log-log) for all three implementations. The naive attention curve should show O(N²) scaling (slope 2 on log-log), while SDPA and tiled attention should show O(N) scaling (slope 1).

### 8. Toy Training Comparison

A minimal demo: a single-layer Transformer block with naive attention vs SDPA, trained on random data for 20 epochs. Shows that both produce equivalent training loss, confirming SDPA is a drop-in replacement that does not change model behavior. The training demo uses `EPOCHS = 20` and `LR = 1e-3` with AdamW.

### 9. Summary

Prints a table of results: for each sequence length, the speedup factor (naive time / SDPA time) and memory reduction factor (naive memory / SDPA memory).

## Key Functions

| Function / Class | Purpose |
|---|---|
| `naive_attention` | Standard O(N²) attention that materializes the full attention matrix |
| `tiled_attention` | Pure-Python tiled attention using online softmax — demonstrates the tiling concept |
| `correctness_check` | Verifies all three implementations produce identical outputs |
| `benchmark_speed` | Measures forward-pass wall-clock time across sequence lengths |
| `benchmark_memory` | Measures peak GPU memory across sequence lengths |
| `TransformerBlock` | Single-layer Transformer block for the toy training comparison |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Implementation | Hand-written CUDA/CUTLASS kernel | PyTorch SDPA (dispatches to FlashAttention-2) + pure-Python tiled attention |
| Scope | Forward + backward pass + end-to-end GPT training | Forward pass speed + memory only, plus toy training demo |
| Hardware | A100 80GB and H100 GPUs | Kaggle T4 GPU (16 GB) |
| Sequence lengths | 512 to 16k | 512 to 8192 |
| Head dimensions | 64 and 128 | 64 only |
| Metrics | TFLOPs/s, % of theoretical max, MFU | Wall-clock ms, peak MB, speedup factor |
| Parallelism analysis | Thread block and warp partitioning details | Not measured (uses PyTorch's built-in kernel) |
