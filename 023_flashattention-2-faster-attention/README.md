---
title: "023 FlashAttention-2"
nav_order: 24
permalink: "/023_flashattention-2-faster-attention/"
---
# FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (FlashAttention-2)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/flashattention-2-faster-attention)

- **Paper:** [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- **Authors:** Tri Dao
- **Lab, year:** Princeton University and Stanford University, 2023 (arXiv preprint, July 2023)

## Summary

Transformers use an attention mechanism whose runtime and memory grow quadratically with sequence length — doubling the sequence quadruples the cost. FlashAttention (Dao et al., 2022) solved the memory problem by tiling: loading blocks of Q, K, V from slow GPU memory (HBM) to fast on-chip memory (SRAM), computing attention block by block, and never writing the full N-by-N attention matrix back to HBM. This reduced memory from O(N²) to O(N) and gave a 2-4x speedup, with no approximation. However, FlashAttention still reached only 25-40% of the GPU's theoretical maximum FLOPs/s (floating-point operations per second), far below the 80-90% that optimized matrix multiply (GEMM) achieves.

FlashAttention-2 closes this gap through three improvements. First, it tweaks the algorithm to reduce non-matmul FLOPs (each non-matmul FLOP is 16x slower than a matmul FLOP on A100, since GPUs have dedicated matrix-multiply units). Second, it parallelizes attention across the sequence length dimension in addition to batch and heads, increasing GPU occupancy for long sequences. Third, it reorganizes how work is split among warps (groups of 32 threads) within each thread block, eliminating the "split-K" scheme that forced expensive shared-memory communication. Together these yield about 2x speedup over FlashAttention, reaching 50-73% of theoretical max FLOPs/s on A100. End-to-end GPT-style training reaches up to 225 TFLOPs/s per A100 (72% model FLOPs utilization).

This repo's notebook benchmarks naive attention against PyTorch's scaled dot-product attention (SDPA, which uses FlashAttention-2 internally) across sequence lengths from 512 to 8192, measuring both speed and peak memory. On GPU, SDPA is 5-10x faster than naive attention and uses O(N) instead of O(N²) memory.

## What Problem Does It Solve

Imagine you are cooking from a recipe book. The attention computation is like looking up ingredients: for every step of the recipe (query), you check every ingredient list (keys) to find matches, then combine the matching amounts (values). If the recipe has 1000 steps, you make a 1000-by-1000 lookup table — a million entries. Double the recipe to 2000 steps and you need 4 million entries. This quadratic growth is the bottleneck.

FlashAttention's trick is like cooking one page at a time: load a few ingredients to your cutting board (SRAM), combine what you need, and never write the full million-entry table back to the shelf (HBM). FlashAttention-2 makes the cutting-board work more efficient: it assigns each cook (warp) a separate set of recipe steps so they never need to share results, and it splits the work across more cooks (thread blocks) when the recipe is long but there are few batches to keep them busy.

## Core Idea

- **Reduce non-matmul FLOPs (Section 3.1):** Maintain an unscaled running output and only rescale at the end; store logsumexp instead of separate max and sum. This minimizes the slower non-matrix operations.
- **Parallelize over sequence length (Section 3.2):** In FlashAttention, each thread block handles one (batch, head) pair. FlashAttention-2 additionally splits the sequence dimension across thread blocks, increasing GPU occupancy when batch size and head count are small (the long-sequence regime).
- **Better warp partitioning (Section 3.3):** FlashAttention splits K and V across 4 warps ("split-K"), forcing warps to write intermediate results to shared memory and synchronize. FlashAttention-2 instead splits Q across warps, so each warp computes an independent slice of the output with no inter-warp communication.

## Key Method Details

- **Forward pass (Algorithm 1):** Online softmax with tiling. Q is divided into row blocks Q_i; K, V into column blocks K_j, V_j. For each Q_i, iterate over K_j, V_j blocks: compute S = Q_i @ K_j^T, update running max m and sum ell, rescale running output O, accumulate. Store only logsumexp L = m + log(ell) for the backward pass.
- **Backward pass (Algorithm 2):** Recompute S and P from Q, K, V in SRAM (no N-by-N matrix stored). Use L to reconstruct softmax probabilities. Five matmuls per block (vs two in forward). Parallelize over column blocks; use atomic adds for dQ updates.
- **Causal masking:** Skip entirely blocks where all column indices exceed row indices (~half the blocks), giving 1.7-1.8x speedup over non-causal attention.
- **Results — Attention benchmark (Figs. 4-6, A100 80GB):** FlashAttention-2 reaches 230 TFLOPs/s forward (73% of theoretical max), 1.7-3.0x faster than FlashAttention, 3-10x faster than PyTorch standard attention. On H100: up to 335 TFLOPs/s.
- **Results — End-to-end training (Table 1, 8xA100):** GPT3-2.7B at 8k context: 225 TFLOPs/s (72% MFU) with FlashAttention-2 vs 175 TFLOPs/s with FlashAttention vs 80 TFLOPs/s without FlashAttention.

### Simplifications in This Notebook

- No custom CUDA kernel: uses PyTorch's built-in `scaled_dot_product_attention` (which dispatches to FlashAttention-2 on GPU) instead of the hand-written CUDA kernel in the paper
- Benchmark only forward pass timing and peak memory; no backward pass or end-to-end training
- Sequence lengths 512-8192 (paper benchmarks up to 16k)
- Head dimension 64 only (paper tests 64 and 128)
- Also implements a pure-Python tiled attention (online softmax) to illustrate the tiling concept, not as a performance baseline

### What the Notebook Result Shows

On a Kaggle T4 GPU (15.6 GB), across sequence lengths 512 to 8192 (batch size scaled to keep total tokens at 16k, head dim 64, 16 heads), PyTorch SDPA (FlashAttention-2 backend) is 1.7-1.9x faster than naive attention in the forward pass and uses up to 25.7x less peak memory. Naive attention memory grows as O(N²): 1291 MB at seq=512, 8523 MB at seq=4096, and **OOMs at seq=8192** — the 8 GB scores matrix exceeds the T4's capacity, directly demonstrating the quadratic memory bottleneck FlashAttention-2 solves. SDPA memory stays flat at ~331 MB across all sequence lengths, confirming O(N) scaling.

The tiled Python implementation is slower than both (due to Python loop overhead) but achieves the same O(N) memory as SDPA, demonstrating that the memory saving comes from tiling, not from CUDA optimization. The correctness check confirms all three implementations produce identical results (max difference: 8.9e-7). The toy training comparison shows naive and SDPA produce identical training loss (difference: 0.000000) over 20 epochs.

## Influence

FlashAttention-2 (July 2023) was adopted almost universally: it is the default attention implementation in PyTorch (via `scaled_dot_product_attention`), Hugging Face Transformers, vLLM, and most major training frameworks. The FlashAttention library on GitHub has over 25,000 stars ([Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). The techniques (online softmax, tiling, warp-level work partitioning) became standard curriculum for GPU kernel optimization, inspiring the GPU Mode community talk series. Tri Dao subsequently extended the approach to FlashAttention-3 (2024) for H100 GPUs with FP8 support. The paper has about 143 citations on [OpenAlex](https://openalex.org/W4384648639) (checked October 2026) and about 1,700 on Google Scholar.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/023_flashattention-2-faster-attention/solution.ipynb) | Pure PyTorch benchmark of naive attention vs tiled online-softmax attention vs PyTorch SDPA (FlashAttention-2) across sequence lengths 512-8192, measuring speed and peak memory on GPU |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
