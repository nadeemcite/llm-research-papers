---
title: "FlashAttention-2: Talk & Q&A"
parent: "023 FlashAttention-2"
nav_order: 2
---
# TALK — FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (FlashAttention-2)

## Talks, Slides and Press Coverage

1. **GPU Mode — "FlashAttention" community talk series:** The GPU Mode community (founded by Tri Dao and Mark Saroufim) regularly hosts talks on FlashAttention and GPU kernel optimization. The YouTube channel features detailed technical discussions of FlashAttention-2's tiling and warp partitioning techniques. ([YouTube channel](https://www.youtube.com/@GPUMODE))

2. **PyTorch Documentation — `scaled_dot_product_attention`:** PyTorch's official API documentation for `torch.nn.functional.scaled_dot_product_attention` describes FlashAttention-2 as one of the backends dispatched automatically, making the algorithm accessible without any custom kernel code. ([PyTorch docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html))

3. **Hugging Face Documentation — Flash Attention:** Hugging Face's documentation describes FlashAttention-2 integration in Transformers, noting that it speeds up training and inference while reducing memory usage, with instructions for enabling it via `attn_implementation="flash_attention_2"`. ([Hugging Face docs](https://huggingface.co/docs/transformers/en/flash_attention))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2307.08691).

**Q1: Why does FlashAttention-2 reduce non-matmul FLOPs?**
A1 (from the paper, Section 3.1): "modern GPUs have specialized compute units (e.g., Tensor Cores on Nvidia GPUs) that makes matmul much faster. As an example, the A100 GPU has a max theoretical throughput of 312 TFLOPs/s of FP16/BF16 matmul, but only 19.5 TFLOPs/s of non-matmul FP32. Another way to think about this is that each non-matmul FLOP is 16x more expensive than a matmul FLOP."

**Q2: Why parallelize over the sequence length dimension?**
A2 (from the paper, Section 3.2): "The first version of FlashAttention parallelizes over batch size and number of heads... This scheduling is efficient when this number is large (say >= 80). In the case of long sequences (which usually means small batch sizes or small number of heads), to make better use of the multiprocessors on the GPU, we now additionally parallelize over the sequence length dimension."

**Q3: What was wrong with the original FlashAttention's warp partitioning?**
A3 (from the paper, Section 3.3): "FlashAttention splits K and V across 4 warps while keeping Q accessible by all warps... all warps need to write their intermediate results out to shared memory, synchronize, then add up the intermediate results. These shared memory reads/writes slow down the forward pass. In FlashAttention-2, we instead split Q across 4 warps while keeping K and V accessible by all warps... There is no need for communication between warps."

**Q4: How much faster is FlashAttention-2 compared to FlashAttention?**
A4 (from the paper, Section 4): "FlashAttention-2 is 1.7-3.0x faster than FlashAttention, 1.3-2.5x faster than FlashAttention in Triton, and 3-10x faster than a standard attention implementation. FlashAttention-2 reaches up to 230 TFLOPs/s, 73% of the theoretical maximum TFLOPs/s on A100 GPUs."

**Q5: What is the memory saving?**
A5 (from the paper, Section 2.3.2): "FlashAttention yields 10-20x memory saving depending on sequence length (memory required is linear in sequence length N instead of quadratic)."

## Common Misconceptions

1. **"FlashAttention approximates attention for speed."** — No. FlashAttention and FlashAttention-2 compute exact attention with no approximation. The speedup comes purely from reducing memory reads/writes (IO-awareness) and better GPU work partitioning. The output is mathematically identical to standard attention (as the notebook's correctness check confirms).

2. **"FlashAttention-2 is a completely new algorithm."** — FlashAttention-2 is an optimization of the original FlashAttention (Dao et al., 2022). The core algorithm (online softmax with tiling) is the same; the improvements are in work partitioning (how GPU threads are assigned), parallelism (adding sequence-length parallelism), and reducing non-matmul FLOPs. As the paper states in Section 3, these yield "2-3x speedup" over FlashAttention.

3. **"FlashAttention only helps with very long sequences."** — While the benefit grows with sequence length, FlashAttention-2 provides speedup even at moderate lengths (512-2k). The paper benchmarks from 512 tokens upward and shows consistent improvement. The memory saving is most dramatic at long sequences, but the speed benefit comes from reduced HBM access at all lengths.

## Real Citations

1. Dao, T. (2023). "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) — the original paper.
2. Dao, T., Fu, D.Y., Ermon, S., Rudra, A., and Re, C. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." In *NeurIPS 2022*. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) — the original FlashAttention that FlashAttention-2 builds upon.
3. Dao, T. (2024). "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision." [arXiv:2407.08608](https://arxiv.org/abs/2407.08608) — the successor optimized for H100 GPUs with FP8 support.

**Citation count:** about 143 citations on [OpenAlex](https://openalex.org/W4384648639) (checked October 2026).
