---
title: "PagedAttention: Talk & Q&A"
parent: "024 PagedAttention"
nav_order: 2
---
# TALK — Efficient Memory Management for Large Language Model Serving with PagedAttention (PagedAttention/vLLM)

## Talks, Slides and Press Coverage

1. **SOSP 2023 — "Efficient Memory Management for Large Language Model Serving with PagedAttention" (Woosuk Kwon, 23 Oct 2023):** The paper was presented at the ACM SIGOPS 29th Symposium on Operating Systems Principles in Koblenz, Germany, where it received the Best Paper Award. The talk covered the PagedAttention algorithm, vLLM system architecture, and the OS-inspired memory management approach. ([conference page](https://sigops.org/s/conferences/sosp/2023/toc.html))

2. **vLLM project — GitHub repository:** The open-source vLLM project, built on PagedAttention, is available at [vllm-project/vllm](https://github.com/vllm-project/vllm) with over 30,000 stars, making it one of the most widely adopted LLM serving systems.

3. **UC Berkeley Sky Computing blog — "vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention" (Woosuk Kwon et al., 20 Jun 2023):** The announcement post introducing vLLM to the community, explaining the PagedAttention concept and showing throughput benchmarks. ([blog post](https://vllm.ai/blog/2023-06-20-vllm))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2309.06180).

**Q1: Why is the KV cache such a critical bottleneck in LLM serving?**
A1 (from the paper, Section 3): For the 13B parameter OPT model, "the KV cache of a single token demands 800 KB of space, calculated as 2 (key and value vectors) × 5120 (hidden state size) × 40 (number of layers) × 2 (bytes per FP16)." Since OPT can generate up to 2,048 tokens, "the memory required to store the KV cache of one request can be as much as 1.6 GB."

**Q2: How much KV cache memory is wasted in existing systems?**
A2 (from the paper, Figure 2): "only 20.4% - 38.2% of the KV cache memory is used to store the actual token states in the existing systems." The rest is wasted to internal fragmentation, external fragmentation, and reserved slots.

**Q3: What inspired PagedAttention?**
A3 (from the paper, Section 4.1): PagedAttention is "an attention algorithm inspired by the classic idea of paging in operating systems." The authors draw a direct analogy: "one can think of blocks as pages, tokens as bytes, and requests as processes."

**Q4: Does PagedAttention's block indirection slow down the attention kernel?**
A4 (from the paper, Section 7.1): Yes. "This leads to 20–26% higher attention kernel latency, compared to the highly-optimized FasterTransformer implementation." However, "the overhead is small as it only affects the attention operator but not the other operators in the model" and "PagedAttention makes vLLM significantly outperform FasterTransformer in end-to-end performance."

**Q5: What is the default block size and why?**
A5 (from the paper, Section 7.2): "In practice, we find that the block size 16 is large enough to efficiently utilize the GPU and small enough to avoid significant internal fragmentation in most workloads. Accordingly, vLLM sets its default block size as 16."

**Q6: How does vLLM handle running out of GPU memory during generation?**
A6 (from the paper, Section 4.5): vLLM uses "an all-or-nothing eviction policy" — either evicting all or none of a sequence's blocks. It recovers evicted blocks via two methods: "Swapping" (copying blocks to CPU memory) and "Recomputation" (re-running the prompt phase to regenerate KV cache). "Recomputation is more efficient when the block size is small, while swapping is more efficient when the block size is large."

**Q7: Can PagedAttention be applied to other GPU workloads?**
A7 (from the paper, Section 8): Not universally. "In DNN training, the tensor shapes are typically static, and thus memory allocation can be optimized ahead of time. For serving DNNs that are not LLMs, an increase in memory efficiency may not result in any performance improvement since the performance is primarily compute-bound."

## Common Misconceptions

1. **"PagedAttention approximates attention to save memory."** — No. PagedAttention computes exact attention; it only changes how the KV cache is stored in memory (non-contiguous blocks instead of one contiguous chunk). The paper states: "without affecting the model accuracy at all" (Section 1).

2. **"vLLM's speedup comes from faster attention kernels."** — The attention kernel is actually 20–26% slower (Section 7.1). The throughput improvement comes from better memory utilization allowing larger batch sizes — more requests fit into the same GPU memory.

3. **"PagedAttention is just about memory, not throughput."** — Memory and throughput are directly linked in LLM serving. The paper explains: "the number of requests that can be batched together is still constrained by GPU memory capacity" (Section 3). By reducing waste, vLLM batches more requests, increasing throughput 2–4x.

4. **"Block size should be as small as possible to minimize fragmentation."** — Not always. The paper finds that "if the block size is too small, vLLM may not fully utilize the GPU's parallelism for reading and processing KV cache" (Section 7.2). Block size 16 balances GPU utilization and low fragmentation.

## Real Citations

1. Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." *SOSP 2023 (ACM SIGOPS 29th Symposium on Operating Systems Principles).* [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) — the original paper.

2. Woosuk Kwon et al. (2023). "vLLM: Easy, Fast, and Cheap LLM Serving." [blog.vllm.ai](https://vllm.ai/blog/2023-06-20-vllm) — the project announcement blog post accompanying the paper.

**Citation count:** about 1,785 citations on [OpenAlex](https://openalex.org/W4387321091) (checked October 2026).
