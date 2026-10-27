---
title: "024 PagedAttention"
nav_order: 25
permalink: "/024_pagedattention-vllm-memory-management/"
---
# Efficient Memory Management for Large Language Model Serving with PagedAttention (PagedAttention/vLLM)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/pagedattention-vllm-memory-management)

- **Paper:** [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- **Authors:** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica
- **Lab, year:** UC Berkeley (Sheng: Stanford University; Yu: Independent Researcher; Zhang: UC San Diego), 2023 (SOSP 2023)

## Summary

When an LLM generates text, it stores the key and value vectors (the KV cache) for every token it has seen so far. This cache is large — for a 13B parameter OPT model, a single token's KV cache takes 800 KB, and one request can need up to 1.6 GB (Section 3). When serving many requests at once (batching), the KV cache consumes 20–30% of GPU memory.

Existing systems store each request's KV cache as one contiguous block of memory, pre-allocated to the maximum possible sequence length (e.g. 2048 tokens). Since actual outputs are usually much shorter, most of this reserved space is wasted: the paper finds that only 20.4%–38.2% of KV cache memory stores actual token states (Figure 2).

PagedAttention borrows the operating system's virtual memory and paging idea. Instead of one contiguous chunk, the KV cache is split into fixed-size blocks (like OS pages), each holding 16 tokens. Blocks are allocated on demand as new tokens are generated, and they need not be contiguous in physical memory. This eliminates external fragmentation (all blocks are the same size), limits internal fragmentation to at most one partially-filled block per request, and enables memory sharing across sequences (e.g. parallel sampling, beam search) via copy-on-write — exactly like forking a process in an OS. On top of PagedAttention, the authors build vLLM, a serving system that achieves 2–4x higher throughput than FasterTransformer and Orca at the same latency.

This repo's notebook simulates both contiguous and paged KV cache allocation on synthetic workloads, measuring memory utilization, fragmentation breakdown, and the effect of block size. The paged allocator achieves much higher effective utilization and admits more concurrent requests into the same memory budget.

## What Problem Does It Solve

Imagine a library where each reader must reserve an entire shelf for their books, sized for the most they could possibly borrow — even if they end up taking just two. When twenty readers show up, most shelves sit half-empty but can't be shared, and the library runs out of shelf space long before it runs out of books to store. That's the KV cache problem in LLM serving: each request grabs a contiguous chunk sized for its maximum possible output (say 2048 tokens), but the actual output might be 50 tokens. The reserved-but-unused space can't be borrowed by other requests, and gaps between differently-sized chunks (external fragmentation) waste more space on top.

PagedAttention solves this the way an operating system manages memory: instead of one big contiguous shelf, the library gives each reader small, identical boxes (blocks) as they need them. Boxes are stored wherever there's room — they don't need to be next to each other. When a reader finishes, their boxes go back to the pool immediately, not a giant shelf held until the end.

## Core Idea

- **Paged KV cache (Section 4.1):** The KV cache for each sequence is divided into fixed-size blocks, each holding B tokens (B = 16 by default). Attention is computed block-by-block: the query is matched against each key block, then the attention scores are applied to the corresponding value block. Blocks can be stored in non-contiguous physical memory.
- **Virtual memory mapping (Section 4.2):** Each request has a block table mapping its logical blocks to physical blocks — exactly like an OS page table. Logical blocks are contiguous; physical blocks are scattered. This eliminates external fragmentation and allows on-demand allocation.
- **Copy-on-write sharing (Section 4.4):** When multiple sequences share a prefix (e.g. parallel sampling, beam search), their logical blocks point to the same physical blocks. When one sequence needs to modify a shared block, vLLM copies only that block (copy-on-write), saving up to 55% memory on beam search.

## Key Method Details

- **KV block size:** B = 16 tokens (default). The paper studies block sizes 1–256 and finds 16–128 optimal for ShareGPT; larger blocks degrade on short sequences (Section 7.2, Figure 18b).
- **Memory waste in existing systems (Figure 2):** Only 20.4%–38.2% of KV cache memory stores actual token states. The rest is split among reserved slots (held for future tokens), internal fragmentation (over-provisioning), and external fragmentation (gaps between chunks).
- **Throughput improvement (Section 6.2):** vLLM sustains 1.7x–2.7x higher request rates than Orca (Oracle) and 2.7x–8x higher than Orca (Max) on ShareGPT, at the same latency. Up to 22x over FasterTransformer.
- **Parallel sampling and beam search (Section 6.3):** 6.1%–9.8% memory saving on parallel sampling, 37.6%–55.2% on beam search (Alpaca). Even higher on ShareGPT: 16.2%–30.5% (parallel) and 44.3%–66.3% (beam search).
- **Shared prefix (Section 6.4):** 1.67x throughput with one-shot prefix, 3.58x with few-shot prefix, over Orca (Oracle).
- **Kernel overhead (Section 7.1):** PagedAttention's attention kernel is 20–26% slower than FasterTransformer's due to block table indirection, but end-to-end vLLM still outperforms because memory savings allow larger batch sizes.
- **Scheduling and preemption (Section 4.5):** FCFS scheduling with all-or-nothing eviction. Two recovery methods: swapping (copy blocks to CPU) and recomputation (re-run prompt phase). Recomputation is better for small block sizes; swapping for large ones.

### Simplifications in This Notebook

- Pure memory allocation simulation — no actual LLM, attention computation, GPU, or CUDA kernels
- Synthetic request traces (log-normal distributions) instead of real ShareGPT/Alpaca datasets
- Block size 16 (vLLM default) with a sweep from 1 to 256
- No copy-on-write sharing, no beam search, no parallel sampling simulation
- No distributed execution, no preemption, no swapping/recomputation
- 200 requests per trial, 5 trials, 600 blocks (9600 slots) of simulated GPU memory

### What the Notebook Result Shows

On 200 synthetic ShareGPT-like requests with 600 blocks (9600 token slots) of simulated KV cache memory, the paged allocator achieves 98.2% effective utilization while the contiguous allocator (which pre-allocates 2048 slots per request, like Orca Max) achieves only 21.5% — consistent with the paper's finding of 20.4%–38.2% for existing systems. The paged allocator admits 20 concurrent requests vs 5 for contiguous, a 4x improvement matching the paper's 2–4x throughput gain.

On the shorter Alpaca-like workload, the gap is even wider: 89.1% vs 3.4% utilization, because contiguous allocation wastes nearly all memory pre-allocating 2048 slots for sequences that average 63 tokens. In practice, systems like Orca use smarter allocation (e.g. power-of-2 sizing) that narrows this gap; our simulation uses the worst-case max-length pre-allocation (Orca Max) to illustrate the upper bound of waste.

The block size sweep shows utilization dropping from 98.4% at block size 16 to 76.4% at block size 256 on ShareGPT, confirming the paper's observation that smaller blocks reduce fragmentation. On Alpaca, the drop is steeper (89.3% to 24.5%) because shorter sequences are more affected by block-size rounding, matching the paper's finding that large block sizes degrade performance on short-sequence workloads.

## Influence

vLLM became one of the most widely adopted LLM serving systems in the world. The vLLM project on GitHub has over 30,000 stars ([vllm-project/vllm](https://github.com/vllm-project/vllm)). PagedAttention's block-based memory management was adopted or emulated by TensorRT-LLM, SGLang, and other serving frameworks. The paper was presented at SOSP 2023 (ACM SIGOPS 29th Symposium on Operating Systems Principles) and received the Best Paper Award. It has about 1,785 citations on [OpenAlex](https://openalex.org/W4387321091) (checked October 2026). The operating-system analogy (pages, block tables, copy-on-write) made the technique intuitive and influenced how the ML systems community thinks about LLM serving memory management.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/024_pagedattention-vllm-memory-management/solution.ipynb) | Pure Python simulation of block-based (paged) vs contiguous KV cache allocation, measuring memory utilization, fragmentation breakdown, and block-size effects on synthetic ShareGPT and Alpaca workloads |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, key classes, and design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
