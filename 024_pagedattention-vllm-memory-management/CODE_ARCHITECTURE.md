---
title: "PagedAttention: Code architecture"
parent: "024 PagedAttention"
nav_order: 1
---
# Code Architecture — PagedAttention/vLLM Notebook

## Overview

This notebook simulates two KV cache memory allocation strategies — contiguous pre-allocation (mimicking existing systems like FasterTransformer and Orca) and block-based paged allocation (mimicking PagedAttention/vLLM) — on synthetic request traces. It measures effective memory utilization, waste breakdown, and the effect of block size. The paged allocator achieves 98.2% effective utilization vs 21.5% for contiguous on ShareGPT, and admits 20 requests vs 5 (4x more) into the same memory budget.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copied from the code cell):
- `BLOCK_SIZE = 16` — tokens per KV block (vLLM default, Section 7.2)
- `MAX_SEQ_LEN = 2048` — maximum possible sequence length for contiguous pre-allocation
- `NUM_REQUESTS = 200` — number of requests to simulate per trial
- `GPU_KV_BLOCKS = 600` — total physical KV blocks available on simulated GPU (9600 slots)
- `TRIALS = 5` — number of simulation trials for variance estimation

### 2. Data (Synthetic Traces)

**Function: `generate_traces(num_requests, workload, seed)`** — returns a list of (prompt_len, output_len) tuples.

Two workloads are generated:
- **ShareGPT-like:** log-normal distributions with mean ~5.1 for prompts (~190 tokens avg) and ~5.0 for outputs (~180 tokens avg), mimicking the longer ShareGPT conversations.
- **Alpaca-like:** log-normal with mean ~3.4 for prompts (~32 tokens avg) and ~3.3 for outputs (~31 tokens avg), mimicking shorter Alpaca instructions.

The paper reports ShareGPT has 8.4x longer prompts and 5.8x longer outputs than Alpaca (Section 6.1).

A histogram plot shows the length distributions for both workloads.

### 3. Memory Allocation Models

**Class: `ContiguousAllocator`** — simulates existing systems
- Pre-allocates `MAX_SEQ_LEN` (2048) slots per request as a contiguous chunk
- Tracks: actual tokens needed, slots allocated, slots wasted (internal fragmentation), peak memory
- Simulates gradual request admission and memory freeing as requests complete
- Returns: effective utilization, waste percentage, num admitted, peak memory, memory timeline

**Class: `PagedAllocator`** — simulates PagedAttention/vLLM
- Allocates fixed-size blocks of `BLOCK_SIZE` (16) tokens on demand
- `_blocks_needed(n)` = ceil(n / BLOCK_SIZE)
- No external fragmentation (all blocks same size)
- Internal fragmentation limited to at most one partially-filled block per request
- Returns same metrics as ContiguousAllocator plus peak blocks used

### 4. Memory Utilization Metric

The key metric is **effective utilization** = actual_tokens / total_allocated. The paper (Figure 2) reports 20.4%–38.2% for existing systems. This cell explains the three waste categories: reserved (future tokens), internal fragmentation (over-provisioning), external fragmentation (gaps).

### 5. Fragmentation Breakdown

**Function: `detailed_fragmentation(traces, ...)`** — breaks down contiguous allocation waste into:
- Internal fragmentation: allocated slots beyond actual sequence length
- Reserved slots: held for output not yet generated at start
- External fragmentation: leftover memory too small for new requests

Prints a detailed breakdown for both ShareGPT and Alpaca workloads.

### 6. Utilization Comparison Plots

- **Bar chart:** Effective utilization for contiguous vs paged, both workloads, with error bars from 5 trials. Annotated with percentage values. Includes a reference line at 38.2% (paper's best baseline).
- **Stacked horizontal bar:** Memory waste breakdown for contiguous allocation, showing actual tokens (green), internal frag (red), reserved (orange), external frag (gray) as fractions of total GPU KV memory.

### 7. Block Size Analysis

Sweeps block sizes [1, 2, 4, 8, 16, 32, 64, 128, 256] for the paged allocator on both workloads. Plots effective utilization vs block size on a log-scale x-axis with a reference line at block size 16 (vLLM default). Prints a table of utilization at each block size.

### 8. Memory Usage Over Time

**Function: `simulate_memory_over_time(traces, ...)`** — step-by-step simulation showing how memory fills up as requests are admitted under each strategy. Plots memory used vs request number for both contiguous and paged on the ShareGPT workload.

### 9. Final Results

Prints a comprehensive summary of all metrics: effective utilization, waste, waste reduction, requests admitted, throughput improvement ratio, and block-size sweep highlights.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `generate_traces` | Generate synthetic (prompt_len, output_len) request traces for ShareGPT or Alpaca workloads |
| `ContiguousAllocator` | Simulate contiguous KV cache pre-allocation (existing systems baseline) |
| `PagedAllocator` | Simulate block-based paged KV cache allocation (PagedAttention/vLLM) |
| `run_simulation` | Run both allocators across multiple trials and aggregate results |
| `detailed_fragmentation` | Break down contiguous allocation waste into internal, reserved, external categories |
| `simulate_memory_over_time` | Step-by-step memory fill simulation for time-series visualization |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| System | Full vLLM serving engine with CUDA kernels, distributed execution | Pure Python memory allocation simulation |
| Workload | Real ShareGPT and Alpaca datasets with Poisson arrival times | Synthetic log-normal traces, no arrival times |
| Model | OPT 13B/66B/175B and LLaMA-13B on A100 GPUs | No model, no GPU, no attention computation |
| Memory sharing | Copy-on-write for parallel sampling, beam search, shared prefix | Not simulated |
| Scheduling | FCFS with preemption, swapping, recomputation | Simple sequential admission, no preemption |
| Throughput | Measured end-to-end (requests/sec at target latency) | Estimated from requests admitted into fixed memory |
| Block size study | Figure 18b with real end-to-end latency | Utilization only (no latency measurement) |
