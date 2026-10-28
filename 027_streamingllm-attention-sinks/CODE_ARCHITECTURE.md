---
title: "StreamingLLM: Code architecture"
parent: "027 StreamingLLM"
nav_order: 1
---
# Code Architecture — StreamingLLM Notebook

## Overview

The notebook visualises attention sinks in GPT-2 small (124M), implements three KV-cache strategies (dense, window, StreamingLLM), and compares their perplexity on a synthetic 2000-token sequence. The result shows that StreamingLLM (4 sink tokens + 252 recent) maintains stable perplexity (~6,300–14,000) while plain window attention (0+256) collapses catastrophically (PPL rises to billions) once the cache fills and sink tokens are evicted. Dense attention diverges beyond the 1024-token training window (PPL ~37,000).

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copy the values from the code cell, never from memory):
- `CACHE_SIZE = 256` — total KV cache size (paper uses 1024–2048)
- `NUM_SINK_TOKENS = 4` — number of initial tokens retained as attention sinks (paper default)
- `MAX_SEQ_LEN = 2000` — length of the synthetic test sequence
- `DEVICE = cuda` (on Kaggle) / `cpu` (local fallback)

### 2. Model & Tokenizer

**Class: `GPT2LMHeadModel` (from HuggingFace transformers)** — GPT-2 small, 124M parameters, 12 layers, 12 heads, hidden dim 768, vocab 50257. Loaded from `gpt2` checkpoint and frozen (eval mode, no gradient). Uses learned positional embeddings (not RoPE — GPT-2 predates RoPE, but the attention sink phenomenon is position-encoding-agnostic as the paper shows across both RoPE and ALiBi models).

### 3. Attention Sink Visualisation

**Function: `visualize_attention_sinks(model, tokenizer, text)`** — runs a forward pass with `output_attentions=True`, extracts attention weights from all 12 layers × 12 heads, and plots the average attention logit map for selected layers (0, 1, 5, 11). Shows the characteristic pattern: lower layers attend locally, deeper layers concentrate on the first token.

**Data flow:**
```
Input text: "The future of AI is..."
  -> tokenizer -> input_ids (1, seq_len)
  -> GPT-2 forward(output_attentions=True)
  -> attentions: tuple of 12 tensors, each (1, 12, seq, seq)
  -> average over heads -> (12, seq, seq)
  -> select layers -> matplotlib heatmap
```

### 4. KV Cache Implementations

Three cache strategies, all operating on a sliding window of KV pairs:

**Class: `DenseKVCache`** — keeps all KV pairs. Memory grows linearly with sequence length. Baseline that fails when seq_len > training window (1024 for GPT-2).

**Class: `WindowKVCache`** — keeps only the most recent `CACHE_SIZE` KV pairs. Evicts the oldest token when full, including the first tokens. This is the strategy that collapses.

**Class: `StreamingKVCache`** — keeps `NUM_SINK_TOKENS` initial KV pairs + the most recent `(CACHE_SIZE - NUM_SINK_TOKENS)` KV pairs. When the cache is full, evicts the oldest *non-sink* token. Position IDs are assigned contiguously (0, 1, ..., CACHE_SIZE-1) within the cache.

**Data flow (StreamingKVCache):**
```
As tokens arrive: [t0, t1, t2, t3, t4, t5, ...]
Cache fills: [t0, t1, t2, t3, t4, ..., t255]  (size=256)
Next token t256: evict t4 (oldest non-sink)
Cache becomes: [t0, t1, t2, t3, t5, ..., t256]
Position IDs:   [0,  1,  2,  3,  4,  ..., 255]  (contiguous, not original positions)
```

### 5. Perplexity Evaluation

**Function: `compute_streaming_ppl(model, input_ids, cache_strategy, cache_size, num_sinks)`** — feeds tokens one at a time through GPT-2 with the specified cache strategy. At each step:
1. Get the current KV cache contents (sink KVs + recent KVs)
2. Run a forward pass with `past_key_values` set to the cache contents and position_ids assigned contiguously
3. Compute cross-entropy loss on the next-token prediction
4. Update the cache with the new token's KV

Returns per-token perplexity (exp(loss)) for the entire sequence.

### 6. Results and Plots

Three outputs:
1. **Attention sink heatmap** (Figure 2 analogue): shows GPT-2 layer 0 (local), layer 1 (local), layer 5 (first-token dominant), layer 11 (first-token dominant) — matching the paper's Figure 2 pattern.
2. **Perplexity vs position curve** (Figure 3 analogue): dense attention diverges beyond 1024 tokens; window attention collapses at the cache boundary (256); StreamingLLM stays stable.
3. **Bar chart of average PPL** in three regions: before cache fills, after cache fills, and beyond training window — showing StreamingLLM matches recomputation while window attention is an order of magnitude worse.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `visualize_attention_sinks` | Extract and plot attention weights from GPT-2 layers, showing the first-token sink pattern |
| `DenseKVCache` | Baseline cache that keeps all KV pairs (grows without bound) |
| `WindowKVCache` | Sliding window cache that evicts oldest tokens including sinks (collapses) |
| `StreamingKVCache` | Sink + rolling window cache; evicts oldest non-sink token, assigns contiguous positions |
| `compute_streaming_ppl` | Token-by-token perplexity evaluation with a given cache strategy |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | Llama-2-[7,13,70]B, MPT, Falcon, Pythia | GPT-2 small (124M) |
| Sequence length | Up to 4 million tokens | ~2000 tokens (synthetic) |
| Cache size | 1024–2048 | 256 |
| Dataset | Concatenated PG19 test set | Synthetic repeated text |
| Position encoding | RoPE / ALiBi | Learned positional embeddings (GPT-2) |
| Sink token pre-training | 160M models trained from scratch | Not included |
| Streaming QA evaluation | ARC-Easy/Challenge, Llama-2-Chat | Not included |
| Speedup measurement | 22.2x vs recomputation | Not included (perplexity only) |
