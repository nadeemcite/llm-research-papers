---
title: "027 StreamingLLM"
nav_order: 28
permalink: "/027_streamingllm-attention-sinks/"
---
# Efficient Streaming Language Models with Attention Sinks (StreamingLLM)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/streamingllm-attention-sinks)

- **Paper:** [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)
- **Authors:** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis
- **Lab, year:** Massachusetts Institute of Technology / Meta AI / Carnegie Mellon University, 2023 (ICLR 2024)

## Summary

Deploying large language models in streaming applications — multi-round dialogue, continuous document generation, always-on assistants — poses two challenges. First, the standard KV cache grows with every token generated, eventually exhausting GPU memory. Second, models trained with a finite attention window (e.g. 4K tokens for Llama-2) degrade sharply when the input exceeds that window. A natural fix is window attention: keep only the most recent K key-value pairs and discard the rest. But the paper shows this collapses — perplexity explodes the moment the cache evicts even the first token.

The authors discover why: autoregressive LLMs develop "attention sinks" — the first few tokens absorb a disproportionate share of attention score across all layers and heads, regardless of their semantic content. This happens because softmax forces attention weights to sum to one; when a query has no strong match among recent tokens, the model "dumps" the residual attention onto the globally visible initial tokens. Removing these tokens from the cache breaks the softmax denominator and destabilises the entire attention distribution.

StreamingLLM is the fix: keep the KV of just four initial tokens (the sinks) alongside a rolling window of recent tokens. This simple combination lets Llama-2, MPT, Falcon, and Pythia generate stable text over 4 million tokens — 1000x beyond their training window — with no fine-tuning. The authors also show that pre-training with a dedicated learnable sink token lets a model stream with just that one token, and that StreamingLLM achieves up to 22.2x speedup over the sliding-window-recomputation baseline.

This repo's notebook visualises attention sinks in GPT-2, implements a rolling KV cache with sink tokens, and shows that streaming perplexity stays stable beyond the training window while plain window attention collapses.

## What Problem Does It Solve

Imagine you are having a day-long chat with an AI assistant. Every message you send and every reply it generates gets stored in the assistant's "short-term memory" so it can reference earlier parts of the conversation. But by hour six, that memory is enormous — the assistant slows down and eventually crashes because it cannot hold everything. You could tell it to only remember the last few messages, but then it suddenly starts producing gibberish. Why?

The paper's answer: the very first thing the assistant saw — even something trivial like a newline — acts as an anchor. The model has learned to "park" excess attention there, like a person who always keeps one hand on a railing for balance. Remove the railing and they stumble. StreamingLLM simply keeps that anchor (the first few tokens) while everything else rolls forward, so the model stays balanced indefinitely.

## Core Idea

- **Attention sinks (Section 3.1):** The first few tokens of any sequence receive disproportionately high attention scores across all layers and heads, not because they are semantically important but because softmax requires attention to sum to one. Initial tokens are visible to all subsequent tokens in autoregressive decoding, making them the easiest "dumping ground" for residual attention.
- **Rolling KV cache with sinks (Section 3.2):** Retain the KV of four initial tokens (the sinks) plus a sliding window of the most recent N tokens. When the cache is full and a new token arrives, evict the oldest *non-sink* token. Position IDs are assigned contiguously within the cache (0, 1, 2, ..., N-1), not according to the original text positions.
- **Sink token pre-training (Section 3.3):** Adding a single learnable placeholder token at the start of every training sample gives the model a dedicated sink. A model trained this way streams stably with just that one sink token, whereas vanilla models need four initial tokens.

## Key Method Details

- **Cache structure:** x + y configuration — x sink tokens (default 4) + y recent tokens. For Llama-2, cache size = 2048; for Falcon/Pythia/MPT, cache size = 1024 (half the pre-training window for visualisation).
- **Position encoding in the cache:** Positions are assigned within the cache, not the original sequence. If the cache holds tokens [0, 1, 2, 3, 6, 7, 8] and is decoding token 9, positions are [0, 1, 2, 3, 4, 5, 6, 7] — contiguous. For RoPE, keys are cached pre-rotation and re-rotated at each step.
- **Key result — Table 1 (Llama-2-13B, PG19):** Window attention (0+1024) PPL = 5158.07; StreamingLLM (4+1020) PPL = 5.40; with linebreak substitutes (4"\n"+1020) PPL = 5.60 — confirming content does not matter.
- **Key result — Table 2 (Llama-2-7B):** 0+4096 PPL = 3359.95; 1+4095 = 11.88; 4+4092 = 9.59 — four sinks suffice, more give diminishing returns.
- **Long-range stability (Figure 5):** Stable perplexity over 4 million tokens across Llama-2-[7,13,70]B, Falcon-[7,40]B, Pythia-[2.8,6.9,12]B, and MPT-[7,30]B.
- **Efficiency (Section 4.5):** Up to 22.2x speedup vs sliding window recomputation (the only viable quality-matching baseline).
- **Sink token pre-training (Table 3, 160M models):** Vanilla needs 4+ tokens (PPL 18.05); Learnable Sink needs only 1 (PPL 18.01). Normal downstream benchmarks (Table 4) are unaffected.

### Simplifications in This Notebook

- GPT-2 small (124M, 12 layers) instead of Llama-2-[7,13,70]B, MPT, Falcon, or Pythia
- Synthetic long sequences (repeated text) up to ~2000 tokens instead of 4 million tokens on PG19
- Cache size of 256 tokens instead of 1024–2048
- 4 sink tokens (matching the paper's default); no sink-token pre-training experiment
- Perplexity comparison on a single long text rather than the concatenated PG19 test set
- No streaming question-answering evaluation; language modelling perplexity only

### What the Notebook Result Shows

On a synthetic 2000-token sequence with GPT-2 small (cache size 256): dense attention diverges (perplexity — a measure of how well the model predicts the next token, lower is better — jumps from ~1 to ~37,000 beyond the 1024-token training window). Plain window attention (0+256) collapses catastrophically (PPL rises to billions once the cache fills and sink tokens are evicted). StreamingLLM (4+252) maintains stable PPL in the thousands — the same order of magnitude as before the cache fills — throughout the entire 2000-token sequence.

The attention visualisation confirms that all 12 of GPT-2's layers allocate disproportionate attention to the first token (5x–200x more than to other tokens), matching the paper's observation in Llama-2-7B (Figure 2). StreamingLLM's PPL is higher than dense within the training window (6,304 vs 1.08) because it only retains 256 tokens of context — this matches the paper's comparison to the recomputation baseline, not to full-context dense attention.

## Influence

StreamingLLM was developed at MIT (Song Han's lab) in collaboration with Meta AI and CMU, published at ICLR 2024. The attention sink concept influenced subsequent work on efficient long-context inference, including H2O (Heavy-Hitter Oracle, Zhang et al., 2023) which builds on the observation that certain tokens dominate attention. The idea of a learnable sink token has been adopted in streaming serving frameworks. The official code at [github.com/mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm) has been widely integrated into LLM inference engines.

About 55 citations on [OpenAlex](https://openalex.org/works/W4387294617) (checked October 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/027_streamingllm-attention-sinks/solution.ipynb) | PyTorch notebook that visualises attention sinks in GPT-2, implements a rolling KV cache with sink tokens, and compares streaming perplexity vs window attention on a synthetic long sequence |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
