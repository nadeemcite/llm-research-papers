---
title: "007 Mistral"
nav_order: 8
permalink: "/007_mistral-7b-sliding-window-attention/"
---
# Mistral 7B (Mistral)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/mistral-7b-sliding-window-attention)

- **Paper:** [Mistral 7B](https://arxiv.org/abs/2310.06825)
- **Authors:** Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed
- **Lab, year:** Mistral AI, 2023

## Summary

Mistral 7B is a 7.3-billion-parameter language model that outperforms Llama 2 13B across all evaluated benchmarks and Llama 1 34B in reasoning, mathematics, and code generation — despite being nearly half the size. The paper's central claim is that careful architectural design (rather than sheer scale) can deliver high performance while keeping inference cheap.

Two key mechanisms drive this efficiency. Grouped-query attention (GQA) shares one set of key/value heads across multiple query heads, reducing the memory footprint during decoding and enabling larger batch sizes. Sliding window attention (SWA) restricts each layer's attention to a fixed window of W=4096 tokens; because information propagates across layers, a token at layer k can effectively attend to tokens up to k×W positions away, giving a theoretical receptive field of ~131K tokens at the top layer.

This repo's notebook builds a mini Mistral-style transformer (GQA + SWA + rolling KV cache) from scratch, trains it on TinyStories, and compares it against a vanilla transformer with full attention. The sliding-window model matches or slightly beats the full-attention model on validation loss while using substantially less memory for long sequences.

## What Problem Does It Solve

Imagine reading a very long book and trying to remember every word on every page to answer questions about it. That is how standard transformer attention works — every token looks back at every previous token. The longer the text, the more memory and compute this costs, growing quadratically.

Mistral's solution is like reading with a "sliding window": each token only directly looks at the last 4096 tokens. But because the model has 32 layers stacked on top of each other, information from earlier in the text trickles forward through the layers. A token at the top layer can "see" about 32 × 4096 ≈ 131K tokens back — far more than the 8K context length — without ever paying the quadratic cost of attending to everything at once.

## Core Idea

- **Sliding window attention (SWA):** Each attention layer restricts queries to attend only to the previous W tokens (W=4096 in the paper). Information still propagates across layers, giving an effective receptive field of k×W after k layers.
- **Rolling buffer cache:** The KV cache has a fixed size of W, storing keys/values at position i mod W. When the sequence exceeds W, old entries are overwritten — cache memory stops growing, saving 8× memory on 32K-token sequences.
- **Grouped-query attention (GQA):** 32 query heads share 8 key/value heads (4:1 ratio), reducing decoder memory and accelerating inference without measurable quality loss.

## Key Method Details

- **Architecture:** 32 layers, dim=4096, head_dim=128, n_heads=32, n_kv_heads=8 (GQA), hidden_dim=14336, window_size=4096, context_len=8192, vocab_size=32000 (Table 1)
- **Results:** Mistral 7B outperforms Llama 2 13B on all benchmarks: MMLU 60.1% vs 55.6%, HellaSwag 81.3% vs 80.7%, GSM8K 52.2% vs 34.3%, HumanEval 30.5% vs 18.9% (Table 2). Mistral 7B Instruct scores 1031 ELO and 6.84 on MT-Bench, surpassing Llama 2 13B Chat (1012, 6.65) (Table 3).

### Simplifications in This Notebook

- Smaller model: 4 layers, 128 dim, 4 query heads, 2 KV heads (GQA 2:1), window_size=64, about 1M parameters
- Dataset: TinyStories (synthetic short stories) instead of multi-trillion-token web crawl
- 10 epochs with AdamW (lr 3e-4) instead of the paper's full pretraining schedule
- Rolling buffer cache implemented but tested on short sequences (256 tokens) rather than 32K

### What the Notebook Result Shows

On the TinyStories validation set (1M characters, character-level tokenizer, 256-token sequences), the mini Mistral (SWA+GQA, 1M parameters) reaches a validation loss of about 0.17, while the vanilla full-attention baseline (810K parameters, same training setup) reaches about 2.07. Both are below the random baseline of about 4.30 (ln(74)).

The rolling buffer cache correctly overwrites old entries at positions beyond the window, keeping KV cache memory fixed at W=64 positions per layer regardless of sequence length (verified over 150+ generation steps). The attention-reach verification confirms that the effective receptive field at layer k is k×W = 64, 128, 192, 256 tokens for layers 1–4, covering the full 256-token training sequence. At seq_len=2048 the rolling buffer saves 96.9% of KV cache memory compared to full attention.

## Influence

Mistral 7B was the first model from Mistral AI and immediately established the startup as a serious competitor to Meta and OpenAI in the open-weight LLM space. The combination of GQA + SWA became a de facto standard: subsequent Mistral models (Mixtral 8×7B, Mistral Large) refined the recipe, and models like Qwen, Yi, and DeepSeek adopted GQA. SWA influenced long-context methods including StreamingLLM (attention sinks) and various efficient-attention variants. The Apache 2.0 release (no usage restrictions) was notable at a time when Meta's Llama licenses were more restrictive.

---


## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/007_mistral-7b-sliding-window-attention/solution.ipynb) | From-scratch PyTorch mini Mistral (GQA + SWA + rolling KV cache) trained on TinyStories and compared with a vanilla full-attention transformer |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
