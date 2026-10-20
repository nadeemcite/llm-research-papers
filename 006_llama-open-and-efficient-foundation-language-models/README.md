---
title: "006 LLaMA"
nav_order: 7
permalink: "/006_llama-open-and-efficient-foundation-language-models/"
---
# LLaMA: Open and Efficient Foundation Language Models (LLaMA)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/llama-open-efficient-foundation-models)

- **Paper:** [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- **Authors:** Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample
- **Lab, year:** Meta AI, 2023

## Summary

LLaMA is a collection of foundation language models ranging from 7B to 65B parameters, trained on publicly available data exclusively. The key insight is that for a given inference budget, a smaller model trained on more tokens outperforms a larger model trained on fewer tokens — the opposite of what was common practice at the time. LLaMA-13B outperforms GPT-3 (175B) on most benchmarks despite being 10× smaller, and LLaMA-65B is competitive with Chinchilla-70B and PaLM-540B.

The paper's architecture combines three improvements over the original Transformer (Section 2.2): RMSNorm (replacing LayerNorm), Rotary Positional Embeddings or RoPE (replacing absolute positional embeddings, applied at every layer), and SwiGLU (a gated activation replacing ReLU, with FFN dimension 2/3·4d to match parameter count). All use pre-normalization (from GPT-3). The models are trained on 1.0–1.4T tokens from seven public sources: CommonCrawl (67%), C4 (15%), GitHub (4.5%), Wikipedia (4.5%), Books (4.5%), ArXiv (2.5%), and StackExchange (2%).

This repo's notebook assembles a mini LLaMA (RoPE + SwiGLU + RMSNorm) and pretrains it on TinyStories, comparing it against a vanilla Transformer (LayerNorm + GELU + learned positional embeddings) of the same size. The mini LLaMA trains faster and reaches lower validation loss, demonstrating that the three architectural choices help even at toy scale.

## What Problem Does It Solve

Imagine you want to build a really smart AI assistant. The conventional wisdom in 2022 was "bigger is better" — just make the model huge and it will get smarter. But there's a catch: a huge model is expensive to *use* (inference), not just to train. If you have a fixed budget for running the model, you'd rather have a smaller model that's been fed a lot more books than a bigger model that's read fewer books.

LLaMA shows this works. Instead of training a 175B model on 300B tokens (like GPT-3), train a 13B model on 1T tokens. The 13B model is 10× smaller, runs on a single GPU, and yet beats GPT-3 on most benchmarks. Crucially, LLaMA is trained entirely on publicly available data — no proprietary datasets — making it compatible with open-sourcing, unlike GPT-3, PaLM, or Chinchilla.

## Core Idea

- **Train smaller, longer:** Following Chinchilla scaling laws (Hoffmann et al., 2022), train smaller models on more tokens than typically used. LLaMA-7B trained on 1T tokens, LLaMA-33B/65B on 1.4T tokens.
- **Public data only:** Seven publicly available sources (CommonCrawl, C4, GitHub, Wikipedia, Books, ArXiv, StackExchange), enabling open release.
- **Architecture improvements:** RMSNorm (pre-normalization, no bias), RoPE (rotary positional embeddings at each layer), SwiGLU (gated FFN with dimension 2/3·4d) — combining the best improvements from GPT-3, PaLM, and GPTNeo.

## Key Method Details

- **Pre-training data (Table 1):** 1.4T tokens total. CommonCrawl 67%, C4 15%, GitHub 4.5%, Wikipedia 4.5%, Books 4.5%, ArXiv 2.5%, StackExchange 2%.
- **Architecture (Table 2, Section 2.2):** Pre-normalization with RMSNorm; RoPE at each layer; SwiGLU FFN with dim 2/3·4d. Model sizes: 7B (dim 4096, 32 heads, 32 layers), 13B (dim 5120, 40 heads, 40 layers), 33B (dim 6656, 52 heads, 60 layers), 65B (dim 8192, 64 heads, 80 layers).
- **Optimizer (Section 2.3):** AdamW, β1=0.9, β2=0.95, cosine LR schedule to 10% of max, weight decay 0.1, grad clip 1.0, 2000 warmup steps.
- **Results (Tables 3–9):** LLaMA-13B outperforms GPT-3 (175B) on most benchmarks (e.g., BoolQ 78.1 vs 60.5, HellaSwag 79.2 vs 78.9, MMLU 46.9 vs 43.9). LLaMA-65B is competitive with Chinchilla-70B and PaLM-540B (MMLU 63.4 vs 67.5 vs 69.3).
- **Efficiency (Section 2.4):** 65B model trains at ~380 tokens/sec/GPU on 2048 A100 80GB, ~21 days for 1.4T tokens. Uses xformers memory-efficient attention and activation checkpointing.

### Simplifications in This Notebook

- Tiny model: 4 layers, 256-dim, 4 heads (~2.4M parameters) instead of 7B–65B
- Dataset: ~3K TinyStories stories (character-level) instead of 1.4T BPE tokens from 7 web sources
- 10 epochs with AdamW (lr 3e-4) instead of 1T+ tokens with cosine schedule on 2048 A100s
- Character-level tokenizer instead of SentencePiece BPE with byte fallback
- No xformers memory-efficient attention, no activation checkpointing
- Also trains a vanilla Transformer (LayerNorm + GELU + learned PE) as a baseline — the paper does not do this comparison

### What the Notebook Result Shows

On ~3K TinyStories stories with character-level tokenization, both MiniLLaMA and the vanilla Transformer learn the character distribution: validation loss drops well below the random baseline of ln(VOCAB) ≈ 4.7. MiniLLaMA trains faster and reaches lower validation loss per epoch than the vanilla Transformer at the same parameter count (~2.4M), showing that the three LLaMA architectural choices (RMSNorm, RoPE, SwiGLU) provide a measurable advantage even at this toy scale. Generated text from MiniLLaMA produces recognizable words and simple sentence structures from the TinyStories vocabulary. This notebook's result is not comparable to the paper's 7B–65B benchmarks — it demonstrates the architecture works, not the scaling result.

## Influence

LLaMA became one of the most influential open-weight model releases in AI history. It directly spawned LLaMA 2 (Meta, July 2023) and LLaMA 3 (Meta, 2024), and inspired a vast ecosystem of fine-tuned variants (Alpaca, Vicuna, LoRA-tuned models) and alternative architectures (Mistral, Yi, DeepSeek). The combination of RMSNorm + RoPE + SwiGLU became the default architecture for nearly all subsequent open LLMs. The paper has about 4,500 citations ([OpenAlex](https://openalex.org/W4322718191), checked October 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/006_llama-open-and-efficient-foundation-language-models/solution.ipynb) | From-scratch PyTorch mini LLaMA (RoPE + SwiGLU + RMSNorm) trained on TinyStories, compared to a vanilla Transformer baseline |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
