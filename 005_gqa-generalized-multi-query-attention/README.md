---
title: "005 GQA"
nav_order: 6
permalink: "/005_gqa-generalized-multi-query-attention/"
---
# GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (GQA)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/gqa-generalized-multi-query-attention)

- **Paper:** [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- **Authors:** Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai
- **Lab, year:** Google Research (Ainslie at University of Southern California, work done at Google Research), 2023 (EMNLP 2023)

## Summary

Autoregressive decoding in Transformers is bottlenecked not by computation but by memory bandwidth — loading all the cached keys and values (the KV cache) from memory at every generation step. Multi-query attention (MQA), proposed by Shazeer (2019), addresses this by using a single key-value head shared across all query heads, slashing the KV cache size by a factor of H (the number of heads). The problem: MQA can degrade quality and cause training instability, and training a separate fast model from scratch is expensive.

This paper makes two contributions. First, it shows that existing multi-head attention (MHA) checkpoints can be "uptrained" into MQA models using only about 5% of the original pre-training compute, by mean-pooling the key and value heads and then continuing pre-training. Second, it introduces grouped-query attention (GQA), which uses an intermediate number of key-value heads — more than one (MQA) but fewer than the number of query heads (MHA). GQA with 8 groups (GQA-8) on T5-XXL achieves quality close to MHA-XXL (average 45.5 vs 46.0 across seven benchmarks) while being nearly as fast as MQA (0.28s vs 0.24s per sample, compared to MHA's 1.51s).

This repo's notebook implements MHA, MQA, and GQA attention from scratch in three identical tiny Transformers trained on synthetic text. At this scale (~1.07M parameters, 30 epochs, CPU/Kaggle GPU), all three variants converge to similar validation loss (within 0.001), while GQA's KV cache is 4× smaller than MHA's — confirming the paper's core trade-off: GQA preserves quality while reducing inference memory.

## What Problem Does It Solve

Imagine a restaurant kitchen where eight chefs (query heads) each need ingredients from a pantry. In a standard kitchen (MHA), each chef has their own pantry shelf (key-value head) — lots of variety but you have to restock eight shelves every time you add a dish. In a one-shelf kitchen (MQA), all eight chefs share a single pantry shelf — restocking is fast, but the chefs lose access to their specialized ingredients, and the food quality drops. GQA is the middle ground: group the eight chefs into pairs, each pair sharing one shelf. You restock four shelves instead of eight (fast), but each group still has its own specialized ingredients (better quality).

In Transformer inference, each step generates one token and must load all previously computed keys and values from memory. With 32 heads and a long context, that KV cache is huge. MQA shrinks it 32× but hurts quality. The paper shows you don't have to choose: GQA with a moderate number of groups gives you most of the speedup with almost no quality loss, and you can get there from an existing model without retraining from scratch.

## Core Idea

- **Grouped-query attention (GQA):** Divide H query heads into G groups, each sharing a single key and value head. GQA-1 is MQA; GQA-H is MHA. An intermediate G gives a favorable trade-off between quality and inference speed.
- **Uptraining recipe:** Convert an MHA checkpoint to GQA by mean-pooling the key and value heads within each group, then continue pre-training for a small fraction (α = 0.05) of the original compute. Mean-pooling outperforms selecting a single head or random initialization.
- **Speed-quality trade-off:** GQA-8 on T5-XXL matches MHA-XXL quality (average 45.5 vs 46.0) while being 5.4× faster (0.28s vs 1.51s per sample), nearly matching MQA's 0.24s.

## Key Method Details

- **GQA structure (Section 2.2):** H query heads are divided into G groups. Group i contains query heads {i·H/G, ..., (i+1)·H/G − 1}, all sharing a single key head and value head. GQA-g refers to GQA with g groups.
- **Checkpoint conversion (Section 2.1, Figure 1):** Key and value projection matrices from all original heads within a group are mean-pooled into a single head. Mean-pooling beats selecting the first head or random initialization (Figure 4).
- **Uptraining (Section 2.1):** After conversion, the model is pre-trained for α proportion of original steps. α = 0.05 (5%) is sufficient; gains diminish beyond α = 0.10 (Figure 5).
- **Main results (Table 1, T5-XXL):** MHA-XXL: avg 46.0, 1.51s/sample. MQA-XXL: avg 45.3, 0.24s. GQA-8-XXL: avg 45.5, 0.28s. GQA-8 achieves 98.9% of MHA quality at 18.6% of the inference time.
- **KV cache reduction:** Going from MHA to MQA reduces the KV cache by a factor of H. GQA-8 with 32 heads reduces it by 4× (32/8), preserving the same proportional decrease as model size increases.
- **Training stability (Appendix A):** MQA trained from scratch suffers from loss spikes and divergence on long-input tasks. Uptrained GQA models are stable.

### Simplifications in This Notebook

- Tiny model: 4 Transformer layers, 128-dim embedding, 8 query heads (~1.07M parameters) instead of T5-XXL (11B parameters, 128 heads)
- Synthetic template-generated text (~8K sentences, 32-char vocab) instead of T5's C4 pre-training corpus (750GB)
- 30 epochs with AdamW (lr 3e-4, batch 128, seq 64) instead of 5% uptraining with Adafactor on TPUv3 (600 TPUv3 chip-days)
- Character-level next-token prediction instead of encoder-decoder summarization/translation with ROUGE/BLEU evaluation
- Three attention variants (MHA-8, MQA-1, GQA-2) trained from scratch instead of uptraining from a pre-trained checkpoint
- KV cache size measured in tensor elements at a fixed sequence length instead of wall-clock inference time on TPUv4

### What the Notebook Result Shows

On the synthetic text corpus, all three variants learn the character-level distribution: validation loss of about 0.060 for MHA, MQA, and GQA-2 (well below the random baseline of ln(32) ≈ 3.47; perplexity ~1.06 for all). The differences are within 0.001 nats, within run-to-run noise, confirming that at this toy scale the attention variant does not materially affect quality. 

The KV cache sizes scale as expected: MHA caches 16,384 elements (8 KV heads), MQA caches 2,048 (1 KV head, 8x reduction), and GQA-2 caches 4,096 (2 KV heads, 4x reduction vs MHA). This notebook's result is not comparable to the paper's T5-XXL benchmarks or TPUv4 timing, but it demonstrates the KV cache size trade-off that drives the paper's inference speed gains, and shows that GQA preserves quality at this scale.

## Influence

GQA was rapidly adopted as a standard architecture component in major open-weight LLMs. Llama 2 (Meta, 2023) uses GQA for its 34B and 70B models, citing the paper directly (Touvron et al., 2023, Table 1 and Section A.2.1). Mistral 7B (Mistral AI, 2023) also adopts GQA. Subsequent models including Llama 3, Qwen, DeepSeek-V2, and Gemma use GQA or its variants. The paper has about 440 citations ([OpenAlex](https://openalex.org/W4389518760), checked October 2026). The paper was accepted at EMNLP 2023.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/005_gqa-generalized-multi-query-attention/solution.ipynb) | From-scratch PyTorch MHA, MQA, and GQA attention implemented in three identical tiny Transformers, trained on synthetic text to compare quality and KV-cache size |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
