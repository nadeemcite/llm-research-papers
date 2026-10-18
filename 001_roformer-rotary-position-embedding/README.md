---
title: "001 RoPE"
nav_order: 2
permalink: "/001_roformer-rotary-position-embedding/"
---
# RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/roformer-rotary-position-embedding)

- **Paper:** [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- **Authors:** Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu
- **Lab, year:** Zhuiyi Technology Co., Ltd., Shenzhen, 2021 (arXiv preprint; later published in Neurocomputing, 2024)

## Summary

Transformers process all tokens in parallel, which makes them fast but blind to word order — the model needs a separate "position encoding" to know which token came first. Earlier methods either added a learned position vector to each token (absolute position embedding, as in BERT and GPT) or added relative-position bias terms inside the attention computation (as in T5 and NEZHA). RoPE (Rotary Position Embedding) takes a different route: it *rotates* the query and key vectors by an angle that depends on the token's absolute position, so that when the query and key are dotted to produce an attention score, the result depends only on their *relative* distance.

The rotation matrix is block-diagonal, with each 2-D sub-space rotated by a different frequency θᵢ = 10000⁻²ⁱ/ᵈ.

This gives RoPE three properties the paper highlights: it handles variable sequence lengths naturally, the attention score decays as tokens get farther apart (long-term decay), and it works with linear attention (a faster attention variant). On the CAIL2019-SCM Chinese legal text matching benchmark, RoFormer (the paper's RoPE-equipped model) at 1024-token length outperforms WoBERT by 1.5% absolute accuracy (69.79% vs 68.10%), while at the standard 512 length it matches WoBERT (68.29% vs 68.10%).

This repo's notebook implements RoPE in a small character-level Transformer and shows its length-extrapolation advantage: a model trained on length-64 sequences maintains about 98.7% next-token accuracy at 128 tokens (2× the training length), while the same model with learned absolute positions collapses to about 59%.

## What Problem Does It Solve

Imagine reading a sentence where someone shuffled all the words: "the bit dog the man" instead of "the dog bit the man." The words are the same, but the meaning is gone — order matters. A Transformer reads all words at once (in parallel), so without help it can't tell which word came first.

Earlier solutions gave each position a numbered tag (position 1, position 2, ...) and added that tag to the word — like writing page numbers on a stack of papers. This works, but the model has to *learn* what each number means from scratch, and if you give it a page number it's never seen before (say, page 128 when it only trained up to page 64), it has no idea what to do.

RoPE's insight: instead of adding a position tag, *rotate* each word's vector by an angle proportional to its position. When two vectors are compared (dotted), only the *difference* in their rotation angles matters — so the model naturally sees relative position, and a rotation pattern learned on short sequences extends to longer ones.

## Core Idea

- **Rotation instead of addition:** Position information is injected by multiplying (rotating) the query and key vectors with a rotation matrix, not by adding a position vector to the input embedding.
- **Relative position from absolute rotation:** Because qₘᵀkₙ = (RₘW_qxₘ)ᵀ(RₙW_kxₙ) = xₘᵀW_q Rₙ₋ₘ W_kxₙ, the attention score depends only on the relative distance (n − m), even though each vector is rotated by its *absolute* position.
- **Multi-frequency rotation:** The d-dimensional head is split into d/2 pairs, each rotated by a different frequency θᵢ = 10000⁻²ⁱ/ᵈ. Low-frequency pairs capture long-range dependencies; high-frequency pairs capture short-range ones.
- **Long-term decay:** With this frequency choice, the inner product between query and key decays as relative distance grows — matching the intuition that distant tokens should attend less to each other.

## Key Method Details

- **Rotation matrix (Eq. 15):** Rᵈ_Θ,m is a block-diagonal matrix of 2×2 rotation blocks, where each block i rotates by angle mθᵢ with θᵢ = 10000⁻²⁽ⁱ⁻¹⁾/ᵈ.
- **2D case (Eq. 12–13):** For d=2, f_q(xₘ, m) = (W_qxₘ)e^{imθ}, i.e. the query is rotated by mθ on the 2D plane. The general d-dimensional case divides the space into d/2 independent 2D sub-spaces.
- **Efficient implementation:** Instead of the sparse d×d matrix multiplication, RoPE is computed by splitting each head dimension into even/odd pairs and applying (x₁cos(mθ) − x₂sin(mθ), x₁sin(mθ) + x₂cos(mθ)) — an O(d) elementwise operation.
- **Results (Table 4, v1):** On CAIL2019-SCM, RoFormer-512: 68.29% test accuracy; RoFormer-1024: 69.79%; WoBERT-512: 68.10%; BERT-512: 67.77%. At 1024 length, RoFormer outperforms WoBERT by 1.5% absolute.
- **Pretraining (Table 2, v1):** Pre-training accuracy increases with sequence length (65.0% at 512 → 67.4% at 1536), demonstrating RoPE's ability to handle long texts.

### Simplifications in This Notebook

- Tiny model: 4 Transformer layers, 4 heads, 128-dim (812,800 params for LearnedPE, 796,416 for RoPE) instead of a 12-layer BERT-scale model
- Synthetic modular-counting sequences (token at position t = (a·t + b) mod 20) instead of Chinese long-text classification (CAIL2019-SCM)
- 100 epochs with AdamW (lr 3e-4), batch size 256, on 4096 sequences — instead of multi-stage pretraining on 34 GB of data
- Next-token prediction (causal language modeling) instead of masked language modeling + semantic text matching

### What the Notebook Result Shows

On 512 unseen test sequences, both models reach about 98.5% next-token accuracy at the training length (64): LearnedPE 98.48%, RoPE 98.51%. At the extrapolation length (128, 2× training), RoPE maintains 98.68% accuracy (loss 0.038) while LearnedPE collapses to 59.15% (loss 2.355) — a 39.9% relative accuracy degradation for LearnedPE versus −0.2% for RoPE. The long-term decay plot shows the RoPE inner product decreasing in magnitude with relative distance. This toy task is not comparable to the paper's CAIL2019-SCM result; it demonstrates the *mechanism* (relative position → better extrapolation) on a controlled setup.

## Influence

RoPE became the default position encoding for most open-source large language models. LLaMA (Meta, 2023), Mistral 7B (2023), Qwen, DeepSeek, and many others adopted it in place of learned absolute positions. The paper was integrated into Hugging Face Transformers ([RoFormer model page](https://huggingface.co/docs/transformers/model_doc/roformer)) and the official code is on [GitHub (ZhuiyiTechnology/roformer)](https://github.com/ZhuiyiTechnology/roformer). It has about 1,900 citations ([OpenAlex](https://openalex.org/works/W4388979610), checked October 2026). The long-term decay property and the multi-frequency design influenced later context-extension methods like Position Interpolation, NTK-aware scaling, and YaRN.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/001_roformer-rotary-position-embedding/solution.ipynb) | From-scratch PyTorch nanoGPT with RoPE vs learned position embedding, trained on synthetic counting sequences to compare length extrapolation (RoPE 98.7% vs LearnedPE 59% at 2x training length) |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
