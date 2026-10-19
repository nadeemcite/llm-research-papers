---
title: "003 SwiGLU"
nav_order: 4
permalink: "/003_swiglu-glu-variants-improve-transformer/"
---
# GLU Variants Improve Transformer (SwiGLU)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/glu-variants-improve-transformer)

- **Paper:** [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- **Authors:** Noam Shazeer
- **Lab, year:** Google, 2020 (arXiv preprint; no formal venue)

## Summary

The Transformer's feed-forward network (FFN) — the part that processes each position independently after attention — traditionally uses a single linear layer, an activation function (ReLU or GELU), then another linear layer. This paper asks a simple question: what if we replace that activation-and-linear path with a gated mechanism, where two parallel projections are multiplied element-wise?

Gated Linear Units (GLUs), introduced by Dauphin et al. (2016), compute the element-wise product of two linear projections, one of which passes through a sigmoid. Shazeer extends this idea: instead of sigmoid, try ReLU, GELU, or Swish (a smooth function defined as $x \cdot \sigma(x)$). The resulting "SwiGLU" variant — Swish-Gated Linear Unit — feeds one projection through Swish and multiplies it by a second raw projection, then applies the output linear layer.

The paper tests these variants in a T5-base model (12 encoder + 12 decoder layers, 220M parameters) on the C4 denoising task. SwiGLU achieves a heldout log-perplexity (a measure of how well the model predicts held-out text — lower is better) of 1.636 at 524,288 training steps, compared to 1.679 for the GELU baseline — a 0.043 improvement (Table 1). The GLU variants also improve average scores on the GLUE and SuperGLUE benchmarks after fine-tuning. To keep the parameter count matched (three matrices instead of two), the hidden dimension is reduced by a factor of 2/3.

This repo's notebook implements a tiny GPT with GELU-FFN and SwiGLU-FFN side by side on synthetic text. At this scale (~808K parameters, 30 epochs), the FFN variant has negligible impact: GELU achieves validation loss 1.6755 and SwiGLU 1.7051 — a tiny difference in the opposite direction from the paper. The paper's improvement appears at the 220M-parameter scale on a real text corpus, not at this toy scale.

## What Problem Does It Solve

Imagine a factory assembly line where each worker processes one item at a time. A standard worker receives the item, applies one transformation (say, sharpening), then passes it on. But what if the worker had *two hands* — one to decide "should I sharpen this?" and the other to hold the item steady — and combined both judgments? That is what gated units do: instead of a single transform-and-activate path, they split the signal into two paths, use one as a "gate" to control the other, then combine them.

The Transformer's FFN is that assembly-line worker. The original design (Vaswani et al., 2017) used a single path: linear → ReLU → linear. Later work swapped ReLU for GELU (a smoother variant). This paper says: add a second path and gate. The gate learns to selectively amplify or suppress features, giving the FFN more expressive power at the same parameter and compute cost.

## Core Idea

- **Gated FFN structure:** Instead of $\text{GELU}(xW_1)W_2$, compute $(\text{Swish}_1(xW) \otimes xV)W_2$ — three matrices instead of two, where $\otimes$ is element-wise product and $\text{Swish}_1(x) = x \cdot \sigma(x)$.
- **Parameter matching by 2/3 shrink:** The GLU variants have three weight matrices instead of two. To keep total parameters and FLOPs constant, the hidden dimension $d_{ff}$ is reduced by a factor of $\frac{2}{3}$ (e.g., from 3072 to 2048 in T5-base).
- **Swish as the gate activation:** Swish$_1$ (also called SiLU) is a smooth, non-monotonic function $x \cdot \sigma(x)$ that is negative for large negative inputs but smoothly approaches zero — unlike ReLU which is exactly zero for all negatives. The paper finds that Swish and GELU-based gates (SwiGLU and GEGLU) outperform ReLU, sigmoid, and bilinear gates.

## Key Method Details

- **Standard FFN (Section 1, Eq. 2):** $\text{FFN}_{\text{ReLU}}(x, W_1, W_2) = \max(xW_1, 0)W_2$ — no bias terms (following the T5 codebase).
- **SwiGLU FFN (Section 2, Eq. 6):** $\text{FFN}_{\text{SwiGLU}}(x, W, V, W_2) = (\text{Swish}_1(xW) \otimes xV)W_2$ — three matrices, no bias.
- **Hidden size reduction (Section 2):** "To keep the number of parameters and the amount of computation constant, we reduce the number of hidden units $d_{ff}$ by a factor of $\frac{2}{3}$."
- **Results (Section 3.2, Table 1):** At 524,288 training steps, SwiGLU achieves 1.636 log-perplexity vs FFN_GELU's 1.679 vs FFN_ReLU's 1.677 — SwiGLU and GEGLU are the best variants. At 65,536 steps, GEGLU achieves 1.942 vs SwiGLU's 1.944.
- **Fine-tuning (Section 3.3, Tables 2–4):** GLU variants improve on most GLUE and SuperGLUE tasks. On SQuAD v1.1, FFN_ReGLU achieves the best F1 (91.18) vs FFN_ReLU's 90.87; results are described as "noisy."

### Simplifications in This Notebook

- Tiny model: 4 Transformer layers, 4 heads, 128-dim (808,192 params GELU, 807,680 SwiGLU) instead of T5-base (12+12 layers, 220M params)
- Synthetic template-generated text (~84K chars, 24-char vocab) instead of the C4 dataset (750 GB, 32K-token vocab)
- 30 epochs with AdamW (lr 3e-4, batch 64, seq 128) instead of 524,288 steps with Adafactor on a 32-core TPUv2 cluster
- Next-token character prediction instead of span-filling (denoising objective)
- GELU vs SwiGLU only, instead of all 8 variants (ReLU, GELU, Swish, GLU, Bilinear, ReGLU, GEGLU, SwiGLU)

### What the Notebook Result Shows

On the synthetic text corpus, both models learn the character-level distribution (validation loss well below the random baseline of ln(24) = 3.18). GELU achieves final validation loss 1.6755 and SwiGLU 1.7051 — a difference of 0.030 in the opposite direction from the paper. At this model scale (~808K parameters) and task simplicity (24-character vocab, template-generated sentences), the FFN variant has negligible impact on overall quality. The paper's 0.043 log-perplexity improvement (1.636 vs 1.679) was measured on a 220M-parameter T5-base model trained for 524K steps on the 750 GB C4 dataset; the notebook's result does not reproduce this improvement, which is expected given the orders-of-magnitude difference in model size, data, and training duration.

## Influence

SwiGLU was adopted as the default FFN activation in most major open-weight LLMs. LLaMA (Meta, 2023) uses SwiGLU in its feed-forward layers, as do Mistral 7B, Qwen, DeepSeek, and PaLM (Google, 2022). The 2/3 hidden-dimension reduction trick is also standard practice in these models. The paper's GitHub repository is not linked from the arXiv page, but the implementation is a few lines of code. It has about 302 citations ([OpenAlex](https://openalex.org/works/W3006439205), checked October 2026). The paper is short (4 pages) and deliberately offers no theoretical explanation for why gating helps, closing with "we attribute their success, as all else, to divine benevolence."

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/003_swiglu-glu-variants-improve-transformer/solution.ipynb) | From-scratch PyTorch tiny GPT with GELU-FFN vs SwiGLU-FFN at matched parameter counts (~808K), trained on synthetic text (GELU val loss 1.6755, SwiGLU 1.7051 — improvement not reproduced at this scale) |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
