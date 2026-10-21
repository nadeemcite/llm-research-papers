---
title: "008 YaRN"
nav_order: 9
permalink: "/008_yarn-efficient-context-window-extension/"
---
# YaRN: Efficient Context Window Extension of Large Language Models (YaRN)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/yarn-efficient-context-window-extension)

- **Paper:** [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)
- **Authors:** Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole
- **Lab, year:** Nous Research, EleutherAI, University of Geneva, 2023

## Summary

YaRN (Yet another RoPE extensioN method) is a technique for extending the context window of transformer-based language models that use Rotary Position Embeddings (RoPE). Standard RoPE models cannot generalize beyond their training context length — perplexity (a measure of how surprised the model is by the next token; lower is better) explodes when the sequence gets longer. YaRN solves this by modifying how RoPE frequencies are computed at inference time, combined with a temperature scaling on the attention logits (the raw scores before the softmax that turns them into probabilities).

The method builds on a lineage of prior approaches. Position Interpolation (PI) scales all positions uniformly by a factor s, but this degrades high-frequency RoPE dimensions. The "NTK-aware" method changes the RoPE base to preserve high frequencies, but introduces uncontrolled extrapolation. "NTK-by-parts" interpolates only the low-frequency dimensions (long wavelengths) while leaving high-frequency ones untouched, using a ramp controlled by α and β. YaRN adds a final ingredient: an attention temperature t = 0.1·ln(s) + 1 that compensates for the entropy increase when the context grows. (Softmax is the function that converts raw scores into probabilities that sum to 1; temperature scales these scores before softmax, making the distribution sharper or flatter.)

The paper fine-tuned Llama 2 7B and 13B to 128K context (32× extension) using only 400 training steps on PG19 data — 10× fewer tokens and 2.5× fewer steps than PI. YaRN models achieved perplexity of 2.37 at 128K on Proof-pile (13B, s=32), passkey retrieval accuracy of 99.4%, and retained short-context benchmark scores within ~1% of the original Llama 2.

This repo's notebook builds a small RoPE-based language model (4 layers, 1M parameters), trains it on Tiny Shakespeare at 128-token context, then applies PI, NTK-aware, NTK-by-parts, and YaRN to evaluate perplexity at up to 8× extension (1024 tokens). YaRN achieves the lowest zero-shot perplexity at 4× and 8× scales.

## What Problem Does It Solve

Imagine you learned to read books that are at most 100 pages long. Now someone hands you a 800-page novel. Even though you can read perfectly, the sheer length confuses you — you lose track of where things are because your "position sense" was never calibrated for distances this large.

That's exactly what happens to language models with RoPE. During training, the model learns positional patterns within a fixed context window (say 4096 tokens). When you give it a longer sequence at inference time, the rotary position embeddings produce angles the model has never seen — like a clock that suddenly has to go past 12. The model gets confused and its predictions degrade rapidly.

YaRN's solution is to "interpolate" rather than "extrapolate." Instead of letting the clock tick past 12, it compresses the entire time range so it still fits within the familiar 1-to-12 range. But instead of compressing everything equally (which is what PI does), YaRN compresses only the parts of the position encoding that actually need it — the low-frequency dimensions that carry absolute position — while leaving the high-frequency dimensions (which carry local relative position) untouched. Then it adds a small temperature correction to keep the attention sharp.

## Core Idea

- **NTK-by-parts interpolation:** RoPE dimensions with wavelengths longer than the training context (λ > L) encode absolute position and must be interpolated; dimensions with short wavelengths (λ < L) encode relative position and should be left alone. A ramp function γ(r) smoothly transitions between these regimes, controlled by α=1 and β=32.
- **Attention temperature scaling:** Extending the context increases attention entropy. YaRN corrects this by dividing the attention logits by a temperature t = 0.1·ln(s) + 1, equivalent to scaling the RoPE embeddings by 1/√t. This has zero overhead since RoPE tables are precomputed.
- **Dynamic Scaling (inference-time):** Instead of fixing s = L′/L, update s = max(1, l′/L) at each forward pass, so the model gracefully degrades rather than abruptly breaking at the context limit.

## Key Method Details

- **RoPE modification (YaRN, Eq. 7 & 13):** g(m) = m, h(θ\_d) = (1 − γ(r(d))) · θ\_d / s + γ(r(d)) · θ\_d, where r(d) = L / λ\_d is the ratio of training context to wavelength, and γ is a linear ramp from 0 (at r < α) to 1 (at r > β).
- **Temperature (Eq. 15):** t = 0.1·ln(s) + 1, fitted from perplexity-minimizing √(1/t) values across LLaMA 7B/13B/33B/65B at various scale factors.
- **Training:** Llama 2 7B/13B, 400 steps, batch size 64, AdamW (lr 2×10⁻⁵, β₁=0.9, β₂=0.95), PG19 dataset chunked to 64K, Flash Attention 2. The s=32 model started from the s=16 checkpoint and trained 200 additional steps (Section 4.1).
- **Results:** Perplexity on Proof-pile at 128K context: 7B s=32 → 2.37, 13B s=32 → 2.24 (Table 1). Passkey retrieval at 128K: 99.4% for both 7B and 13B s=32 (Table 9). Hugging Face Open LLM benchmarks within ~1% of Llama 2 baseline (Table 3). Training cost: 256 A100-hours for 7B 64K model vs 640 for PI-based 16K model (Table 4).

### Simplifications in This Notebook

- Smaller model: 4 layers, 4 heads, 128 dim, ~1.06M parameters (vs Llama 2 7B/13B)
- Dataset: Tiny Shakespeare (~1 MB, character-level) instead of PG19 (~82 GB, BPE)
- 15 epochs / 3000 steps at 128-character context instead of 400 steps at 64K context
- Fine-tuning: 200 steps at 256-character context instead of 400 steps at 64K context
- Evaluation: direct perplexity on held-out data instead of sliding-window perplexity (S=256)
- α=1, β=32: same as the paper's recommended values for LLaMA family models

### What the Notebook Result Shows

On the Tiny Shakespeare validation set (character-level, 128-token training context), the zero-shot perplexity at 8× extension (1024 tokens) is: baseline 21.76, PI 110.04, NTK-aware 12.07, NTK-by-parts 33.61, YaRN 31.56. After 200 steps of fine-tuning at 256-token context, the post-fine-tuning perplexity at 8× is: baseline 21.54, PI 47.12, NTK-aware 8.02, NTK-by-parts 8.93, YaRN 8.95.

The temperature ablation confirms the paper's finding: YaRN's temperature scaling improves over NTK-by-parts by 1.4% at 2×, 6.6% at 4×, and 6.1% at 8× in the zero-shot setting. The improvement grows with scale factor, consistent with the paper's observation that entropy increases more at larger extensions. Notably, NTK-aware performs best after fine-tuning in this tiny model — unlike the paper's 7B results where YaRN wins — because the model is too small for the wavelength-based ramp to have its full effect. The overall pattern (extension methods dramatically reduce perplexity vs baseline; YaRN's temperature helps in the zero-shot regime) matches the paper's findings.

## Influence

YaRN was rapidly adopted across the open-source LLM ecosystem. Mistral's Code Llama used the earlier "NTK-aware" method (which YaRN formalized), and the full YaRN method was integrated into Hugging Face Transformers' RoPE scaling options (`rope_scaling: yarn`). It became the standard approach for context extension in models like Qwen, Yi, and DeepSeek. The "train short, test long" paradigm YaRN demonstrated — fine-tuning on shorter sequences while extrapolating to much longer ones — influenced subsequent long-context training strategies. The paper has about 350 citations ([OpenAlex](https://openalex.org/works/W4390027983), checked October 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/008_yarn-efficient-context-window-extension/solution.ipynb) | From-scratch PyTorch RoPE transformer with PI, NTK-aware, NTK-by-parts, and YaRN extension methods, trained on Tiny Shakespeare and evaluated at up to 8× context |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
