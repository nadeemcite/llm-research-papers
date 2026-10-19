---
title: "004 RMSNorm"
nav_order: 5
permalink: "/004_rmsnorm-root-mean-square-layer-norm/"
---
# Root Mean Square Layer Normalization (RMSNorm)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/rmsnorm-root-mean-square-layer-normalization)

- **Paper:** [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- **Authors:** Biao Zhang, Rico Sennrich
- **Lab, year:** University of Edinburgh and University of Zurich, 2019 (NeurIPS 2019)

## Summary

Layer normalization (LayerNorm), introduced by Ba et al. (2016), stabilizes deep network training by normalizing both the mean (re-centering) and variance (re-scaling) of each layer's activations. While effective, the mean computation adds overhead that becomes costly as networks grow deeper — especially in RNNs (recurrent neural networks, which process text one step at a time) where normalization runs at every time step. This paper asks: is the re-centering (mean subtraction) step actually necessary?

The authors hypothesize that re-scaling invariance — keeping the output stable when inputs or weights are scaled — is the property that matters, and that re-centering invariance is dispensable. They propose RMSNorm, which normalizes by the root mean square (RMS) of the inputs alone, dropping the mean subtraction entirely. RMSNorm is $\bar{a}_i = \frac{a_i}{\text{RMS}(a)} g_i$ where $\text{RMS}(a) = \sqrt{\frac{1}{n}\sum_i a_i^2}$, compared to LayerNorm's $\bar{a}_i = \frac{a_i - \mu}{\sigma} g_i$.

Experiments on machine translation (WMT14 En-De), image-caption retrieval (MS COCO), reading comprehension (CNN/DailyMail), and image classification (CIFAR-10) show that RMSNorm matches LayerNorm's quality while reducing running time by 7–64% across different models. On the Transformer (Vaswani et al., 2017), RMSNorm achieves BLEU (a 0-100 translation quality score) 26.8 vs LayerNorm's 26.6 on newstest2014 with a 7% speedup. The paper also introduces pRMSNorm, which estimates RMS from just 6.25% of the inputs.

This repo's notebook implements RMSNorm and LayerNorm as norm layers in two identical tiny Transformers trained side by side on synthetic text. At this scale (~866K parameters, 25 epochs, CPU), both models converge to nearly identical validation loss (0.191 vs 0.192), confirming the paper's core finding of comparable quality. The speed difference at this toy scale on CPU is negligible (within noise), because the normalization overhead is a tiny fraction of total compute — the paper's 7–64% speedup was measured on GPUs with large models where normalization is a significant bottleneck.

## What Problem Does It Solve

Imagine you are adjusting the volume on a music player. LayerNorm is like a sound engineer who both centers the audio (removing any DC offset so the waveform swings equally above and below zero) and normalizes the loudness (so no track is too quiet or too loud). The centering step is extra work. This paper asks: what if we skip the centering and just adjust the loudness? As long as the volume is consistent, does it matter if there is a slight offset?

In neural networks, each layer's activations can drift in scale (become too large or too small) during training, destabilizing learning. LayerNorm fixes this by computing both the mean and the variance of each layer's outputs and using them to normalize. The mean computation requires summing all activations, subtracting, then computing variance — three passes over the data. RMSNorm just computes the root mean square (a single pass) and divides by it. The paper shows this simpler operation is enough: the mean subtraction that LayerNorm does is not what makes training stable.

## Core Idea

- **Drop the mean, keep the RMS:** RMSNorm simplifies LayerNorm by removing the mean subtraction ($\mu$) and normalizing only by $\text{RMS}(a) = \sqrt{\frac{1}{n}\sum_i a_i^2}$. The output is $\bar{a}_i = \frac{a_i}{\text{RMS}(a)} g_i$, where $g$ is a learnable gain parameter initialised to 1.
- **Re-scaling invariance is sufficient:** The paper proves that RMSNorm preserves re-scaling invariance (output unchanged when inputs or weights are scaled by a constant) but drops re-centering invariance (output changes under shifts). Experiments show re-centering is not needed for training stability.
- **Implicit learning rate adaptation:** The gradient of the weight matrix $W$ in RMSNorm is negatively correlated with weight scaling — this acts as an automatic learning rate controller that prevents weight norms from exploding (Section 4.2, Eq. 9–10).
- **pRMSNorm for further speed:** When neurons are i.i.d., RMS can be estimated from just $p\%$ of the inputs. The paper uses $p = 6.25\%$ and shows competitive results, though implementation overhead sometimes negates the theoretical speedup.

## Key Method Details

- **LayerNorm (Ba et al., 2016, Eq. 2–3):** $\bar{a}_i = \frac{a_i - \mu}{\sigma} g_i$, where $\mu = \frac{1}{n}\sum_i a_i$ and $\sigma = \sqrt{\frac{1}{n}\sum_i (a_i - \mu)^2}$.
- **RMSNorm (Section 4, Eq. 4):** $\bar{a}_i = \frac{a_i}{\text{RMS}(a)} g_i$, where $\text{RMS}(a) = \sqrt{\frac{1}{n}\sum_i a_i^2}$. No bias term; gain $g$ initialised to 1.
- **pRMSNorm (Section 5):** $\text{RMS}(a) = \sqrt{\frac{1}{k}\sum_{i=1}^{k} a_i^2}$ where $k = \lceil n \cdot p \rceil$. Uses the first $p\%$ of inputs. The paper uses $p = 6.25\%$.
- **Invariance properties (Table 1):** RMSNorm is invariant to weight matrix re-scaling, input re-scaling, and dataset re-scaling, but NOT to re-centering (shifts). LayerNorm is invariant to both re-scaling and re-centering.
- **Transformer results (Table 4, Section 6.1):** RMSNorm achieves BLEU 26.8 (newstest2014) and 27.7 (newstest2017) vs LayerNorm's 26.6 and 27.7. Speedup: 7–9% per 1K steps on Tesla V100.
- **RNNSearch results (Table 2, TensorFlow):** RMSNorm 22.4 BLEU (newstest2014) vs LayerNorm 22.6, with 24.7% speedup. Baseline (no norm): 21.7.
- **Speed range (Section 7):** 7–64% speedup across models and frameworks. The largest speedups (40–64%) are on the image-caption retrieval model (Table 8); the smallest (7–9%) on the Transformer.

### Simplifications in This Notebook

- Tiny model: 4 Transformer layers, 4 heads, 128-dim embedding (~866K parameters) instead of the paper's Transformer base (6+6 layers, 512-dim, 8 heads, ~40M params)
- Synthetic template-generated text (~8K sentences, 31-char vocab) instead of WMT14 En-De (4.5M sentence pairs, 32K BPE vocab)
- 25 epochs with AdamW (lr 3e-4, batch 128, seq 64) instead of 300K steps with Adam (warmup 4000, inverse-sqrt schedule)
- Character-level next-token prediction instead of word-level machine translation with BLEU evaluation
- RMSNorm vs LayerNorm only, instead of all 6 methods (Baseline, LayerNorm, L2-Norm, RMSNorm, pRMSNorm, BatchNorm, WeightNorm)

### What the Notebook Result Shows

On the synthetic text corpus, both models learn the character-level distribution (validation loss 0.191 for LayerNorm vs 0.192 for RMSNorm, well below the random baseline of $\ln(31) \approx 3.43$; perplexity 1.21 for both). The difference of 0.0006 in validation loss is within run-to-run noise, confirming the paper's core finding that RMSNorm achieves "comparable performance" to LayerNorm. RMSNorm trains stably without mean subtraction.

On this toy model on CPU, the total training time is comparable (97s vs 107s), with no meaningful speedup — the paper's 7–64% speedup was measured on GPUs (TITAN X, Tesla V100, RTX 2080 Ti) with large models where the normalization overhead is a significant fraction of total compute, and PyTorch's built-in LayerNorm is C++-optimized while our RMSNorm is a pure-Python module. This notebook's result is not comparable to the paper's BLEU scores or speedup percentages, but it demonstrates that RMSNorm is a correct, stable, drop-in replacement for LayerNorm.

## Influence

RMSNorm was adopted as the default normalization layer in most major open-weight LLMs. LLaMA (Meta, 2023) uses RMSNorm in place of LayerNorm, as do Mistral 7B, Qwen, DeepSeek, and PaLM (Google, 2022). The removal of the mean computation and bias term became standard practice in modern Transformer architectures. The paper has about 203 citations ([OpenAlex](https://openalex.org/W2970903692), checked October 2026). The official code is at [github.com/bzhangGo/rmsnorm](https://github.com/bzhangGo/rmsnorm). The paper was accepted at NeurIPS 2019.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/004_rmsnorm-root-mean-square-layer-norm/solution.ipynb) | From-scratch PyTorch RMSNorm and LayerNorm implemented as norm layers in two identical tiny Transformers, trained on synthetic text to compare quality and speed |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
