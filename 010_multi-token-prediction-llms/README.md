---
title: "010 Multi-token Pred"
nav_order: 11
permalink: "/010_multi-token-prediction-llms/"
---
# Better & Faster Large Language Models via Multi-token Prediction (Multi-token Pred)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/multi-token-prediction-llms)

- **Paper:** [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
- **Authors:** Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, Gabriel Synnaeve
- **Lab, year:** FAIR at Meta (Gloeckle also at CERMICS, Ecole des Ponts ParisTech; Youbi Idrissi also at LISN, Université Paris-Saclay), 2024 (ICML 2024)

## Summary

Large language models such as GPT and Llama are trained with a next-token prediction loss: at each position in the training text, the model learns to predict only the very next token. This paper proposes multi-token prediction, where the model instead predicts the next n tokens (n = 4 works best) simultaneously using n independent output heads on top of a shared transformer trunk. The shared trunk produces a hidden representation of the context, and each head independently predicts one of the next n future tokens. This forces the model to plan further ahead and learn richer representations from the same data.

The method improves sample efficiency — models learn more from the same amount of training data — with no overhead in training time or memory (thanks to a memory-efficient implementation that processes heads sequentially during forward/backward passes). The gains scale with model size: at 13B parameters, 4-token prediction models solve 12% more problems on HumanEval (a Python coding benchmark) and 17% more on MBPP (another coding benchmark) than comparable next-token models. As a bonus, the extra heads enable self-speculative decoding, making inference up to 3x faster on code and 2.7x faster on text, since the model can propose multiple tokens per forward pass and verify them with the next-token head.

This repo's notebook implements multi-token prediction from scratch in PyTorch with n=4 heads, trains a next-token baseline and a 4-token model on Tiny Shakespeare, and demonstrates self-speculative decoding with a 5x inference speedup.

## What Problem Does It Solve

Imagine you are learning to write by copying sentences one word at a time. After each word, you only check whether you got the very next word right. You might pick the next word correctly but have no idea where the sentence is going — you are reacting locally rather than planning ahead. This is how next-token prediction trains language models: it latches onto local patterns and overlooks "hard" decisions, the choice points that actually determine whether the text makes sense. That is why state-of-the-art models need orders of magnitude more text than a human child to reach fluency.

Multi-token prediction fixes this by asking the model to predict several words ahead at once. Now the model cannot just react — it has to think about where the sentence is going to get all n predictions right. This shifts the learning signal toward consequential decisions and long-range patterns.

## Core Idea

- **n independent output heads on a shared trunk:** A shared transformer trunk produces a hidden representation z from the input context. n independent heads each project z to vocabulary logits, predicting tokens at positions t+1, t+2, ..., t+n. The loss is the average cross-entropy across all n heads.
- **Memory-efficient forward/backward:** Instead of materializing all n heads' logits (shape (n, V), where V is vocabulary size) simultaneously in GPU memory, the implementation sequentially computes forward and backward for each head, accumulating gradients at the trunk. Peak memory drops from O(nV + d) to O(V + d) with no runtime cost.
- **Self-speculative decoding:** At inference, the next-token head (head 0) is the main predictor. The extra heads propose future tokens, which are verified by head 0 — if they match, those tokens are accepted without separate forward passes. This is speculative decoding without needing a separate draft model.

## Key Method Details

- **Multi-token prediction loss (Eq. 2):** L_n = -sum_t log P_theta(x_{t+n:t+1} \| x_{t:1}), factored as sum over n independent heads: -sum_t sum_{i=1}^{n} log P_theta(x_{t+i} \| z_{t:1}) * P_theta(z_{t:1} \| x_{t:1}).
- **Architecture (Section 2):** Shared transformer trunk f_s producing hidden z, n independent output heads (transformer layers f_{h_i}), shared unembedding matrix f_u. P(x_{t+i} \| x_{t:1}) = softmax(f_u(f_{h_i}(f_s(x_{t:1})))) for i = 1..n.
- **Parameter budget:** To keep comparisons fair, when adding n-1 layers in future prediction heads, the authors remove n-1 layers from the shared trunk (Section 3).
- **Results (Table 1, 7B models on 200B tokens of code):** 4-token prediction: MBPP pass@1 33.8% (vs 30.0% next-token), HumanEval pass@1 24.0% (vs 22.8%). At 13B on 1T tokens: MBPP pass@1 43.1% (vs 40.7%), HumanEval pass@1 31.6% (vs 31.7%).
- **Inference speedup (Section 3.2):** 3.0x on code (2.5 accepted out of 3), 2.7x on text. 8-byte prediction model: 6.4x speedup.
- **Optimal n (Section 3.4):** n = 4 is optimal for token-level models; n = 8 for byte-level models.

### Simplifications in This Notebook

- Smaller model: 4 layers, 4 heads, 128 dim, about 0.82M (next-token) / 0.85M (multi-token) parameters (vs 300M-13B in the paper)
- Dataset: Tiny Shakespeare (~1 MB, character-level, vocab 65) instead of 200B+ tokens of code
- Output heads: single linear layers (vs transformer-layer heads in the paper)
- 12 epochs, 100 steps per epoch, batch 64, lr 3e-4 with AdamW — instead of full-scale pretraining
- No memory-efficient sequential head processing (not needed at this scale); no MoE; no byte-level experiments
- Self-speculative decoding: simplified greedy acceptance (accept all proposed, no rollback) rather than full blockwise parallel decoding with Medusa-style tree attention

### What the Notebook Result Shows

On Tiny Shakespeare (character-level, 128-token context, 12 epochs on Kaggle T4 GPU), the next-token model (823,680 params) achieves validation loss 2.0794 (perplexity 7.98, top-1 accuracy 38.5%). The multi-token model (848,640 params, 3.0% extra for 3 additional heads) achieves head-0 validation loss 2.2304 (perplexity 9.30, accuracy 33.7%). Per-head accuracy degrades as expected: 33.7% (t+1), 23.9% (t+2), 18.4% (t+3), 16.9% (t+4). The next-token model is slightly better at this small scale — consistent with the paper's finding that "multi-token prediction models are worse than the baseline for small model sizes, but outperform the baseline at scale" (Section 3.1).

Self-speculative decoding achieves a 5.04x wall-clock speedup (200 tokens in 0.11s vs 0.57s), with 58.8% acceptance rate and 1.40 tokens accepted per 4-token proposal. The paper reports 3x speedup with 2.5 accepted tokens out of 3 on code — our acceptance rate is lower because our tiny model's speculative heads are less accurate, but the speedup is still substantial since we generate 4 tokens per forward pass.

## Influence

Multi-token prediction was adopted in Meta's production code models and influenced subsequent work on auxiliary training losses for LLMs. The self-speculative decoding approach (using a model's own extra heads as a draft model) connected to the broader speculative decoding literature (Leviathan et al., 2023; Stern et al., 2018; Medusa by Cai et al., 2024). The finding that multi-token prediction promotes induction head formation linked this work to interpretability research on in-context learning (Olsson et al., 2022). Byte-level multi-token prediction (n=8) opened a path toward efficient byte-level models that can compete with token-based models. The paper has about 15 citations ([OpenAlex](https://openalex.org/W4396821494), checked October 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/010_multi-token-prediction-llms/solution.ipynb) | From-scratch PyTorch implementation of multi-token prediction (n=4 heads) vs next-token baseline on Tiny Shakespeare, with self-speculative decoding speedup demonstration |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
