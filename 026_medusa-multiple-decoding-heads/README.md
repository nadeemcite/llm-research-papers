---
title: "026 Medusa"
nav_order: 27
permalink: "/026_medusa-multiple-decoding-heads/"
---
# Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads (Medusa)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/medusa-multiple-decoding-heads)

- **Paper:** [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
- **Authors:** Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, Tri Dao
- **Lab, year:** Princeton University / Together AI / University of Illinois Urbana-Champaign / Carnegie Mellon University / University of Connecticut, 2024 (ICML 2024)

## Summary

Large language models generate text one token at a time — each step requires a full forward pass through the model, and the next step cannot begin until the current one finishes. This serial bottleneck is memory-bandwidth-bound: each forward pass moves the entire model's parameters from high-bandwidth memory (HBM) to the accelerator's compute units, only to produce a single token. Methods like speculative decoding address this by using a separate small "draft" model to predict multiple tokens, but finding, training, and serving a good draft model is difficult and adds system complexity.

Medusa takes a simpler approach. Instead of a separate draft model, it adds extra "Medusa heads" — lightweight decoding layers — on top of the frozen backbone model. Each head k predicts the token at position t+k+1 from the hidden state at position t, so the model can speculate on multiple future tokens in parallel. A tree-based attention mechanism then constructs multiple candidate continuations (the Cartesian product of each head's top-k predictions) and verifies them all in a single forward pass. The longest verified prefix is accepted, giving 1 to K+1 tokens per decoding step instead of 1.

The paper presents two training strategies: Medusa-1 trains only the extra heads on a frozen backbone (parameter-efficient, lossless), while Medusa-2 jointly fine-tunes the backbone and heads with a combined loss, differential learning rates, and heads warmup to preserve the backbone's capabilities. The paper also introduces a typical acceptance scheme (inspired by truncation sampling) that is more efficient than rejection sampling at higher temperatures, and a self-distillation pipeline for when no training data is available.

Experiments on Vicuna-7B/13B/33B and Zephyr-7B show Medusa-1 achieves over 2.2x speedup without quality loss, while Medusa-2 reaches 2.3-2.8x. Table 3 breaks down the incremental contributions: heads alone ~1.5x, adding tree attention ~1.9x, optimized tree configuration ~2.2x, Medusa-2 training ~2.8x. The coding and extraction categories benefit most (3.29x and 3.62x respectively).

This repo's notebook implements Medusa with GPT-2 small (124M) as the frozen backbone, 3 Medusa heads trained for 3 epochs, batched candidate verification, and measures an acceleration rate of 2.13 tokens per step (53% fewer forward passes).

## What Problem Does It Solve

Imagine a restaurant where the head chef tastes every dish before it goes out. The chef is excellent but slow — each tasting takes a full round trip to the kitchen. If you need 60 dishes, that's 60 trips. Now imagine the chef has a few "assistant tasters" who can predict what the chef will approve. They prepare multiple candidate dishes at once, and the chef verifies them all in a single trip. Most of the time, the chef agrees with several of the assistants' picks — so 60 dishes might take only 28 trips instead of 60.

That's Medusa: the "assistants" are lightweight prediction heads bolted onto the model itself (not a separate kitchen), and the "verification trip" is a single batched forward pass that checks all candidates at once. The key advantage over speculative decoding is that there's no separate draft model to train, host, or keep in sync — the heads are just a few extra layers that share the backbone's representations.

## Core Idea

- **Medusa heads (Section 2.1.1):** Instead of a separate draft model, add K extra decoding heads on top of the backbone's last hidden states. Each head k predicts position t+k+1 from h_t using a single MLP layer with residual connection: p^(k)_t = softmax(W2_k · (SiLU(W1_k · h_t) + h_t)). W2_k is initialized to the original LM head weight, W1_k to zero — so initially every head mimics the original model.
- **Tree attention (Section 2.1.2):** Each head's top-k predictions form the levels of a tree. The Cartesian product gives k^K candidate paths. A tree-structured attention mask allows all candidates to be verified in a single forward pass — each token attends only to its ancestors in the tree, not to tokens from other branches.
- **Typical acceptance (Section 2.3.1):** Instead of rejection sampling (which becomes less efficient at higher temperatures), accept a candidate token if its probability under the original model exceeds a threshold: min(ε, δ · exp(-H(p))). This is inspired by truncation sampling and reverts to greedy decoding at temperature=0.

## Key Method Details

- **Medusa-1 loss (Eq. 1):** L = Σ_k -λ_k · log p^(k)_t(y_{t+k+1}), where λ_k = 0.8^k. Backbone is frozen; only heads receive gradients. Training takes ~5 hours on a single A100 for Vicuna-7B on 60k ShareGPT samples.
- **Medusa-2 loss (Eq. 2):** L = L_LM + λ_0 · L_Medusa-1, with differential learning rates (lower for backbone, higher for heads) and two-stage training (heads-only warmup, then joint). Uses LoRA + KL-divergence self-distillation when the original training data is unavailable.
- **Speedup results (Figure 3, Table 1):** Vicuna-7B: Medusa-1 2.18x, Medusa-2 2.83x. Vicuna-13B: Medusa-1 2.33x, Medusa-2 2.83x. Vicuna-33B: 2.35x. Zephyr-7B: 2.66x. Speculative decoding with draft models: only 1.47-1.60x.
- **Table 3 — incremental impact:** Heads without tree attention ~1.5x; +tree attention ~1.9x; +optimized tree ~2.2x; +Medusa-2 training ~2.8x.
- **Per-category speedup (Figure 3b):** Coding 3.29x, Extraction 3.62x, Math 3.01x — structured/predictable tasks benefit most.
- **AlpacaEval (Table 4):** Vicuna-7B 3.47x acceleration rate, 2.83x speedup; Zephyr-7B 3.14x rate, 2.66x speedup.

### Simplifications in This Notebook

- GPT-2 small (124M, 12 layers) as the frozen backbone instead of Vicuna-7B/13B/33B
- 3 Medusa heads instead of 5; top-2 per head (8 candidates) instead of optimized sparse trees with 64+ nodes
- 200 training prompts for 3 epochs instead of 60k ShareGPT samples for 2 epochs
- Medusa-1 training only (frozen backbone); no Medusa-2 joint training
- Batched candidate verification (functionally equivalent to tree attention) instead of custom tree attention masks
- No self-distillation pipeline; no typical acceptance used in the main decoding loop (demoed separately)

### What the Notebook Result Shows

On 10 test prompts with GPT-2 small as the frozen backbone and 3 Medusa heads (top-2 per head, 8 candidates per step), Medusa decoding achieves an **acceleration rate of 2.13 tokens per forward pass** (averaged across prompts), reducing the number of forward passes by **53%** (from 60 to ~28 per prompt). The tokens-per-step distribution shows 86% of steps accept 2 tokens, 12% accept 3, and 1% accept 4.

However, the **wall-clock speedup is 0.52x** — Medusa is slower in wall time because GPT-2 small (124M) is so small that the overhead of processing 8 candidate sequences in a batch outweighs the savings from fewer forward passes. The paper's speedups (2.2-2.8x) are on Vicuna-7B/13B/33B where each forward pass is far more expensive, making the overhead negligible. This confirms the paper's insight that Medusa is most beneficial for large models where the memory-bandwidth bottleneck dominates.

Head accuracy after 3 epochs on 45 training sequences: Head 1 (t+2) 14.9%, Head 2 (t+3) 6.8%, Head 3 (t+4) 15.0% — far below the paper's ~60% on Vicuna, because GPT-2 small's hidden representations are weaker and the training corpus is much smaller.

## Influence

Medusa was developed at Together AI (Tri Dao's group) and released in January 2024. It directly influenced the adoption of multi-token prediction for inference acceleration in production LLM serving systems. The tree-attention idea was independently explored by SpecInfer (Miao et al., 2023) and Sequoia (Chen et al., 2024). Medusa is integrated into vLLM and other inference frameworks. The typical acceptance scheme connects speculative decoding to truncation sampling literature (Hewitt et al., 2022).

The paper builds on Blockwise Parallel Decoding (Stern et al., 2018), which first proposed extra heads for parallel token prediction but without tree attention or the typical acceptance scheme. Medusa's key contribution is making this practical for modern LLMs by combining it with tree-based candidate verification and efficient training recipes.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/026_medusa-multiple-decoding-heads/solution.ipynb) | PyTorch Medusa heads trained on frozen GPT-2 small, with batched tree verification and speedup measurement (acceleration rate 2.13 tokens/step, 53% pass reduction) |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
