---
title: "067 Tiny Recursive Model"
nav_order: 68
permalink: "/067_tiny-recursive-model-reasoning/"
---
# Less is More: Recursive Reasoning with Tiny Networks (Tiny Recursive Model)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/tiny-recursive-model-reasoning)

- **Paper:** [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871)
- **Authors:** Alexia Jolicoeur-Martineau
- **Lab, year:** Samsung SAIL Montréal, 2025 (preprint)

## Summary

Large language models struggle on hard puzzle tasks like Sudoku and ARC-AGI, even with chain-of-thought reasoning and massive test-time compute. In October 2025, Alexia Jolicoeur-Martineau at Samsung SAIL Montréal proposed the Tiny Recursive Model (TRM), a radically simple approach that beats LLMs on these benchmarks using only 7 million parameters — less than 0.01% of the parameters of models like Deepseek R1 or Gemini 2.5 Pro.

TRM builds on the Hierarchical Reasoning Model (HRM, Wang et al. 2025), which used two separate networks recursing at different frequencies with deep supervision. TRM simplifies this to a single tiny 2-layer network that recursively updates a latent reasoning feature z and a proposed answer y. Over multiple supervision steps, the model progressively improves its answer. The key insight: full backpropagation through the recursion (instead of HRM's 1-step gradient approximation) and a single shared network (instead of two) yield dramatically better generalization while cutting parameters by 4×.

With 7M parameters, TRM achieves 87.4% test accuracy on Sudoku-Extreme (vs HRM's 55%), 85.3% on Maze-Hard (vs 74.5%), 44.6% on ARC-AGI-1 (vs 40.3%), and 7.8% on ARC-AGI-2 (vs 5.0%) — all beating the 27M-parameter HRM. On ARC-AGI-1, TRM outperforms Deepseek R1 (15.8%), o3-mini-high (34.5%), and Gemini 2.5 Pro (37.0%).

This notebook implements TRM from scratch in PyTorch, training a 530K-parameter recursive model on 4×4 Sudoku puzzles with 1,000 training examples and data augmentation, demonstrating the core recursive reasoning and deep supervision mechanism.

## What Problem Does It Solve

Imagine solving a really hard Sudoku puzzle. A big AI model like GPT-4 tries to write out its reasoning step by step in words — "if the 7 goes in row 3, then column 5 must have a 2..." — but one wrong word early on ruins the entire chain. It is like trying to solve a puzzle by talking out loud; one slip of the tongue and you lose the thread.

TRM takes a different approach: instead of reasoning in words, it reasons in a compressed internal representation. It starts with a guess at the answer, then quietly refines that guess many times inside a tiny network — like scribbling and erasing on a scratchpad until the answer looks right. Because the network is so small (only 2 layers), it cannot memorize the training data, so it is forced to actually learn the reasoning pattern. The paper shows this tiny recursive approach generalizes far better than either large language models or the more complex HRM.

## Core Idea

- **Single tiny network with recursion:** Instead of HRM's two separate networks, TRM uses one shared 2-layer network. Given the input x, current answer y, and latent z, the network recursively updates z (latent reasoning) and then y (answer refinement) over n steps. The same network handles both tasks — the inclusion of x in the input distinguishes "update reasoning" from "update answer."
- **Full backpropagation through recursion:** HRM assumed a fixed-point was reached and used a 1-step gradient approximation. TRM simply backpropagates through the entire recursion (T−1 no-grad passes followed by one with-grad pass), eliminating the need for fixed-point theorems while achieving much better generalization (56.5% → 87.4% on Sudoku-Extreme).
- **Deep supervision with EMA:** The model runs up to N_sup = 16 supervision steps, each improving y and z. Between steps, y and z are detached and carried forward. Exponential Moving Average (EMA) of weights with decay 0.999 prevents overfitting collapse on small datasets.

## Key Method Details

- **Architecture:** Single 2-layer transformer with RMSNorm, SwiGLU activation, rotary embeddings, no bias. For Sudoku (9×9 grid, fixed context length), self-attention is replaced by an MLP-Mixer-style layer applied on the sequence dimension, improving accuracy from 74.7% to 87.4%.
- **Recursion:** T = 3 supervision blocks, each containing n = 6 latent updates + 1 answer update. T−1 = 2 blocks run without gradients, then 1 block with full backpropagation. Effective depth per supervision step: T(n+1) = 3×7 = 42 layers.
- **Training:** AdamW (β1=0.9, β2=0.95), batch size 768, hidden size 512, learning rate 1e-4 with 2K warmup, weight decay 1.0, stablemax loss, 60K epochs. EMA decay 0.999. N_sup = 16 max supervision steps with ACT early stopping (BCE halting loss, single forward pass).
- **Data:** 1,000 Sudoku-Extreme training examples, 423K test examples, with 1000× shuffling augmentation per example.
- **Results:** Sudoku-Extreme 87.4% (TRM-MLP, 5M params) or 74.7% (TRM-Att, 7M params); Maze-Hard 85.3%; ARC-AGI-1 44.6%; ARC-AGI-2 7.8%.

### Simplifications in This Notebook

- Smaller model: 2-layer network, 530K parameters, hidden size 128 instead of 512
- 4×4 Sudoku (paper: 9×9 Sudoku-Extreme), 1,000 train / 200 test puzzles (paper: 423K test)
- Training: 140 epochs instead of 60K, batch size 128 instead of 768, to fit within Kaggle GPU time limits
- N_sup = 8 supervision steps instead of 16, T = 2 instead of 3, n = 4 instead of 6, to reduce memory and runtime
- EMA decay 0.999 (same as paper), stablemax loss (same as paper)
- Self-attention architecture used (not MLP-Mixer), since it generalizes better across tasks and is simpler to implement

### What the Notebook Result Shows

On 200 held-out 4×4 Sudoku puzzles, the 530K-parameter TRM achieves 73.2% per-cell accuracy (vs 24.3% random baseline) after 140 epochs. Accuracy improves across supervision steps from 71.4% at step 1 to 73.5% at step 5, validating the recursive reasoning mechanism. Full-puzzle solve rate is 0% (all 16 cells correct is hard at this scale). Results are not comparable to the paper's 87.4% because we use 4×4 instead of 9×9 Sudoku, 430× fewer epochs, a 6× smaller batch size, and reduced recursion depth (T=2, n=4 vs T=3, n=6).

## Influence

TRM is a very recent paper (October 2025) and has already garnered significant attention, with about 221 citations on [OpenAlex](https://openalex.org/works/W4404948530) (checked November 2026). It directly improves on HRM (Wang et al. 2025) and has inspired unofficial implementations, including one by [lucidrains](https://github.com/lucidrains/tiny-recursive-model) (192 stars). The paper's finding that 2-layer networks outperform deeper ones on small-data reasoning tasks challenges conventional scaling assumptions and may influence future research on parameter-efficient reasoning architectures.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/067_tiny-recursive-model-reasoning/solution.ipynb) | From-scratch PyTorch TRM trained on synthetic 4×4 Sudoku puzzles with recursive reasoning and deep supervision |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
