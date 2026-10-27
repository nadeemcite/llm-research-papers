---
title: "022 Titans"
nav_order: 23
permalink: "/022_titans-learning-to-memorize-test-time/"
---
# Titans: Learning to Memorize at Test Time (Titans)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/titans-learning-to-memorize-at-test-time)

- **Paper:** [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)
- **Authors:** Ali Behrouz, Peilin Zhong, Vahab Mirrokni
- **Lab, year:** Google Research, 2024 (arXiv preprint, December 2024)

## Summary

Transformers excel at sequence modeling because attention captures direct dependencies between all tokens in the context window. But this comes at a quadratic cost: doubling the context length quadruples the memory and computation. Linear recurrent models (like Mamba and DeltaNet) compress the past into a fixed-size state, enabling efficient inference, but lose information because a small matrix cannot faithfully store very long histories. Titans bridge this gap with a **neural long-term memory module** — a small MLP that learns to memorise data at test time by treating each incoming token as a training example.

The key insight is a **surprise metric**: the memory module updates its weights using the gradient of an associative memory loss with respect to the current input. When an input is surprising (high gradient), the memory updates more aggressively, mimicking how surprising events are more memorable in human cognition. The update rule combines momentum (past surprise) with weight decay (a forgetting gate), which the authors show generalises the gating mechanisms in modern RNNs (recurrent neural networks) like Mamba2. Titans also include **persistent memory** (learnable, data-independent parameters that encode task knowledge) and a **core attention module** (short-term memory for the current context window).

The authors propose three architectural variants — Memory as a Context (MAC), Memory as a Gate (MAG), and Memory as a Layer (MAL) — each combining the long-term memory with sliding-window attention differently. At 760M parameters trained on 30B tokens, Titans (MAG) achieves average accuracy of 52.50 on commonsense reasoning benchmarks, outperforming Transformer++ (48.69) and Gated DeltaNet (49.69). Titans scale to over 2M token context windows with higher accuracy on needle-in-haystack tasks than all baselines. This repo's notebook implements the neural long-term memory module from scratch and compares it against linear attention and full attention on associative recall.

## What Problem Does It Solve

Imagine you are at a party that lasts for hours. You cannot remember every conversation word-for-word (that would be like a Transformer — perfect recall but your brain would overflow). Instead, you naturally remember the surprising moments — the unexpected joke, the shocking news — and let the routine small talk fade. This is exactly how Titans work: the memory module pays more attention to surprising inputs (measured by gradient magnitude) and forgets the rest.

The problem with existing efficient models is that they compress all past information uniformly. A fixed-size matrix state, whether in Mamba or a linear Transformer, treats all tokens equally — there is no mechanism to say "this token is surprising, remember it more" or "this topic is over, forget the old context." Titans solve this by making the memory update data-dependent through the surprise metric, and by using a deep (multi-layer) memory module that can store non-linear abstractions of the past, not just linear ones.

## Core Idea

- **Surprise-based memory update (Section 3.1):** The memory module M is a small MLP whose weights are updated at test time using gradient descent on an associative memory loss. The gradient measures how "surprising" the current input is — surprising inputs cause larger updates, mimicking human memory.
- **Momentum + weight decay (Eq. 13-14):** The update rule combines past surprise (momentum S_t) with an adaptive forgetting gate (alpha_t), which the authors prove generalises the gating in Mamba2, GLA, and DeltaNet. This gives the memory both persistence and the ability to forget.
- **Three memory architectures (Section 4):** MAC (memory as context prepended to attention), MAG (memory combined with sliding-window attention via a gate), and MAL (memory as a layer before attention). MAC handles longest contexts best; MAG is most efficient.

## Key Method Details

- **Memory architecture:** MLP with L_M >= 1 layers; deeper memory (L_M = 2-4) is more expressive and scales better to long sequences (Section 5.5)
- **Associative memory loss (Eq. 12):** ℓ(M_{t-1}; x_t) = \|\|M_{t-1}(k_t) - v_t\|\|^2, where k_t = x_t W_K and v_t = x_t W_V
- **Update rule (Eq. 13-14):** M_t = (1 - α_t) M_{t-1} + S_t, where S_t = η_t S_{t-1} - θ_t ∇ℓ(M_{t-1}; x_t); α_t is the forgetting gate, η_t is surprise decay, θ_t is the learning rate — all data-dependent
- **Parallel training (Section 3.2):** The sequential update is reformulated using mini-batch gradient descent with chunked matmuls and parallel associative scan for the momentum term, enabling O(N) training with GPU-friendly operations
- **Results (760M / 30B tokens):** Titans (MAC) avg reasoning accuracy 52.51, WikiText perplexity 19.93; Titans (MAG) 52.50, 18.61. S-NIAH accuracy at 16K: MAC 95.2-98.4 vs Mamba2 0-5.4. BABILong: Titans outperform GPT-4 with ~70x fewer parameters

### Simplifications in This Notebook

- Toy MLP memory with 1-2 layers and d_model=64 (paper uses deeper memory with d_model up to 4096)
- Synthetic associative recall task with vocab size 16, seq len 64 (paper uses FineWeb-Edu with 30B tokens)
- 20 epochs, batch 32, lr 1e-3 with AdamW (paper uses AdamW lr 4e-4 with cosine annealing, batch 0.5M tokens)
- Sequential (non-parallelized) memory update for clarity (paper uses chunkwise parallel algorithm)
- No persistent memory, no sliding window attention, no convolution — pure LMM module only (paper combines all three)

### What the Notebook Result Shows

On a synthetic multi-query associative recall task (sequence length 64, 16-token vocab, 4 key-value pairs, 2000 training examples, 20 epochs), the neural long-term memory (LMM) module achieves validation accuracy of 0.97, compared to 0.26 for linear attention and 1.00 for full attention. The LMM with momentum and weight decay significantly outperforms plain linear attention, confirming the paper's ablation finding (Table 5: removing weight decay drops long-context accuracy from 92.68 to 85.60, removing momentum drops it to 87.12).

The random baseline is 1/16 = 0.0625. These toy results are not comparable to the paper's 760M-parameter benchmarks — the advantage of deep neural memory is most pronounced at scale with real-world long contexts.

## Influence

Titans (December 2024) introduced the concept of a neural long-term memory that learns at test time via gradient-based surprise, directly inspiring follow-up work such as ATLAS (2025), which extends the memorization framework with optimal context compression. The surprise-based update mechanism influenced the broader trend of test-time training in sequence models, connecting to TTT (Test-Time Training) and fast-weight program literature. The paper has about 470 citations on [Google Scholar](https://scholar.google.com/scholar?q=%22Titans+Learning+to+Memorize+at+Test+Time%22) (checked October 2026) and 23 on [OpenAlex](https://openalex.org/W4406032263). Multiple unofficial PyTorch implementations exist on GitHub, including [Aedelon/titans-pytorch-mlx](https://github.com/Aedelon/titans-pytorch-mlx) with 33 stars. Google has not released an official code repository as of October 2026.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/022_titans-learning-to-memorize-test-time/solution.ipynb) | From-scratch PyTorch implementation of the Titans neural long-term memory module (surprise-based gradient updates with momentum and weight decay), compared against linear attention and full attention on synthetic associative recall (LMM 0.97, linear-attn 0.26, full-attn 1.00) |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
