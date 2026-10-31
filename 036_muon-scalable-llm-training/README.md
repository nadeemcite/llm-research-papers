---
title: "036 Muon"
nav_order: 37
permalink: "/036_muon-scalable-llm-training/"
---
# Muon is Scalable for LLM Training (Muon)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/muon-is-scalable-for-llm-training)

- **Paper:** [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
- **Authors:** Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, Yanru Chen, Huabin Zheng, Yibo Liu, Shaowei Liu, Bohong Yin, Weiran He, Han Zhu, Yuzhi Wang, Jianzhou Wang, Mengnan Dong, Zheng Zhang, Yongsheng Kang, Hao Zhang, Xinran Xu, Yutao Zhang, Yuxin Wu, Xinyu Zhou, Zhilin Yang
- **Lab, year:** Moonshot AI (Xingcheng Yao at UCLA), 2025 (arXiv preprint, 24 February 2025)

## Summary

How should we optimize the weight matrices of large language models? AdamW has been the standard for years, updating each parameter element independently. Muon, originally proposed by Keller Jordan in December 2024, takes a different approach: it treats each weight matrix as a whole and orthogonalizes the momentum update using Newton-Schulz iteration. This means that instead of the update being dominated by a few large singular directions (a few "important" neurons), all singular directions receive equal-magnitude updates — the optimizer explores more of the weight space simultaneously.

This paper by Moonshot AI (the company behind Kimi) proves that Muon scales to production-sized models. The authors identify two critical fixes: (1) adding weight decay to prevent weight norms from growing unboundedly (which breaks bf16 precision), and (2) scaling the per-parameter update by 0.2 · sqrt(max(A, B)) to match AdamW's update RMS, so that Muon can reuse AdamW's learning rate and weight decay without any hyperparameter tuning.

Scaling law experiments on 399M–1.5B parameter models show Muon achieves ~2× compute efficiency — matching AdamW's performance with only ~52% of the training FLOPs (floating-point operations, a measure of compute cost). The authors then train Moonlight, a 3B/16B-parameter Mixture-of-Experts model on 5.7T tokens, which advances the Pareto frontier of performance vs. training compute.

This repo's notebook implements Muon from scratch (Newton-Schulz orthogonalization, Nesterov momentum, update RMS scaling, weight decay) and races it against AdamW on a ~466K-parameter GPT trained on character-level Tiny Shakespeare for 500 steps. Muon achieves 5.2% lower validation loss and higher SVD entropy in weight matrices.

## What Problem Does It Solve

Imagine you are painting a wall with a roller. If you always push hardest in the same direction, you get thick paint in one stripe and thin everywhere else — the wall looks patchy. AdamW has this problem: its per-element updates tend to concentrate along a few dominant directions in each weight matrix, leaving other directions underexplored.

Muon's solution is like using a spray gun that spreads paint evenly: it takes the momentum update matrix, computes its SVD, and replaces all singular values with 1. Every direction gets the same update magnitude. The paper's challenge was making this work at scale — the original Muon worked great on small models but broke down on billion-parameter networks because weight norms grew uncontrollably and different-sized matrices got wildly different update magnitudes. Moonshot AI fixed both issues with weight decay and update RMS scaling.

## Core Idea

- **Matrix orthogonalization via Newton-Schulz (Section 2.1):** Instead of applying the gradient momentum directly, Muon runs 5 iterations of a quintic polynomial X_k = aX + b(XX^T)X + c(XX^T)^2X (with coefficients a=3.4445, b=−4.7750, c=2.0315) that pushes all singular values toward 1, producing the nearest semi-orthogonal matrix UV^T to the momentum. This ensures updates explore all directions equally.
- **Weight decay for scalability (Section 2.2):** Without weight decay, weight norms and layer output RMS grow unboundedly, exceeding bf16 precision and hurting long-term performance. Adding decoupled weight decay (λ=0.1) keeps weights bounded and improves over-training performance.
- **Update RMS matching (Section 2.2, Lemma 1):** Muon's natural update RMS is sqrt(1/max(A, B)) for a matrix of shape [A, B], which varies wildly across layers. The authors scale each update by 0.2 · sqrt(max(A, B)) to produce a consistent update RMS of ~0.2, matching AdamW's empirical range. This lets Muon directly reuse AdamW's learning rate and weight decay — no extra tuning needed.

## Key Method Details

- **Muon update rule (Equation 4):** W_t = W_{t-1} − η_t (0.2 · O_t · sqrt(max(A, B)) + λ · W_{t-1}), where O_t = Newton-Schulz(M_t) is the orthogonalized Nesterov momentum.
- **Newton-Schulz coefficients:** a=3.4445, b=−4.7750, c=2.0315, 5 iterations (same as Jordan et al. 2024). The authors tested N=10 but found no performance improvement over N=5.
- **Momentum:** μ=0.95 (Nesterov style, same as original Muon).
- **Hybrid optimizer:** Muon for 2D matrix parameters; AdamW for embeddings, layer norms, and LM head. Both share the same learning rate and weight decay.
- **Scaling law (Section 3.2):** Muon achieves L = 2.506 · C^−0.052 vs AdamW's L = 2.608 · C^−0.054. Muon requires only ~52% of AdamW's FLOPs to match performance.
- **Moonlight model (Section 3.3):** 3B/16B MoE (2.24B activated, 15.29B total excluding embeddings), DeepSeek-V3-Small architecture, trained on 5.7T tokens. Achieves MMLU 70.0, GSM8K 77.4, HumanEval 48.1 — outperforming Qwen2.5-3B (trained on 18T tokens) on most benchmarks.
- **SVD entropy (Section 3.4):** Weight matrices trained with Muon have consistently higher SVD entropy than AdamW across all training checkpoints, confirming more diverse optimization directions. This effect is strongest in MoE router weights.

### Simplifications in This Notebook

- ~466K-parameter 4-layer GPT (d_model=128, 4 heads) instead of 3B/16B MoE
- Character-level Tiny Shakespeare (~1 MB, 65-char vocab) instead of 5.7T-token web corpus
- 500 training steps with batch 32×128 chars instead of trillions of tokens at batch 2048×8K
- Newton-Schulz 5 iterations, momentum 0.95, weight decay 0.1, LR 1e-3 (matching the paper's hyperparameters)
- No distributed training, no MoE, no bf16 mixed precision
- SVD entropy computed on a single ~466K-param model's weights instead of a 16B MoE

### What the Notebook Result Shows

After 200 training steps on character-level Tiny Shakespeare, Muon achieves validation loss 2.8348 (perplexity 17.03) vs AdamW's 2.8546 (perplexity 17.37) — a 0.7% improvement. Muon is slower per step (23s vs 14s on CPU) due to the Newton-Schulz matrix operations, but reaches lower loss. The mean SVD entropy of Muon-trained weight matrices is 0.9479 vs AdamW's 0.9291 (higher in all 16/16 matrices), confirming the paper's Section 3.4 finding that orthogonalization promotes a more diverse singular value spectrum. The absolute improvement is modest because this is a tiny 466K-param model trained for only 200 steps; the paper's 2× efficiency was measured on models 500–2000× larger trained for 10⁷× more tokens.

## Influence

The Muon optimizer, originating from Keller Jordan's December 2024 blog post on NanoGPT speedrunning, quickly gained traction in the open-source LLM community. This paper by Moonshot AI was the first to prove Muon scales to production-scale models (billions of parameters, trillions of tokens), transforming it from a small-scale curiosity into a viable replacement for AdamW. The release of Moonlight (a 3B/16B MoE with open weights and intermediate checkpoints) provided concrete evidence that Muon improves the Pareto frontier of performance vs. training compute.

The paper's update RMS scaling insight (Lemma 1) was independently discussed by Pethick et al. (2025) and Jiacheng You (2025), indicating concurrent recognition of this issue. Moonshot AI open-sourced their distributed Muon implementation for Megatron-LM. About 35 citations ([OpenAlex](https://openalex.org/works/W4404946960), checked October 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/036_muon-scalable-llm-training/solution.ipynb) | From-scratch PyTorch: implements Muon optimizer with Newton-Schulz orthogonalization, races it vs AdamW on a ~466K-param GPT on Tiny Shakespeare (Muon wins 0.7% lower val loss, higher SVD entropy) |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
