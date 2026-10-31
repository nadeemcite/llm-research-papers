---
title: "Muon: Talk & Q&A"
parent: "036 Muon"
nav_order: 2
---
# TALK — Muon is Scalable for LLM Training (Muon)

## Talks, Slides and Press Coverage

1. **Keller Jordan — "Muon: An optimizer for hidden layers in neural networks" (8 December 2024):** The original blog post that introduced Muon, describing its design, Newton-Schulz orthogonalization, and NanoGPT speedrun results. ([blog post](https://kellerjordan.github.io/posts/muon/))
2. **Keller Jordan — NanoGPT Speedrun leaderboard:** The ongoing community competition where Muon was first validated, training GPT-2-level models in record time on 8×H100 GPUs. ([leaderboard](https://github.com/KellerJordan/modded-nanogpt))
3. **Moonshot AI — "Moonlight" model release (February 2025):** Moonshot AI released the 3B/16B MoE model trained with Muon, along with intermediate checkpoints and their distributed Muon implementation. ([GitHub](https://github.com/MoonshotAI/Moonlight))
4. **Jeremy Bernstein and Laker Newhouse — "Old Optimizer, New Norm: An Anthology" (2024):** The theoretical framework connecting Muon to steepest descent under norm constraints, which the paper builds upon. ([arXiv:2409.20325](https://arxiv.org/abs/2409.20325))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2502.16982).

**Q1: Why does Muon need weight decay when the original version did not use it?**
A1 (from the paper, Section 2.2): "We found the performance gains diminish when we scale up to train a larger model with more tokens. We observed that both the weight and the layer output's RMS keep growing to a large scale, exceeding the high-precision range of bf16, which might hurt the model's performance." Adding weight decay "addressed this issue — the results demonstrate that Muon with weight decay outperforms both vanilla Muon and AdamW, achieving lower validation loss in the over-train regime."

**Q2: What is the update RMS problem and how is it fixed?**
A2 (from the paper, Section 2.2, Lemma 1): For a full-rank matrix of shape [A, B], "its theoretical Muon update RMS is sqrt(1/max(A, B))." This means large matrices get tiny updates and small matrices get huge updates. The fix is to "scale the Muon update for each matrix by its sqrt(max(A, B)) to cancel the effect," then multiply by 0.2 to match AdamW's empirical update RMS of 0.2–0.4.

**Q3: How much compute does Muon save compared to AdamW?**
A3 (from the paper, Section 1 and Figure 1a): "Muon only requires about 52% training FLOPs to match the performance of AdamW under compute-optimal setting." The fitted scaling laws are L = 2.506 × C^−0.052 for Muon vs L = 2.608 × C^−0.054 for AdamW.

**Q4: Can Muon be used for supervised fine-tuning (SFT) as well?**
A4 (from the paper, Section 3.5): "A model that is both Muon-pretrained and Muon-finetuned outperforms others in the ablation studies. However, we also observe that when the SFT optimizer differs from the pretraining optimizer, SFT with Muon does not show a significant advantage over AdamW." The authors note this "pretraining-finetuning mismatch" as an open problem.

**Q5: Does Muon work with non-matrix parameters?**
A5 (from the paper, Section 2.2): "Muon is designed to update matrix-based parameters. In practice, AdamW is used in couple with Muon to handle non-matrix based parameters, like RMSNorm, LM head, and embedding parameters." The authors propose extending Muon to all parameters as future work (Section 4).

**Q6: Why does Muon especially help on math and code tasks?**
A6 (from the paper, Section 3.3): "We observed that Muon especially excels on Math and Code related tasks, and we encourage the research community to further investigate this phenomena." The paper does not provide a definitive explanation, though the higher SVD entropy in router weights (Section 3.4) may contribute for MoE models.

## Common Misconceptions

1. **"Muon replaces AdamW entirely."** — No. Muon only optimizes 2D matrix parameters. Embeddings, layer norms, and the LM head still use AdamW. The paper describes this as a "hybrid approach" (Section 4) and notes extending Muon to all parameters as future work.

2. **"Muon is just Shampoo with a different name."** — While both are matrix-aware optimizers, they differ fundamentally. Shampoo uses the matrix's inverse fourth root as a preconditioner; Muon orthogonalizes the update (sets all singular values to 1). The paper discusses this connection via Bernstein & Newhouse (2024)'s norm-constraint framework: Muon corresponds to the spectral norm constraint, while AdamW corresponds to a dynamically adjusted Max-of-Max norm.

3. **"The 2× efficiency means Muon trains 2× faster in wall-clock time."** — The 2× refers to compute efficiency (FLOPs), not wall-clock time. Muon has higher per-step latency due to Newton-Schulz iterations and additional communication in distributed settings. The paper notes this overhead is "1% to 3%" of the forward-backward pass time (Section 2.3), but it is not zero.

4. **"More Newton-Schulz iterations are always better."** — The paper explicitly tested this: "when setting N to 10, the iterative process will yield a more accurate orthogonalization result than N=5, but it won't lead to better performances. Hence we set N=5 in this work for the sake of efficiency" (Section 2.2).

## Real Citations

1. Jingyuan Liu, Jianlin Su, Xingcheng Yao, et al. (2025). "Muon is Scalable for LLM Training." *arXiv preprint.* [arXiv:2502.16982](https://arxiv.org/abs/2502.16982) — the original paper.
2. Keller Jordan (2024). "Muon: An optimizer for hidden layers in neural networks." [blog post](https://kellerjordan.github.io/posts/muon/) — the original Muon optimizer, which this paper scales up.
3. Jeremy Bernstein and Laker Newhouse (2024). "Old Optimizer, New Norm: An Anthology." [arXiv:2409.20325](https://arxiv.org/abs/2409.20325) — the theoretical framework connecting Muon to steepest descent under spectral norm constraints.
4. Thomas Pethick et al. (2025). "Training Deep Learning Models with Norm-Constrained LMOs." [arXiv:2502.07529](https://arxiv.org/abs/2502.07529) — concurrently discussed the update RMS scaling issue.

**Citation count:** about 35 citations on [OpenAlex](https://openalex.org/works/W4404946960) (checked October 2026).
