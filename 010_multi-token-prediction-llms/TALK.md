---
title: "Multi-token Pred: Talk & Q&A"
parent: "010 Multi-token Pred"
nav_order: 2
---
# TALK — Better & Faster Large Language Models via Multi-token Prediction (Multi-token Pred)

## Talks, Slides and Press Coverage

1. **ICML 2024 — Conference presentation:** The paper was accepted at ICML 2024 and presented as a conference talk. The paper is listed in the ICML 2024 proceedings. ([link](https://icml.cc/virtual/2024/calendar))
2. **Hugging Face Daily Papers — "Better & Faster Large Language Models via Multi-token Prediction":** The paper was featured as a daily paper on Hugging Face, receiving community upvotes and discussion. ([link](https://huggingface.co/papers/2404.19737))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2404.19737).

**Q1: Why does multi-token prediction help with sample efficiency?**
A1 (from the paper, Section 5.1): "Multi-token prediction implicitly assigns weights to training tokens depending on how closely they are correlated with their successors." Not all tokens are equally important — some are "choice points" that determine the direction of the text. Multi-token prediction gives these consequential tokens higher implicit weight in the loss because their consequences (the future tokens) are also hard to predict.

**Q2: Why does the benefit only appear at larger model scales?**
A2 (from the paper, Section 3.1): "We believe this usefulness only at scale to be a likely reason why multi-token prediction has so far been largely overlooked as a promising training loss for large language model training." Figure 3 shows that multi-token prediction models are worse than the baseline for small model sizes (300M) but outperform the baseline at larger sizes (3B+).

**Q3: How does the memory-efficient implementation work?**
A3 (from the paper, Section 2, "Memory-efficient implementation"): "After the forward pass through the shared trunk f_s, we sequentially compute the forward and backward pass of each independent output head f_i, accumulating gradients at the trunk. While this creates logits (and their gradients) for the output head f_i, these are freed before continuing to the next output head f_{i+1}." This reduces peak GPU memory from O(nV + d) to O(V + d).

**Q4: What is self-speculative decoding and how is it different from regular speculative decoding?**
A4 (from the paper, Section 3.2): The additional output heads can be used for "self-speculative decoding methods such as blockwise parallel decoding (Stern et al., 2018) — a variant of speculative decoding (Leviathan et al., 2023) without the need for an additional draft model." The model uses its own extra heads as the "draft" model, avoiding the need to train and run a separate smaller model.

**Q5: What is the optimal number of tokens to predict?**
A5 (from the paper, Section 3.4): "Training with 4-future tokens outperforms all the other models consistently throughout HumanEval and MBPP for pass at 1, 10 and 100 metrics." For byte-level models, n = 8 is optimal.

## Common Misconceptions

1. **"Multi-token prediction slows down training because of the extra heads"** — No. The paper explicitly states "no overhead in training time" because the shared trunk does one forward pass, and the heads are processed sequentially during the backward pass, so there is no additional forward computation for the trunk. The memory-efficient implementation ensures no GPU memory penalty either.

2. **"Multi-token prediction always improves over next-token prediction"** — Not at small scales. The paper clearly shows (Figure 3, Section 3.1) that multi-token prediction is worse than the baseline for small model sizes (300M parameters) and only outperforms at larger scales (3B+). This is likely why the method was overlooked previously — early experiments at small scale would have shown no benefit.

3. **"The extra heads are only useful for speculative decoding"** — The heads serve a dual purpose. During training, they improve the shared trunk's representations by forcing it to encode information useful for multi-step-ahead prediction. Even when the extra heads are removed and the model is finetuned with next-token prediction, it still outperforms a model that was pretrained with next-token prediction alone (Section 3.6, Figure 4).

## Real Citations

1. Gloeckle, F., Youbi Idrissi, B., Rozière, B., Lopez-Paz, D., & Synnaeve, G. (2024). "Better & Faster Large Language Models via Multi-token Prediction." *ICML 2024.* [arXiv:2404.19737](https://arxiv.org/abs/2404.19737) — the original paper.
2. Stern, M., Shazeer, N., & Uszkoreit, J. (2018). "Blockwise Parallel Decoding for Deep Autoregressive Models." [arXiv:1811.03115](https://arxiv.org/abs/1811.03115) — the blockwise parallel decoding method used for self-speculative decoding.
3. Leviathan, Y., Kalman, M., & Matias, Y. (2023). "Fast Inference from Transformers via Speculative Decoding." *ICML 2023.* [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) — speculative decoding, which multi-token prediction enables without a separate draft model.
4. Cai, T. et al. (2024). "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads." [arXiv:2401.10774](https://arxiv.org/abs/2401.10774) — Medusa-style tree attention for speculative decoding, referenced by the paper.
5. Olsson, C. et al. (2022). "In-context Learning and Induction Heads." [arXiv:2209.11895](https://arxiv.org/abs/2209.11895) — induction heads, which multi-token prediction promotes in small models (Section 4.1).

**Citation count:** about 15 citations on [OpenAlex](https://openalex.org/W4396821494) (checked October 2026).
