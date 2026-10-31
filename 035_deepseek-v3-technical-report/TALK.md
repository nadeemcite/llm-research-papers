---
title: "DeepSeek-V3: Talk & Q&A"
parent: "035 DeepSeek-V3"
nav_order: 2
---
# TALK — DeepSeek-V3 Technical Report (DeepSeek-V3)

## Talks, Slides and Press Coverage

1. **Simon Willison's Weblog — "DeepSeek-V3" (Simon Willison, 26 Dec 2024):** Willison highlights the training cost ($5.58M in H800 GPU-hours, ~11× less than Llama 3 405B) and benchmarks comparable to Claude 3.5 Sonnet, calling it "a highly impressive display of research and engineering under resource constraints." ([link](https://simonwillison.net/2024/Dec/26/deepseek-v3/))
2. **Hugging Face — DeepSeek-V3 model page:** Hosts the open-weight model with model card, tokenizer, and usage instructions for the 671B-parameter MoE model. ([link](https://huggingface.co/deepseek-ai/DeepSeek-V3))
3. **DeepSeek GitHub — "DeepSeek-V3" repository:** Official source for model weights, inference scripts, and the technical report PDF. ([link](https://github.com/deepseek-ai/DeepSeek-V3))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2412.19437).

**Q1: Why does the auxiliary-loss-free strategy perform better than conventional auxiliary loss?**
A1 (from the paper, Section 2.1.2): The conventional auxiliary loss "will impair the model performance" when too large. The aux-free strategy avoids this by using a bias term that is "only used for routing" — "the gating value, which will be multiplied with the FFN output, is still derived from the original affinity score." Since the bias does not enter the gradient, it balances load without interfering with the model's learning objective.

**Q2: How much did DeepSeek-V3 cost to train?**
A2 (from the paper, Abstract): "DeepSeek-V3 requires only 2.788M H800 GPU hours for its full training." Table 1 breaks this down: 2,664K hours for pre-training, 119K for context extension, 5K for post-training. At $2/H800-hour, the total is approximately $5.58 million.

**Q3: Was the training stable?**
A3 (from the paper, Abstract): "Throughout the entire training process, we did not experience any irrecoverable loss spikes or perform any rollbacks."

**Q4: How does DeepSeek-V3 compare to closed-source models?**
A4 (from the paper, Section 4.4): "DeepSeek-V3 outperforms all other open-source models, achieving 88.5 on MMLU, 75.9 on MMLU-Pro, and 59.1 on GPQA." The chat version "achieves performance comparable to leading closed-source models, including GPT-4o and Claude-3.5-Sonnet, on a series of standard and open-ended benchmarks."

**Q5: What is Multi-Token Prediction and why use it?**
A5 (from the paper, Section 2.2): "DeepSeek-V3 employs a multi-token prediction training objective, which we have observed to enhance the overall performance on evaluation benchmarks." The MTP depth D is set to 1, meaning each token predicts the next token and one additional token. "It can also be used for speculative decoding for inference acceleration."

## Common Misconceptions

1. **"DeepSeek-V3 is just a bigger DeepSeek-V2."** — While V3 inherits MLA and DeepSeekMoE from V2, the auxiliary-loss-free load balancing (Section 2.1.2) and Multi-Token Prediction (Section 2.2) are new contributions. V3 also uses sigmoid gating (V2 used softmax), and introduces FP8 mixed precision training at scale.

2. **"The aux-free strategy completely replaces auxiliary loss."** — The paper still uses a complementary sequence-wise auxiliary loss (Eq. 17) alongside the bias-tuning strategy, "to prevent extreme imbalance within any single sequence." The aux-free strategy is the primary mechanism, but the conventional loss is kept at a much smaller weight as a safety net.

3. **"DeepSeek-V3 doesn't drop tokens during training."** — This is actually correct, and the paper explicitly states it (Section 2.1.2): "Due to the effective load balancing strategy, DeepSeek-V3 keeps a good load balance during its full training. Therefore, DeepSeek-V3 does not drop any tokens during training." Token dropping is a common technique in other MoE models to handle overflow, but V3's load balancing makes it unnecessary.

## Real Citations

1. DeepSeek-AI (2024). "DeepSeek-V3 Technical Report." *arXiv preprint.* [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) — the original paper.
2. DeepSeek-AI (2024). "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) — the predecessor that introduced MLA and DeepSeekMoE, which V3 inherits.
3. Wang et al. (2024). "Auxiliary-Loss-Free Load Balancing for Mixture-of-Experts Models." [arXiv:2404.14831](https://arxiv.org/abs/2404.14831) — the standalone paper describing the aux-free strategy that V3 pioneers in a large-scale model.
4. Dai et al. (2024). "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models." [arXiv:2401.06066](https://arxiv.org/abs/2401.06066) — the fine-grained MoE architecture V3 adopts.

**Citation count:** about 293 citations on [OpenAlex](https://openalex.org/works/W4405903187) (checked October 2026).
