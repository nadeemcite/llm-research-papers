---
title: "YaRN: Talk & Q&A"
parent: "008 YaRN"
nav_order: 2
---
# TALK — YaRN: Efficient Context Window Extension of Large Language Models (YaRN)

## Talks, Slides and Press Coverage

1. **Hugging Face — Paper page for YaRN:** Hugging Face hosts the paper page with community discussion, where YaRN was the #1 paper of the day on September 3, 2023. ([link](https://huggingface.co/papers/2309.00071))
2. **Nous Research — YaRN GitHub repository:** The official implementation and training code, released alongside the paper. ([link](https://github.com/jquesnelle/yarn))
3. **Reddit r/LocalLLaMA — "NTK-Aware Scaled RoPE" thread (bloc97, June 2023):** The original community thread that led to the NTK-aware method, which YaRN formalized and improved. ([link](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2309.00071).

**Q1: Why does YaRN need a temperature scaling on top of NTK-by-parts?**
A1 (from the paper, Section 3.3): "we also observe that introducing a temperature t on the logits before the attention softmax has a uniform impact on perplexity regardless of the data sample and the token position over the extended context window." The temperature compensates for increased attention entropy when the context is extended.

**Q2: How much training data does YaRN need compared to Position Interpolation?**
A2 (from the paper, Abstract): "requiring 10x less tokens and 2.5x less training steps than previous methods."

**Q3: Can YaRN extrapolate beyond the fine-tuning context?**
A3 (from the paper, Section 4.2): The s=32 model "was only further trained for 200 steps using the s = 16 checkpoint with 64k data and is able to extrapolate to 128k context." This is the "train short, test long" capability — trained on 64K data but tested at 128K.

**Q4: Why does YaRN work with Flash Attention 2?**
A4 (from the paper, Section 3.3): "YaRN can effectively alter the attention mechanism without modifying its code. Furthermore, it has zero overhead during both inference and training, as rotary position embeddings are generated in advance and are reused for all forward passes."

**Q5: What are the recommended α and β values?**
A5 (from the paper, Section 3.2): "for the Llama family of models, good values for α and β are α = 1 and β = 32."

## Common Misconceptions

1. **"YaRN is just NTK-by-parts with a different name."** — No. YaRN combines NTK-by-parts interpolation with an attention temperature correction (t = 0.1·ln(s) + 1). The temperature is the novel contribution; NTK-by-parts was an earlier method that the paper formalized. The ablation in this notebook shows the temperature provides 6.6% perplexity improvement at 4× scale.

2. **"YaRN requires retraining from scratch."** — No. YaRN is applied to existing pretrained models and requires only a small fine-tuning (400 steps on PG19 in the paper). The Dynamic-YaRN variant works with zero fine-tuning at all.

3. **"PI is always worse than YaRN."** — PI performs well at small scale factors (s ≤ 4) and after fine-tuning at the matching context. Its weakness is at large scale factors (s ≥ 8) where it degrades, and in the zero-shot setting. The paper recommends YaRN as a drop-in replacement with "no downsides," but PI remains a valid baseline.

## Real Citations

1. Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole (2023). "YaRN: Efficient Context Window Extension of Large Language Models." [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) — the original paper.
2. Shouyuan Chen, Sherman Wong, Liangjian Chen, Yuandong Tian (2023). "Extending Context Window of Large Language Models via Positional Interpolation." [arXiv:2306.15595](https://arxiv.org/abs/2306.15595) — the PI method that YaRN builds upon and improves.
3. Jianlin Su, Yu Lu, Shengfeng Pan, et al. (2022). "RoFormer: Enhanced Transformer with Rotary Position Embedding." [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) — the RoPE method that is the foundation of all methods discussed.

**Citation count:** about 350 citations on [OpenAlex](https://openalex.org/works/W4390027983) (checked October 2026).
