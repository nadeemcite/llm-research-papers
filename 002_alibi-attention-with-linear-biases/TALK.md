---
title: "ALiBi: Talk & Q&A"
parent: "002 ALiBi"
nav_order: 2
---
# TALK — Train Short, Test Long: Attention with Linear Biases (ALiBi)

## Talks, Slides and Press Coverage

1. **OpenReview — ICLR 2022 submission and reviews:** The paper was submitted to and accepted at ICLR 2022, with public reviews and discussion available on OpenReview. ([link](https://openreview.net/forum?id=R8s-pp-TtIU))
2. **MosaicML — "MPT-7B" model announcement (2023):** MosaicML's MPT models used ALiBi for position encoding, enabling context lengths up to 65K tokens. The blog post describes the choice of ALiBi over other position methods. ([link](https://www.mosaicml.com/blog/mpt-7b))
3. **Papers with Code — "Train Short, Test Long" entry:** The paper's page on Papers with Code, linking to the official GitHub repository and results. ([link](https://paperswithcode.com/paper/train-short-test-long-attention-with-linear))
4. **Official code repository — GitHub (ofirpress/attention_with_linear_biases):** The authors' implementation, including the WikiText-103 and CC100+RoBERTa experiments. ([link](https://github.com/ofirpress/attention_with_linear_biases))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2108.12409).

**Q1: Why does ALiBi not use any position embeddings?**
A1 (from the paper, Section 3): "When using ALiBi, we do not add position embeddings at any point in the network. The only modification we apply is after the query-key dot product, where we add a static, non-learned bias."

**Q2: How are the ALiBi slopes chosen?**
A2 (from the paper, Section 3): "For our models with 8 heads, the slopes that we used are the geometric sequence: 1/2^1, 1/2^2, ..., 1/2^8. For models that require 16 heads, we interpolate those 8 slopes by geometrically averaging every consecutive pair... In general, for n heads, our set of slopes is the geometric sequence that starts at 2^(-8/n) and uses that same value as its ratio."

**Q3: Are the slopes learned during training?**
A3 (from the paper, Section 3): "We initially experimented with making the slopes trainable, but this did not yield strong extrapolation results." The slopes are fixed before training and never updated. The authors note that "this set of slopes works on a wide variety of text domains and model sizes" and that "we do not believe that it is necessary to tune these slope values every time a new model is trained on a new dataset."

**Q4: What is ALiBi's inductive bias?**
A4 (from the paper, Section 3): "ALiBi has an inductive bias towards recency; it penalizes attention scores between distant query-key pairs, with the penalty increasing as the distance between a key and a query grows. The different heads increase their penalties at different rates, depending on the slope magnitude."

**Q5: How far can ALiBi extrapolate?**
A5 (from the paper, Section 1 and Figure 6): "Though performance peaks at around two times the number of tokens that the model was trained on, ALiBi maintains strong performance even on sequences of length 10,000." The L=512 model achieves its best perplexity at 1012 tokens; the L=1024 model at twice its training length.

**Q6: Does ALiBi slow down training or add parameters?**
A6 (from the paper, Section 1 and Figure 2): "Compared to a sinusoidal model trained on the same input length, our method requires no additional runtime or parameters and incurs a negligible (0-0.7%) memory increase. ALiBi can be implemented by changing only a few lines of existing transformer code."

## Common Misconceptions

1. **"ALiBi is a relative position encoding."** — The paper describes it as a relative position method (Section 3: "Since ALiBi is a relative position method, we add position information at every layer to the keys and queries but not to the values"). But it is important to understand that ALiBi does not encode *positions* at all — it encodes *distances*. There is no learned or fixed embedding for position 1 vs position 100; there is only a linear penalty for distance \|j - i\|. This is why it extrapolates: distance is defined for any length.

2. **"ALiBi was widely adopted by major LLMs."** — While ALiBi influenced the field (MPT, BloombergGPT used it), RoPE (from the RoFormer paper) became the dominant choice for most large language models (LLaMA, Mistral, Qwen, DeepSeek). ALiBi's main impact was demonstrating that position representation is the key to extrapolation, which motivated later work on RoPE-based context extension (YaRN, Position Interpolation).

3. **"ALiBi works at any length without degradation."** — The paper shows that ALiBi's best performance is around 2L (twice the training length), not at arbitrary lengths. Beyond 2L, performance plateaus rather than improving (Figure 6). The paper notes: "When inference is performed on subsequences of length 2L+1 or longer, less than half of the predictions the model makes are on subsequences of lengths seen during training, and that might degrade performance."

## Real Citations

1. Ofir Press, Noah A. Smith, Mike Lewis (2021). "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." *ICLR 2022.* [arXiv:2108.12409](https://arxiv.org/abs/2108.12409) — the original paper.
2. MosaicML (2023). "MPT-7B." — adopted ALiBi for position encoding, enabling 65K-token context. [blog post](https://www.mosaicml.com/blog/mpt-7b)
3. Peng et al. (2023). "YaRN: Efficient Context Window Extension of Large Language Models." [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) — builds on the extrapolation problem ALiBi addressed, using a different approach (RoPE frequency scaling).

**Citation count:** about 161 citations on [OpenAlex](https://openalex.org/works/W3196318247) (checked October 2026).
