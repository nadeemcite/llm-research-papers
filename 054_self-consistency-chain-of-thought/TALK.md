---
title: "Self-Consistency: Talk & Q&A"
parent: "054 Self-Consistency"
nav_order: 2
---
# TALK — Self-Consistency Improves Chain of Thought Reasoning in Language Models (Self-Consistency)

## Talks, Slides and Press Coverage

1. **Google Research Blog — "Language Models Perform Reasoning via Chain of Thought" (Jason Wei and Denny Zhou, 16 May 2022):** This blog post by two of the self-consistency co-authors introduces chain-of-thought prompting and explicitly mentions the self-consistency follow-up: "follow-up work on self-consistency shows that the performance of chain of thought prompting can be improved further by taking a majority vote over multiple sampled reasoning paths." ([link](https://research.google/blog/language-models-perform-reasoning-via-chain-of-thought/))

2. **ICLR 2023 — "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (poster):** The paper was accepted to ICLR 2023 and presented as a poster. The work originated at Google Research, Brain Team. ([arXiv page](https://arxiv.org/abs/2203.11171))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2203.11171).

**Q1: Why does sampling multiple reasoning paths help over greedy decoding?**
A1 (from the paper, Section 2): "Since language models are not perfect reasoners, the model might also produce an incorrect reasoning path or make a mistake in one of the reasoning steps... but such solutions are less likely to arrive at the same answer. That is, we hypothesize that correct reasoning processes, even if they are diverse, tend to have greater agreement in their final answer than incorrect processes."

**Q2: How is the final answer selected from multiple sampled paths?**
A2 (from the paper, Section 2): "self-consistency applies a marginalization over rᵢ by taking a majority vote over aᵢ, i.e., arg max_a Σᵢ 1(aᵢ = a), or as we defined as the most 'consistent' answer among the final answer set." The paper also shows (Table 1) that majority vote performs nearly identically to normalized weighted sum, because "the normalized conditional probabilities P(rᵢ, aᵢ \| prompt, question) are quite close to each other."

**Q3: Is self-consistency sensitive to the sampling strategy?**
A3 (from the paper, Section 3.5): "Self-consistency is compatible with most existing sampling algorithms, including temperature sampling, top-k sampling, and nucleus sampling." Figure 4 shows the method is robust to various T, k, and p values on PaLM-540B GSM8K.

**Q4: Does self-consistency require training or fine-tuning?**
A4 (from the paper, Section 1): "Self-consistency is far simpler than prior approaches that either train an additional verifier or train a re-ranker given additional human annotations... Instead, self-consistency is entirely unsupervised, works off-the-shelf with pre-trained language models, requires no additional human annotation, and avoids any additional training, auxiliary models or fine-tuning."

**Q5: Does self-consistency work with zero-shot CoT (not just few-shot)?**
A5 (from the paper, Table 8): Yes. When combined with zero-shot CoT (Kojima et al., 2022) on PaLM-540B for GSM8K, self-consistency improved accuracy from 43.0% to 69.2% (+26.2%).

**Q6: How does model scale affect self-consistency gains?**
A6 (from the paper, Section 3.5, Figure 4 right): "Self-consistency robustly improves performance across all scales." However, "the gain is relatively lower for smaller models due to certain abilities (e.g., arithmetic) only emerge when the model reaches a sufficient scale." The paper found +3-6% gains for UL2-20B but +9-23% for LaMDA-137B and GPT-3.

## Common Misconceptions

1. **"Self-consistency is the same as an ensemble of multiple models."** — The paper explicitly distinguishes the two (Section 1): "self-consistency also differs from a typical ensemble approach where multiple models are trained and the outputs from each model are aggregated, it acts more like a 'self-ensemble' that works on top of a single language model." Table 7 shows that multi-model ensembles achieve much smaller gains than self-consistency on a single model.

2. **"Self-consistency can be applied to any text generation task."** — The paper notes a limitation (Section 2): "self-consistency can be applied only to problems where the final answer is from a fixed answer set, but in principle this approach can be extended to open-text generation problems if a good metric of consistency can be defined between multiple generations."

3. **"Majority voting is inferior to weighted probability aggregation."** — Table 1 shows that simple unweighted majority vote (74.4% on GSM8K with PaLM-540B) performs nearly identically to normalized weighted sum (74.1%), and the paper adopted majority vote as the default due to its simplicity.

## Real Citations

1. Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou (2022). "Self-Consistency Improves Chain of Thought Reasoning in Language Models." *ICLR 2023.* [arXiv:2203.11171](https://arxiv.org/abs/2203.11171) — the original paper.
2. Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) — the few-shot CoT prompting method that self-consistency builds upon.
3. Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, Yusuke Iwasawa (2022). "Large Language Models are Zero-Shot Reasoners." [arXiv:2205.11916](https://arxiv.org/abs/2205.11916) — zero-shot CoT, which was combined with self-consistency in Table 8.
4. Shunyu Yao, Dian Yu, Jeffrey Zhao, et al. (2023). "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." [arXiv:2305.10601](https://arxiv.org/abs/2305.10601) — extends self-consistency's multi-path idea with tree-structured search over reasoning states.

**Citation count:** about 2,236 citations on [OpenAlex](https://openalex.org/W4221161695) (checked November 2026).
