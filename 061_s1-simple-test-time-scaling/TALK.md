---
title: "s1: Talk & Q&A"
parent: "061 s1"
nav_order: 2
---
# TALK — s1: Simple test-time scaling (s1)

## Talks, Slides and Press Coverage

1. **simplescaling/s1 — GitHub repository (Muennighoff et al., Jan 2025):** The official open-source release of s1, including the s1K dataset, training code, and evaluation scripts. The repository README documents the budget forcing method and provides reproduction instructions. ([GitHub](https://github.com/simplescaling/s1))
2. **Hugging Face — "s1" model page:** The s1-32B model weights are publicly available on Hugging Face, with a model card describing the training procedure and benchmark results. ([Hugging Face](https://huggingface.co/simplescaling/s1-32B))
3. **Niklas Muennighoff — "s1: Simple test-time scaling" (EMNLP 2025):** The paper was accepted to the main conference track of EMNLP 2025, published in the proceedings with DOI 10.18653/v1/2025.emnlp-main.1025. ([ACL Anthology](https://aclanthology.org/2025.emnlp-main.1025/))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2501.19393).

**Q1: Why does supervised fine-tuning on just 1,000 samples lead to such large performance gains?**
A1 (from the paper, Section 6.1): "We hypothesize that the model is already exposed to large amounts of reasoning data during pretraining which spans trillions of tokens. Thus, the ability to perform reasoning is already present in our model. Our sample-efficient finetuning stage just activates it and we scale it further at test time with budget forcing." The authors draw a parallel to the "Superficial Alignment Hypothesis" from LIMA, where 1,000 examples sufficed for alignment.

**Q2: What exactly is budget forcing?**
A2 (from the paper, Section 3.1): "We propose a simple decoding-time intervention by forcing a maximum and/or minimum number of thinking tokens at test time. Specifically, we enforce a maximum token count by simply appending the end-of-thinking token delimiter and 'Final Answer:' to early exit the thinking stage and make the model provide its current best answer. To enforce a minimum, we suppress the generation of the end-of-thinking token delimiter and optionally append the string 'Wait' to the model's current reasoning trace to encourage the model to reflect on its current generation."

**Q3: Why is "Wait" the best string for extending reasoning?**
A3 (from the paper, Table 4): The authors compared no extrapolation, 2x without string, and 2x with "Alternatively", "Hmm", or "Wait". "Wait" achieved the best AIME24 score (53.3% vs 50.0% for no extrapolation), while also maintaining MATH500 and GPQA performance. The other strings either matched or underperformed.

**Q4: What are the limitations of budget forcing?**
A4 (from the paper, Section 6.2): "it has two key limitations when scaling further: it eventually flattens out (Figure 4), and the context window of the underlying language model constrains it." The authors note that "suppressing the end-of-thinking token delimiter too often can lead the model into repetitive loops instead of continued reasoning."

**Q5: Did the paper use reinforcement learning?**
A5 (from the paper, Abstract and Section 1): No. Unlike DeepSeek R1, which uses large-scale RL, s1 uses only supervised fine-tuning on 1,000 examples. The paper states: "we seek the simplest approach to achieve test-time scaling and strong reasoning performance" and contrasts itself with approaches "relying on techniques like Monte Carlo Tree Search" and "reinforcement learning via millions of samples."

## Common Misconceptions

1. **"s1 uses reinforcement learning to train the reasoning model."** — No, s1 uses only supervised fine-tuning (SFT) on 1,000 examples. Budget forcing is a decoding-time intervention applied at inference, not a training method. The paper explicitly contrasts its SFT-only approach with RL-based methods like DeepSeek R1 (Section 1).

2. **"Budget forcing always improves performance with more 'Wait' insertions."** — Not true. The paper shows that performance eventually flattens out and can degrade. Figure 4 shows the AIME24 curve flattening at 6x, and Section 6.2 notes that "suppressing the end-of-thinking token delimiter too often can lead the model into repetitive loops instead of continued reasoning."

3. **"The 1,000 samples can be any reasoning examples."** — The ablations (Table 2) show this is wrong. Random 1K selection scores 36.7% on AIME24 vs 50.0% for s1K. Selecting only the longest traces (1K-longest) gives 33.3%. The combination of quality, difficulty, and diversity filtering is crucial — removing any one criterion degrades performance significantly.

## Real Citations

1. Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, Tatsunori Hashimoto (2025). "s1: Simple test-time scaling." *EMNLP 2025.* [arXiv:2501.19393](https://arxiv.org/abs/2501.19393) — the original paper.
2. DeepSeek-AI et al. (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) — concurrent work using RL instead of SFT, achieving higher performance with 800x more data.
3. Zhou et al. (2023). "LIMA: Less Is More for Alignment." [arXiv:2305.11206](https://arxiv.org/abs/2305.11206) — the "Superficial Alignment Hypothesis" paper that inspired s1's minimal-data approach.

**Citation count:** about 1,400 citations on [OpenAlex](https://openalex.org/works/W4416036482) (checked November 2026).
