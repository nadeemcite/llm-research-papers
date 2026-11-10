---
title: "Dr. GRPO: Talk & Q&A"
parent: "063 Dr. GRPO"
nav_order: 2
---
# TALK — Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO)

## Talks, Slides and Press Coverage

1. **Hugging Face Papers — "Understanding R1-Zero-Like Training: A Critical Perspective" (April 2025):** The paper was featured on Hugging Face's Daily Papers page (#3 Paper of the Day on April 3, 2025) with 60 upvotes and community discussion. ([paper page](https://huggingface.co/papers/2503.20783))
2. **GitHub — sail-sg/understand-r1-zero:** The open-source code, trained models, and training framework (Oat) are available on GitHub, with over 1,290 stars. ([repository](https://github.com/sail-sg/understand-r1-zero))
3. **Oat RL Framework — sail-sg/oat:** The modular LLM RL framework used in the paper, open-sourced by Sea AI Lab. ([repository](https://github.com/sail-sg/oat))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2503.20783).

**Q1: Why does GRPO cause response length to increase during training?**
A1 (from the paper, Section 3.1): GRPO introduces a "response-level length bias" from dividing by \|o_i\|. "For positive advantages (indicating a correct response), this bias results in greater gradient updates for shorter responses, leading the policy to favor brevity in correct answers. Conversely, for negative advantages (indicating an incorrect response), longer responses are penalized less due to their larger \|o_i\|, causing the policy to prefer lengthier responses among incorrect ones."

**Q2: What is the difficulty bias, and why is it a problem?**
A2 (from the paper, Section 3.1): The difficulty bias "is caused by dividing the centered outcome reward by std({R(q,o_1),...,R(q,o_G)}). Questions with lower standard deviations (e.g., those that are too easy or too hard, with the outcome rewards being almost all 1 or 0) are given higher weights during policy updates." While advantage normalization is common in RL, "question-level normalization results in varying weights in the objective for different questions, leading to a difficulty bias in optimization."

**Q3: How does Dr. GRPO fix these biases?**
A3 (from the paper, Section 3.2): "To avoid the aforementioned optimization bias in GRPO, we propose to simply remove the 1/\|o_i\| and std({R(q,o_1),...,R(q,o_G)}) normalization terms." The paper notes that "these simple modifications recover the PPO objective in Eq. 2, with the advantage estimated by Monte Carlo return with an unbiased baseline."

**Q4: Do Qwen2.5-Math models really work best without any prompt template?**
A4 (from the paper, Section 2.2): "not using any template can drastically boost the average performance, resulting in an improvement of about 60% compared to the traditional 4-shot prompting." The authors hypothesize "they might pretrain on the concatenated text to maximize log p_theta(q, o) directly" and caution: "we shall be more careful about using Qwen2.5 models to reproduce DeepSeek-R1-Zero, since the base models are already SFT-like without templates."

**Q5: Does the "Aha moment" emerge from RL, or was it already in the base model?**
A5 (from the paper, Section 2.3): "Nearly all base models already exhibit the 'Aha moment', including DeepSeek-V3-Base." The authors tested DeepSeek-V3-Base-685B and found it "also generates a decent amount of self-reflections." They also note that in R1-Zero, "self-reflection behaviors are not positively correlated with higher accuracy" (Appendix F).

**Q6: Does the length bias also exist in PPO implementations?**
A6 (from the paper, Section 3.1): Yes. "We also examined several popular open-source implementations of vanilla PPO algorithms for LLM post-training. To our surprise, all of these implementations normalize the loss by response length." The paper lists trl, OpenRLHF, verl, SimpleRL-Zero, and Open-Reasoner-Zero as all containing this bias (Table 2).

## Common Misconceptions

1. **"The increase in response length during R1-Zero training proves the model is developing deeper reasoning"** — The paper argues this is confounded by GRPO's length bias. Section 3.1 states: "the observed increase in response length may also be attributed to a bias inherent in the GRPO objective function." Figure 5 shows Dr. GRPO prevents "the response length from growing wildly during training" while maintaining performance.

2. **"Qwen2.5 base models are pure base models suitable for reproducing R1-Zero"** — The paper's Section 2.2 shows Qwen2.5-Math models achieve ~60% improvement without any template, suggesting they were "pretrained on concatenated question-answer texts" and are "already SFT-like without templates." The authors warn this means "we should be more conservative in claiming the huge gains brought about by pure RL."

3. **"Dr. GRPO is a complex new algorithm"** — It is the opposite: Dr. GRPO is simpler than GRPO. It just removes two normalization terms (1/\|o_i\| and 1/std(R)). The paper calls it "simple yet significant modifications" (Figure 1 caption).

## Real Citations

1. Zichen Liu et al. (2025). "Understanding R1-Zero-Like Training: A Critical Perspective." *arXiv preprint.* [arXiv:2503.20783](https://arxiv.org/abs/2503.20783) — the original paper.
2. Zhihong Shao et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) — GRPO, the algorithm whose biases this paper identifies and fixes.
3. Daya Guo et al. (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) — R1-Zero, the training paradigm this paper critically examines.
4. Qiying Yu et al. (2025). "DAPO: An Open-Source LLM Reinforcement Learning System at Scale." [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) — follow-up that further improved GRPO with Clip-Higher, Dynamic Sampling, and Token-Level Loss.

**Citation count:** about 130 citations on [OpenAlex](https://openalex.org/works/W4400753527) (checked November 2026).
