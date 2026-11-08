---
title: "GRPO: Talk & Q&A"
parent: "059 GRPO"
nav_order: 2
---
# TALK — DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)

## Talks, Slides and Press Coverage

1. **DeepSeek-AI — "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (DeepSeek-AI, 5 Feb 2024):** The official announcement of DeepSeekMath and GRPO, describing the 7B model's 51.7% MATH score and the GRPO algorithm. ([GitHub](https://github.com/deepseek-ai/DeepSeek-Math))
2. **DeepSeek-AI GitHub — "DeepSeekMath" repository:** The official code release including the GRPO training scripts and model weights, with documentation of the data selection pipeline and RL training setup. ([GitHub](https://github.com/deepseek-ai/DeepSeek-Math))
3. **Hugging Face TRL — GRPO Trainer documentation:** The open-source TRL library includes a GRPO trainer implementation based on this paper, with documentation explaining the algorithm. ([docs](https://huggingface.co/docs/trl/en/grpo_trainer))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2402.03300).

**Q1: Why does GRPO eliminate the value model?**
A1 (from the paper, Section 4.1.1): "As the value function employed in PPO is typically another model of comparable size as the policy model, it brings a substantial memory and computational burden. Additionally, during RL training, the value function is treated as a baseline in the calculation of the advantage for variance reduction. While in the LLM context, usually only the last token is assigned a reward [...] To address this, [...] GRPO [...] obviates the need for additional value function approximation as in PPO, and instead uses the average reward of multiple sampled outputs, produced in response to the same question, as the baseline."

**Q2: How are the advantages computed in GRPO?**
A2 (from the paper, Section 4.1.2): "Outcome supervision provides the normalized reward at the end of each output $o_i$ and sets the advantages $\hat{A}_{i,t}$ of all tokens in the output as the normalized reward, i.e., $\hat{A}_{i,t} = \tilde{r}_i = (r_i - \text{mean}(\mathbf{r})) / \text{std}(\mathbf{r})$."

**Q3: Why does GRPO add the KL penalty directly to the loss instead of in the reward?**
A3 (from the paper, Section 4.1.1): "Instead of adding KL penalty in the reward, GRPO regularizes by directly adding the KL divergence between the trained policy and the reference policy to the loss, avoiding complicating the calculation of $\hat{A}_{i,t}$."

**Q4: What reward does the paper use for RL training?**
A4 (from the paper, Section 4.2): "The training data of RL are chain-of-thought-format questions related to mathematics [...] the reward model is based on the rule judgment." — The paper uses both a trained reward model and rule-based (verifiable) rewards depending on the experiment.

## Common Misconceptions

1. **"GRPO is just PPO without a value model"** — While GRPO does remove the value model, the advantage estimation is fundamentally different. PPO uses a learned value function to estimate expected returns; GRPO uses the empirical average of group rewards. This means GRPO's advantage estimates have higher variance with small group sizes but require no value model training. The paper notes (Section 5.2.1) that GRPO's gradient coefficient differs from Online RFT, which "does not penalize incorrect responses."

2. **"GRPO requires a trained reward model"** — Not necessarily. The paper shows that for math problems, rule-based (verifiable) rewards work well: "the reward model is based on the rule judgment" (Section 4.2). The verifiable reward simply checks whether the final answer is correct.

3. **"GRPO was designed specifically for math"** — While introduced in the math context, GRPO is a general RL algorithm. It was later adopted for general reasoning training in DeepSeek-R1 (2025), which used GRPO to incentivize reasoning capability through RL on diverse problems.

## Real Citations

1. Zhihong Shao, Peiyi Wang, Qihao Zhu, et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." *arXiv preprint.* [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) — the original paper.
2. DeepSeek-AI (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) — directly adopts GRPO for large-scale reasoning training without SFT.
3. Zichen Liu et al. (2025). "Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO)." [arXiv:2503.20783](https://arxiv.org/abs/2503.20783) — analyzes and simplifies GRPO further.

**Citation count:** about 10,800 citations on [OpenAlex](https://www.openalex.org/works/W4391631327) (checked November 2026).
