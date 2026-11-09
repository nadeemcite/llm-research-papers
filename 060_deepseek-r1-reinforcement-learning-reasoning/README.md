---
title: "060 DeepSeek-R1"
nav_order: 61
permalink: "/060_deepseek-r1-reinforcement-learning-reasoning/"
---
# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (DeepSeek-R1)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/deepseek-r1)

- **Paper:** [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- **Authors:** DeepSeek-AI (Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, et al.) — 80+ authors
- **Lab, year:** DeepSeek-AI, 2025

## Summary

In January 2025, DeepSeek-AI published DeepSeek-R1-Zero and DeepSeek-R1, two reasoning models trained primarily through reinforcement learning. DeepSeek-R1-Zero is the more radical of the two: it applies GRPO (Group Relative Policy Optimization, introduced in DeepSeekMath) directly to the DeepSeek-V3-Base model — a 671B-parameter Mixture-of-Experts model — with **no supervised fine-tuning** on reasoning traces at all. The only supervision is a rule-based reward: accuracy (does the final answer match the known correct answer?) and format (does the model use `<think>` and `<answer>` tags?).

Despite this minimal supervision, DeepSeek-R1-Zero developed sophisticated reasoning behaviors spontaneously. Its pass@1 score on AIME 2024 (a competition mathematics benchmark) increased from 15.6% to 71.0%, and with majority voting to 86.7%, matching OpenAI's o1-0912. The model learned to self-verify, reflect on its work, and generate increasingly long chain-of-thought reasoning — a phenomenon the authors called the "aha moment," where the model spontaneously learned to reevaluate its initial approach.

DeepSeek-R1, the full model, adds a multi-stage pipeline: cold-start data fine-tuning, reasoning-oriented RL, rejection sampling and SFT, and a final RL stage for all scenarios. It achieves 79.8% pass@1 on AIME 2024, 97.3% on MATH-500, and a 2,029 Elo rating on Codeforces (outperforming 96.3% of human participants), matching OpenAI's o1-1217. The paper also distills reasoning into six dense models (1.5B to 70B), with the 32B distilled model scoring 72.6% on AIME 2024.

This notebook implements R1-Zero style training on a toy Countdown task with a small transformer (~108K parameters), demonstrating the GRPO algorithm, rule-based rewards, and the emergence of format compliance through pure RL.

## What Problem Does It Solve

Imagine teaching someone to solve puzzles, but instead of showing them worked examples, you just tell them whether their answer is right or wrong. Most approaches to training reasoning models rely on large amounts of human-written reasoning traces — step-by-step solutions that a teacher demonstrates. Collecting these is expensive and time-consuming.

R1-Zero asks: what if we skip the examples entirely? Give the model a problem, let it generate a reasoning process and an answer, and reward it only if the answer is verifiably correct. The model has to figure out *how* to reason on its own. This is pure reinforcement learning — the model explores different reasoning strategies and keeps the ones that lead to correct answers.

## Core Idea

- **Pure RL without SFT:** R1-Zero applies GRPO directly to the base model with no supervised fine-tuning on reasoning traces (Section 2.2). The only training signal is rule-based rewards.
- **Rule-based rewards:** Accuracy reward (is the final answer correct?) and format reward (does the model use `<think>`/`<answer>` tags?). No neural reward model is used, because "the neural reward model may suffer from reward hacking" (Section 2.2.2).
- **Emergent reasoning behaviors:** Through RL, the model spontaneously develops self-verification, reflection, and longer chain-of-thought — the "aha moment" where it learns to reevaluate its approach (Section 2.2.4).

## Key Method Details

- **GRPO objective (Equation 1):** For each question $q$, sample $G$ outputs from $\pi_{\theta_{old}}$. The objective is $\frac{1}{G}\sum_{i=1}^{G}\left[\min\left(r_{i}(\theta)\hat{A}_{i},\;\text{clip}(r_{i}(\theta),\,1-\varepsilon,\,1+\varepsilon)\hat{A}_{i}\right)\right] - \beta D_{KL}[\pi_\theta \| \pi_{ref}]$, where $r_{i}(\theta)$ is the importance ratio and $\hat{A}_{i}$ is the group-relative advantage.
- **Advantage (Equation 3):** $\hat{A}_{i} = (r_i - \text{mean}(\mathbf{r})) / \text{std}(\mathbf{r})$ — the normalized reward within the group.
- **Training template (Table 1):** A simple prompt instructing the model to produce reasoning in `<think>` tags followed by an answer in `<answer>` tags, with no content-specific constraints.
- **Base model:** DeepSeek-V3-Base (671B MoE, 37B activated parameters).
- **Results:** R1-Zero: AIME 2024 pass@1 15.6% → 71.0%, MATH-500 95.9%, Codeforces 1444 rating. R1: AIME 2024 79.8%, MATH-500 97.3%, Codeforces 2029 Elo, MMLU 90.8%, GPQA Diamond 71.5%.

### Simplifications in This Notebook

- Smaller model: 2-layer transformer (~108K parameters) instead of DeepSeek-V3-Base (671B MoE)
- Toy Countdown task (combine 4 numbers to reach a target) instead of AIME / MATH / coding benchmarks
- Light pre-training on 8,000 mixed arithmetic + formatted examples (base-model analog) instead of trillions of tokens
- 120 GRPO iterations with group size G=8 instead of thousands of RL steps with G=64
- Learning rate 3e-4 (much higher than the paper's 1e-6, to compensate for fewer iterations)
- Rule-based rewards (accuracy 0.9 + format 0.1) — same principle as the paper, no neural reward model

### What the Notebook Result Shows

On 50 held-out Countdown problems, the R1-Zero model (after 120 GRPO iterations) generates responses averaging about 15.5 tokens, compared to 11.6 tokens for the base model — a 33% increase in response length through pure RL. Format compliance rises from about 90% to 100%. Sampling accuracy reaches 0.5% (the model occasionally solves a problem correctly). These results are not comparable to the paper's AIME 15.6% → 71.0% improvement; our ~108K-parameter toy model on simple arithmetic cannot match a 671B model trained over thousands of RL steps. The key observation is the mechanism: the model learns the template format and generates longer responses through pure RL with verifiable rewards, without any supervised reasoning data.

## Influence

DeepSeek-R1 became one of the most influential papers of 2025, sparking widespread interest in RL-based reasoning training. It directly influenced follow-up work including DAPO (2025, an open-source RL system at scale), Dr. GRPO (2025, which analyzed and simplified GRPO further), s1 (2025, simple test-time scaling), and the broader open-source reasoning model ecosystem. The paper's key insight — that reasoning can emerge from pure RL with verifiable rewards — became a foundational principle for the field. About 2,100 citations ([OpenAlex](https://www.openalex.org/works/W4391631327), checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/060_deepseek-r1-reinforcement-learning-reasoning/solution.ipynb) | PyTorch R1-Zero implementation (GRPO from scratch, tiny transformer) that RL-trains a 2-layer transformer on the Countdown task with rule-based rewards via pure RL |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
