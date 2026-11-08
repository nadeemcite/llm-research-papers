---
title: "059 GRPO"
nav_order: 60
permalink: "/059_grpo-deepseek-mathematical-reasoning/"
---
# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/grpo-deepseek-mathematical-reasoning)

- **Paper:** [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- **Authors:** Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo
- **Lab, year:** DeepSeek-AI (Shao: Tsinghua University; Wang and Zhu: Peking University, work done during internship at DeepSeek-AI), 2024

## Summary

In February 2024, researchers at DeepSeek-AI introduced DeepSeekMath 7B, a language model that pushed open-source mathematical reasoning to near-GPT-4 levels. The model achieves 51.7% on the competition-level MATH benchmark (a dataset of high-school competition math problems) without external tools or voting, and 60.9% with self-consistency over 64 samples. This was achieved by continuing pre-training DeepSeek-Coder-Base-v1.5 7B with 120 billion math-related tokens mined from Common Crawl using a fastText classifier, combined with a new reinforcement learning algorithm.

The paper's second contribution is **Group Relative Policy Optimization (GRPO)**, a variant of Proximal Policy Optimization (PPO) that eliminates the critic (value) model. In standard PPO, a separate value network — typically as large as the policy model itself — estimates a baseline for variance reduction in the advantage calculation. GRPO instead samples a group of G outputs for each question and uses the group's average reward as the baseline, computing advantages as normalized group-relative rewards. This halves the memory footprint of RL training while still improving performance: GRPO boosts DeepSeekMath-Instruct from 82.9% to 88.2% on GSM8K and 46.8% to 51.7% on MATH.

This notebook implements GRPO from scratch in PyTorch and applies it to a small MLP policy on synthetic arithmetic problems with verifiable rewards, following the paper's SFT-to-RL pipeline. It demonstrates group-relative advantage estimation and the KL-regularized policy update without a value model.

## What Problem Does It Solve

Imagine a teacher grading math homework. If the teacher already knows the correct answer, they can simply check whether each student's answer matches — no need for a second "grading assistant" to estimate how well each student was expected to do. PPO, the standard RL algorithm for language models, is like a school that hires a full-time grading assistant (the value model) for every teacher (the policy model). This doubles the compute and memory cost.

GRPO's insight is that for math problems with verifiable answers, you can skip the grading assistant entirely. Instead, you ask the same student to attempt the problem multiple times (a "group"), then compare each attempt against the group's average. Attempts that score above average get reinforced; those below average get discouraged. This is cheaper and works just as well.

## Core Idea

- **No value model:** GRPO eliminates the critic network that PPO uses for baseline estimation, halving the memory footprint of RL training (Section 4.1.1).
- **Group-relative advantages:** For each question, sample G outputs from the old policy. Compute rewards, normalize by the group mean and standard deviation, and use the normalized reward as the advantage (Section 4.1.2).
- **KL regularization via direct penalty:** Instead of adding the KL penalty inside the reward (as PPO does), GRPO adds it directly to the loss function using an unbiased estimator (Schulman, 2020), simplifying the advantage calculation (Section 4.1.1).

## Key Method Details

- **GRPO objective (Equation 3):** For each question $q$, sample $G$ outputs from $\pi_{\theta_{old}}$. The objective is $\frac{1}{G}\sum_{i=1}^{G}\left[\min\left(r_{i}(\theta)\hat{A}_{i},\;\text{clip}(r_{i}(\theta),\,1-\varepsilon,\,1+\varepsilon)\hat{A}_{i}\right)\right] - \beta D_{KL}[\pi_\theta \| \pi_{ref}]$, where $r_{i}(\theta)$ is the importance ratio and $\hat{A}_{i}$ is the group-relative advantage.
- **Advantage (Equation 5):** $\hat{A}_{i} = \tilde{r}_i = (r_i - \text{mean}(\mathbf{r})) / \text{std}(\mathbf{r})$ — the normalized reward (outcome supervision).
- **Training hyperparameters (Section 4.2):** Policy learning rate 1e-6, KL coefficient $\beta = 0.04$, group size $G = 64$, max length 1024, batch size 1024 questions, single policy update per exploration stage.
- **Results:** DeepSeekMath 7B scores 51.7% on MATH (greedy), 60.9% with self-consistency (64 samples), 64.2% on GSM8K. GRPO improves GSM8K from 82.9% to 88.2% and MATH from 46.8% to 51.7% over the instruction-tuned baseline.

### Simplifications in This Notebook

- Smaller model: 3-layer MLP (~19K parameters) instead of DeepSeekMath 7B (7B parameters)
- Synthetic single-digit arithmetic (1-9 +/- 1-9) instead of competition-level MATH problems
- Group size G=16 instead of 64
- 20 GRPO iterations with 50 questions each, single policy update per iteration
- Learning rate 2e-4 (much higher than the paper's 1e-6, to compensate for fewer iterations)
- Verifiable rule-based reward (exact match) instead of a trained reward model
- SFT for 10 epochs on 3000 examples before GRPO, instead of pre-training on 120B math tokens

### What the Notebook Result Shows

On 100 held-out test arithmetic problems, the MLP's greedy and sampling accuracies after SFT and after GRPO are printed in the Final Evaluation section. The training curves show mean group reward and both accuracy metrics per iteration. With a ~19K-parameter model and 20 iterations, the improvement is modest — the paper uses a 7B model trained over many more iterations with group size 64 and 1024 questions per batch. The result is not comparable to the paper's GSM8K or MATH numbers.

## Influence

GRPO became one of the most influential RL algorithms for LLM reasoning. It was adopted directly in DeepSeek-R1 (2025), which used GRPO to train reasoning models via pure RL without supervised fine-tuning, sparking widespread interest in RL-based reasoning training. The algorithm's simplicity — no value model needed — made it a popular choice for open-source RL training frameworks. Follow-up work includes Dr. GRPO (2025), which analyzed and simplified GRPO further, and DAPO (2025), which scaled GRPO-based RL to larger models. About 10,800 citations ([OpenAlex](https://www.openalex.org/works/W4391631327), checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/059_grpo-deepseek-mathematical-reasoning/solution.ipynb) | PyTorch GRPO implementation (algorithm from scratch, MLP policy) that RL-trains a 3-layer MLP on synthetic arithmetic with verifiable rewards via SFT then GRPO |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
