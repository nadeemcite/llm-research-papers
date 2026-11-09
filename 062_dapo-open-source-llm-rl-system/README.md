---
title: "062 DAPO"
nav_order: 63
permalink: "/062_dapo-open-source-llm-rl-system/"
---
# DAPO: An Open-Source LLM Reinforcement Learning System at Scale (DAPO)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/dapo-open-source-llm-rl-system)

- **Paper:** [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- **Authors:** Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, Mingxuan Wang
- **Lab, year:** ByteDance Seed and Institute for AI Industry Research (AIR), Tsinghua University, 2025

## Summary

In March 2025, researchers at ByteDance Seed and Tsinghua University's AIR released DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization), a fully open-source reinforcement learning system for training reasoning LLMs. DAPO achieves 50 points on AIME 2024 (a competition-level math benchmark) using the Qwen2.5-32B base model, outperforming the previous state-of-the-art DeepSeek-R1-Zero-Qwen-32B (47 points) using only 50% of the training steps. Unlike prior reasoning models that concealed their RL training details (OpenAI o1, DeepSeek R1), DAPO open-sourced its algorithm, training code (built on the verl framework), and a curated dataset (DAPO-Math-17K).

The paper identifies four critical problems in naive GRPO training: entropy collapse (the policy becomes deterministic too early), gradient dilution (all-correct or all-wrong groups produce zero gradients), length bias (sample-level loss unfairly weights short and long responses equally), and reward noise from truncated long outputs. DAPO introduces four corresponding techniques: Clip-Higher (decoupled clipping with a higher upper bound to encourage exploration), Dynamic Sampling (filtering out prompts where all G responses are correct or all wrong), Token-Level Policy Gradient Loss (weighting by tokens rather than samples), and Overlong Reward Shaping (a soft length-based penalty instead of a flat punishment for truncated outputs).

This notebook implements all four DAPO techniques on top of a GRPO baseline, using a small MLP policy on synthetic arithmetic problems with verifiable rewards, and compares DAPO against vanilla GRPO to show the incremental benefit of each technique.

## What Problem Does It Solve

Imagine coaching a team for a math competition. If you only reward correctness, the team quickly converges to one strategy — everyone solves problems the same way and stops exploring new approaches. This is "entropy collapse." Worse, for easy problems everyone gets them right (no learning signal) and for hard problems everyone fails (also no signal) — "gradient dilution." And if one student writes a two-line solution while another writes ten pages of gibberish, treating them as equally important ("sample-level loss") gives too much weight to the gibberish.

DAPO fixes all four issues: it encourages exploration by allowing low-probability "creative" moves to grow more (Clip-Higher), it skips problems that are too easy or too hard (Dynamic Sampling), it weights learning by individual reasoning steps rather than per-problem (Token-Level Loss), and it gently penalizes overly long answers instead of harshly punishing them (Overlong Reward Shaping).

## Core Idea

- **Clip-Higher (decoupled clipping):** Standard PPO clips the importance ratio to [1-ε, 1+ε]. DAPO uses separate bounds ε_low=0.2 and ε_high=0.28, giving the upper bound more room so low-probability "exploration" tokens can increase their probability more freely (Section 3.1).
- **Dynamic Sampling:** Before each training step, over-sample prompts and filter out those where all G responses are correct (accuracy=1) or all wrong (accuracy=0), keeping only prompts with non-zero, non-trivial gradients. This maintains a consistent effective batch size (Section 3.2).
- **Token-Level Policy Gradient Loss:** Instead of averaging per-sample then across samples (which gives each sample equal weight), DAPO averages across all tokens in the batch, so longer responses contribute proportionally more and quality patterns within them get proper reinforcement (Section 3.3).
- **Overlong Reward Shaping:** Rather than assigning a flat -1 to truncated outputs, DAPO uses a soft penalty that linearly decreases the reward as the response length approaches the maximum, reducing reward noise from valid-but-long reasoning (Section 3.4).

## Key Method Details

- **DAPO objective (Equation 8):** $\mathcal{J}_{DAPO}(\theta) = \mathbb{E}\left[\frac{1}{\sum_{i=1}^{G}\|o_i\|}\sum_{i=1}^{G}\sum_{t=1}^{\|o_i\|}\min\left(r_{i,t}(\theta)\hat{A}_{i,t},\;\text{clip}\left(r_{i,t}(\theta),\,1-\varepsilon_{low},\,1+\varepsilon_{high}\right)\hat{A}_{i,t}\right)\right]$, subject to $0 < \|\{o_i \mid \text{is\_equivalent}(a, o_i)\}\| < G$.
- **Advantage (Equation 9):** $\hat{A}_{i,t} = (R_i - \text{mean}(\{R_i\})) / \text{std}(\{R_i\})$ — group-relative, same as GRPO but without a KL penalty (Section 2.3 removes KL).
- **Clip-Higher (Section 3.1):** ε_low=0.2, ε_high=0.28. The paper notes the mean probability of up-clipped tokens is <0.2, confirming that the upper clip constrains exploration.
- **Soft Overlong Punishment (Equation 13):** $R_{length}(y) = 0$ if $\|y\| \leq L_{max} - L_{cache}$; linearly decreasing in $[L_{max} - L_{cache}, L_{max}]$; $-1$ if $\|y\| > L_{max}$. Paper uses L_max=16384, L_cache=4096.
- **Training (Section 4.1):** AdamW optimizer, lr=1e-6, warmup 20 rollout steps, prompt batch 512, G=16 responses per prompt, 16 gradient updates per rollout, max generation 20480 tokens.
- **Progressive results (Table 1):** Naive GRPO: 30 → +Overlong Filtering: 36 → +Clip-Higher: 38 → +Soft Overlong: 41 → +Token-Level Loss: 42 → +Dynamic Sampling (full DAPO): 50 on AIME 2024 avg@32.

### Simplifications in This Notebook

- Smaller model: 3-layer MLP (~19K parameters) instead of Qwen2.5-32B (32B parameters)
- Synthetic single-digit arithmetic (1-9 +/- 1-9) instead of competition-level AIME problems
- Group size G=16 (same as paper), 30 DAPO iterations with 50 questions per iteration
- Learning rate 2e-4 (much higher than the paper's 1e-6, to compensate for fewer iterations and smaller model)
- Verifiable rule-based reward (exact match, +1/-1) instead of a trained reward model
- SFT for 10 epochs on 3000 examples before RL, instead of starting from a 32B pre-trained model
- Overlong reward shaping applied with a toy max-length of 8 "steps" (responses are single-token answers) to demonstrate the mechanism
- No KL penalty (DAPO also removes KL, so this is faithful to the paper)

### What the Notebook Result Shows

On 200 held-out test arithmetic problems, the SFT baseline reaches 97.5% greedy accuracy. After 30 RL iterations, both GRPO and DAPO achieve 100% greedy accuracy, but DAPO reaches 91.5% sampling accuracy vs GRPO's 87.5% (+4%). The progressive ablation shows incremental sampling accuracy gains: GRPO 87.5% → +Clip-Higher 89.5% → +Dynamic Sampling 90.0% → +Token-Level 88.5% → Full DAPO 91.5%, mirroring the paper's Table 1 pattern. The entropy plot shows DAPO maintains slightly higher policy entropy than GRPO, consistent with Clip-Higher's exploration-promoting effect. With a ~19K-parameter model and toy arithmetic, these improvements are modest and not comparable to the paper's AIME 2024 results (30→50 with Qwen2.5-32B).

## Influence

DAPO became one of the most widely adopted open-source RL systems for LLM reasoning training. Its full release of algorithm, code (on verl), and dataset (DAPO-Math-17K) enabled many groups to reproduce and extend RL-based reasoning. Follow-up work includes Dr. GRPO (2025), which simplified GRPO's length bias, and subsequent open-source reasoning models that built on DAPO's techniques. About 2,811 citations ([OpenAlex](https://openalex.org/works/W7196953391), checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/062_dapo-open-source-llm-rl-system/solution.ipynb) | From-scratch PyTorch implementation of DAPO's four key techniques (Clip-Higher, Dynamic Sampling, Token-Level Loss, Overlong Reward Shaping) on a small MLP policy with verifiable arithmetic rewards, comparing against vanilla GRPO |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
