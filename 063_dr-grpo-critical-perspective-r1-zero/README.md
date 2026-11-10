---
title: "063 Dr. GRPO"
nav_order: 64
permalink: "/063_dr-grpo-critical-perspective-r1-zero/"
---
# Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/dr-grpo-critical-perspective-r1-zero)

- **Paper:** [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)
- **Authors:** Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, Min Lin
- **Lab, year:** Sea AI Lab (Lee: National University of Singapore; Lin: also Singapore Management University), 2025

## Summary

In March 2025, researchers at Sea AI Lab critically examined the R1-Zero-like training paradigm — applying reinforcement learning directly to base LLMs without supervised fine-tuning, as introduced by DeepSeek-R1-Zero. The paper analyzes two core components: base models and the RL algorithm. On base models, they find that DeepSeek-V3-Base already exhibits the "Aha moment" (self-reflection behavior) before any RL, and that Qwen2.5-Math models perform best without any prompt template, suggesting they were pre-trained on concatenated question-answer texts (making them implicitly SFT-like).

The paper's key algorithmic contribution is identifying two biases in Group Relative Policy Optimization (GRPO), the RL algorithm used in DeepSeek-R1. First, a length bias: dividing the per-token loss by response length 1/\|o_i\| causes the policy to favor longer incorrect responses (they get penalized less) and shorter correct ones. Second, a difficulty bias: normalizing advantages by 1/std(R) over-weights questions that are too easy or too hard. The authors propose Dr. GRPO ("GRPO Done Right") — simply removing both normalization terms — which improves token efficiency while maintaining reasoning performance. Their minimalist recipe achieves 43.3% on AIME 2024 (a competition-level math benchmark) with a 7B base model, establishing a new state-of-the-art at the time.

This notebook demonstrates GRPO's length and difficulty biases on a small MLP policy solving synthetic arithmetic, then implements the Dr. GRPO fix and compares training dynamics. The length bias and difficulty bias are shown quantitatively, and Dr. GRPO is shown to reduce the gap between incorrect and correct response lengths.

## What Problem Does It Solve

Imagine a student who gets math problems wrong. Instead of learning to solve them correctly, they start writing longer and longer wrong answers — filling pages with gibberish that looks like reasoning. That's what happens with GRPO: the algorithm's loss function inadvertently penalizes long wrong answers less than short wrong ones, so the model "learns" to be verbose when it's confused. This wastes compute and gives a false impression that the model is developing deeper reasoning.

Dr. GRPO fixes this by removing two normalization terms that cause the bias. It's like switching from a grading system that averages per-student (where one student's ten pages of wrong work gets the same weight as another's two lines) to one that counts each reasoning step equally. The result: the model still learns just as well, but its wrong answers stay concise instead of ballooning — saving tokens and compute.

## Core Idea

- **Length bias (Section 3.1):** GRPO divides the per-token loss by response length \|o_i\|. For correct responses (positive advantage), shorter responses get larger gradient updates. For incorrect responses (negative advantage), longer responses are penalized less. Net effect: the model generates progressively longer incorrect responses during training.
- **Difficulty bias (Section 3.1):** GRPO normalizes advantages by the group's reward standard deviation std(R). Questions with low reward variance (too easy or too hard) get higher weights, distorting the optimization toward extreme-difficulty questions.
- **Dr. GRPO fix (Section 3.2):** Simply remove both the 1/\|o_i\| and 1/std(R) normalization terms. The advantage becomes the unbiased baseline-subtracted return: A = R - mean(R). This recovers the PPO objective with Monte Carlo return and an unbiased baseline.

## Key Method Details

- **GRPO objective (Equation 3):** $J_{GRPO}(\\theta) = \\frac{1}{G}\\sum_{i=1}^{G}\\frac{1}{\|o_i\|}\\sum_{t=1}^{\|o_i\|}\\min\\left[r_{i,t}(\\theta)\\hat{A}_{i,t},\\;\\text{clip}(r_{i,t}(\\theta),1-\\varepsilon,1+\\varepsilon)\\hat{A}_{i,t}\\right]$, where $\\hat{A}_{i,t} = \\frac{R_i - \\text{mean}(R)}{\\text{std}(R)}$.
- **Dr. GRPO (Section 3.2):** Remove 1/\|o_i\| (replace mask.sum with a constant like the generation budget) and remove 1/std(R). The advantage becomes $\\tilde{A}_{i,t} = R_i - \\text{mean}(R)$ — the unbiased Monte Carlo return with group mean as baseline.
- **No KL penalty (Section 3):** The paper sets β=0 for R1-Zero-like training, since rule-based verifiers eliminate distributional shift concerns and removing KL saves memory and computation.
- **Base model findings (Section 2):** Qwen2.5-Math models achieve ~60% improvement when no template is used (Table 1), suggesting pretraining on concatenated QA text. DeepSeek-V3-Base already exhibits "Aha moment" self-reflection before RL (Section 2.3, Figure 3 right).
- **Minimalist recipe (Section 1):** RL-tune Qwen2.5-Math-7B with Dr. GRPO on MATH level 3-5 questions with the Qwen-Math template, 27 hours on 8×A100 GPUs.
- **Results:** 43.3% accuracy on AIME 2024 with a 7B base model (new state-of-the-art at the time). Dr. GRPO maintains reasoning performance while reducing the length of incorrect responses (Figure 5, Plot 4).

### Simplifications in This Notebook

- Smaller model: 3-layer MLP (~19,475 parameters) instead of Qwen2.5-Math-7B (7B parameters)
- Synthetic single-digit arithmetic (1-9 +/- 1-9) instead of competition-level MATH problems
- Group size G=16 (paper uses 16-64), 20 RL iterations with 50 questions each
- Learning rate 5e-5 (much higher than the paper's 1e-6, to compensate for fewer iterations and smaller model)
- Verifiable rule-based reward (+1 correct, 0 incorrect) instead of Math-Verify
- No KL penalty (faithful to the paper, which also sets β=0)
- Simulated variable-length responses (repeating answer tokens) to demonstrate the length bias, instead of actual multi-token generation
- SFT for 20 epochs on 3000 examples before RL, instead of starting from a pre-trained 7B base model

### What the Notebook Result Shows

On 200 held-out test arithmetic problems, the SFT baseline reaches 84.0% greedy accuracy and 65.0% sampling accuracy. After 20 RL iterations with 50 questions each, both GRPO (67.0% greedy) and Dr. GRPO (63.5% greedy) show some accuracy degradation — expected with a tiny 19K-parameter model and very few RL questions.

The key results are the bias demonstrations: the length bias demo shows GRPO applies 10x less gradient penalty to a length-20 incorrect response vs a length-2 one (-0.047 vs -0.468), while Dr. GRPO gives equal weight. The difficulty bias demo shows GRPO assigns 2.83x higher advantage magnitude to easy/hard questions (std=0.354) vs medium ones (std=0.535), while Dr. GRPO treats all questions more equally.

With a ~19K-parameter model and toy arithmetic, these effects are modest and not comparable to the paper's AIME 2024 results (43.3% with Qwen2.5-Math-7B).

## Influence

Dr. GRPO became widely adopted in the open-source LLM reasoning community as a drop-in fix for GRPO's length bias. The paper's analysis of base model pretraining biases (particularly the Qwen2.5-Math "no template" finding) influenced how researchers interpret R1-Zero replications, prompting more careful claims about what RL alone contributes versus what pretraining already provides. Follow-up work includes DAPO (2025), which further improved GRPO with Clip-Higher, Dynamic Sampling, and Token-Level Loss. The open-source code (built on the Oat framework) and released models enabled many groups to reproduce R1-Zero-like training efficiently. About 130 citations ([OpenAlex](https://openalex.org/works/W4400753527), checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/063_dr-grpo-critical-perspective-r1-zero/solution.ipynb) | PyTorch implementation demonstrating GRPO's length and difficulty biases on a 3-layer MLP with synthetic arithmetic, and the Dr. GRPO fix that removes both normalization terms |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
