---
title: "064 RL vs Base Model"
nav_order: 65
permalink: "/064_rl-reasoning-beyond-base-model/"
---
# Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? (RL vs Base Model)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/rl-reasoning-beyond-base-model)

- **Paper:** [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837)
- **Authors:** Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, Gao Huang
- **Lab, year:** Tsinghua University (LeapLab; Huang: corresponding author) and Shanghai Jiao Tong University (Wang), 2025 (NeurIPS 2025 Oral — Best Paper Runner-Up; ICML 2025 AI4Math Workshop Best Paper)

## Summary

In April 2025, researchers at Tsinghua University and Shanghai Jiao Tong University asked a provocative question: does Reinforcement Learning with Verifiable Rewards (RLVR) — the technique behind reasoning models like DeepSeek-R1 — actually teach LLMs new reasoning skills, or does it just make them better at sampling answers they already knew? Using pass@k (the probability that at least one of k sampled responses is correct) as the probing metric, they systematically compared base models against their RLVR-trained counterparts across math (GSM8K, MATH500, AIME24, AMC23), coding (LiveCodeBench, HumanEval+), and visual reasoning (MathVista, MathVision) benchmarks.

The surprise: while RLVR-trained models outperform base models at small k (e.g., k=1, greedy decoding), base models consistently catch up and surpass the RL models as k grows to hundreds. The reasoning paths that RLVR models generate are already present in the base model's sampling distribution — RLVR sharpens the distribution toward rewarded paths but does not expand the set of solvable problems.

In fact, RLVR narrows the reasoning boundary: problems the base model could solve (at large k) become unsolvable after RL training. Coverage and perplexity (a measure of how well a model predicts text — lower perplexity means the text is more likely under the model's distribution) analyses confirm that RLVR operates within the base model's prior rather than beyond it. Six popular RL algorithms (PPO, GRPO, Reinforce++, RLOO, ReMax, DAPO) all perform similarly and remain far from optimal. By contrast, distillation from a stronger teacher genuinely introduces new reasoning patterns and expands the boundary.

This notebook reproduces the paper's core pass@k experiment on a small scale: it trains a tiny MLP policy on synthetic arithmetic with an RLVR-style reward, then plots pass@k curves for the "base" (SFT-only) and "RL-trained" models to show the crossover phenomenon — RL improves pass@1 but reduces pass@k at large k.

## What Problem Does It Solve

Imagine you have a big jar of marbles, and you want to find the red ones (correct answers). A base model is like searching the jar with your eyes closed — if you pull out enough marbles (large k), you'll eventually find the red ones. RLVR training is like putting on glasses that make red marbles easier to spot, so you find them faster with fewer pulls (small k). But here's the catch: the glasses also make you pull fewer marbles overall, so some red marbles that you would have found blind now get missed entirely.

The paper shows that RLVR doesn't add new red marbles to the jar — it just changes which ones you grab first. The total number of red marbles (the base model's reasoning capacity) stays the same or even shrinks. This matters because everyone assumed RLVR was teaching models to reason in fundamentally new ways, when really it's just making them more efficient at finding answers they already could produce.

## Core Idea

- **pass@k as a capability probe (Section 2.1):** Instead of measuring only greedy accuracy (pass@1), sample k responses and check if any is correct. pass@k at large k (e.g., 256) approximates the model's total reasoning coverage — the full set of problems it can solve given enough tries.
- **The crossover phenomenon (Section 3):** RLVR models beat base models at small k (better sampling efficiency) but lose at large k (narrower coverage). The base model's pass@k curve crosses above the RL model's as k increases — across every benchmark and model family without exception.
- **Reasoning paths already in the base (Section 4.1):** Perplexity analysis shows RL-generated responses are highly likely under the base model's distribution. RLVR sharpens the distribution within the base prior; it does not generate paths outside it.
- **Distillation genuinely expands (Section 4.2):** Unlike RLVR, distillation from a stronger teacher introduces new reasoning patterns that are not in the base model's distribution, genuinely expanding the reasoning boundary.
- **Sampling efficiency gap (Section 4.3):** The paper defines $\Delta_{\text{SE}} = \text{pass@1}_{\text{RL}} - \text{pass@k}_{\text{base}}$ (with k=256 as proxy for upper bound). All six RL algorithms show similar, large $\Delta_{\text{SE}}$ — they are far from optimal in leveraging the base model's potential.

## Key Method Details

- **pass@k formula (Section 2.1):** $\text{pass@k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$, where $n$ is the total number of samples and $c$ is the number of correct ones. The paper uses a low-variance estimator (Appendix A.2) for stable estimation at large k.
- **Experimental setup (Table 1):** Mathematics — LLaMA-3.1-8B, Qwen2.5-7B/14B/32B-Base, Qwen2.5-Math-7B trained with SimpleRLZoo, Oat-Zero, DAPO (GRPO); benchmarks GSM8K, MATH500, Minerva, Olympiad, AIME24, AMC23. Coding — Qwen2.5-7B-Instruct, DeepSeek-R1-Distill-Qwen-14B trained with Code-R1, DeepCoder (GRPO); benchmarks LiveCodeBench, HumanEval+. Visual — Qwen2.5-VL-7B trained with EasyR1 (GRPO); benchmarks MathVista, MathVision. Deep analysis — Qwen2.5-7B-Base, Qwen2.5-7B-Instruct, DeepSeek-R1-Distill-Qwen-7B trained with VeRL (PPO, GRPO, Reinforce++, RLOO, ReMax, DAPO); benchmark MATH500 with Omni-Math-Rule verifier.
- **Sampling protocol:** Temperature 0.6, top-p 0.95, max 16,384 tokens. Base models evaluated with few-shot examples (consistent with common practice); RLVR models with their trained templates.
- **Coverage analysis (Table 2):** On AIME24 (k=1024) and MATH500 (k=128), comparing SimpleRLZoo vs base: 63.3% of AIME24 problems are solvable by both; 13.3% by base only; 0.0% by RL only; 23.3% by neither. On MATH500: 92.4% both, 3.6% base only, 1.0% RL only, 3.0% neither.
- **Results:** RLVR improves pass@1 across all settings but base models achieve higher pass@k at large k. Six RL algorithms perform similarly ($\Delta_{\text{SE}}$ shows only minor variation yet remains consistently large). Distillation expands the reasoning boundary beyond the base model.

### Simplifications in This Notebook

- Smaller model: 3-layer MLP (~19,475 parameters) instead of 7B-32B LLMs
- Synthetic single-digit arithmetic (1-9 +/- 1-9) instead of competition-level math/code/vision benchmarks
- pass@k computed over n=64 samples per problem for k=1..64, instead of k up to 1024
- "Base model" = SFT-pretrained MLP on 3000 arithmetic examples (10 epochs, lr 1e-3); "RL model" = same MLP after GRPO-style RL training
- 40 RL iterations with 50 questions each, group size G=16, learning rate 1e-3
- Verifiable reward: +1 for correct answer, 0 for incorrect (same as paper's rule-based verifier)
- No KL penalty (faithful to the paper, which also sets β=0 for R1-Zero-like training)
- No temperature/top-p sampling — the MLP outputs a discrete distribution directly, and sampling is exact

### What the Notebook Result Shows

On 200 held-out test arithmetic problems, the SFT "base" model achieves pass@1 of 0.3855 and the RL-trained model achieves pass@1 of 0.3864 — the RL model wins at k=1 (better sampling efficiency). However, at k=2 the base model overtakes (0.5002 vs 0.4004), and the gap widens dramatically: at k=64, the base model reaches 0.9950 while the RL model only reaches 0.4150. The pass@k curves cross at k=2, reproducing the paper's crossover phenomenon.

The coverage analysis shows the base model solves 58% of problems that the RL model cannot (at k=64), while 0% are solvable by RL only — directly reproducing the paper's Table 2 finding that base models cover a wider set of problems. The RL model shows drastically reduced response diversity (1.51 vs 9.38 mean unique answers per problem), consistent with the paper's finding that RLVR reduces output entropy (Section 4.5). With a ~19K-parameter model and single-digit arithmetic, the crossover is qualitatively reproduced but not quantitatively comparable to the paper's results on AIME24 with 7B-32B models.

## Influence

The paper received the NeurIPS 2025 Best Paper Runner-Up award and the ICML 2025 AI4Math Workshop Best Paper award, making it one of the most impactful critical analyses of the RLVR paradigm. Its central finding — that RLVR improves sampling efficiency but does not expand reasoning capacity beyond the base model — reshaped the debate around what RL actually contributes to reasoning models like DeepSeek-R1 and o1. Follow-up work on improving RL exploration, continual data scaling, and multi-turn agent-environment interaction cited this paper's findings as motivation. The distinction between RLVR (distribution sharpening) and distillation (boundary expansion) influenced subsequent research into hybrid approaches. About 800 citations ([OpenAlex](https://openalex.org/works/W7196963086), checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/064_rl-reasoning-beyond-base-model/solution.ipynb) | PyTorch implementation of the pass@k crossover experiment: trains a tiny MLP on synthetic arithmetic with SFT then GRPO-style RL, and plots pass@k curves showing the base model overtaking the RL model at large k |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
