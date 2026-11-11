---
title: "066 Absolute Zero"
nav_order: 67
permalink: "/066_absolute-zero-reinforced-self-play-reasoning/"
---
# Absolute Zero: Reinforced Self-play Reasoning with Zero Data (AZR)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/absolute-zero-reinforced-self-play-reasoning)

- **Paper:** [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://arxiv.org/abs/2505.03335)
- **Authors:** Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Yang Yue, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, Gao Huang
- **Lab, year:** Tsinghua University (Zhao, Yue, Xu, Lin, Huang), Beijing Institute for General Artificial Intelligence (Wu T., Yue, Zheng), Penn State University (Wu Y., Wu Q.), 2025 (ICML 2025)

## Summary

In May 2025, researchers at Tsinghua University, BIGAI, and Penn State introduced the **Absolute Zero** paradigm — a way to train reasoning models that requires no external data at all. Prior RLVR (Reinforcement Learning with Verifiable Rewards) methods like GRPO and R1-Zero eliminated the need for human reasoning traces, but still depended on manually curated question-answer pairs for training. The Absolute Zero paradigm goes further: a single model simultaneously *proposes* tasks that maximize its own learning progress and *solves* them, with a code executor providing all the verification needed.

Their instantiation, the **Absolute Zero Reasoner (AZR)**, works with (program, input, output) triplets. The model plays two roles: as a *proposer*, it generates new (program, input) pairs; as a *solver*, it predicts the missing element — the output (deduction), the input (abduction), or the program itself (induction). A Python interpreter serves as the sole verifier, executing code to validate tasks and check answers.

The proposer is rewarded via a **learnability reward**: it performs G Monte Carlo rollouts of the solver, and tasks where the solver succeeds sometimes (but not always) get the highest reward. This steers task generation toward the frontier of the model's current ability. Training uses **Task-Relative REINFORCE++ (TRR++)**, which computes separate baselines for each of the 6 task-role configurations, providing structured variance reduction for the multitask setting.

Despite training entirely without external data, AZR-Coder-7B achieves state-of-the-art on the combined average of coding and math benchmarks, outperforming models trained on tens of thousands of expert-curated examples. It surpasses the previous best by 1.8 absolute points in average, and by 0.3 points in coding. Cross-domain transfer is striking: AZR trained on self-proposed code tasks improves math average by 15.2 points (vs only 0.65 for expert code models trained with human data). The method scales: 3B, 7B, and 14B coder models gain +5.7, +10.2, and +13.2 points respectively.

This notebook implements AZR with three small MLP solvers (~93K parameters total) on simple Python lambda expressions, demonstrating the full self-play loop: propose tasks via stochastic mutation, verify with Python, compute learnability rewards, and update with TRR++.

## What Problem Does It Solve

Imagine teaching a child math by giving them worksheets. You have to write the problems, work out the answers, and grade their work. If the child is smarter than you, the problems you create become too easy — you can't challenge them anymore. This is the scalability problem facing RLVR: it depends on human experts to curate training data, and as models approach or surpass human intelligence, human-provided tasks offer diminishing learning potential.

AZR's insight is that if you have a Python interpreter, you don't need a human to write problems. The model can *invent its own problems* — programs with inputs — and the interpreter provides ground truth by executing them. The only question is: which problems should it generate? AZR answers this with the learnability reward: generate problems where the solver *sometimes* succeeds. Too easy (always solves) or too hard (never solves) means no learning signal. Moderate difficulty — the frontier of the model's ability — is where learning happens.

## Core Idea

- **Self-play without external data:** A single model proposes tasks and solves them. No human-curated questions, answers, or reasoning traces are needed — only a code executor for verification (Section 2.2).
- **Three reasoning modes from one triplet:** AZR reasons about (program, input, output) triplets in three ways: deduction (predict output from program+input), abduction (find input from program+output), and induction (synthesize program from input-output examples) (Section 3.2).
- **Learnability reward for the proposer:** The proposer is rewarded for generating tasks of moderate difficulty. G solver rollouts estimate the success rate $\bar{r}_{solve}$; the reward is $r_{propose} = 0$ if $\bar{r}_{solve} = 0$, otherwise $1 - \bar{r}_{solve}$. Tasks that are too easy or too hard get zero reward (Equation 4, Section 3.1).
- **Task-Relative REINFORCE++ (TRR++):** Instead of a single global baseline, TRR++ computes separate mean/std baselines for each of the 6 task-role configurations (3 task types × 2 roles), interpolating between per-question baselines (GRPO) and global baselines (REINFORCE++) (Section 3.3.5, Equation 8).

## Key Method Details

- **AZR objective (Equation 3):** $\mathcal{J}(\theta) = \max_\theta \mathbb{E}_{z \sim p(z)} \left[ \mathbb{E}_{(x, y^\star) \sim f_e(\cdot\|\tau), \tau \sim \pi_\theta^{propose}(\cdot\|z)} \left[ \lambda r_e^{propose}(\tau, \pi_\theta) + \mathbb{E}_{y \sim \pi_\theta^{solve}(\cdot\|x)} \left[ r_e^{solve}(y, y^\star) \right] \right] \right]$, where $\lambda$ balances proposer and solver rewards.
- **Solver reward (Equation 5):** $r_{solve} = \mathbb{I}(y = y^\star)$ — binary correctness via Python value equality.
- **Proposer reward (Equation 4):** $r_{propose} = 0$ if $\bar{r}_{solve} = 0$, else $1 - \bar{r}_{solve}$, where $\bar{r}_{solve} = \frac{1}{G}\sum_{i=1}^G r_{solve}^{(i)}$.
- **TRR++ advantage (Equation 8):** $A^{norm}_{task,role} = (r - \mu_{task,role}) / \sigma_{task,role}$, computed separately for each of the 6 task-role pairs.
- **Training (Section 4.1):** Batch size 64×6 (2 roles × 3 task types), constant learning rate 1e-6, AdamW optimizer. No KL penalty in the final system.
- **Results (Table 1):** AZR-Coder-7B achieves SOTA on combined average (coding + math), outperforming the previous best by 1.8 points. Cross-domain: math average improves by 15.2 points despite training only on self-proposed code tasks. Scaling: 3B +5.7, 7B +10.2, 14B +13.2 points.

### Simplifications in This Notebook

- Three small MLP classifiers (~93K parameters total) instead of Qwen2.5-7B (7B parameters)
- Simple Python lambda expressions (5 operators × 19 operands) instead of arbitrary programs
- 80 AZR iterations with batch size 8 per task type instead of hundreds of steps with batch 64
- 4 Monte Carlo rollouts (G=4) for learnability estimation instead of G=8
- No KL penalty (the paper also omits KL for the final system)
- Seeded buffer with 16 hand-crafted triplets instead of LLM-generated seed programs
- Program proposer uses stochastic mutation instead of LLM text generation

### What the Notebook Result Shows

On 40 held-out test tasks per reasoning mode, the trained solvers show mixed results. Induction improves from 0.0 to about 0.13, while deduction and abduction show no improvement or slight regression. The proposer learnability reward stabilizes around 0.09–0.12. The average accuracy is roughly flat (about 0.09 for both baseline and trained).

With ~93K parameters and 80 iterations, the improvement is modest and inconsistent — the paper uses a 7B model trained over hundreds of iterations with batch size 64 and G=8 rollouts, achieving state-of-the-art performance on coding and math benchmarks. The result is not comparable to the paper's LiveCodeBench or AIME numbers, but demonstrates the core self-play loop: propose, verify with Python, solve, compute learnability reward, update with TRR++.

## Influence

AZR introduced the "Absolute Zero" paradigm — training reasoning models with zero external data — which sparked significant interest in self-play RL for LLMs. It was accepted at ICML 2025. The key insight that a code executor can serve as a unified verifier for open-ended yet grounded self-play has influenced subsequent work on self-improving reasoning systems. The learnability reward concept (drawing from autotelic agents and unsupervised environment design) has been adopted in follow-up self-play approaches. The paper also raised safety concerns: the "uh-oh moment" — emergent concerning chains of thought from the Llama-3.1-8B variant — highlighted risks in autonomous self-play training. About 340 citations ([OpenAlex](https://www.openalex.org/works/W4416010183), checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/066_absolute-zero-reinforced-self-play-reasoning/solution.ipynb) | From-scratch PyTorch AZR implementation: three MLP solvers that self-play propose-and-solve code reasoning tasks with a Python verifier, learnability reward, and TRR++ updates |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
