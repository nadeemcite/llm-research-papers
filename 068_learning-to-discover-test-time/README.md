---
title: "068 Learning to Discover"
nav_order: 69
permalink: "/068_learning-to-discover-test-time/"
---
# Learning to Discover at Test Time (TTT-Discover)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/learning-to-discover-at-test-time)

- **Paper:** [Learning to Discover at Test Time](https://arxiv.org/abs/2601.16175)
- **Authors:** Mert Yuksekgonul, Daniel Koceja, Xinhao Li, Federico Bianchi, Jed McCaleb, Xiaolong Wang, Jan Kautz, Yejin Choi, James Zou, Carlos Guestrin, Yu Sun
- **Lab, year:** Stanford University, UC San Diego, Together AI, Astera Institute, NVIDIA, 2026 (preprint)

## Summary

How can AI discover a new state of the art for a scientific problem? Prior work in test-time scaling, such as AlphaEvolve, performs search by prompting a frozen LLM — the model generates many candidate solutions and picks the best, but the LLM itself never improves. TTT-Discover takes a different approach: it continues to *train* the LLM at test time on the single problem it is trying to solve, using reinforcement learning so the policy gets better with experience specific to that problem.

The key insight is that discovery problems differ from standard RL in two ways. First, the policy only needs to solve *this one* problem, not generalize. Second, we only need *one* great solution, not good average performance. The paper's entropic objective favors the maximum-reward action (approximating max as temperature grows), and its PUCT-based state reuse prioritizes the most promising previous solutions as starting points for new attempts. The method uses an open model (gpt-oss-120b) with LoRA fine-tuning via the Tinker API, costing about $500 per problem.

TTT-Discover sets new state-of-the-art results in mathematics (Erdős' minimum overlap problem: 0.380876 vs 0.380924, and a first autocorrelation inequality: 1.50287 vs 1.50317), GPU kernel engineering (TriMul competition: up to 2× faster than prior art across A100, H100, B200, and MI300X), algorithm design (AtCoder Heuristic Contests: 1st place scores on both ahc039 and ahc058), and biology (single-cell denoising: 0.71 vs 0.64 on PBMC). Expert reviews from mathematicians, GPUMode organizers, and biologists validate the results.

This notebook implements the core TTT-Discover algorithm — entropic objective with adaptive beta, PUCT-based buffer reuse, and test-time policy training — on a Traveling Salesman Problem (TSP-20) where the policy learns to propose 2-opt moves. TTT-Discover finds a shorter tour (57.4) than both Best-of-N (80.2) and No-TTT PUCT-only search (57.6) with the same sampling budget.

## What Problem Does It Solve

Imagine you're given a really hard programming assignment that requires ideas beyond anything in the textbook. You could try guessing the solution many times — eventually you might stumble on something that works. But each guess doesn't make you any smarter; you're just rolling dice. Alternatively, you could *learn from your failed attempts*: each try teaches you something about the problem, and your next attempt is more intelligent than the last.

Prior AI approaches to scientific discovery are like the first strategy — they prompt a frozen LLM to generate many solutions and pick the best, but the LLM never learns from its attempts. TTT-Discover is like the second strategy: the model actually trains on its own attempts, getting better at generating solutions specific to this one problem. The paper shows this learning-based approach beats pure search, mirroring how learning superseded pure search in AI history (AlphaGo learning to play Go rather than just searching moves).

## Core Idea

- **Test-time training for discovery:** Instead of keeping the LLM frozen, continue to update its weights at test time using RL on the single problem being solved. Each attempt provides training data specific to that problem — the most valuable data for an out-of-distribution challenge.
- **Entropic objective:** Standard RL optimizes average reward, but discovery only needs one great solution. The entropic objective J_beta = E_s[log E_a[exp(beta * R(s,a))]] approximates the maximum as beta grows, focusing learning on the best outcomes rather than the average. An adaptive beta per initial state prevents instability (too large beta early) and vanishing advantages (too small beta late).
- **PUCT state reuse:** A PUCT-inspired rule selects which previous solution to reuse as the starting point for new rollouts, balancing exploitation (high-reward states are more likely to seed high-reward children) with exploration (under-visited states get a bonus). Unlike standard RL's mean, Q(s) uses the *maximum* child reward — we care about the best outcome, not the average.

## Key Method Details

- **Algorithm:** TTT-Discover (Algorithm 1): for each of N steps, select an initial state via PUCT, generate a batch of rollouts, evaluate rewards, update the buffer, then train the policy with the entropic objective.
- **Policy:** gpt-oss-120b with LoRA rank 32, run on Tinker (Thinking Machines API). 50 training steps, 512 rollouts per step (8 groups of 64). High reasoning effort. Context window limited to 32,768 tokens on Tinker.
- **Entropic objective:** J_beta(theta) with adaptive beta(s) set per initial state by constraining KL divergence of the induced policy. KL penalty to initial policy pi_theta0 with coefficient lambda. As beta → ∞, objective → max.
- **PUCT:** score(s) = Q(s) + c · P(s) · sqrt(1+T) / (1+n(s)), where Q(s) = max child reward, P(s) ∝ rank by reward, n(s) = expansion count, T = total expansions.
- **Cost:** ~$500 per problem on Tinker (50 steps × 512 rollouts, ~3000-token prompts, ~16000 sampling tokens average).
- **Results:** Erdős' min. overlap 0.380876 (prev. best 0.380924); AC1 inequality 1.50287 (prev. best 1.50317); TriMul A100 2198 µs (prev. best 4531 µs, 50% faster); AtCoder ahc039 567,062 (1st place); ahc058 848,414,228 (1st place); denoising PBMC 0.71 (prev. best 0.64).

### Simplifications in This Notebook

- Toy 4D multimodal optimization (mixture of Gaussian wells) instead of Erdős, GPU kernels, or AtCoder contests
- Tiny MLP policy (2 hidden layers of 128 units, ~18K parameters) instead of gpt-oss-120b with LoRA rank 32
- 100 TTT steps with 256 rollouts per step (16 groups of 16) instead of 50 steps with 512 rollouts (8 groups of 64)
- Incremental delta proposals instead of LLM-generated code/algorithms
- Adaptive beta schedule (linear increase from 0.5 to 4.0) instead of the paper's KL-constrained adaptive scheme
- Adam optimizer with lr 5e-4 instead of LoRA fine-tuning on a 120B model

### What the Notebook Result Shows

On the 4D multimodal optimization problem, TTT-Discover finds a solution with reward 0.987, improving from an initial sota reward of ~0 (a deliberately poor starting point far from all wells). This beats both Best-of-25600 (reward ~0, the frozen policy cannot reach the wells from the starting point in a single step) and No-TTT PUCT-only search (reward 0.973) — demonstrating that test-time training adds value beyond pure search.

The mean reward per step increases over TTT steps, showing the policy genuinely improves from its attempts. The adaptive beta increases from 0.5 to 4.0, confirming the paper's observation that beta should start small for stability and grow to focus on the maximum. The gap to the true optimum (1.002) is only 0.015 — the toy policy can nearly solve the problem with enough TTT steps. Results are not comparable to the paper's state-of-the-art discoveries in mathematics, kernel engineering, algorithm design, or biology.

## Influence

TTT-Discover is a very recent paper (January 2026) from Stanford, Together AI, and NVIDIA. It builds on the test-time training framework (Sun et al. 2020) and directly extends AlphaEvolve (Novikov et al. 2025) by adding RL-based policy improvement to evolutionary search. Concurrent works include EvoTune (Surina et al. 2025), MiGrATe (Phan et al. 2025), and ThetaEvolve (Wang et al. 2025a). The paper's finding that learning beats search at test time for discovery problems may influence future research on AI-driven scientific discovery, following the historical pattern where learning superseded search in Go and protein folding.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/068_learning-to-discover-test-time/solution.ipynb) | From-scratch PyTorch TTT-Discover on a 4D multimodal optimization problem with entropic objective, PUCT reuse, and Best-of-N / No-TTT baselines |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
