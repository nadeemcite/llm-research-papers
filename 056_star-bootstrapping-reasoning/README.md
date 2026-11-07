---
title: "056 STaR"
nav_order: 57
permalink: "/056_star-bootstrapping-reasoning/"
---
# STaR: Bootstrapping Reasoning With Reasoning (STaR)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/star-bootstrapping-reasoning)

- **Paper:** [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)
- **Authors:** Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman
- **Lab, year:** Stanford University and Google Research, 2022 (NeurIPS 2022)

## Summary

In March 2022, researchers at Stanford University and Google Research introduced **STaR (Self-Taught Reasoner)**, a method that lets a language model improve its own reasoning ability by bootstrapping from a small set of examples with chain-of-thought rationales. The core idea is a simple iterative loop: generate rationales (step-by-step reasoning) for many questions, keep only the rationales that lead to correct answers, fine-tune the model on those correct rationales, and repeat. Each iteration, the improved model generates better rationales, which become better training data, creating a positive feedback loop.

A key innovation is **rationalization**: when the model answers a question incorrectly, the correct answer is given as a hint, and the model is asked to produce a rationale that justifies that answer. These "rationalized" examples are added to the training set as if the model had discovered the rationale on its own. This lets the model learn from problems it could not initially solve, by reasoning backward from the answer.

The authors applied STaR to arithmetic, CommonsenseQA (a 5-way multiple-choice commonsense reasoning dataset), and GSM8K (grade-school math word problems). Using GPT-J (6B parameters) as the base model, STaR improved CommonsenseQA dev-set accuracy from 36.6% (few-shot chain-of-thought) to 72.5%, comparable to a fine-tuned GPT-3 (73.0%) that is 30× larger. On arithmetic, STaR reached near-perfect accuracy on 2-5 digit addition after 16 iterations, from a starting point of less than 1%. On GSM8K, STaR improved from 3.1% (few-shot CoT) to 10.7%.

This notebook implements a simplified STaR loop with GPT-2 small (124M parameters) on 2-digit addition, demonstrating the bootstrap cycle — rationale generation, filtering by correctness, rationalization for failures, and iterative fine-tuning — with a model 50× smaller than the paper's GPT-J.

## What Problem Does It Solve

Imagine a student who can solve easy problems but gets stuck on hard ones. If the student only practices what they already know, they never improve on the hard problems. But if a teacher gives them the answer to a hard problem and asks "can you explain why that's the answer?", the student can learn the reasoning process by working backward. Next time, they might solve a similar problem on their own.

This is exactly what STaR does for language models. Before STaR, getting a language model to produce step-by-step reasoning (chain-of-thought rationales) required either expensive human annotation of thousands of examples, or using few-shot prompting — showing a few examples in the prompt — which improved reasoning but still underperformed models fine-tuned on large datasets. STaR bridges this gap: it starts with just a few hand-written rationale examples (the few-shot prompt), generates rationales for many more questions, keeps the correct ones, and iteratively trains on them, bootstrapping from the model's own successes.

## Core Idea

- **Rationale generation bootstrapping:** Few-shot prompt the model to generate a rationale and answer for each question. Filter to keep only rationales that produce the correct answer. Fine-tune the original pre-trained model on these correct rationales. Repeat with the improved model. This is the core loop — the model teaches itself by learning from its own correct reasoning.
- **Rationalization:** For questions the model gets wrong, provide the correct answer as a hint and ask the model to generate a rationale justifying it. Add these to the training set without the hint, as if the model reasoned it out independently. This exposes the model to harder problems and breaks the plateau where it can only learn from problems it already solves.
- **Iterative refinement:** Each outer-loop iteration produces a better model, which generates better rationales, creating higher-quality training data for the next iteration. The process continues until performance plateaus.

## Key Method Details

- **Base model (Section 4.1):** GPT-J (6B parameters), fine-tuned using the GPT-J repository script with a 100-step learning rate warmup and constant learning rate. Training steps start at a small number and increase each outer-loop iteration.
- **Arithmetic task (Section 4.2):** 50,000 randomly sampled n-digit addition problems (1-5 digits) in the scratchpad format from Nye et al. (2021). Each problem shows the input (e.g., "6 2 4 + 2 5 9"), and the model generates a scratchpad (step-by-step digit addition with carry) followed by the final answer. 10,000 problems sampled per outer-loop iteration.
- **CommonsenseQA (Section 4.4, Table 1):** 9,741 train questions, 1,221 dev questions, 5 choices each. Few-shot direct GPT-J: 20.9%. Few-shot CoT GPT-J: 36.6%. GPT-J Direct Finetuned: 60.0%. STaR without rationalization: 68.8%. STaR with rationalization: 72.5%. GPT-3 Direct Finetuned (30× larger): 73.0%.
- **GSM8K (Section 4.5, Table 2):** 7,473 train, 1,319 test. Few-shot CoT GPT-J: 3.1%. GPT-J Direct Finetuned: 5.8%. STaR without rationalization: 10.1%. STaR with rationalization: 10.7%.
- **Rationalization benefit (Section 4.3):** On arithmetic, rationalization lets the model learn multiple digit lengths simultaneously (rather than the stage-wise progression seen without it), and enables out-of-distribution generalization to 9-10 digit addition never seen in training.

### Simplifications in This Notebook

- **Smaller model:** GPT-2 small (124M parameters) instead of GPT-J (6B) — 50× smaller
- **Dataset:** 2-digit addition only (1,600 training problems) instead of 50,000 problems spanning 1-5 digits
- **Outer-loop iterations:** 3 instead of 16-36
- **Fine-tuning:** 50 steps per iteration with AdamW (lr 5e-5) instead of the paper's increasing step schedule with warmup
- **Few-shot prompt:** 4 hand-written examples instead of 10 per digit
- **Rationalization:** included, following the paper's approach of providing the correct answer as a hint
- **Sampling:** greedy decoding (temperature 0) instead of the paper's low-temperature sampling

### What the Notebook Result Shows

On 2-digit addition with GPT-2 small (124M), the STaR loop produces 0% test accuracy across all 3 iterations — the base model starts at 0% (0/200) and remains at 0% after each round of fine-tuning. However, the number of correctly-generated training rationales grows slightly (10 → 11 → 12 out of 1,600 problems), and rationalization contributes a few additional examples (1, 3, 0 per iteration).

The paper explicitly notes that "GPT-2 was not able to bootstrap (a process where a model learns from its own outputs) from few-shot reasoning in even the arithmetic domain" (Section 6), so this result is consistent with the paper's own finding — GPT-2 small lacks the reasoning capacity to bootstrap. The key output is the qualitative demonstration of the STaR mechanics (rationale generation, filtering, rationalization, iterative fine-tuning), not accuracy improvement, which would require a model large enough to bootstrap.

## Influence

STaR pioneered the paradigm of self-improving reasoning through bootstrapping, directly inspiring a wave of self-play and iterative reasoning methods. Key follow-ups include STaR\* (Huang et al., 2023), which extends STaR with self-consistency filtering; ReST (Gulcehre et al., 2023), which separates growing and improving phases; and the line of work on reasoning via reinforcement learning (DeepSeek-R1, 2025). STaR is also conceptually related to Expert Iteration (Anthony et al., 2017) and Self-Rewarding Language Models (Yuan et al., 2024). The idea of learning from self-generated reasoning has become a standard technique in modern reasoning model training.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/056_star-bootstrapping-reasoning/solution.ipynb) | PyTorch STaR loop with pretrained GPT-2 small on 2-digit addition, demonstrating rationale generation, filtering, rationalization, and iterative fine-tuning (3 iterations) |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
