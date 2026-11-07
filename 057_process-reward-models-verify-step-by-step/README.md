---
title: "057 Process Reward Models"
nav_order: 58
permalink: "/057_process-reward-models-verify-step-by-step/"
---
# Let's Verify Step by Step (Process Reward Models)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/process-reward-models-verify-step-by-step)

- **Paper:** [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- **Authors:** Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, Karl Cobbe
- **Lab, year:** OpenAI, 2023

## Summary

In May 2023, researchers at OpenAI introduced **Process Reward Models (PRMs)**, a method for training reward models that evaluate each intermediate step in a language model's reasoning, rather than just checking whether the final answer is correct. The paper directly compares process supervision (feedback per step) with outcome supervision (feedback only on the final answer), and finds that process supervision significantly outperforms outcome supervision for training models to solve problems from the MATH dataset (a benchmark of competition-level mathematics).

The key idea is simple: when a model makes a multi-step reasoning error, outcome supervision can only tell you the solution is wrong, but not where it went off track. Process supervision labels each step as positive (correct), negative (incorrect), or neutral (ambiguous), so the reward model learns to pinpoint the exact step where reasoning breaks down.

The authors collected 800,000 step-level human feedback labels (released as PRM800K) to train their best PRM, which solves 78% of problems from a representative subset of the MATH test set when used for best-of-N reranking (selecting the best solution from many candidates). They also show that active learning — selectively labeling only the most informative solutions — significantly reduces the cost of human data collection.

This notebook implements a simplified PRM vs ORM comparison with GPT-2 small (124M parameters) on synthetic 2-step arithmetic problems, demonstrating step-level scoring, best-of-N reranking, and the qualitative finding that process supervision outperforms outcome supervision.

## What Problem Does It Solve

Imagine a math teacher grading a student's work. If the teacher only marks the final answer as right or wrong, the student doesn't know where they made a mistake — was it the first step or the last? But if the teacher checks each step and marks where the error occurred, the student can learn exactly what went wrong and fix it. The same is true for language models: telling a model which step was wrong is more informative than just saying the answer is wrong.

Before this paper, most reward models for reasoning used outcome supervision — they checked only the final answer. The authors suspected this was wasteful: a solution that gets the right answer through flawed reasoning (a "false positive") would be mislabeled as correct, teaching the model bad habits. Process supervision avoids this by checking every step, so a solution with correct reasoning throughout gets a high score, while one that reaches the right answer by luck gets a low score.

## Core Idea

- **Step-level labeling:** Human annotators label each reasoning step as positive (correct), negative (incorrect), or neutral (ambiguous). This gives the reward model far more signal than a single correct/incorrect label for the whole solution.
- **PRM scoring:** The PRM predicts correctness at the last token of each step. The solution-level score is the product of per-step correctness probabilities (the best-performing reduction, per Appendix F.2), so a single wrong step drags the score down dramatically.
- **Best-of-N reranking:** To evaluate a reward model, generate N candidate solutions per problem, select the one the reward model ranks highest, and check if it's correct. This directly measures whether the reward model can identify good solutions — the property that matters for downstream RL training.

## Key Method Details

- **Base model (Section 2.2):** All large-scale models are fine-tuned from GPT-4, pretrained solely on next-token prediction (no RLHF). Small-scale models are similar in design but use roughly 200× less compute. All models are additionally fine-tuned on ~1.5B math-relevant tokens (MathMix).
- **Generator (Section 2.3):** The generator is fine-tuned to produce solutions in a newline-delimited step-by-step format, by few-shot generating solutions to MATH training problems, filtering for correct final answers, and fine-tuning for one epoch.
- **ORM (Section 2.5):** Trained to predict whether each solution is correct or incorrect, using the ORM's prediction at the final token as the overall score. Training labels come from automatic final-answer checking.
- **PRM (Section 2.6):** Trained to predict correctness at the last token of each step. The solution-level score is the product of per-step probabilities (Appendix F.2). Trained on PRM800K (800K step-level human labels).
- **Results (Figure 3):** At best-of-1860, the ORM solves 72.4% of test problems, the PRM solves 78.2%, and majority voting solves 69.6%. The PRM outperforms both the ORM and majority voting.
- **Active learning (Section 4.2):** A selector model (PRM_selector) trained on 1 sample per problem scores 1000 samples; 80% of selected samples are the most convincing wrong-answer samples, 20% are the most convincing remaining samples. This focuses labeling budget on the most informative solutions.
- **OOD generalization (Table 1):** On a held-out set of 224 STEM questions (AP Physics, AP Calculus, AP Chemistry, AMC10/12), the PRM achieves 72.9% (best-of-100), vs 63.8% for the ORM and 61.3% for majority voting.

### Simplifications in This Notebook

- **Smaller model:** GPT-2 small (124M parameters) instead of GPT-4 — roughly 1000× smaller
- **Dataset:** Synthetic 2-step arithmetic problems (200 training, 50 test) instead of the MATH dataset
- **Labels:** Ground-truth step labels from problem generation instead of 800K human annotations (PRM800K)
- **Training scale:** 1,000 training solutions, 15 epochs with AdamW (lr 5e-5) instead of large-scale training
- **Best-of-N:** N up to 5 instead of N up to 1,860
- **No active learning, no OOD evaluation, no MathMix pretraining**

### What the Notebook Result Shows

On 50 held-out test problems with 5 candidate solutions each (20% correct, 80% incorrect), the PRM (process supervision) outperforms the ORM (outcome supervision) at N=3 (28% vs 16%) and N=5 (36% vs 20%). At N=1 all methods score 22% (the fraction of correct solutions in the pool). The PRM's product-of-step-probabilities scoring correctly penalizes solutions with any incorrect step — for example, a solution with step labels [1, 0] gets a PRM score of 0.069, while a fully correct solution [1, 1] gets 0.208. The ORM, which only sees the final answer, scores the fully correct solution at 0.9999 but struggles to distinguish solutions that reach wrong answers through different error patterns.

The PRM's step-level visualization shows high scores for correct steps and low scores for incorrect steps, demonstrating the core advantage of process supervision: pinpointing where reasoning goes wrong. Training accuracy reaches 78.3% for the PRM (step-level) and 94.9% for the ORM (outcome-level), reflecting that outcome labeling is easier to learn but less informative for reranking.

These results are not comparable to the paper's 78.2% solve rate on MATH, which used GPT-4, 800K human labels, and up to 1,860 candidate solutions. The notebook demonstrates the qualitative finding (process > outcome supervision at higher N) at a miniature scale.

## Influence

"Let's Verify Step by Step" established process supervision as a principled alternative to outcome supervision for reward modeling in mathematical reasoning. The release of PRM800K — the first large-scale public dataset of step-level human feedback — catalyzed research on process reward models. Key follow-ups include Math-Shepherd (Wang et al., 2024), which automates step-level labeling without human annotation; OpenAI's o1 system (2024), which uses process-based reward models for reasoning; and a wave of work integrating PRMs into RL training pipelines (e.g., DeepSeek-R1, 2025). The paper also influenced the broader trend of verifiable rewards in LLM training, where step-level or outcome-level correctness can be checked automatically (as in GRPO, 2024).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/057_process-reward-models-verify-step-by-step/solution.ipynb) | PyTorch PRM and ORM trained on synthetic 2-step arithmetic with GPT-2 small, compared via best-of-N reranking (process > outcome supervision) |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
