---
title: "048 ORPO"
nav_order: 49
permalink: "/048_orpo-monolithic-preference-optimization/"
---
# ORPO: Monolithic Preference Optimization without Reference Model (ORPO)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/orpo-monolithic-preference-optimization)

- **Paper:** [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)
- **Authors:** Jiwoo Hong, Noah Lee, James Thorne
- **Lab, year:** KAIST AI, 2024 (preprint)

## Summary

In March 2024, researchers at KAIST introduced **ORPO (Odds Ratio Preference Optimization)**, a method that combines supervised fine-tuning (SFT) and preference alignment into a single training stage — eliminating the need for a separate reference model. Prior methods like DPO (Direct Preference Optimization) require two stages: first an SFT warm-up, then a preference alignment phase that uses a frozen copy of the SFT model as a reference. ORPO collapses these into one step by adding an odds-ratio penalty to the standard SFT loss.

The key insight is that ordinary SFT, which maximizes the likelihood of chosen responses, inadvertently increases the likelihood of rejected responses too — because cross-entropy loss has no mechanism to penalize undesired tokens. ORPO fixes this by appending an odds-ratio term that contrasts the odds of the chosen response against the rejected response. The odds of a response y given input x is defined as P(y\|x) / (1 − P(y\|x)), and the odds ratio measures how much more likely the model is to generate the chosen response than the rejected one. This ratio is wrapped in a log-sigmoid function, so minimizing it pushes up the chosen odds and pushes down the rejected odds.

Fine-tuning Phi-2 (2.7B), Llama-2 (7B), and Mistral (7B) with ORPO on UltraFeedback alone surpassed models with more than 7B and 13B parameters: achieving up to 12.20% on AlpacaEval 2.0, 66.19% on IFEval (instruction-level loose), and 7.32 on MT-Bench. Mistral-ORPO-α and Mistral-ORPO-β outperformed Zephyr β, which uses a two-stage SFT+DPO pipeline.

This notebook implements the ORPO loss from scratch in PyTorch on a tiny character-level GRU language model, trains it on synthetic preference-pair data, and compares it with DPO on the same data. The odds-ratio penalty successfully pushes down rejected log-probability while pushing up chosen log-probability, without any reference model.

## What Problem Does It Solve

Imagine you are teaching a student to write good essays. The standard approach (SFT) is to show them good essays and say "write like this." But the student also picks up general writing habits — including bad ones — because showing good essays doesn't explicitly tell them what *not* to do. DPO fixes this by showing pairs of essays (one good, one bad) and saying "prefer the good one," but it requires keeping a copy of the student before training (the reference model) to measure how much their preferences have changed — doubling memory and compute.

ORPO is like grading essays with a simple rule: "make the good essay more likely and the bad essay less likely, all in one go." You don't need a "before" snapshot of the student. The odds ratio — which compares how likely each essay is — does the job of the reference model, but using only the current model's own probabilities. This halves the number of forward passes per batch (2 instead of 4) and removes the need for an SFT warm-up stage entirely.

## Core Idea

- **Monolithic training:** ORPO combines SFT and preference alignment into a single training run. There is no separate SFT warm-up phase and no reference model — one model, one loss, one stage.
- **Odds-ratio penalty:** The ORPO loss adds an odds-ratio term to the standard NLL loss. The odds ratio contrasts the odds of the chosen response against the rejected response, pushing the model to favor chosen over rejected without needing a reference model's probabilities.
- **Reference-free contrast:** Unlike DPO, which uses the log-ratio log π_θ(y\|x) / π_ref(y\|x) requiring a frozen reference model, ORPO uses the odds P(y\|x) / (1 − P(y\|x)) which only depends on the current model. This halves the forward passes per batch (2 instead of 4) and saves memory.

## Key Method Details

- **ORPO loss (Eq. 6):** L_ORPO = E\_{(x, y_w, y_l)}[L_SFT + λ · L_OR], where L_SFT is the standard NLL on the chosen response and L_OR is the odds-ratio penalty
- **Odds-ratio loss (Eq. 7):** L_OR = −log σ(log(odds_θ(y_w\|x) / odds_θ(y_l\|x))), where odds_θ(y\|x) = P_θ(y\|x) / (1 − P_θ(y\|x))
- **λ (odds-ratio weight):** Controls the trade-off between SFT adaptation and preference alignment. The paper uses λ = 0.1 for Mistral, 0.2 for Llama-2, 0.25 for Phi-2
- **Gradient structure (Eq. 8):** ∇_θ L_OR = δ(d) · h(d), where δ(d) acts as a penalty that shrinks as the model learns to prefer chosen over rejected, and h(d) is a weighted contrast of the chosen and rejected gradients
- **Results:** Mistral-ORPO-β achieves 12.20% on AlpacaEval 2.0, 66.19% on IFEval, 7.32 on MT-Bench — surpassing Zephyr β and Llama-2-Chat 13B (Table 1, Figures 4, 12)
- **Computational efficiency:** ORPO requires 2 forward passes per batch (chosen + rejected) vs DPO's 4 (policy chosen + policy rejected + reference chosen + reference rejected), and no reference model in memory (Section 7.3)

### Simplifications in This Notebook

- **Tiny model:** A 2-layer character-level GRU language model (179,358 parameters) instead of OPT-125M to Mistral-7B
- **Dataset:** 400 synthetic preference pairs (chosen = repetitive pattern, rejected = random noise) instead of HH-RLHF or UltraFeedback
- **50 epochs** with AdamW (lr 5e-4), λ = 0.5 — the paper uses a single epoch on full-scale models with various λ values (0.1–0.25)
- **DPO baseline:** Uses a 30-epoch SFT reference model pre-trained on chosen completions, instead of full instruction fine-tuning
- **Sequence probability:** Uses geometric mean of per-token probabilities for odds calculation, instead of exact per-token odds

### What the Notebook Result Shows

On 100 synthetic test preference pairs, both ORPO and DPO achieve 100% preference accuracy (chosen odds > rejected odds, or chosen reward > rejected reward), while an untrained baseline scores 45%. ORPO's chosen log-probability reaches about −13.5 while rejected log-probability drops to about −107, giving a log-prob margin of about 93.5. The log odds ratio grows to about 5.4. DPO achieves a reward margin of about 6.5.

The λ ablation (0.1, 0.5, 1.0) shows all values achieve 100% accuracy with log-prob margins of about 92.8, 94.0, and 94.8 respectively, confirming that higher λ produces stronger preference separation. The key observation is that ORPO's rejected log-probability decreases during training (unlike pure SFT, which increases both), confirming the paper's Figure 7 finding. These are toy results with synthetic data and a tiny model, and are not comparable to the paper's 7B-scale AlpacaEval and MT-Bench results.

## Influence

ORPO is integrated into Hugging Face's TRL library and has been adopted as a practical alternative to DPO for resource-efficient alignment. Its key contribution — eliminating the reference model — directly influenced later reference-free alignment methods including SimPO (2024). The released Mistral-ORPO-α and Mistral-ORPO-β checkpoints were among the top-performing 7B instruction-following models on the AlpacaEval leaderboard at the time of release. About 200 citations on [OpenAlex](https://openalex.org/W4390839461) (checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/048_orpo-monolithic-preference-optimization/solution.ipynb) | From-scratch PyTorch implementation of the ORPO loss with a tiny character-level GRU LM trained on synthetic preference pairs, compared against DPO on the same data |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Q&A answered from the paper, citations, and common misconceptions about ORPO |
