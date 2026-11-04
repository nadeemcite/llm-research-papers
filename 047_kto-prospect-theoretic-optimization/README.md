---
title: "047 KTO"
nav_order: 48
permalink: "/047_kto-prospect-theoretic-optimization/"
---
# KTO: Model Alignment as Prospect Theoretic Optimization (KTO)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/kto-prospect-theoretic-optimization)

- **Paper:** [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)
- **Authors:** Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, Douwe Kiela
- **Lab, year:** Stanford University (first author was an intern at Contextual AI) and Contextual AI, 2024 (ICML 2024)

## Summary

In February 2024, researchers at Stanford and Contextual AI introduced **Kahneman-Tversky Optimization (KTO)**, a method that aligns language models with human feedback using only a binary thumbs-up / thumbs-down signal — no preference pairs required. Prior methods like DPO (Direct Preference Optimization) need paired comparisons (output A is better than output B), which are expensive to collect. KTO only needs to know whether a single output is desirable or not.

The key insight comes from prospect theory (Kahneman & Tversky, 1979), a Nobel Prize-winning framework from behavioral economics that describes how humans perceive gains and losses asymmetrically — humans are more sensitive to losses than equivalent gains (loss aversion). The authors show that successful alignment losses like DPO and PPO implicitly incorporate these human biases, and define a family of such losses called **HALOs** (human-aware losses). They then derive KTO directly from the Kahneman-Tversky value function, using a logistic function (concave in gains, convex in losses) to model how humans perceive the quality of model outputs relative to a reference point.

KTO matches or exceeds DPO performance at scales from 1B to 30B parameters despite using a weaker binary signal. On GSM8K (a math reasoning benchmark), swapping DPO for KTO when aligning Zephyr-β-SFT on UltraFeedback improved performance by 13.5 points. KTO can also handle extreme data imbalances, matching DPO while using up to 90% fewer desirable examples.

This notebook implements the KTO loss from scratch in PyTorch on a tiny character-level GRU language model, trains it on synthetic binary feedback data, and compares it with DPO on the same data. An ablation over the risk-aversion parameter β shows how it controls the learning dynamics.

## What Problem Does It Solve

Imagine you are training a chef by tasting their dishes. The old way (DPO) requires tasting two dishes side by side and saying which is better — a preference comparison. This is expensive: you need to taste pairs every time, and the feedback is only useful relative to the other dish.

KTO is like giving a simple thumbs-up or thumbs-down after each dish. You don't need to compare two dishes — just say "good" or "bad." This is much cheaper and faster to collect, and there is vastly more data available (every Yelp review, every Reddit upvote, every "like" button). The surprising finding is that this simpler signal can be just as effective, because the loss function — derived from how humans actually perceive quality — compensates for the weaker signal.

## Core Idea

- **Binary feedback instead of preferences:** KTO only needs a binary signal (desirable / undesirable) per example, not paired preferences. This makes data collection cheaper, faster, and more scalable.
- **Prospect-theoretic value function:** KTO uses the Kahneman-Tversky value function from prospect theory, which models human loss aversion — humans are more sensitive to losses than gains of the same magnitude. The logistic value function is concave in gains (diminishing returns) and convex in losses.
- **Human-aware losses (HALOs):** The paper defines HALOs as a family of losses that implicitly model human biases. DPO and PPO-Clip are proven to be HALOs; KTO is a new HALO derived directly from prospect theory. Among existing methods, HALOs consistently outperform non-HALOs.

## Key Method Details

- **KTO loss (Eq. 8):** L_KTO = E\_{x,y}[λ_y − v(x, y)], where v is the value function:
  - Desirable y: v = λ_D · σ(β · (r_θ(x,y) − z_0))
  - Undesirable y: v = λ_U · σ(β · (z_0 − r_θ(x,y)))
- **Implied reward:** r_θ(x,y) = log(π_θ(y\|x) / π_ref(y\|x)) — the log-ratio of policy to reference model probabilities, measured in nats
- **Reference point z_0:** Estimated as a biased KL divergence surrogate via batch-shuffled mismatched pairs, clamped to be non-negative
- **β (risk aversion):** Default 0.1; controls how quickly the value function saturates. Higher β = more risk-averse in gains, more risk-seeking in losses
- **λ_D, λ_U (loss aversion):** Both default to 1.0; should be set so that λ_D · n_D / (λ_U · n_U) ∈ [1, 3/2]
- **Results:** KTO matches or exceeds DPO at 1B–30B scale (§4.3); on GSM8K, KTO improves by 13.5 points over DPO when aligning Zephyr-β-SFT on UltraFeedback (Table 2); KTO with 90% fewer desirable examples still outperforms DPO (Figure 5)

### Simplifications in This Notebook

- **Tiny model:** A character-level GRU language model (80,286 parameters) instead of 1B–30B scale models
- **Dataset:** 300 synthetic training examples with patterned (desirable) vs random (undesirable) completions, instead of real human feedback datasets (HH, SHP, OpenAssistant, UltraFeedback)
- **β = 0.1** (paper default), λ_D = λ_U = 1.0, 60 epochs with AdamW (lr 5e-4) — the paper uses RMSprop with lr 1e-6 for fair comparison with DPO
- **Reference model:** Pre-trained via SFT on desirable completions for 30 epochs, instead of full instruction fine-tuning
- **KL estimate z_0:** Uses a simple batch-shuffled mean clamped to 0, instead of the paper's full mismatched-pair estimator

### What the Notebook Result Shows

On 100 synthetic test examples (50 desirable, 50 undesirable), KTO achieves 74% test accuracy (desirable reward > undesirable reward), compared to DPO's 78% on paired preferences from the same data. KTO loss decreases from about 0.46 to about 0.16 over 60 epochs, and the reward margin grows from about 3.0 to about 103.7. The β ablation shows that higher β (0.5) yields 96% test accuracy with smaller reward margins (17.6), while lower β (0.01) yields 50% accuracy with larger margins (105.5), confirming the paper's finding that β controls the risk-aversion/saturation tradeoff. These are toy results with synthetic data and a tiny model, and are not comparable to the paper's 1B–30B scale experiments judged by GPT-4 and human evaluators.

## Influence

KTO is integrated into Hugging Face's TRL library and has been adopted as a practical alternative to DPO for settings where only binary feedback is available. It directly inspired follow-up work on alignment with non-paired feedback, including SimPO (2024, reference-free preference optimization) and further exploration of prospect-theoretic objectives for alignment. The HALO framework introduced in this paper provides a theoretical lens for understanding why certain alignment losses succeed, influencing how researchers design new objectives. About 253 citations on [OpenAlex](https://openalex.org/W4391555989) (checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/047_kto-prospect-theoretic-optimization/solution.ipynb) | From-scratch PyTorch implementation of the KTO loss with a tiny character-level GRU LM trained on synthetic binary feedback, compared against DPO on the same data |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Q&A answered from the paper, citations, and common misconceptions about KTO |
