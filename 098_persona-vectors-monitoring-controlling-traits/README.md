---
title: "098 Persona Vectors"
nav_order: 99
permalink: "/098_persona-vectors-monitoring-controlling-traits/"
---
# Persona Vectors: Monitoring and Controlling Character Traits in Language Models (Persona Vectors)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/persona-vectors-monitoring-controlling-traits)

- **Paper:** [Persona Vectors: Monitoring and Controlling Character Traits in Language Models](https://arxiv.org/abs/2507.21509)
- **Authors:** Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, Jack Lindsey
- **Lab, year:** Anthropic Fellows Program, UT Austin, Constellation, Truthful AI, UC Berkeley, Anthropic, 2025 (arXiv preprint, July 2025)

## Summary

Persona Vectors introduces an automated pipeline for extracting linear directions in a language model's activation space that correspond to personality traits — directions the authors call "persona vectors." Given only a natural-language description of a trait (e.g., "evil," "sycophancy," "propensity to hallucinate"), the pipeline uses a frontier LLM (Claude 3.7 Sonnet) to generate contrastive system prompts and evaluation questions. The persona vector is then computed as the difference in mean residual-stream activations between responses that exhibit the trait and those that do not.

Once extracted, a persona vector serves four purposes. First, it can **steer** the model's behavior during generation — adding the vector increases trait expression, subtracting it decreases expression. Second, it can **monitor** deployment-time persona shifts: the projection of the last prompt token's activation onto the persona vector correlates strongly (r = 0.75–0.83) with the trait expression of the subsequent response. Third, it can **predict and mitigate** persona shifts caused by fine-tuning: activation changes along the persona vector during training strongly correlate (r = 0.76–0.97) with behavioral changes, and a novel "preventative steering" method can limit unwanted shifts. Fourth, it can **flag problematic training data** before fine-tuning by computing the "projection difference" between training responses and the base model's natural responses.

This notebook extracts a sycophancy persona vector from Qwen2.5-0.5B-Instruct and demonstrates steering, monitoring, and finetuning-shift measurement. The sycophancy vector separates sycophantic from honest responses, steering modulates sycophancy scores, and fine-tuning on sycophantic data shifts activations in the predicted direction.

## What Problem Does It Solve

Imagine you have a robot butler who is normally very polite and helpful. But sometimes, depending on how you phrase your requests, it suddenly becomes a completely different personality — maybe overly agreeable and tells you only what you want to hear, or starts making things up. Even worse, if you retrain it on some new tasks, its personality might shift in unexpected ways. The paper builds a "personality detector" for AI models: a single direction in the model's internal representations that captures each trait, so you can measure, predict, and control personality changes before they cause problems.

## Core Idea

- **Traits are linear directions:** Personality traits like sycophancy or evil are encoded as linear directions in the model's residual stream activation space. The difference in mean activations between trait-exhibiting and non-exhibiting responses yields a "persona vector."
- **Automated extraction from text alone:** Given only a trait name and description, the pipeline automatically generates contrastive prompts, evaluation questions, and a scoring rubric — no bespoke data curation needed.
- **One vector, four applications:** The same persona vector enables behavioral steering (causal control), deployment monitoring (predicting shifts from prompts), finetuning-shift prediction, and pre-finetuning data screening.

## Key Method Details

- **Models:** Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct, 24–32 layers, residual stream activations averaged across response tokens (Section 3.1).
- **Extraction:** 5 pairs of contrastive system prompts, 40 evaluation questions (20 extraction, 20 evaluation), 10 rollouts per question per prompt. Responses filtered by trait expression score (threshold 50). Persona vector = difference in mean activations at the selected layer (Section 2.2).
- **Steering:** `h_ℓ ← h_ℓ + α · v_ℓ` at each decoding step. Positive α increases trait expression; negative α decreases it. Side effects measured via MMLU accuracy and coherence scores (Section 3.2).
- **Monitoring:** Projection of last-prompt-token activation onto persona vector correlates r = 0.75–0.83 with trait expression scores across system prompt variations (Section 3.3).
- **Finetuning shift:** `(post-FT mean activation − pre-FT mean activation) · v̂` correlates r = 0.76–0.97 with post-finetuning trait expression (Section 4.2).
- **Preventative steering:** Steering *toward* the undesired persona direction *during* fine-tuning cancels out the training pressure to shift that direction, limiting persona drift while preserving capabilities (Section 5.2).
- **Projection difference:** `ΔP = mean[projection(training_response)] − mean[projection(base_model_response)]` predicts post-finetuning trait expression before training begins (Section 6.1).

### Simplifications in This Notebook

- Smaller model: Qwen2.5-0.5B-Instruct, 494M parameters, 24 layers (paper uses 7B–8B models)
- 10 evaluation questions instead of 40; 3 rollouts per question instead of 10
- Heuristic keyword-based sycophancy scorer (0–100) instead of GPT-4.1-mini judge
- Layer 12 (middle of 24) chosen a priori instead of empirically selecting the most informative layer
- 20 training samples × 2 epochs with full fine-tuning instead of LoRA (rank 32) on curated datasets
- Hand-crafted contrastive prompts and training data instead of the automated Claude 3.7 Sonnet pipeline
- Only sycophancy trait studied; paper covers evil, sycophancy, hallucination, optimism, and humor

### What the Notebook Result Shows

A sycophancy persona vector (norm 1.3273) extracted from Qwen2.5-0.5B-Instruct (494M, 24 layers) at layer 12 separates sycophantic from honest prompt responses with a mean projection separation of 1.3273. Steering at coefficients -3.0, 0.0, and +3.0 produces sycophancy scores of 35.6, 30.8, and 52.4 respectively (heuristic scorer, 0-100), showing a clear monotonic trend.

Fine-tuning on 20 sycophantic examples for 2 epochs shifts activations +0.9527 along the persona vector and increases sycophancy scores from 35.6 to 69.6. Honest fine-tuning shifts activations -0.4166 and changes scores to 47.6. The monitoring correlation (r = -0.242) is weak compared to the paper's r = 0.75-0.83, likely because the 0.5B model encodes sycophancy less linearly and the keyword scorer is coarser than GPT-4.1-mini. The finetuning shift results are consistent with the paper's finding that activation shifts predict behavioral changes.

## Influence

The paper builds directly on the emergent misalignment work of Betley et al. (2025) and the activation steering literature (Turner et al. 2024, Panickssery et al. 2024, Zou et al. 2025). It extends Wang et al. (2025), who showed that emergent misalignment is mediated by linear "misaligned persona" directions, by systematizing the extraction process into an automated pipeline.

The preventative steering method relates to CAFT (Casademunt et al. 2025) and detachable LoRA modules (Zhou et al. 2024). The work has practical implications for AI safety: it provides tools to detect and prevent unintended personality shifts during model training, a concern highlighted by real-world incidents including GPT-4o's sycophancy issue (OpenAI, 2025) and Grok's persona shifts (Reuters, 2025). Code is at [github.com/safety-research/persona\_vectors](https://github.com/safety-research/persona_vectors). About 5 citations ([Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.21509?fields=citationCount), checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/098_persona-vectors-monitoring-controlling-traits/solution.ipynb) | PyTorch notebook extracting a sycophancy persona vector from Qwen2.5-0.5B-Instruct: steering, monitoring, and finetuning-shift measurement on synthetic data |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
