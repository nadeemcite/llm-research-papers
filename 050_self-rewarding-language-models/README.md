---
title: "050 Self-Rewarding LM"
nav_order: 51
permalink: "/050_self-rewarding-language-models/"
---
# Self-Rewarding Language Models (Self-Rewarding LM)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/self-rewarding-language-models)

- **Paper:** [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020)
- **Authors:** Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, Jason Weston
- **Lab, year:** Meta AI (FAIR) and New York University, 2024 (preprint)

## Summary

In January 2024, researchers at Meta AI and NYU introduced **Self-Rewarding Language Models**, a method where the language model itself provides its own training rewards via LLM-as-a-Judge prompting — eliminating the need for a separate, frozen reward model. Prior RLHF approaches train a reward model from human preferences, which is bottlenecked by human performance and cannot improve during LLM training. Self-rewarding closes this loop: the same model that is being improved also serves as its own judge.

The method works through **Iterative DPO** (Direct Preference Optimization). Each iteration has two steps: (1) the model generates multiple responses per prompt and judges them using an LLM-as-a-Judge prompt that evaluates response quality on a 1–5 scale, producing preference pairs, and (2) the model is fine-tuned with DPO on these self-generated pairs, using the current model as a frozen reference. A small auxiliary SFT (supervised fine-tuning) loss is added to the DPO objective to prevent the model from degrading on instruction-following while learning preferences.

Fine-tuning Llama 2 70B over three iterations (M1, M2, M3) yielded a model that outperformed Claude 2, Gemini Pro, and GPT-4 0613 on the AlpacaEval 2.0 leaderboard, reaching a 20.44% length-controlled win rate. Crucially, both instruction-following ability and reward model quality (measured by the RewardBench agreement with human labels) improved across iterations — the model gets better at both following instructions and judging them simultaneously.

This notebook implements the self-rewarding pipeline from scratch in PyTorch on a tiny character-level GRU language model, runs 2 iterations of self-reward DPO on synthetic instruction data, and demonstrates that both generation quality and judge accuracy improve across iterations.

## What Problem Does It Solve

Imagine a student who is learning to write essays. The usual approach is to have a teacher grade the essays and give feedback. But what if the teacher is no better at writing than the student? That is the problem with human-feedback RLHF: the reward model is trained on human preferences, so it cannot guide the student beyond human-level quality. And if you freeze the teacher (as RLHF does with the reward model), the teacher cannot learn and improve alongside the student.

Self-rewarding is like having the student grade their own essays — but with a rubric that gets better each round. The student writes several drafts, picks the best and worst using their own judgment, then practices making the "best" ones more likely and the "worst" ones less likely. As the student improves at writing, they also improve at judging (because judging is itself a writing task), creating a virtuous cycle where both skills co-improve without an external teacher.

## Core Idea

- **LLM-as-a-Judge for rewards:** Instead of a separate frozen reward model, the language model itself evaluates response quality using an LLM-as-a-Judge prompt. The model judges pairs of responses and assigns scores, which are used to construct preference pairs for DPO training.
- **Iterative DPO:** Training proceeds in iterations (M1, M2, M3). Each iteration generates new preference data using the current model as both generator and judge, then trains with DPO using the current model as the frozen reference. This creates a self-improving loop.
- **Dual improvement:** The key finding is that both instruction-following ability and reward model quality improve across iterations. The model does not just get better at following instructions — it also gets better at judging responses, because judging is itself an instruction-following task that benefits from the same training.

## Key Method Details

- **Self-creation pipeline (Section 3):** For each prompt, the model generates N responses, then uses an LLM-as-a-Judge prompt to evaluate and rank them. The highest-scored response becomes the "chosen" and the lowest becomes "rejected" in a preference pair.
- **DPO + SFT objective (Eq. 7):** L = L_DPO + α · L_SFT, where L_DPO is the standard DPO loss with a frozen reference model and L_SFT is the NLL of the chosen response. The paper uses α = 1 for the SFT weight.
- **LLM-as-a-Judge prompt (Section 3.1):** The judge evaluates responses on a 1–5 scale across dimensions like relevance, clarity, completeness, and helpfulness. The evaluation is itself a generation task, so the judge improves as the model improves.
- **Iterations:** The paper runs 3 iterations (M1, M2, M3) starting from Llama 2 70B SFT. Each iteration generates 1,000 prompts with multiple responses, judges them, and trains with DPO + SFT.
- **Results:** M3 achieves 20.44% length-controlled win rate on AlpacaEval 2.0 (surpassing Claude 2 at 13.17%, Gemini Pro at 14.51%, and GPT-4 0613 at 15.79%), and 55.3% on RewardBench (improving from M1's 53.0%), demonstrating dual improvement in instruction following and judging.

### Simplifications in This Notebook

- **Tiny model:** A 2-layer character-level GRU language model (179,358 parameters) instead of Llama 2 70B
- **Dataset:** 400 synthetic instruction pairs (repeating-pattern and low-entropy tasks) instead of real instruction-following data
- **2 iterations** with 40 epochs each, instead of 3 iterations at full scale
- **Judge proxy:** A structural-quality scorer (transition count, unique-token count) instead of a full LLM-as-a-Judge prompt with 1–5 scoring
- **SFT weight:** 0.1 instead of the paper's 1.0, to keep the DPO preference signal dominant in the toy setting
- **DPO temperature:** β = 0.5 instead of the paper's 0.1

### What the Notebook Result Shows

On the Kaggle GPU run, the baseline (SFT-warmup only) model starts with 100% preference accuracy, generation quality score of −7.58, and judge accuracy of 49% (model's log-prob ranking barely better than random at matching the oracle quality ranking). After 2 self-reward iterations:
- **Generation quality improves** from −7.58 to −5.95 (iteration 1) to −4.91 (iteration 2) — the model's generated responses increasingly follow the target structural patterns.
- **Judge accuracy improves** from 49% to 52% (iteration 1) to 56% (iteration 2) — the model's implicit log-probability ranking aligns more closely with the oracle quality ranking.
- **Preference accuracy** stays high at 99–100% throughout.
- The DPO reward margin within each iteration grows steadily (0.15 → 7.58 in iteration 1; 0.28 → 5.37 in iteration 2), showing the preference signal is being learned.

These are toy results with synthetic data and a tiny GRU model, and are not comparable to the paper's 70B-scale AlpacaEval and RewardBench results. The key qualitative finding — that both generation quality and judging ability improve across self-reward iterations — is what this notebook demonstrates.

## Influence

The paper influenced the field of self-improving AI systems by showing that models can generate their own training signal without external human annotation, opening research directions in iterative self-improvement and scalable oversight. Follow-up work includes SPIN (Self-Play fIne-tuning, 2024), which uses a similar self-play mechanism, and CRINGE (2024), which explores self-rewarding with contrastive objectives. The LLM-as-a-Judge paradigm was further studied in the "Judging the Judges" line of work. About 300 citations on [OpenAlex](https://openalex.org/W4396956981) (checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/050_self-rewarding-language-models/solution.ipynb) | From-scratch PyTorch implementation of the self-rewarding pipeline (generate → judge → DPO+SFT) with a tiny character-level GRU LM trained on synthetic instruction data across 2 iterations |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Q&A answered from the paper, citations, and common misconceptions about Self-Rewarding Language Models |
