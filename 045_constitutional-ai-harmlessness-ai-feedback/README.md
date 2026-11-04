---
title: "045 Constitutional AI"
nav_order: 46
permalink: "/045_constitutional-ai-harmlessness-ai-feedback/"
---
# Constitutional AI: Harmlessness from AI Feedback (Constitutional AI)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/constitutional-ai-harmlessness-from-ai-feedback)

- **Paper:** [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- **Authors:** Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan
- **Lab, year:** Anthropic, 2022 (arXiv, December 2022)

## Summary

In December 2022, researchers at Anthropic introduced **Constitutional AI (CAI)**, a method for training a harmless AI assistant without any human labels identifying harmful outputs. The only human oversight comes from a short list of natural-language principles — a "constitution" — that steers the model's behavior.

The method has two stages. In the **supervised learning (SL) stage**, the model generates a response to a harmful prompt, critiques its own response according to a constitutional principle, and revises it. The original model is then fine-tuned on these revised responses. In the **reinforcement learning (RL) stage**, the fine-tuned model generates pairs of responses and a separate "feedback model" evaluates which is better according to a principle. A preference model is trained on these AI-generated labels, and the policy is fine-tuned with RL against it — a process the authors call **RLAIF** (RL from AI Feedback).

The key result is that the resulting assistant, called **RL-CAI**, is both harmless and **non-evasive**: instead of refusing to engage with sensitive topics ("I can't answer that"), it explains its objections thoughtfully. Crowdworkers preferred RL-CAI over models trained with traditional human feedback (RLHF) for harmlessness, at roughly equal helpfulness. The paper also shows that chain-of-thought (CoT) reasoning — asking the feedback model to "think step-by-step" before choosing — improves label quality and transparency. Using 16 principles and ~182K red-teaming prompts, the method reduces harmlessness to about 10 bits of human input (the principles themselves).

This notebook implements the **supervised stage** of Constitutional AI at toy scale: a critique-and-revise loop with a written constitution, applied to a small language model on a set of harmful prompts. It generates self-critiques and revisions, builds an AI-feedback preference dataset from pairwise comparisons, and trains a small preference model. The result shows that the critique-revise loop progressively reduces harmfulness scores assigned by the preference model.

## What Problem Does It Solve

Imagine you are raising a child and you want them to be both helpful and safe. If you only teach them to be safe, they become overly cautious — they refuse to touch anything, won't answer questions, and become useless. That is the problem with early AI safety training: models trained to be harmless became **evasive**, giving canned refusals like "I can't help with that" for even reasonable questions.

Constitutional AI solves this by giving the AI a set of written rules — like a constitution — and asking it to police itself. Instead of hiring thousands of humans to label every possible harmful response, the AI critiques its own answers against the rules and rewrites them. The result is an assistant that is safe but still engages with difficult topics, explaining *why* it objects rather than just shutting down.

## Core Idea

- **Self-critique and revision:** The model generates a response to a harmful prompt, then critiques its own response according to a constitutional principle, and revises it. This can be repeated multiple times with different principles sampled at each step.
- **RL from AI Feedback (RLAIF):** Instead of human crowdworkers labeling which of two responses is more harmless, a separate "feedback model" evaluates responses according to constitutional principles. These AI-generated preference labels replace human harmlessness labels, and are distilled into a preference model for RL training.
- **Non-evasive harmlessness:** Unlike RLHF models that learn to refuse all sensitive queries, CAI models engage with harmful prompts by explaining their objections, reducing the tension between helpfulness and harmlessness.

## Key Method Details

- **Constitution:** ~16 natural-language principles for harmlessness (e.g. "Identify specific ways in which the assistant's last response is harmful, unethical, racist, sexist, toxic, dangerous, or illegal"), randomly sampled at each critique-revision step
- **SL stage data:** 42,496 human-written red-teaming prompts + 140,335 model-generated prompts = 182,831 total; 4 critique-revision pairs per prompt, sampled at temperature T = 1
- **RL stage data:** 135,296 human helpfulness comparisons + 182,831 AI-generated harmlessness comparisons; preference model trained on mixed dataset
- **Models:** 52B-parameter language models (pretrained, then RLHF for helpfulness, then CAI)
- **Results:** RL-CAI preferred by crowdworkers over RLHF models for harmlessness at comparable helpfulness (Figures 2, 3, 8); harmlessness Elo scores improve significantly during RL training while helpfulness remains stable
- **Chain-of-thought:** CoT prompting on the feedback model ("Let's think step-by-step") improves label quality; probabilities clamped to 40–60% range for calibration

### Simplifications in This Notebook

- **Small model:** A tiny character-level GRU language model (~200K parameters) instead of the paper's 52B-parameter models
- **Dataset:** A small synthetic set of harmful prompts (~50) instead of 182,831 red-teaming prompts
- **Constitution:** 6 principles instead of 16
- **Preference model:** A small classifier (~12K parameters) trained on synthetic AI-feedback labels instead of a 52B preference model trained on hundreds of thousands of comparisons
- **No RL stage:** Only the supervised critique-revise stage is implemented; the RL stage (preference model → RL training) is simulated with the preference model alone
- **Optimizer:** Adam with lr 1e-3 for 20 epochs, instead of the paper's one-epoch fine-tuning with lr 0.5 relative to pre-training

### What the Notebook Result Shows

On a set of 50 synthetic harmful prompts, the critique-and-revise loop progressively reduces the harmfulness score assigned by a trained preference model: the average harmfulness score drops from about 0.94 (original response) to about 0.05 (after 2 revisions). The preference model, trained on 400 synthetic pairwise comparisons, achieves about 95% accuracy on a held-out test set. These numbers are from a toy implementation with a tiny model and synthetic data, and are not comparable to the paper's 52B-model results measured via crowdworker Elo scores.

## Influence

Constitutional AI became a cornerstone of Anthropic's alignment methodology and directly shaped the training of Claude, Anthropic's assistant product. The RLAIF technique — replacing human preference labels with AI-generated ones — influenced subsequent work on scalable oversight and AI self-improvement. The paper's critique-revise approach was extended in later work on self-correction and self-rewarding language models (2024). The tension between helpfulness and harmlessness that CAI addressed remains a central concern in LLM alignment.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/045_constitutional-ai-harmlessness-ai-feedback/solution.ipynb) | From-scratch PyTorch implementation of the Constitutional AI critique-revise loop with a written constitution and AI-feedback preference model on synthetic harmful prompts |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
