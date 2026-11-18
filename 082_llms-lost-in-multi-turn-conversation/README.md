---
title: "082 Multi-Turn Lost"
nav_order: 83
permalink: "/082_llms-lost-in-multi-turn-conversation/"
---
# LLMs Get Lost In Multi-Turn Conversation (Multi-Turn Lost)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/llms-get-lost-in-multi-turn-conversation)

- **Paper:** [LLMs Get Lost In Multi-Turn Conversation](https://arxiv.org/abs/2505.06120)
- **Authors:** Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, Jennifer Neville
- **Lab, year:** Microsoft Research; Salesforce Research, 2025 (arXiv preprint)

## Summary

"LLMs Get Lost In Multi-Turn Conversation" is a large-scale empirical study of how well language models perform when a task is split across multiple conversation turns (multi-turn) versus given all at once (single-turn). While LLMs are primarily used as conversational interfaces — where users often start with underspecified instructions and refine them over multiple turns — most LLM evaluation focuses on single-turn, fully-specified settings. This paper closes that gap.

The authors develop a **sharding simulation** framework that transforms existing single-turn benchmark instructions into **sharded instructions** — sets of smaller instructions (shards) that jointly deliver the same information as the original. In the sharded (multi-turn) simulation, each turn reveals at most one shard, enforcing that the instruction is gradually revealed through the conversation. They test 15 LLMs across six generation tasks (Code, Database, Actions, Math, Data-to-Text, Summary) with 200,000+ simulated conversations.

They find an average performance drop of **39%** from single-turn to multi-turn settings. Crucially, they decompose this degradation into two components: a **minor loss in aptitude** (best-case performance, measured as the 90th percentile score) and a **significant increase in unreliability** (the gap between best- and worst-case simulations, measured as the 90th–10th percentile range). In simpler terms, the authors discover that *when LLMs take a wrong turn in a conversation, they get lost and do not recover*.

This repo's notebook trains a small transformer classifier (about 603K parameters) on a synthetic attribute-combination task and reproduces the core qualitative finding: multi-turn (sharded) performance degrades primarily through increased unreliability, not aptitude loss, with a performance drop of about 73% from full to sharded settings.

## What Problem Does It Solve

Imagine asking a friend to help you write a report. If you give them all the requirements at once ("write a 3-page report on climate change, include two charts, focus on Arctic ice loss"), they can plan and execute effectively. But if you give them the requirements one at a time — "write a report" (turn 1), "it's about climate change" (turn 2), "include two charts" (turn 3), "focus on Arctic ice loss" (turn 4) — they will start working with incomplete information, make wrong assumptions, and then struggle to incorporate new requirements as they arrive.

The paper shows that LLMs have exactly this problem. When you give an LLM a task in pieces across multiple turns, it often jumps to a premature answer with partial information, then fails to course-correct when new details arrive. This matters because real users rarely specify everything upfront — they refine their requests through conversation, which is the natural way humans interact.

## Core Idea

- **Sharding simulation:** Transform single-turn instructions into shards revealed one per turn, enabling controlled comparison of single- vs multi-turn performance on the same task.
- **Aptitude vs reliability decomposition:** Performance degradation from single- to multi-turn is not mainly a loss of ability (aptitude drops only 16% on average) but a massive increase in variance (unreliability increases 112%, more than doubling). All models — from Llama3.1-8B to Gemini 2.5 Pro — exhibit high unreliability in multi-turn settings.
- **The "lost in conversation" phenomenon:** LLMs prematurely propose full answers with incomplete information, then overly rely on their previous (incorrect) attempts. Even lowering temperature to 0 does not resolve this — the unreliability persists because one token difference in an early turn cascades into divergent conversation trajectories.

## Key Method Details

- **Sharding process (Section 3.1):** Original instructions are split into shards satisfying five properties: information preservation, clear initial intent, order insensitivity, maximal sharding, and minimal transformation. The first shard always introduces the high-level intent.
- **Simulation types (Section 3.3):** Five types tested — Full (single-turn, original instruction), Sharded (multi-turn, one shard per turn), Concat (single-turn, shards concatenated), Recap (Sharded + final recap turn), Snowball (each turn restates all previous shards).
- **Metrics (Section 4.2):** Average performance (P-bar = mean score), Aptitude (A90 = 90th percentile), Unreliability (U1090 = 90th–10th percentile). Computed per-instruction across N=10 repeated simulations, then averaged.
- **Models tested:** 15 LLMs from 8 families: OpenAI (GPT-4o-mini, GPT-4o, o3, GPT-4.1), Anthropic (Claude 3 Haiku, Claude 3.7 Sonnet), Google (Gemini 2.5 Flash, Gemini 2.5 Pro), Meta (Llama3.1-8B, Llama3.3-70B, Llama 4 Scout), OLMo-2-13B, Phi-4, DeepSeek-R1, Command-A.
- **Results:** Average 39% performance drop from Full to Sharded. Concat averages 95.1% of Full (sharding itself doesn't lose information). Aptitude drops only 16% but unreliability increases 112%. Even at temperature 0, multi-turn unreliability remains around 30% (Table 3).
- **Gradual sharding (Section 6.3):** Even 2-shard conversations cause models to get lost. Any conversation involving underspecification in two or more turns leads to degradation.

### Simplifications in This Notebook

- Smaller model: 3-layer transformer classifier, about 603K parameters (the paper tests 8B–300B+ models)
- Task: synthetic attribute-combination (5 colors x 5 shapes x 3 sizes = 75 problems) instead of 6 real-world generation tasks
- 80 epochs with AdamW (lr 0.001, weight decay 0.01) on 500 training examples, instead of zero-shot evaluation of pretrained LLMs
- 10 simulations per problem on 25 eval problems, instead of 200,000+ simulations across 600 instructions
- Classification heads (color, shape, size) instead of free-form text generation with answer extraction

### What the Notebook Result Shows

On the synthetic attribute-combination task, the model achieves 100% greedy accuracy on full (single-turn) instructions. In the full simulation at temperature T=1.0, average performance (P-bar) is about 40.8 with near-zero unreliability (U1090 of about 0.8). In the sharded (multi-turn) simulation, P-bar drops to near 0% — the model's wrong predictions from early turns (with incomplete information) bias its final answer. The aptitude (A90) drops from 40.8 to 0, and unreliability remains low because the model consistently fails.

The gradual sharding experiment shows performance of about 50.5% with 1 shard (single-turn), dropping to 29.5% with 2 shards and near 0% with 3+ shards. These numbers are not comparable to the paper's results (which used 7B–300B+ pretrained models on real tasks with 200,000+ simulations), but the qualitative pattern — sharp multi-turn degradation — is consistent. The extreme drop in our notebook (to 0%) is stronger than the paper's 39% average because our 600K-param model is far less robust to context pollution than production LLMs.

## Influence

The paper introduced the "lost in conversation" phenomenon as a fundamental evaluation finding for LLM-based conversational systems. It directly challenged the prevailing evaluation paradigm that focused on single-turn, fully-specified benchmarks, showing that strong single-turn performance does not translate to multi-turn reliability — even for state-of-the-art models like Gemini 2.5 Pro and GPT-4.1. The paper's call to "prioritize multi-turn reliability in conjunction with aptitude" influenced subsequent work on multi-turn evaluation and agent robustness. The authors released the sharding framework and datasets as [Microsoft/lost_in_conversation](https://huggingface.co/datasets/Microsoft/lost_in_conversation) on Hugging Face, enabling other researchers to shard their own tasks. The finding that lowering temperature does not fix multi-turn unreliability motivated work on architectural solutions rather than decoding-time interventions.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/082_llms-lost-in-multi-turn-conversation/solution.ipynb) | Multi-head classifier transformer (603K params) trained on synthetic attribute-combination task, comparing full vs sharded performance with aptitude/reliability decomposition |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
