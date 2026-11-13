---
title: "070 Toolformer"
nav_order: 71
permalink: "/070_toolformer-teaching-lms-to-use-tools/"
---
# Toolformer: Language Models Can Teach Themselves to Use Tools (Toolformer)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/toolformer-teaching-lms-to-use-tools)

- **Paper:** [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
- **Authors:** Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom
- **Lab, year:** Meta AI (FAIR), 2023 (NeurIPS 2023)

## Summary

Toolformer teaches a language model to **teach itself** when and how to call external tools — a calculator, a question-answering system, a search engine, a translator, and a calendar — through simple API calls. The key insight is a **self-supervised** labeling process: given a few hand-written demonstrations of each API, the model samples positions in its training text where it *might* insert a tool call, executes the call, and then checks whether the result actually helps it predict the *next* tokens better. Only calls that reduce a filtered loss are kept; the model is then fine-tuned on this self-labeled augmented dataset.

The approach requires nothing more than a handful of demonstrations per API (the paper uses 20 for most tools). After fine-tuning, the model autonomously decides which API to call, what arguments to pass, and how to incorporate the response — all during inference, without human intervention.

Toolformer is based on GPT-J (a 6.7-billion-parameter model). On the LAMA knowledge benchmark (a task where models complete factual statements like "Paris is the capital of [France]"), it improves the SQuAD subset from 19.2 to 33.8 and the T-REx subset from 33.2 to 53.5, outperforming the much larger OPT-66B and GPT-3 (175B). On math benchmarks (ASDiv, SVAMP, MAWPS — grade-school arithmetic word problems), it more than doubles performance, e.g. ASDiv from 9.6 to 40.4. Crucially, these gains come without sacrificing the model's core language modeling ability: perplexity (a measure of how well the model predicts text) is virtually unchanged.

This repo's notebook builds a tiny Toolformer from scratch with a calculator API on toy math problems, and shows that the self-supervised filtering + fine-tuning pipeline improves test accuracy from 27% (baseline) to 31.5% (Toolformer with API).

## What Problem Does It Solve

Large language models are remarkably good at writing essays and answering questions, yet they fail at things a pocket calculator does trivially — arithmetic, looking up today's date, translating a sentence into Hindi. This is paradoxical: a model with billions of parameters can write a poem but cannot reliably compute 47 × 83.

The paper's analogy is straightforward: humans don't do everything in their heads. When you need to multiply large numbers, you reach for a calculator. When you need today's exchange rate, you look it up. Language models should do the same — call external tools when their own knowledge falls short — but they need to learn *when* a tool would help and *how* to use its output. Toolformer automates this learning: the model discovers useful tool calls on its own, without humans labeling thousands of examples.

## Core Idea

- **Self-supervised API call labeling:** The model samples candidate positions for inserting API calls in its training text, executes the calls, and filters them by whether the result reduces a weighted loss on subsequent tokens. No human annotation is needed beyond a few initial demonstrations.
- **Loss-based filtering criterion:** For each candidate call, compute L⁺ (loss with the API call and its result inserted) and L⁻ (the minimum of the loss without any API call, and the loss with the call but without the result). Keep the call only if L⁻ − L⁺ ≥ τ_f (the filtering threshold). This ensures only calls that genuinely help the model's predictions are kept.
- **Fine-tuning on augmented data:** After filtering, the surviving API calls are merged into an augmented dataset. The model is fine-tuned on this data, learning to autonomously insert API calls and use their results at inference time.

## Key Method Details

- **Base model:** GPT-J (6.7B parameters), pretrained on the Pile.
- **Sampling threshold τ_s:** Controls which positions are candidate API call sites (positions where the model assigns > τ_s probability to the `<API>` token).
- **Filtering threshold τ_f:** Only API calls where L⁻ − L⁺ ≥ τ_f are kept. The paper uses τ_f = 1.0 as the main setting.
- **Tools:** Calculator, question answering (based on a 3B-parameter fine-tuned T5), Wikipedia search (via BM25), machine translation (using a 600M NLLB model), and a calendar API.
- **Dataset:** CCNet (a subset of Common Crawl); over 1 million documents processed, yielding tens of thousands of filtered API call examples per tool.
- **Results (GPT-J 6.7B):** LAMA SQuAD 33.8 (baseline 19.2), T-REx 53.5 (baseline 33.2); ASDiv 40.4 (baseline 9.6), SVAMP 29.4 (baseline 5.0), MAWPS 44.0 (baseline 9.3). Outperforms OPT-66B and GPT-3 (175B) on most tasks.
- **Language modeling:** Perplexity on WikiText and CCNet is virtually unchanged, confirming that tool use does not degrade core LM ability.

### Simplifications in This Notebook

- Toy math problems (500 train, 200 test) instead of CCNet (millions of documents)
- Tiny transformer: 3 layers, d_model=128, about 657K parameters (instead of GPT-J 6.7B)
- Calculator API only (instead of 5 tools: calculator, QA, search, translation, calendar)
- 80 pretrain + 60 fine-tune epochs with Adam (lr 3e-4) instead of fine-tuning a 6.7B model
- 150 demonstrations with API calls (instead of a handful per tool); demonstrations repeated 3x during pretraining to emphasize the API pattern
- τ_f = 0.01 (very lenient) instead of the paper's τ_f = 1.0, because the tiny model's loss landscape is much flatter

### What the Notebook Result Shows

On 200 test math questions (addition, subtraction, multiplication with results 0–199), the Toolformer model achieves **31.5%** accuracy with the calculator API enabled, versus **27.0%** for the baseline model (no API). The model calls the tool on **100%** of test examples. When the API is disabled at test time, the fine-tuned model scores 23.5% — lower than the baseline, showing that fine-tuning on augmented data shifts the model toward tool use at some cost to direct prediction.

By operation: subtraction improves most (6.7% → 24.0%), addition improves modestly (3.2% → 8.1%), and multiplication slightly regresses (74.6% → 63.5%) because the pretrained model already predicts small products well and the API insertion sometimes confuses the result-copying pathway. The self-supervised filter keeps 92 of 415 sampled calls (22.2% pass rate) — calls where the calculator result demonstrably reduced the model's loss.

These toy numbers are not comparable to the paper's LAMA or ASDiv results; a 657K-parameter transformer on 500 toy problems cannot match a 6.7B model on millions of real documents. The notebook demonstrates the mechanism — sample, execute, filter by loss reduction, fine-tune — not the paper's scale.

## Influence

Toolformer became one of the most influential papers on LLM tool use. Its self-supervised approach to teaching models when to call external APIs influenced subsequent work on function-calling LLMs, including OpenAI's function calling feature (2023), Gorilla (Patil et al., 2023), and NexusRaven (Nexusflow, 2023). The paper's filtering criterion — keeping only tool calls that reduce the model's loss — became a standard idea in tool-augmented LLM training. Follow-up work includes Chain-of-Tools and CHEER (2024), which extend the approach to chained and interactive tool use, addressing limitations the paper itself identified. The paper has been cited about 3,245 times ([OpenAlex](https://openalex.org/works/W7133214447), checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/070_toolformer-teaching-lms-to-use-tools/solution.ipynb) | From-scratch PyTorch Toolformer with a calculator API on toy math problems, self-supervised call labeling with loss-based filtering, fine-tuning, and evaluation (31.5% vs 27% baseline) |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
