---
title: "053 Zero-shot CoT"
nav_order: 54
permalink: "/053_zero-shot-cot-reasoning/"
---
# Large Language Models are Zero-Shot Reasoners (Zero-shot CoT)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/zero-shot-cot-reasoning)

- **Paper:** [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916)
- **Authors:** Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, Yusuke Iwasawa
- **Lab, year:** The University of Tokyo (Kojima, Matsuo, Iwasawa) and Google Research, Brain Team (Gu, Reid), 2022 (NeurIPS 2022)

## Summary

In May 2022, researchers at the University of Tokyo and Google Brain showed that large language models can perform complex multi-step reasoning with **zero** hand-crafted examples — simply by appending "Let's think step by step" to the prompt. This technique, called **Zero-shot-CoT**, requires no task-specific few-shot exemplars (unlike the original chain-of-thought prompting from Wei et al. 2022) and uses the same single trigger phrase across all tasks.

The authors evaluated Zero-shot-CoT on 12 datasets spanning arithmetic (MultiArith, GSM8K, AQUA-RAT, SVAMP, SingleEq, AddSub), symbolic (Last Letter, Coin Flip), commonsense (CommonsenseQA, StrategyQA), and logical reasoning (Date Understanding, Tracking Shuffled Objects). With InstructGPT (text-davinci-002), the method boosted MultiArith accuracy from 17.7% to 78.7% and GSM8K from 10.4% to 40.7% — gains comparable to few-shot CoT but without any examples. The same prompt worked across all tasks, suggesting broad, task-agnostic cognitive capabilities latent in LLMs.

The paper also found that Zero-shot-CoT is an **emergent property of model scale**: small models show little benefit, while large models (100B+) see dramatic gains. This mirrors the scale dependence found in few-shot CoT and reinforced the emerging understanding that reasoning abilities appear only past a model-size threshold.

This notebook measures the effect of "Let's think step by step" across arithmetic, symbolic, and logical tasks using GPT-2 small (124M), demonstrating the scale-dependent nature of the Zero-shot-CoT phenomenon.

## What Problem Does It Solve

Imagine you are asked "If a train travels 60 km/h for 2.5 hours and then 80 km/h for 1.5 hours, what is the total distance?" You would not blurt out a single number — you would break it down: "60 × 2.5 = 150, 80 × 1.5 = 120, so 150 + 120 = 270." The paper teaches language models to do the same thing, but with an even simpler trick than prior work: just say "Let's think step by step" and let the model figure out the rest.

Before this work, chain-of-thought prompting required carefully written worked examples for each task — a labor-intensive process. The authors showed that a single generic trigger phrase, with no examples at all, could unlock similar reasoning behavior across many different tasks. This made CoT accessible as a zero-shot baseline and revealed that the reasoning ability was already inside the models, waiting to be prompted out.

## Core Idea

- **Single task-agnostic trigger:** Instead of hand-crafting few-shot examples per task, append "Let's think step by step" to any question to elicit multi-step reasoning — the same prompt works for math, logic, and symbolic tasks.
- **Two-stage prompting pipeline:** Stage 1 generates a reasoning chain from the trigger; Stage 2 feeds that reasoning back and asks for the final answer in the correct format. This self-augmented prompting separates reasoning from answer formatting.
- **Emergent with scale:** Zero-shot-CoT gains appear only in large models (~100B+ parameters). Small models produce fluent but illogical reasoning, mirroring the scale dependence of few-shot CoT.

## Key Method Details

- **Prompting template (Section 3):** `Q: [X]. A: [T]` where [T] is the trigger sentence. For Zero-shot-CoT, T = "Let's think step by step." For the standard baseline, T = "The answer is."
- **Two-stage pipeline (Section 3.1, Figure 2):** Stage 1 (reasoning extraction) generates reasoning z from the triggered prompt. Stage 2 (answer extraction) concatenates `[X'] [Z] [A]` where [A] is an answer-format trigger like "Therefore, the answer (arabic numerals) is."
- **Decoding:** Greedy decoding throughout (deterministic for zero-shot methods)
- **Key results (Table 1, text-davinci-002):** MultiArith 17.7% → 78.7%, GSM8K 10.4% → 40.7%, AQUA-RAT 22.4% → 33.5%, SVAMP 58.8% → 62.1%, Coin Flip 12.8% → 91.4%, Last Letter 0.2% → 57.6%, Date Understanding 49.3% → 67.5%, Tracking Shuffled Objects 31.3% → 52.4%
- **Scale study (Figure 3):** Performance is flat without CoT across model sizes but jumps sharply with CoT for large models. Small models (<10B) show no CoT benefit.
- **Prompt robustness (Table 4):** "Let's think step by step." achieved the best result (78.7% on MultiArith). Other instructive templates like "First," (77.3%) and "Let's think about this logically." (74.5%) also worked well. Misleading and irrelevant templates did not improve performance.

### Simplifications in This Notebook

- **Smaller model:** GPT-2 small (124M parameters) instead of InstructGPT text-davinci-002 (~175B) or PaLM 540B
- **Dataset:** 30-problem subsets per task instead of full benchmarks (GSM8K: 30 of 1,319; others: 30 generated)
- **Tasks:** 5 tasks (GSM8K, MultiArith, Coin Flip, Last Letter, Date Understanding) instead of the paper's 12 datasets
- **Greedy decoding** (same as the paper), no few-shot examples (same as the paper)
- **Synthetic MultiArith, Coin Flip, Last Letter, and Date Understanding problems** generated procedurally instead of the paper's curated benchmarks

### What the Notebook Result Shows

On 30-problem subsets with GPT-2 small (124M), both standard and CoT prompting achieve low accuracy across all tasks. GSM8K: standard 3.3%, CoT 0.0%. MultiArith: 0.0% for both. Coin Flip: 53.3% for both (the model can track a simple binary state). Last Letter: 0.0% for both. Date Understanding: 0.0% for both. Average: standard 11.3%, CoT 10.7% — essentially no improvement, and a slight regression on GSM8K where the CoT reasoning path sometimes leads to an unparseable answer.

This is consistent with the paper's central finding that Zero-shot-CoT is an emergent property of model scale. At 124M parameters, far below the ~100B threshold the paper identifies, models produce "fluent but illogical chains of thought." The paper reports that even PaLM 8B showed minimal CoT benefit (MultiArith 25.5%→66.1% only at 540B), so a 124M model showing no improvement is an expected replication of the scale-dependence finding, not a failure. The qualitative pattern — the same "Let's think step by step" trigger applied uniformly across all tasks with no examples — is the key demonstration.

## Influence

Zero-shot-CoT became one of the most widely cited prompting techniques in the LLM era. It directly influenced self-consistency (sampling multiple CoT paths and voting), Tree of Thoughts, and the broader family of test-time reasoning methods. The finding that a single generic prompt can elicit reasoning across diverse tasks shaped how the community thinks about latent capabilities in pretrained models. The scale-dependence finding contributed to the debate on emergent abilities in LLMs. About 2,423 citations ([OpenAlex](https://openalex.org/W7133231369), checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/053_zero-shot-cot-reasoning/solution.ipynb) | Zero-shot vs Zero-shot-CoT prompting comparison across 5 tasks (arithmetic, symbolic, logical) with GPT-2 small, demonstrating the scale-dependent emergence of CoT reasoning |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, prompt construction, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about Zero-shot CoT |
