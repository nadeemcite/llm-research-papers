---
title: "052 Chain-of-Thought"
nav_order: 53
permalink: "/052_chain-of-thought-prompting-reasoning/"
---
# Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (CoT)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/chain-of-thought-prompting-reasoning)

- **Paper:** [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- **Authors:** Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, Denny Zhou
- **Lab, year:** Google Research, Brain Team, 2022 (NeurIPS 2022)

## Summary

In January 2022, researchers at Google Brain showed that simply augmenting few-shot prompts with intermediate reasoning steps — a "chain of thought" — dramatically improves the ability of large language models to perform complex reasoning. The method, called **chain-of-thought (CoT) prompting**, requires no finetuning: you just include a few worked examples showing the reasoning process in the prompt, and the model learns to generate similar step-by-step reasoning before giving its final answer.

The authors evaluated CoT prompting on three categories of reasoning: arithmetic (GSM8K, SVAMP, ASDiv, AQuA, MAWPS), commonsense (CSQA, StrategyQA, Date Understanding, Sports Understanding, SayCan), and symbolic (last-letter concatenation, coin flip). Across all categories, CoT prompting yielded large gains — but only for sufficiently large models. The headline result: PaLM 540B with just eight CoT exemplars achieved 56.9% accuracy on GSM8K, surpassing the prior state of the art of 55% held by finetuned GPT-3 with a verifier. On SVAMP, CoT boosted PaLM 540B from 69.4% to 79.0%.

Crucially, the paper found that chain-of-thought reasoning is an **emergent property of model scale**: it does not improve performance for small models and only yields gains at ~100B+ parameters. Small models produced "fluent but illogical chains of thought," often performing worse with CoT than with standard prompting. This finding about emergence has shaped how the field understands scaling and reasoning capabilities.

This notebook compares standard vs CoT prompting on a 200-problem GSM8K subset using GPT-2 small (124M), demonstrating the scale-dependent nature of CoT benefits.

## What Problem Does It Solve

Imagine someone asks you "If Jane has 10 flowers and gives 2 to her mom and 3 to her dad, how many does she have left?" You don't just blurt out a number — you break it down: "She had 10, gave away 2, that's 8. Then gave away 3 more, that's 5. So the answer is 5." The paper teaches language models to do the same thing: instead of jumping straight to an answer, they generate intermediate reasoning steps that lead to the final answer.

Before this work, large language models could answer simple questions via few-shot prompting (showing a few examples of question → answer), but they struggled with multi-step reasoning tasks like math word problems. Simply making models bigger didn't help much on these tasks — the scaling curve was flat. The insight was that the prompt format itself was the bottleneck: standard prompts asked for answers directly, leaving no room for the model to "think" through the problem.

## Core Idea

- **Chain-of-thought exemplars:** Augment each few-shot example in the prompt with a chain of intermediate reasoning steps leading to the final answer, so the model learns to produce reasoning before answering.
- **No finetuning required:** CoT prompting works purely through the prompt — a single off-the-shelf model checkpoint can perform many reasoning tasks without any weight updates.
- **Emergent property of scale:** CoT reasoning only emerges in sufficiently large models (~100B+ parameters). Small models produce illogical chains of thought and may perform worse with CoT than without.

## Key Method Details

- **Prompting format:** Standard prompting uses ⟨input, output⟩ pairs; CoT prompting uses ⟨input, chain of thought, output⟩ triples (Section 2)
- **Exemplars:** 8 manually composed CoT exemplars for arithmetic, used across all five math benchmarks (Section 3.1). Full prompts in Appendix G.
- **Decoding:** Greedy decoding (no sampling)
- **Models evaluated:** GPT-3 (350M–175B), LaMDA (422M–137B), PaLM (8B–540B), UL2 20B, Codex (Section 3.1)
- **Key results on GSM8K** (Table 2): PaLM 540B standard 17.9% → CoT 56.9%; GPT-3 175B standard 15.6% → CoT 46.9%; LaMDA 137B standard 6.5% → CoT 14.3%. Small models (<10B) showed no improvement or degradation.
- **Ablation findings** (Section 3.3): Equation-only prompting, variable-computation (dots), and reasoning-after-answer ablations all performed ~same as baseline, confirming that the natural language reasoning steps — not just extra computation or knowledge activation — drive the gains.
- **Robustness** (Section 3.4): CoT outperforms standard prompting across different annotators, exemplar sets, and exemplar orders.

### Simplifications in This Notebook

- **Smaller model:** GPT-2 small (124M parameters) instead of PaLM 540B / GPT-3 175B
- **Dataset:** 200-problem subset of GSM8K test set instead of the full 1,319 problems
- **8 few-shot exemplars** from the GSM8K training set (same count as the paper), greedy decoding
- **No finetuning** — pure few-shot prompting, matching the paper's approach

### What the Notebook Result Shows

On the 200-problem GSM8K subset, GPT-2 small (124M) achieves low accuracy with both standard and CoT prompting, with CoT showing a modest improvement or comparable performance. This is consistent with the paper's central finding that CoT reasoning is an emergent property of model scale — at 124M parameters (far below the ~100B threshold the paper identifies), models produce "fluent but illogical chains of thought." The paper reports that PaLM 8B also showed no CoT benefit (4.9% standard → 4.1% CoT on GSM8K), so a 124M model showing minimal improvement is an expected replication of the scale-dependence finding, not a failure. The qualitative pattern — small models don't benefit from CoT while large models do — is the key takeaway.

## Influence

Chain-of-thought prompting became one of the most influential ideas in the LLM era. It directly inspired zero-shot CoT ("Let's think step by step"), self-consistency (sampling multiple reasoning paths and majority voting), Tree of Thoughts, and the entire family of test-time reasoning methods. CoT prompting is now a standard technique built into modern LLM systems (OpenAI o1, DeepSeek-R1, etc.). The paper's finding that CoT is an emergent property of scale became a foundational reference for the study of emergent abilities in LLMs. About 9,664 citations ([OpenAlex](https://openalex.org/W7133224126), checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/052_chain-of-thought-prompting-reasoning/solution.ipynb) | Few-shot prompting comparison (standard vs chain-of-thought) on GSM8K with GPT-2 small, demonstrating the scale-dependent emergence of CoT reasoning |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, prompt construction, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about Chain-of-Thought Prompting |
