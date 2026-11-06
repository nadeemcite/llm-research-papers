---
title: "054 Self-Consistency"
nav_order: 55
permalink: "/054_self-consistency-chain-of-thought/"
---
# Self-Consistency Improves Chain of Thought Reasoning in Language Models (Self-Consistency)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/self-consistency-chain-of-thought-reasoning)

- **Paper:** [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)
- **Authors:** Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou
- **Lab, year:** Google Research, Brain Team, 2022 (ICLR 2023)

## Summary

In March 2022, researchers at Google Research Brain Team introduced **self-consistency**, a new decoding strategy for chain-of-thought (CoT) prompting. Standard CoT prompting uses greedy decoding — it takes the single most likely reasoning path from the language model. Self-consistency replaces this with a "sample-and-marginalize" procedure: it samples multiple diverse reasoning paths from the model's decoder (using temperature sampling) and then selects the most consistent answer by majority vote. The key intuition is that complex reasoning problems typically have multiple valid approaches leading to the same correct answer, while incorrect reasoning paths are more likely to diverge to different wrong answers.

The authors evaluated self-consistency on four language models — UL2-20B, GPT-3-175B, LaMDA-137B, and PaLM-540B — across arithmetic reasoning (GSM8K, MultiArith, AddSub, ASDiv, AQuA, SVAMP), commonsense reasoning (CommonsenseQA, StrategyQA, ARC), and symbolic reasoning (Last Letter, Coin Flip). With PaLM-540B, self-consistency boosted GSM8K from 56.5% to 74.4% (+17.9%), SVAMP from 79.0% to 86.6% (+11.0%), and MultiArith from 94.7% to 99.3% (+4.6%), achieving new state-of-the-art results on nearly all tasks. The method requires no additional training, fine-tuning, or human annotation — it works entirely off-the-shelf with pre-trained models.

The paper also showed that self-consistency is robust to sampling strategies and parameters, works with imperfect prompts, and even improves performance when CoT hurts (e.g., on NLI tasks). Gains scale with model size: +3-6% for UL2-20B but +9-23% for LaMDA-137B and GPT-3. The method also provides an uncertainty estimate — the consistency (fraction of paths agreeing with the final answer) correlates with accuracy.

This notebook implements self-consistency with GPT-2 small (124M) on GSM8K and MultiArith, demonstrating the method's mechanics and the scale-dependence of its gains. At 124M parameters, greedy CoT accuracy is 3.3% on GSM8K and 0.0% on MultiArith, and self-consistency does not improve over greedy — consistent with the paper's finding that gains are minimal for small models.

## What Problem Does It Solve

Imagine you are solving a math puzzle and you work through it three different ways. Two of your approaches give you the answer 42, and the third gives 38. You would probably trust 42 — the fact that two independent methods agree gives you confidence. The paper applies this same principle to language models.

Before this work, chain-of-thought prompting used greedy decoding, which always picks the single most likely next word at every step. This means if the model starts down a wrong path early, it commits to it — there is no way to recover. Self-consistency instead samples multiple different reasoning paths and takes a majority vote on the final answer, so a single mistake in one path does not drag down the result if other paths reach the correct answer. It is like asking the model to solve the problem several times independently and going with the consensus.

## Core Idea

- **Sample-and-marginalize decoding:** Instead of one greedy reasoning path, sample N diverse reasoning paths using temperature sampling (T=0.5-0.7, top-k=40). Each path produces a final answer; select the most common answer by majority vote.
- **Self-ensemble, not model ensemble:** Self-consistency works on a single language model — it does not require training or combining multiple models. The diversity comes from the stochastic sampling process, not from different models.
- **No training required:** The method is entirely unsupervised, works with any pre-trained language model, and requires no additional data, fine-tuning, or auxiliary models.

## Key Method Details

- **Few-shot CoT prompting (Section 2):** The model is prompted with 8 manually written chain-of-thought exemplars for arithmetic tasks, 4-7 for commonsense tasks. The test question is appended and the model generates a reasoning path + answer.
- **Sampling (Section 3.1):** Temperature sampling with T=0.5, top-k=40 for UL2-20B and LaMDA-137B; T=0.7, top-k=40 for PaLM-540B; T=0.7 without top-k for GPT-3.
- **Answer aggregation (Section 2, Table 1):** Majority vote (unweighted sum) over final answers. Table 1 shows this performs nearly identically to normalized weighted sum, because the model's normalized probabilities are close across sampled paths.
- **Results (Table 2, PaLM-540B):** GSM8K 56.5% → 74.4% (+17.9%), MultiArith 94.7% → 99.3% (+4.6%), SVAMP 79.0% → 86.6% (+11.0%), AQuA 35.8% → 48.3% (+12.5%)
- **Scale dependence (Section 3.5, Figure 4 right):** Gains are larger for bigger models — +3-6% for UL2-20B, +9-23% for LaMDA-137B and GPT-3, and up to +17.9% for PaLM-540B on GSM8K.
- **Robustness (Section 3.5):** Self-consistency is robust to sampling strategies (temperature, top-k, nucleus), imperfect prompts (Table 8), and works with zero-shot CoT (PaLM-540B GSM8K: 43.0% → 69.2%, +26.2%).
- **Consistency as uncertainty (Figure 5):** The fraction of sampled paths agreeing with the final answer correlates with accuracy, providing a built-in confidence estimate.

### Simplifications in This Notebook

- **Smaller model:** GPT-2 small (124M parameters) instead of PaLM-540B / GPT-3-175B / LaMDA-137B / UL2-20B
- **Dataset:** 30 GSM8K + 20 MultiArith problems instead of full benchmarks (GSM8K: 30 of 1,319; MultiArith: 20 generated)
- **Few-shot exemplars:** 8 from GSM8K train set (same count as paper's 8 manually written exemplars)
- **Sampling:** T=0.7, top-k=40 (paper's PaLM-540B setting), 40 sampled paths per problem
- **Single run** (no averaging over 10 runs as in the paper)

### What the Notebook Result Shows

On 30 GSM8K and 20 MultiArith problems with GPT-2 small (124M), greedy CoT achieves 3.3% on GSM8K and 0.0% on MultiArith (average 1.7%). Self-consistency at N=40 achieves 0.0% on both tasks (average 0.0%), a -1.7% change from greedy. The sampled reasoning paths produce wildly different answers (e.g., for a GSM8K problem with ground truth 18, sampled answers include 1, 40, 32, -14, 12, 8, 48, 2.5, 128, 50 — no consensus), so majority voting cannot recover the correct answer.

This is consistent with the paper's central finding that self-consistency gains are scale-dependent (Section 3.5, Figure 4 right): "the gain is relatively lower for smaller models due to certain abilities (e.g., arithmetic) only emerge when the model reaches a sufficient scale." At 124M parameters, far below the 20B-540B models the paper evaluated, the model lacks the arithmetic reasoning ability to produce correct paths, so sampling more paths cannot help — the diversity produces scatter, not consensus. The paper's UL2-20B (the smallest model tested) showed only +3-6% gains; at 124M, no gain is an expected extrapolation of this trend, not a failure. The qualitative demonstration — the sample-and-marginalize procedure, accuracy vs N plot, and vote distribution — is the key output.

## Influence

Self-consistency became one of the most widely adopted decoding strategies for LLM reasoning. It directly influenced Tree of Thoughts (Yao et al., 2023), which extends the multi-path idea to tree-structured search, and the broader family of test-time compute scaling methods. The "self-ensemble" concept — getting better answers from a single model via diverse sampling and aggregation — shaped subsequent work on best-of-N sampling, process reward models, and reasoning scaling laws. The finding that consistency correlates with accuracy inspired work on LLM uncertainty estimation. About 2,236 citations ([OpenAlex](https://openalex.org/W4221161695), checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/054_self-consistency-chain-of-thought/solution.ipynb) | Self-consistency decoding pipeline with GPT-2 small: sample 40 reasoning paths per problem, majority-vote, and plot accuracy vs N on GSM8K and MultiArith |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, prompt construction, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about Self-Consistency |
