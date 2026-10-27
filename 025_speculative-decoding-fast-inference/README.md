---
title: "025 Speculative Decoding"
nav_order: 26
permalink: "/025_speculative-decoding-fast-inference/"
---
# Fast Inference from Transformers via Speculative Decoding (Speculative Decoding)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/speculative-decoding-fast-inference)

- **Paper:** [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- **Authors:** Yaniv Leviathan, Matan Kalman, Yossi Matias
- **Lab, year:** Google Research, Mountain View, CA, 2022 (ICML 2023 Oral)

## Summary

Large autoregressive (next-token-predicting) models like Transformers generate text one token at a time — decoding K tokens requires K sequential forward passes through the model, each dependent on the previous output. This serial bottleneck makes inference slow, especially for large models where each forward pass is expensive. Existing speedup methods (distillation, quantization, early exits) typically change the model architecture, require retraining, or alter the output distribution.

Speculative decoding breaks this serial dependency. The key insight is that many tokens in a sequence are "easy" — a small, fast approximation model can predict them correctly.

The algorithm works in two phases per step: first, a small **draft model** M_q autoregressively generates gamma candidate tokens (cheap, serial but fast). Then, the large **target model** M_p verifies all gamma tokens **in one parallel forward pass** — it processes the prefix plus all gamma draft tokens at once, producing gamma+1 logit distributions (a logit is the raw score a model assigns to each possible next token before being converted to probabilities).

Each draft token is accepted or rejected based on whether the target model agrees. On rejection, a correction token is sampled from an adjusted distribution. This produces between 1 and gamma+1 tokens per target-model call, with a mathematical guarantee that the output distribution is **identical** to standard decoding.

The paper demonstrates 2X-3X wall-clock acceleration on T5-XXL (11B parameters) for English-German translation and CNN/DM summarization, with byte-identical outputs — no retraining, no architecture changes, no distribution shift. The method also works with trivial approximation models like n-gram lookup tables, which still yield non-trivial speedups (1.25X for a bigram model on translation).

This repo's notebook implements speculative decoding with GPT-2 small (124M) as the target and a 4-layer GPT-2 variant (~12M) as the draft, verifying output identity and measuring speedup and acceptance rate.

## What Problem Does It Solve

Imagine a senior chef who can cook any dish perfectly but works slowly, and a junior cook who is fast but sometimes makes mistakes. If you need a 10-course meal, having the senior chef cook everything sequentially takes all day. But what if the junior cook could draft all 10 dishes first, and the senior chef only needs to **taste-test** them all at once? If a dish is correct, it's accepted; if not, the senior chef fixes just that one. The senior chef's tasting is the bottleneck, but now it happens once for 10 dishes instead of 10 times for one dish each.

That's speculative decoding: the draft model (junior cook) speculatively generates several tokens, and the target model (senior chef) verifies them all in one parallel pass. The target model is called fewer times, so inference is faster — and because every token is verified by the target, the output is identical to what the target would have produced alone.

## Core Idea

- **Speculative execution (Section 1):** Borrowing from CPU branch prediction, the algorithm performs work (draft token generation) *before* confirming it's needed, then verifies in parallel. The payoff is increased concurrency: the expensive target model processes gamma+1 positions at once instead of one at a time.
- **Speculative sampling (Section 2.3):** A novel sampling method that guarantees the output distribution is identical to sampling from the target model alone. For each draft token x sampled from q(x), it is accepted with probability min(1, p(x)/q(x)). On rejection, a token is resampled from norm(max(0, p(x) - q(x))). This works for any temperature, top-k, or nucleus sampling.
- **Acceptance rate alpha (Section 3.2):** The expected fraction of draft tokens accepted, alpha = E(min(p, q)) (Corollary 3.6), measures how well the draft approximates the target. The expected tokens per step is (1 - alpha^(gamma+1)) / (1 - alpha) (Equation 1), and the walltime improvement is that divided by (gamma*c + 1) where c is the draft-to-target cost ratio (Theorem 3.8).

## Key Method Details

- **Algorithm 1 (Speculative Decoding Step):** (1) Sample gamma tokens x_1,...,x_gamma from M_q autoregressively. (2) Run M_p in parallel on prefix + all gamma draft tokens, getting gamma+1 distributions p_1,...,p_{gamma+1}. (3) Accept draft tokens left-to-right: accept x_i if r_i < p_i(x_i) / q_i(x_i) where r_i ~ U(0,1). (4) At the first rejection (or after all accepted), sample one bonus token from the adjusted distribution. Return n accepted tokens + 1 bonus token.
- **Walltime improvement (Theorem 3.8):** (1 - alpha^(gamma+1)) / ((1-alpha)(gamma*c + 1)). When alpha > c, a speedup is guaranteed (Corollary 3.9), with minimum improvement (1+alpha)/(1+c).
- **Empirical results (Table 2):** On T5-XXL 11B with T5-small (77M) as draft: 3.4X speedup (argmax) and 2.6X (temperature=1) for EnDe translation; 3.1X (argmax) and 2.3X (temp=1) for CNN/DM summarization. gamma = 7 for translation, gamma = 5 for summarization.
- **Alpha values (Table 3):** Approximation models ~2 orders of magnitude smaller yield alpha between 0.5 and 0.9. Even trivial models: unigram alpha = 0.03-0.13, bigram alpha = 0.05-0.23. Sharper distributions (argmax) give higher alpha than temperature sampling.
- **Negligible-cost models (Section 3.6):** N-gram models with c ~ 0 yield improvement (1-alpha^(gamma+1))/(1-alpha), bounded by 1/(1-alpha). A bigram model on EnDe translation achieves alpha ~ 0.2, giving 1.25X speedup with gamma = 3 — surprisingly high for a table lookup.
- **No retraining needed (Section 1):** The method works with off-the-shelf models. Any model can serve as the approximation model M_q without restriction (Section A.1 guarantees identical distribution for any M_q).

### Simplifications in This Notebook

- GPT-2 small (124M, 12 layers) as target and a 4-layer GPT-2 variant (~12M) as draft, instead of T5-XXL 11B and T5-small 77M
- Short text-completion prompts instead of WMT translation or CNN/DM summarization tasks
- Greedy (argmax, temp=0) decoding primarily; speculative sampling code included for temperature > 0
- gamma = 4 draft tokens (paper uses up to 7); 60 tokens generated per prompt; 10 test prompts
- Draft model is a layer-truncated version of the target (first 4 of 12 blocks, shared embeddings), not a separately trained smaller model as in the paper — this results in low alpha

### What the Notebook Result Shows

On 10 diverse text-completion prompts with GPT-2 small as the target and a 4-layer GPT-2 variant as the draft (gamma = 4, greedy decoding), speculative decoding produces **byte-identical** output to standard autoregressive (one-token-at-a-time) decoding on all 10 prompts — confirming the paper's central guarantee of unchanged outputs.

The 4-layer draft model (built by copying the first 4 transformer blocks of GPT-2, sharing embeddings) achieves a very low acceptance rate (alpha ~ 0.03 under greedy decoding), because a layer-truncated model is a poor approximation of the full model — unlike the paper's separately trained smaller models (T5-small, 77M) which achieve alpha 0.5-0.9.

The draft model's serial calls add overhead without enough accepted tokens to compensate, resulting in a wall-clock slowdown rather than a speedup. The draft model size sweep shows alpha increasing from 0.028 (2 layers) to 0.091 (8 layers), and reaching 1.0 at 12 layers (draft = target, every token accepted). This confirms the algorithm's correctness but also illustrates that speculative decoding's speedup depends critically on having a good draft model — the paper's key requirement that the draft be "a couple of orders of magnitude smaller" and separately trained.

## Influence

Speculative decoding became one of the most widely adopted LLM inference acceleration techniques. It was integrated into production serving frameworks including vLLM, TensorRT-LLM, TGI (Text Generation Inference), and LMDeploy. The method influenced a line of follow-up work: Medusa (Cai et al., 2024) replaced the autoregressive draft with parallel multi-head prediction; EAGLE (Li et al., 2024) improved draft quality with feature-level prediction; and SpecExec (Svirschevski et al., 2024) extended speculative execution to multi-token verification. The paper has about 71 citations on [OpenAlex](https://openalex.org/W4310561894) (checked October 2026). It was presented as an Oral at ICML 2023.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/025_speculative-decoding-fast-inference/solution.ipynb) | PyTorch implementation of speculative decoding (Algorithm 1) with GPT-2 small target and 4-layer draft model, verifying output identity and measuring acceptance rate and speedup |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, key functions, and design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
