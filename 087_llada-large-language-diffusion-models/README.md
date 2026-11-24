---
title: "087 LLaDA"
nav_order: 88
permalink: "/087_llada-large-language-diffusion-models/"
---
# Large Language Diffusion Models (LLaDA)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/llada-large-language-diffusion-models)

- **Paper:** [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)
- **Authors:** Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, Chongxuan Li
- **Lab, year:** Gaoling School of Artificial Intelligence, Renmin University of China (with Ant Group; work done during internship at Ant Group), 2025 (preprint, February 2025)

## Summary

Large language models (LLMs) are almost universally built on autoregressive models (ARMs) — the "predict the next token" paradigm. LLaDA (Large Language Diffusion with mAsking) challenges this assumption by training a diffusion model from scratch at the 8B scale, using the same pre-training and supervised fine-tuning (SFT) pipeline as standard LLMs.

Instead of predicting tokens left-to-right, LLaDA uses a masked diffusion process: during training, each token is independently masked with a random ratio t ~ U(0,1), and a bidirectional Transformer predicts all masked tokens simultaneously. The loss is a cross-entropy on masked positions weighted by 1/t, which is a variational lower bound on the log-likelihood (Eq. 3-4 in the paper). At inference time, generation starts from a fully masked sequence and iteratively unmasks tokens over multiple diffusion steps, with low-confidence remasking at each step (Section 2.4).

LLaDA 8B was pre-trained on 2.3 trillion tokens (0.13M H800 GPU hours) and SFT'd on 4.5M pairs. On 15 zero/few-shot benchmarks it surpasses LLaMA2 7B Base and matches LLaMA3 8B Base on many tasks — notably MMLU 65.9 vs 65.4 and GSM8K 70.3 vs 48.7 (Table 1). It also breaks the "reversal curse" (the inability of ARMs to answer "B is the capital of A" after learning "A is the capital of B"), outperforming GPT-4o on a reversal poem completion task (Table 4).

This repo's notebook implements a tiny masked-diffusion LM (~803K parameters) trained on 129 toy sentences, and visualizes the step-by-step denoising process that is the visual signature of LLaDA.

## What Problem Does It Solve

Imagine filling in a crossword puzzle. A standard LLM fills in words one at a time, left to right — it can only use the words it has already written to guess the next one. If it needs information from the end of the sentence to get the beginning right, it is out of luck. This is the "reversal curse": if an LLM learns "Paris is the capital of France," it often cannot answer "What is France's capital?" because it only learned the left-to-right direction.

LLaDA fills in the whole puzzle at once, like a human solver who looks at all the clues simultaneously. It masks random words, predicts them all using context from both sides, and repeats until the puzzle is complete. This bidirectional approach naturally handles reversal tasks because it never committed to a left-to-right order in the first place.

## Core Idea

- **Masked diffusion instead of autoregression:** LLaDA replaces the autoregressive factorization p(x) = p(x1) \* p(x2\|x1) \* ... with a forward masking process (randomly mask tokens with ratio t) and a reverse generation process (iteratively unmask), parameterized by a bidirectional Transformer (Section 2.1).
- **Likelihood bound training objective:** The training loss (Eq. 3) is cross-entropy on masked tokens weighted by 1/t, which is a principled upper bound on negative log-likelihood (Eq. 4). This is what distinguishes LLaDA from BERT's fixed 15% masking or MaskGIT's heuristic objective — the 1/t weighting gives it a theoretical link to maximum likelihood estimation.
- **Low-confidence remasking at inference:** During generation, the model predicts all masked tokens at each step, keeps the highest-confidence ones, and re-masks the rest (Section 2.4). This iterative refinement produces better results than one-shot prediction and is analogous to the annealing tricks used in LLM sampling.

## Key Method Details

- **Architecture (8B, Table 5):** 32 layers, d_model=4096, 32 attention heads, FFN dim 12,288. Key difference from LLaMA3: no causal mask (bidirectional attention), vanilla MHA instead of GQA (32 KV heads vs 8), no KV cache. Vocab 126,464. ~8.02B total parameters.
- **Pre-training:** 2.3T tokens, fixed sequence length 4096, AdamW (lr 4e-4, wd 0.1, batch 1280), WSD schedule. 0.13M H800 GPU hours.
- **SFT:** 4.5M instruction pairs, 3 epochs, lr 2.5e-5, batch 256. Prompt tokens are never masked; only response tokens are masked (Section 2.3).
- **Results (Table 1, pre-trained):** MMLU 65.9 (5-shot), GSM8K 70.3 (4-shot), HumanEval 35.4, HellaSwag 70.5 — competitive with LLaMA3 8B (MMLU 65.4, GSM8K 48.7, HumanEval 34.8, HellaSwag 79.1).
- **Reversal (Table 4):** On Chinese poem completion, LLaDA 8B Instruct scores 45.6 on reversal vs GPT-4o's 34.3 and Qwen2.5-7B's 38.0, while scoring 51.8 on forward (vs GPT-4o 82.7).
- **Sampling flexibility (Table 7):** LLaDA supports autoregressive, block diffusion, and pure diffusion sampling from the same model. Pure diffusion achieves the best overall performance (BBH 49.7, GSM8K 70.3, HumanEval 35.4).

### Simplifications in This Notebook

- Smaller model: 4 layers, d_model=128, ~803K parameters (vs 32 layers, 8.02B in the paper)
- Dataset: 129 hand-written English sentences with character-level tokenization instead of 2.3T tokens with BPE
- Vocabulary: 29 tokens (26 letters + space + mask + pad) instead of 126,464
- 30 epochs with AdamW (lr=1e-3, wd=0.1) instead of the paper's WSD schedule trained on H800 clusters
- 16 diffusion sampling steps with greedy decoding instead of the paper's configurable multi-step process

### What the Notebook Result Shows

The toy model's diffusion loss decreases from ~65 to ~6 over 30 epochs (with high variance due to the tiny dataset), showing that the masked diffusion objective is learnable even at tiny scale. The denoising visualization shows tokens being progressively unmasked from a fully masked sequence — the core visual signature of LLaDA's bidirectional generation.

Token prediction accuracy is about 0.16 at all masking ratios (vs a random baseline of 0.034), roughly flat across t values because the tiny model has not learned enough to benefit from additional context. The generated text is not coherent at this scale — the model produces character-like patterns but cannot form words or sentences with only 129 training examples. This is not comparable to the paper's 8B results.

## Influence

LLaDA was the first masked diffusion model scaled to 8B parameters from scratch, demonstrating that core LLM capabilities (scalability, in-context learning, instruction-following) are not unique to autoregressive models. It influenced subsequent work on diffusion-based language models, including Mercury (by Inception Labs) and other discrete diffusion approaches. The official code and model checkpoints were released at [github.com/ml-gsai/llada](https://github.com/ml-gsai/llada), and the paper received significant community attention (127 upvotes on Hugging Face Daily Papers).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/087_llada-large-language-diffusion-models/solution.ipynb) | From-scratch PyTorch masked-diffusion LM trained on toy sentences, with step-by-step denoising visualization |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
