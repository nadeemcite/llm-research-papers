---
title: "009 MLA"
nav_order: 10
permalink: "/009_deepseek-v2-multi-head-latent-attention/"
---
# DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (MLA)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/deepseek-v2-multi-head-latent-attention)

- **Paper:** [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
- **Authors:** DeepSeek-AI (Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Damai Dai, Daya Guo, ... Zhihong Shao, et al.)
- **Lab, year:** DeepSeek (Hangzhou, China), 2024

## Summary

DeepSeek-V2 is a Mixture-of-Experts (MoE) language model with 236 billion total parameters, of which only 21 billion are activated per token. Its two architectural innovations are Multi-head Latent Attention (MLA) and DeepSeekMoE. MLA compresses the Key-Value (KV) cache — the memory that stores attention keys and values for previously generated tokens so the model does not recompute them — into a small latent vector, cutting cache size by 93.3% compared to standard attention. DeepSeekMoE splits the feed-forward network into many fine-grained experts with shared experts for common knowledge, enabling strong performance at economical training cost.

The paper trains DeepSeek-V2 on 8.1 trillion tokens and further applies Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL). With only 21B activated parameters, it achieves 78.5 on MMLU (a 57-subject multiple-choice benchmark), 78.9 on BBH (BigBench Hard), and 79.2 on GSM8K (grade-school math), matching or surpassing LLaMA 3 70B and Mixtral 8x22B while using far fewer active parameters. Training costs are 42.5% lower than DeepSeek 67B, and generation throughput is 5.76 times higher.

This repo's notebook implements Multi-head Latent Attention from scratch in PyTorch, alongside standard MHA and GQA, and trains three small character-level transformers on Tiny Shakespeare to compare KV-cache size and language-modelling quality.

## What Problem Does It Solve

Imagine you are reading a long book and need to remember every sentence you've read so far to understand the next one. You write down notes — but the notes pile up, and soon you run out of desk space. In transformer language models, those "notes" are the KV cache: for every token generated, the model stores its key and value vectors so it can attend back to them without recomputing. As the text grows, this cache becomes the bottleneck — it limits how many simultaneous users you can serve and how long the context can be.

MLA solves this by compressing the notes. Instead of storing full keys and values for every head, it projects the hidden state down to a tiny latent vector and only caches that. At inference time, the keys and values can be recovered from the latent — or even better, the up-projection weights can be absorbed into the query and output projections, so the keys and values never need to be explicitly computed at all. This gives MHA-level performance with a fraction of the memory.

## Core Idea

- **Low-rank KV joint compression:** The hidden state is projected down to a compressed latent vector `c_KV` (dimension `d_c`, much smaller than `n_h * d_h`). Only `c_KV` is cached during inference. At training time, keys and values are recovered via up-projection matrices `W_UK` and `W_UV`; at inference, these can be absorbed into the query and output projections to skip key/value computation entirely.
- **Decoupled Rotary Position Embedding (RoPE):** RoPE is position-sensitive and incompatible with low-rank KV compression (it would prevent weight absorption). MLA solves this by splitting the query and key into a content part (`q_C`, `k_C`) that goes through the compressed path, and a separate RoPE part (`q_R`, `k_R`) that carries position information via a shared decoupled key. The final attention uses the concatenation `[q_C; q_R]` and `[k_C; k_R]`.
- **Query compression:** The query is also compressed via `c_Q = W_DQ @ h`, reducing activation memory during training (though it does not affect the KV cache). The decoupled RoPE queries are produced from `c_Q` via `W_QR`.

## Key Method Details

- **KV compression (Eq. 9–11):** `c_KV = W_DKV @ h_t`, where `d_c = 512` for DeepSeek-V2 (set to `4 * d_h`). Keys: `k_C = W_UK @ c_KV`, values: `v_C = W_UV @ c_KV`. KV cache per token: `(d_c + d_h_R) * l` elements, where `d_h_R = 64` (set to `d_h / 2`).
- **Decoupled RoPE (Eq. 14–19):** `q_R = RoPE(W_QR @ c_Q)`, `k_R = RoPE(W_KR @ h_t)`. The shared decoupled key `k_R` is cached alongside `c_KV`. Attention uses scale `1/sqrt(d_h + d_h_R)`.
- **Weight absorption:** During inference, `W_UK` is absorbed into `W_UQ` and `W_UV` into `W_O`, so `k_C` and `v_C` need not be computed for cached tokens.
- **Architecture:** 60 layers, hidden dim 5120, 128 heads, `d_h = 128`, `d_c = 512`, `d_c_q = 1536`, `d_h_R = 64`. DeepSeekMoE: 2 shared experts + 160 routed experts (6 activated), intermediate dim 1536 per expert. 236B total, 21B activated.
- **Results (Table 2):** MMLU 78.5, BBH 78.9, GSM8K 79.2, MATH 43.6. KV cache reduced 93.3%, training cost saved 42.5%, throughput 5.76x vs DeepSeek 67B. MLA vs MHA ablation (Table 9): MLA achieves MMLU 59.0 vs 57.5 for large MoE with MHA, using 4% of the KV cache (34.6K vs 860.2K elements per token).

### Simplifications in This Notebook

- Smaller model: 4 layers, 4 heads, 128 dim, about 0.80-0.98M parameters (vs 236B total, 21B activated)
- Dataset: Tiny Shakespeare (~1 MB, character-level, vocab 65) instead of 8.1T tokens
- 8 epochs, 50 steps per epoch, batch 32, lr 3e-4 instead of full-scale pretraining
- No MoE (focus is on MLA attention comparison); no SFT/RL; no weight absorption optimisation
- MLA dimensions scaled down: `d_c = 128` (4 x `d_h`), `d_h_R = 16` (`d_h / 2`), following the paper's ratios

### What the Notebook Result Shows

On the Tiny Shakespeare validation set (character-level, 128-token context, 8 epochs), the three attention mechanisms achieve: MHA val perplexity 7.61 (KV cache 1024 elements/token), GQA val perplexity 7.81 (512 elements, 50% of MHA), MLA val perplexity 7.94 (576 elements, 56.2% of MHA). MLA reduces KV cache by 43.8% vs MHA at this scale — less than the paper's 93.3% because our `d_c / (2 * n_h * d_h)` ratio is higher than DeepSeek-V2's.

The perplexities are within 4.4% of each other, consistent with the paper's finding that MLA matches MHA quality (Table 9: MLA slightly outperforms MHA at scale). At this tiny scale (under 1M parameters), MHA has a small advantage as expected — the paper's MLA advantage appears at larger scale where the low-rank compression acts as a useful regulariser.

## Influence

DeepSeek-V2's MLA architecture became the foundation for DeepSeek-V3, DeepSeek-R1, and DeepSeek-V3.2, which are among the most capable open-weight models as of 2026. The decoupled RoPE strategy for low-rank attention influenced subsequent efficient attention designs. The combination of MLA with MoE demonstrated that extreme KV-cache compression could coexist with strong performance, encouraging broader adoption of latent attention techniques. The paper has about 108 citations ([OpenAlex](https://openalex.org/W4396815229), checked October 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/009_deepseek-v2-multi-head-latent-attention/solution.ipynb) | From-scratch PyTorch implementation of Multi-head Latent Attention (MLA) with decoupled RoPE, compared against MHA and GQA on Tiny Shakespeare |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
