---
title: "035 DeepSeek-V3"
nav_order: 36
permalink: "/035_deepseek-v3-technical-report/"
---
# DeepSeek-V3 Technical Report (DeepSeek-V3)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/deepseek-v3-technical-report)

- **Paper:** [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- **Authors:** DeepSeek-AI; Liu, Aixin; Feng, Bei; Xue, Bing; Wang, Bingxuan; Wu, Bochao; Lu, Chengda; Zhao, Chenggang; Deng, Chengqi; Zhang, Chenyu; Ruan, Chong; Dai, Damai; Guo, Daya; et al. (200+ authors total)
- **Lab, year:** DeepSeek (Hangzhou, China), 2024 (arXiv preprint, 27 December 2024)

## Summary

DeepSeek-V3 is a Mixture-of-Experts (MoE) language model with 671 billion total parameters, of which 37 billion are activated for each token. It inherits two architectural innovations from DeepSeek-V2: Multi-head Latent Attention (MLA), which compresses the KV cache through low-rank joint compression, and the DeepSeekMoE architecture, which uses fine-grained experts (256 routed + 1 shared) with sigmoid gating and top-K normalization. On top of this, DeepSeek-V3 introduces two new contributions: an **auxiliary-loss-free load balancing strategy** and a **Multi-Token Prediction (MTP)** training objective.

The auxiliary-loss-free strategy replaces the conventional auxiliary loss (an extra loss term that penalizes uneven expert usage) with a simple per-expert bias term. The bias is added to routing scores for top-K selection only — the gating value still uses the original affinity score, so the bias does not distort the model's gradient signal. After each training step, the bias is decreased for overloaded experts and increased for underloaded ones.

This achieves balanced expert load without the performance degradation that comes from a large auxiliary loss. The MTP objective trains the model to predict not just the next token but also one additional token at depth 1, which the authors find improves benchmark performance and can be used for speculative decoding at inference time.

DeepSeek-V3 was pre-trained on 14.8 trillion tokens, followed by supervised fine-tuning and reinforcement learning. The total training cost was 2.788 million H800 GPU hours (about $5.58 million at $2/GPU-hour), roughly 11× less compute than Llama 3 405B. The training was remarkably stable: no irrecoverable loss spikes and no rollbacks throughout the entire run. Evaluations show DeepSeek-V3 outperforms all other open-source models and achieves performance comparable to GPT-4o and Claude-3.5-Sonnet on many benchmarks (88.5 MMLU, 75.9 MMLU-Pro, 59.1 GPQA).

This repo's notebook implements the auxiliary-loss-free load balancing strategy in a small 8-expert MoE transformer trained on Tiny Shakespeare, comparing it against no balancing and conventional auxiliary-loss balancing.

## What Problem Does It Solve

Imagine a hospital with 8 specialist doctors. Patients arrive and a receptionist routes each one to the 2 most relevant specialists. Without any guidance, the receptionist might send almost everyone to Doctor A and Doctor B, while Doctors C through H sit idle. The hospital's capacity is wasted.

This is **routing collapse** in MoE models. Without intervention, the router learns to prefer a few experts and ignores the rest, wasting most of the model's capacity. The traditional fix is to add a penalty term (an "auxiliary loss") that scolds the model whenever expert usage is uneven. But this penalty competes with the main learning objective — too strong a penalty hurts model quality.

DeepSeek-V3's solution is elegant: add a small adjustable bias to each expert's routing score. The bias nudges the routing decision toward underused experts, but since the bias is not part of the gradient, it doesn't interfere with what the model learns. It's like giving the receptionist a gentle nudge toward idle doctors, without changing how they evaluate patients.

## Core Idea

- **Auxiliary-loss-free bias tuning (Section 2.1.2):** A per-expert bias term b_i is added to the affinity score s_{i,t} for top-K routing selection only. The gating value (which multiplies the expert output) uses the original affinity, so gradients flow cleanly. After each step, b_i is decreased by γ if expert i is overloaded and increased by γ if underloaded.
- **DeepSeekMoE with fine-grained experts (Section 2.1.2):** 256 routed experts with 2048-dim intermediate hidden, 1 shared expert always active, top-8 routing with sigmoid gating and normalization among selected affinities. The first 3 layers are dense (no MoE).
- **Multi-Token Prediction (Section 2.2):** Each token predicts the next token and one additional token at depth 1. This provides a richer training signal and enables speculative decoding at inference time.

## Key Method Details

- **Model size (Section 2.1):** 671B total parameters, 37B activated per token. 61 transformer layers, d_model = 7168, 128 attention heads, d_h = 128.
- **MoE config (Section 2.1.2):** N_s = 1 shared expert, N_r = 256 routed experts, K_r = 8 activated. Intermediate dim 2048 per expert. Node-limited routing: each token sent to at most 4 nodes.
- **Affinity computation (Eq. 15):** s_{i,t} = sigmoid(e_i^T · u_t), where e_i is the centroid of expert i. Gating values are normalized among selected experts: g_{i,t} = s_{i,t} / Σ(selected s_{j,t}).
- **Auxiliary-loss-free routing (Eq. 16):** g'_{i,t} = s_{i,t} if s_{i,t} + b_i ∈ Topk({s_{j,t} + b_j}, K_r), else 0. Bias is only used for routing; gating value uses original s_{i,t}.
- **Bias update:** At end of each step, if expert i is overloaded: b_i -= γ; if underloaded: b_i += γ. γ is the bias update speed hyperparameter.
- **Complementary sequence-wise auxiliary loss (Eq. 17):** L_Bal = α Σ f_i P_i, used alongside aux-free to prevent extreme per-sequence imbalance. Much smaller weight than conventional auxiliary loss.
- **Training (Section 4.1):** 14.8T tokens, AdamW (β1=0.9, β2=0.95, weight_decay=0.1), lr 2.2e-4 with 2K warmup steps, constant then cosine decay. Max seq length 4K. FP8 mixed precision.
- **Training cost (Table 1):** 2,664K H800-hours pre-training, 119K context extension, 5K post-training. Total 2,788K H800-hours (~$5.58M).
- **Results (Section 4.4):** 88.5 MMLU, 75.9 MMLU-Pro, 59.1 GPQA, 90.2 HumanEval-Mul, 61.6 GSM8K, 86.1 MATH. Comparable to GPT-4o and Claude-3.5-Sonnet.

### Simplifications in This Notebook

- 8 routed experts + 1 shared expert instead of 256 routed + 1 shared
- top-2 routing instead of top-8
- 4 transformer layers, d_model=128, 2 attention heads instead of 61 layers, d_model=7168, 128 heads
- Tiny Shakespeare char-level (~1 MB, 65 chars) instead of 14.8T tokens
- 15 epochs with AdamW (lr 3e-4) instead of the paper's full pre-training schedule
- No Multi-head Latent Attention (MLA), no Multi-Token Prediction (MTP), no FP8 training
- Sign-based bias update (simple rule) instead of the paper's exact implementation details

### What the Notebook Result Shows

The auxiliary-loss-free strategy achieves a coefficient of variation (CV) of 0.0173 in expert usage, versus 0.6653 for no balancing and 0.6207 for conventional auxiliary loss — confirming that the bias-tuning approach distributes tokens far more evenly across experts. The aux-free model achieves validation perplexity of 7.15, better than the aux-loss model's 7.22 and close to the no-balancing model's 7.08.

The bias terms self-adjust: across layers, bias values range from -0.01 to 0.44, with overloaded experts receiving lower bias and underloaded experts receiving higher bias. Absolute perplexity numbers are not comparable to the paper's 671B model on 14.8T tokens — this is a toy-scale demonstration of the routing principle.

## Influence

DeepSeek-V3 became one of the most discussed open-weight models of early 2025, widely noted for achieving frontier-class performance at a fraction of the training cost of comparable models. The auxiliary-loss-free load balancing strategy has been adopted in subsequent MoE training efforts. DeepSeek followed up with DeepSeek-R1 (January 2025), which distilled reasoning capability into DeepSeek-V3 through reinforcement learning. The open-source release of model weights and the technical report influenced the broader open-source LLM community's approach to efficient MoE training. About 293 citations ([OpenAlex](https://openalex.org/works/W4405903187), checked October 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/035_deepseek-v3-technical-report/solution.ipynb) | From-scratch PyTorch: 4-layer MoE transformer with 8 experts, comparing no-balancing, aux-loss, and DeepSeek-V3's aux-free load balancing on Tiny Shakespeare |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
