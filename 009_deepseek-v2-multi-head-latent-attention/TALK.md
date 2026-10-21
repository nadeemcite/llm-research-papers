---
title: "MLA: Talk & Q&A"
parent: "009 MLA"
nav_order: 2
---
# TALK — DeepSeek-V2: Multi-head Latent Attention (MLA)

## Talks, Slides and Press Coverage

1. **DeepSeek Official — "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" (DeepSeek-AI, May 2024):** The model announcement and technical report on the DeepSeek website and GitHub. ([link](https://github.com/deepseek-ai/DeepSeek-V2))
2. **Hugging Face — "DeepSeek-V2" model page:** The model is hosted on Hugging Face with model cards for DeepSeek-V2 and DeepSeek-V2-Chat, enabling community use and benchmarking. ([link](https://huggingface.co/deepseek-ai/DeepSeek-V2))
3. **Simon Willison — "DeepSeek-V2" (May 2024):** Coverage of DeepSeek-V2's architecture and efficiency claims, noting the MLA innovation and MoE design. ([link](https://simonwillison.net/2024/May/))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2405.04434).

**Q1: Why does MLA use low-rank compression for both KV and queries?**
A1 (from the paper, Section 2.1.2): The core of MLA is the low-rank joint compression for keys and values to reduce KV cache: "c_KV_t = W_DKV h_t, where c_KV_t is the compressed latent vector for keys and values; d_c (<< d_h n_h) denotes the KV compression dimension." Query compression is added for a different reason: "in order to reduce the activation memory during training, we also perform low-rank compression for the queries, even if it cannot reduce the KV cache."

**Q2: Why is RoPE incompatible with low-rank KV compression, and how does decoupled RoPE fix this?**
A2 (from the paper, Section 2.1.3): "RoPE is position-sensitive for both keys and queries. If we apply RoPE for the keys k_C, W_UK will be coupled with a position-sensitive RoPE matrix. In this way, W_UK cannot be absorbed into W_Q any more during inference, since a RoPE matrix related to the currently generating token will lie between W_Q and W_UK and matrix multiplication does not obey a commutative law." The solution: "we propose the decoupled RoPE strategy that uses additional multi-head queries q_R and a shared key k_R to carry RoPE."

**Q3: How much KV cache does MLA save compared to MHA?**
A3 (from the paper, Table 1 and Section 2.1.4): "MLA requires only a small amount of KV cache, equal to GQA with only 2.25 groups, but can achieve stronger performance than MHA." For DeepSeek-V2 specifically: d_c=512, d_h_R=64, l=60 layers, giving (512+64)*60 = 34,560 elements per token, vs MHA's 2*128*128*60 = 1,966,080 — a 93.3% reduction.

**Q4: Does MLA actually perform better than MHA, or just match it?**
A4 (from the paper, Table 9, Appendix D.2): The ablation compares MLA and MHA at two scales. For large MoE models (~250B total, ~21-25B activated), MLA achieves MMLU 59.0 vs 57.5 for MHA, BBH 50.7 vs 46.6, CMMLU 62.5 vs 60.7 — all better — while using only 4% of MHA's KV cache (34.6K vs 860.2K elements per token). The paper states: "MLA shows better performance than MHA."

## Common Misconceptions

1. **"MLA is just another form of GQA/MQA"** — No. GQA and MQA reduce KV cache by sharing K/V heads across query heads, which trades performance for efficiency. MLA instead compresses K/V into a low-rank latent vector and reconstructs full per-head K/V via up-projection. The paper's ablation (Table 8) shows MHA outperforms GQA and MQA; MLA then outperforms MHA (Table 9), so MLA is strictly better, not a compromise.

2. **"MLA cannot use RoPE"** — This is a common misunderstanding. MLA *can* use RoPE; it just requires the decoupled RoPE trick (Section 2.1.3). The content part of the key goes through the compressed path without RoPE, and a separate small decoupled key carries the RoPE positional information. This is why the KV cache is `(d_c + d_h_R) * l` rather than just `d_c * l`.

3. **"The KV cache reduction is negligible because you still need the up-projection weights"** — The up-projection weights (W_UK, W_UV) are model parameters, not per-token cache. The KV cache is per-token memory that grows with sequence length. By caching only `c_KV` (dim d_c) instead of full K and V (dim 2*n_h*d_h), the per-token memory is drastically reduced, enabling larger batch sizes and longer contexts.

## Real Citations

1. DeepSeek-AI (2024). "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) — the original paper.
2. Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) — GQA, the baseline MLA improves upon.
3. Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2024). "RoFormer: Enhanced Transformer with Rotary Position Embedding." [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) — RoPE, which MLA adapts via decoupling.
4. Dai, D., et al. (2024). "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models." [arXiv:2401.06066](https://arxiv.org/abs/2401.06066) — the MoE architecture used in DeepSeek-V2.

**Citation count:** about 108 citations on [OpenAlex](https://openalex.org/W4396815229) (checked October 2026).
