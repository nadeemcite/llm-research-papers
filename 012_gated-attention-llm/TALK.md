---
title: "Gated Attention: Talk & Q&A"
parent: "012 Gated Attention"
nav_order: 2
---
# TALK — Gated Attention for Large Language Models (Gated Attention)

## Talks, Slides and Press Coverage

1. **NeurIPS 2025 Oral Presentation:** The paper was accepted as an Oral at NeurIPS 2025, a distinction given to approximately 1% of submissions. The official implementation and models are released at [github.com/qiuzh20/gated_attention](https://github.com/qiuzh20/gated_attention). ([NeurIPS 2025](https://nips.cc/Conferences/2025))
2. **Hugging Face Papers — "Gated Attention for Large Language Models":** The paper received community attention on Hugging Face Papers, where researchers discuss and upvote new work. ([huggingface.co/papers/2505.06708](https://huggingface.co/papers/2505.06708))
3. **Qwen Team Official Release:** The Qwen Team at Alibaba Group released the paper alongside open-source models, continuing their line of architecture research from the Qwen series. The authors include core Qwen contributors Junyang Lin and Jingren Zhou. ([arXiv:2505.06708](https://arxiv.org/abs/2505.06708))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2505.06708).

**Q1: Why does gating after SDPA work better than gating at other positions?**
A1 (from the paper, Section 4.1): The value and output projections form a low-rank linear mapping (since dk < d_model). Inserting non-linearity between them increases expressiveness. Gating after SDPA (position G1) or after the value layer (G2) both address this, but G1 achieves better benchmark results. Gating after the final output projection (G5) has no effect because it does not address the lack of non-linearity between WV and WO.

**Q2: What is the attention sink and how does gating eliminate it?**
A2 (from the paper, Abstract): "this sparse gating mechanism mitigates 'attention sink' and enhances long-context extrapolation performance." The paper finds that query-dependent sparse gating at the SDPA output reduces first-token attention from 46.7% to 4.8% (Figure 2). The sparsity filters out context irrelevant to the current query, preventing early tokens from accumulating redundant attention.

**Q3: Does the gate add significant computational cost?**
A3 (from the paper, Section 3.1): "Since the parameters and flops introduced by the gating are small, the wall-time latency introduced by gating is less than 2%."

**Q4: Can gating replace other stability techniques like sandwich normalization?**
A4 (from the paper, Table 2 and Section 4.3): Gating and sandwich normalization both reduce massive activations. The gated model at 2x learning rate (8e-3) achieves PPL 7.325, while the baseline at the same rate diverges entirely. The paper attributes stability to reduced massive activations, which makes the model "less susceptible to numerical errors during BF16 training."

## Common Misconceptions

1. **"Gated Attention is just another form of SwiGLU."** — No. SwiGLU gates the feed-forward network (FFN) layer, while Gated Attention gates the attention output. They operate on different parts of the transformer block. The paper explicitly builds on SwiGLU's gating concept but applies it to a new location (Section 2.2, Eq. 5).
2. **"The gate needs to be complex to work."** — The paper's central finding is that a *simple* sigmoid gate is sufficient. Among 30 variants tested, the simplest configuration — head-specific, elementwise, multiplicative, sigmoid — performs best (Table 1, row 5).
3. **"Gated Attention is only for MoE models."** — The paper validates on both 15B MoE and 1.7B dense models. On the dense model, gating improves PPL from 6.180 to 6.130 at 3.5T tokens and enables stable training at 2x learning rate (Table 2).

## Real Citations

1. Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, Junyang Lin (2025). "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free." *NeurIPS 2025 (Oral).* [arXiv:2505.06708](https://arxiv.org/abs/2505.06708) — the original paper.
2. Guangxuan Xiao, Jun Fang, Jiaxin Shi, Shangchen Zhou, Beidi Chen, Xianbiao Qi (2023). "Efficient Streaming Language Models with Attention Sinks." [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) — defines the attention sink phenomenon that Gated Attention eliminates.
3. Noam Shazeer (2020). "GLU Variants Improve Transformer (SwiGLU)." [arXiv:2002.05202](https://arxiv.org/abs/2002.05202) — the FFN gating mechanism that inspired applying gating to attention.

**Citation count:** about 23 citations on [OpenAlex](https://openalex.org/works/W4403324167) (checked October 2026).
