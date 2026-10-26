---
title: "RWKV: Talk & Q&A"
parent: "019 RWKV"
nav_order: 2
---
# TALK — RWKV: Reinventing RNNs for the Transformer Era (RWKV)

## Talks, Slides and Press Coverage

1. **EMNLP 2023 Findings — "RWKV: Reinventing RNNs for the Transformer Era" (Peng et al., 2023):** The paper was published in the Findings of EMNLP 2023 (DOI: 10.18653/v1/2023.findings-emnlp.936), where it was presented as a peer-reviewed contribution. ([ACL Anthology](https://aclanthology.org/2023.findings-emnlp.936/))
2. **Hugging Face RWKV organization page:** The canonical community hub for RWKV model weights and discussion, hosting released checkpoints (RWKV-4 1.5B, 3B, 7B, 14B) and model cards. ([huggingface.co/RWKV](https://huggingface.co/RWKV))
3. **RWKV project GitHub / community — "RWKV-LM":** The open-source implementation and ongoing development (RWKV-4 through RWKV-6) is maintained at [github.com/BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM), with an active Discord community. The repository documents the architecture, training recipes, and the custom CUDA WKV kernel.

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2305.13048).

**Q1: How can RWKV be both a Transformer and an RNN?**
A1 (from the paper, Abstract): "Our approach leverages a linear attention mechanism and allows us to formulate the model as either a Transformer or an RNN, thus parallelizing computations during training and maintains constant computational and memory complexity during inference." The key is that the WKV operator (Eq. 16) can be computed as a parallel scan over the full sequence (Transformer-like training) or as a step-by-step state update with a fixed-size state (RNN-like inference), using the same trained weights.

**Q2: What is the time-decay vector w and why must it be non-negative?**
A2 (from the paper, Section 3, Eq. 10): "We require w to be non-negative to ensure that e^{w_{t,i}} <= 1 and the per-channel weights decay backwards in time." The interaction weight is w_{t,i} = -(t-i)w, where w is a per-channel learnable vector. Non-negativity guarantees that older tokens receive exponentially smaller weights, giving the recurrence stable, fading memory.

**Q3: What is the bonus u for the current token?**
A3 (from the paper, Section 3.1.2): "To circumvent any potential degradation of W, we introduce a vector U that separately attends to the current token." Because the time-decay formulation would attenuate even the present token's contribution, a separate learnable bonus u is added so the current token is not lost — more detail is in Appendix I.

**Q4: Does RWKV follow the same scaling laws as Transformers?**
A4 (from the paper, Section 4.2): "We train 45 RWKV models for a variety of pairs (dataset, parameters) and find that RWKV does follow the same general form of the scaling law that is well established for transformers." The linear fit to Pareto-optimal compute points holds an r^2 of 0.994, and extrapolation an order of magnitude beyond the fit still gives r^2 = 0.875 (Figure 4).

**Q5: Can RWKV handle long contexts?**
A5 (from the paper, Section 4.2): "We are able to teach the model how to efficiently handle substantially larger batch sizes by finetuning with progressively increasing sequence length." Context is progressively doubled from 1024 up to 8192 tokens via fine-tuning, and "increasing context length leads to lower test loss on the Pile" (Figure 6).

**Q6: Something the paper did NOT do?**
A6 (from the 2023 follow-up RWKV-5/6 work, not this paper: [RWKV-LM GitHub](https://github.com/BlinkDL/RWKV-LM)): Later RWKV versions introduced matrix-valued states and data-dependent (input-dependent) time decay, addressing a known limitation of RWKV-4 that its scalar per-channel decay w is not input-dependent. This paper's RWKV uses a fixed (non-input-dependent) decay vector, which is one reason Mamba's selective SSM later achieved stronger results on tasks requiring content-dependent memory.

## Common Misconceptions

1. **"RWKV is just another linear attention variant."** — While RWKV is mathematically a form of linear attention (inspired by AFT, Section 3, Eq. 9), its key contribution is the dual parallel/recurrent formulation with a channel-wise time-decay that admits a fixed-size recurrent state. This is what gives it O(d) inference memory — a property standard linear attention does not have without additional machinery. The paper emphasizes this duality (Table 1: O(Td) time, O(d) space).

2. **"RWKV cannot do parallel training like a Transformer."** — This is the opposite of the paper's central claim. The abstract states RWKV "combines the efficient parallelizable training of transformers with the efficient inference of RNNs." The WKV operator is computed in parallel over all timesteps during training; only inference uses the recurrent form.

3. **"RWKV replaces attention entirely with recurrence, so it can't attend to specific past tokens."** — The time-decay formulation means each channel has a learnable memory timescale; channels with decay near 1 preserve long-range information while channels with decay near 0 act locally (Appendix I, Figure 10). The notebook's decay visualization (Section 14) confirms the model learns this mix of timescales. However, it is true that RWKV-4's decay is not input-dependent — a limitation later addressed in RWKV-5/6 and in Mamba.

## Real Citations

1. Bo Peng, Eric Alcaide, Quentin Anthony, et al. (2023). "RWKV: Reinventing RNNs for the Transformer Era." *Findings of EMNLP 2023.* [arXiv:2305.13048](https://arxiv.org/abs/2305.13048) / [DOI:10.18653/v1/2023.findings-emnlp.936](https://aclanthology.org/2023.findings-emnlp.936/) — the original paper.
2. Albert Gu and Tri Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) — a follow-up line of work on selective state-space models that, like RWKV, maintains a parallel-training / recurrent-inference duality but introduces input-dependent gating.
3. Zhihan Gao, Youzhi Li, et al. (2024). "A Survey of RWKV." [arXiv:2412.14847](https://arxiv.org/abs/2412.14847) — a survey covering RWKV-4 through RWKV-6 and the evolution of the architecture.

**Citation count:** about 389 citations on [OpenAlex](https://openalex.org/works/W4389524555) (checked October 2026).
