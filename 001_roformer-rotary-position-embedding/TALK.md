---
title: "RoPE: Talk & Q&A"
parent: "001 RoPE"
nav_order: 2
---
# TALK — RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)

## Talks, Slides and Press Coverage

1. **Hugging Face — "RoFormer" model documentation:** Hugging Face Transformers integrated RoFormer, noting it "puts rotary position embedding on query and key." ([link](https://huggingface.co/docs/transformers/model_doc/roformer))
2. **Eleuther AI — "Rotary Embeddings: A Relative Revolution" (Ajay Jothiprakash, 2021):** A blog post explaining how RoPE works and why it combines the benefits of absolute and relative position encodings. ([link](https://blog.eleuther.ai/rotary-embeddings/))
3. **Jianlin Su's blog (科学空间 / kexue.fm):** The first author's own write-up of the RoPE idea in Chinese, providing additional intuition behind the rotation-based formulation. ([link](https://kexue.fm/archives/8265))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2104.09864).

**Q1: Why use rotation instead of adding position embeddings?**
A1 (from the paper, Section 3.2): "In contrast to the additive nature of position embedding method adopted in the previous works... our approach is multiplicative. Moreover, RoPE naturally incorporates relative position information through rotation matrix product instead of altering terms in the expanded formulation of additive position encoding."

**Q2: What properties does RoPE have?**
A2 (from the paper, Abstract): "RoPE enables valuable properties, including the flexibility of sequence length, decaying inter-token dependency with increasing relative distances, and the capability of equipping the linear self-attention with relative position encoding."

**Q3: Does RoPE work with linear attention?**
A3 (from the paper, Section 3.3): Yes. The paper shows that RoPE is compatible with linear attention formulations, unlike previous relative position encoding methods which "commonly add the position information to the context representation and thus render them unsuitable for the linear self-attention architecture" (Section 2).

**Q4: How does RoPE handle longer sequences than it was trained on?**
A4 (from the paper, Section 4.2, Table 2): The pre-training accuracy of RoFormer increases with the upper bound of sequence length (65.0% at 512 → 67.4% at 1536). The paper states this "demonstrates the ability of RoFormer in dealing with long texts" and attributes it to "the excellent generalizability of the proposed RoPE."

**Q5: What was the headline experimental result?**
A5 (from the paper, Section 4.4, Table 4): On CAIL2019-SCM, "when increase the maximum input text length to 1024 RoFormer outperforms WoBERT by an absolute improvement of 1.5%" (69.79% vs 68.10% test accuracy).

## Common Misconceptions

1. **"RoPE is a relative position encoding."** — Not exactly. RoPE encodes *absolute* position (each vector is rotated by its own absolute position m), but the *attention score* (dot product) depends only on *relative* position (n − m). The paper is careful about this distinction: "RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation" (Abstract).

2. **"RoPE was invented by Meta/Google for LLaMA."** — No. RoPE was proposed by Jianlin Su and colleagues at Zhuiyi Technology in April 2021. LLaMA (Meta, 2023) adopted it two years later. The RoPE paper predates LLaMA by nearly two years.

3. **"RoPE adds position information to the token embedding."** — No. Unlike absolute position embeddings (BERT, GPT), RoPE does *not* modify the input embeddings at all. It rotates the query and key vectors *inside* the attention computation, after the linear projection. The paper contrasts this explicitly: "our approach is multiplicative" vs "additive" (Section 3.2).

## Real Citations

1. Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." *Neurocomputing, 2024.* [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) — the original paper.
2. Meta AI (2023). "LLaMA: Open and Efficient Foundation Language Models." [arXiv:2302.13971](https://arxiv.org/abs/2302.13971) — adopted RoPE as the position encoding for all LLaMA models, bringing it to wide use.
3. Peng et al. (2023). "YaRN: Efficient Context Window Extension of Large Language Models." [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) — builds on RoPE's rotation properties to extend context windows without retraining.

**Citation count:** about 1,900 citations on [OpenAlex](https://openalex.org/works/W4388979610) (checked October 2026).
