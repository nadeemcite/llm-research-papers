---
title: "SwiGLU: Talk & Q&A"
parent: "003 SwiGLU"
nav_order: 2
---
# TALK — GLU Variants Improve Transformer (SwiGLU)

## Talks, Slides and Press Coverage

1. **Llama 2 model card — Meta (2023):** The LLaMA 2 paper describes using SwiGLU in the feed-forward layers, following the GLU Variants paper. The architecture section states "we use the SwiGLU activation function" with the 2/3 hidden dimension adjustment. ([link](https://arxiv.org/abs/2307.09288))
2. **PaLM paper — Google (2022):** The PaLM (Pathways Language Model) paper uses SwiGLU in its feed-forward layers, citing Shazeer (2020). The paper notes this as one of several architectural modifications. ([link](https://arxiv.org/abs/2204.02311))
3. **Papers with Code — "GLU Variants Improve Transformer" entry:** The paper's page on Papers with Code, with links to methods and related papers. ([link](https://paperswithcode.com/paper/glu-variants-improve-transformer))

No formal conference talks or press interviews were found for this paper.

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2002.05202).

**Q1: Why do the GLU variants use three weight matrices instead of two?**
A1 (from the paper, Section 2): "All of these layers have three weight matrices, as opposed to two for the original FFN. To keep the number of parameters and the amount of computation constant, we reduce the number of hidden units $d_{ff}$ (the second dimension of W and V and the first dimension of W2) by a factor of 2/3 when comparing these layers to the original two-matrix version."

**Q2: Which variants perform best?**
A2 (from the paper, Section 3.2, Table 1): "The GEGLU and SwiGLU variants produce the best perplexities." At 524,288 steps, GEGLU achieves 1.633 and SwiGLU 1.636, both better than the GELU baseline (1.679) and ReLU baseline (1.677).

**Q3: Why does the paper use no bias terms?**
A3 (from the paper, Section 1): "Following the T5 codebase, we use a version with no bias." The paper frames this as a simplification: "Also in the interest of ML fairness."

**Q4: Why does the paper use no dropout during pre-training?**
A4 (from the paper, Section 3.2): "Our main departure from [Raffel et al., 2019] is that we use no dropout during pre-training. We find this to produce superior results." The authors note that the original T5 results were "notably worse" and attribute this to their use of dropout during pre-training.

**Q5: Does the paper offer an explanation for why gating works?**
A5 (from the paper, Section 4): "We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence." The paper is purely empirical.

**Q6: What is the relationship between GLU and the original Gated Linear Units?**
A6 (from the paper, Section 2): "[Dauphin et al., 2016] introduced Gated Linear Units (GLU), a neural network layer defined as the component-wise product of two linear transformations of the input, one of which is sigmoid-activated." Shazeer's contribution is replacing the sigmoid with other activations (ReLU, GELU, Swish) and applying these to the Transformer FFN.

## Common Misconceptions

1. **"SwiGLU was invented by the LLaMA authors."** — SwiGLU was introduced by Noam Shazeer in this 2020 paper. LLaMA (2023) adopted it as one of several standard architectural choices, citing Shazeer. The 2/3 hidden-dimension reduction is also from this paper, not from LLaMA.

2. **"SwiGLU is a new activation function."** — SwiGLU is not a standalone activation function like ReLU or GELU. It is a *gated FFN structure* that uses the Swish$_1$ (SiLU) activation function as part of a gate mechanism. The activation itself was proposed by Ramachandran et al. (2017) as "Swish." What is new in this paper is the combination: gate path (Swish-activated) multiplied by value path (raw), then projected.

3. **"The paper proves that gating is better."** — The paper is explicitly empirical and offers no theoretical explanation. The results are described as "noisy" on fine-tuning benchmarks (Section 3.3), and the paper does not claim universal superiority. The conclusion is that GLU variants "seem to produce better perplexities" and "better results on many downstream language-understanding tasks."

## Real Citations

1. Noam Shazeer (2020). "GLU Variants Improve Transformer." *arXiv preprint.* [arXiv:2002.05202](https://arxiv.org/abs/2002.05202) — the original paper.
2. Yann N. Dauphin, Angela Fan, Michael Auli, David Grangier (2016). "Language Modeling with Gated Convolutional Networks." [arXiv:1612.08083](https://arxiv.org/abs/1612.08083) — introduced Gated Linear Units (GLU), which this paper extends.
3. Prajit Ramachandran, Barret Zoph, Quoc V. Le (2017). "Searching for Activation Functions." [arXiv:1710.05941](https://arxiv.org/abs/1710.05941) — introduced the Swish activation function used in SwiGLU.
4. Hugo Touvron et al. (2023). "LLaMA: Open and Efficient Foundation Language Models." [arXiv:2302.13971](https://arxiv.org/abs/2302.13971) — adopted SwiGLU, popularizing it in open-weight LLMs.

**Citation count:** about 302 citations on [OpenAlex](https://openalex.org/works/W3006439205) (checked October 2026).
