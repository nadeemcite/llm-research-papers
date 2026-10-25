---
title: "Mamba-2: Talk & Q&A"
parent: "018 Mamba-2"
nav_order: 2
---
# TALK — Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2)

## Talks, Slides and Press Coverage

1. **Tri Dao — "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" (ICML 2024):** Tri Dao presented the SSD framework at ICML 2024, covering the semiseparable matrix connection, the block-decomposed algorithm, and Mamba-2 results. ([paper page](https://icml.cc/virtual/2024/poster/34410))

2. **Albert Gu — "Mamba and SSMs" (Simons Institute, UC Berkeley, 2024):** Albert Gu gave a talk on structured state space models including Mamba-2 at the Simons Institute workshop on foundation models. ([video](https://www.youtube.com/watch?v=vrF3MtGwD0Y))

3. **Hazy Research — "Efficient language models as arithmetic circuits" (blog post, June 2024):** The Hazy Research group at Stanford discussed the connection between SSMs, structured matrices, and arithmetic circuits in the context of Mamba-2's theoretical framework. ([link](https://hazyresearch.stanford.edu/blog/2024-06-22-ac))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2405.21060).

**Q1: What is the core relationship between SSMs and attention?**
A1 (from the paper, Section 5.3): The paper shows that structured state-space models and structured masked attention intersect at a large class of dual models. "A natural special case of each one — more precisely, state space models with scalar-identity structure on the A matrices, and structured masked attention with 1-semiseparable structure on its L mask — are duals of each other with the exact same linear and quadratic forms."

**Q2: Why is the SSD algorithm faster than Mamba-1's scan?**
A2 (from the paper, Section 6): The block-decomposed algorithm uses matrix multiplications for diagonal blocks (exploiting GPU tensor cores) and a smaller recurrence for inter-chunk propagation. "SSD, thanks to its reformulation to use matrix multiplication as a subroutine, can exploit specialized matrix multiplication (matmul) units on GPUs, also known as tensor cores. As a result, it is 2-8× faster than Mamba's fused associative scan."

**Q3: How does Mamba-2 differ from Mamba-1 architecturally?**
A3 (from the paper, Section 7.1): Mamba-2 moves the A, B, C, X projections to happen in parallel at the start of the block (like Q/K/V in attention), instead of sequentially after the X projection. It also adds an extra normalization layer before the output projection. "In Mamba-2, the SSD layer is viewed as a map from A, X, B, C → Y. It therefore makes sense to produce A, X, B, C in parallel with a single projection at the beginning of the block."

**Q4: What does the theory say about attention with efficient autoregression?**
A4 (from the paper, Theorem 5.2, Section 5.2): "For any instantiation of structured masked attention that is an autoregressive process with bounded order, the structured mask L must be a semiseparable matrix. In other words, efficient autoregressive attention is general semiseparable SMA." This means any attention variant that has a fast recurrent form must be an SSM.

**Q5: What is the relationship between linear attention and SSD?**
A5 (from the paper, Section 5.2): Linear attention is the special case of structured masked attention where the mask L is the causal mask (a 1-semiseparable mask with a_t = 1, i.e., no decay). SSD generalizes this by allowing input-dependent decay through the scalar A parameter, making the mask a general 1-semiseparable matrix instead of just the cumsum mask.

## Common Misconceptions

1. **"Mamba-2 is just a faster Mamba-1"** — While Mamba-2 is 2–8× faster, the main contribution is the theoretical SSD framework showing SSMs and attention are duals. The speedup is a consequence of the theory, not the only result. The framework also enables multi-head patterns, tensor parallelism, and hybrid SSD+attention architectures that were not possible with Mamba-1.

2. **"SSD replaces attention entirely"** — The paper shows SSD and attention are complementary. Section 9.2.3 finds that a hybrid with about 10% attention layers outperforms both pure Mamba-2 and pure Transformer++. The authors hypothesize SSM layers handle general sequence mapping while attention layers provide precise in-context retrieval.

3. **"The quadratic form is just for theory — you never use it in practice"** — The quadratic (attention) form is actually used for the diagonal blocks in the chunked SSD algorithm (Section 6.1). For small chunk sizes Q, the quadratic form is more hardware-efficient than the recurrence. The algorithm deliberately combines both forms.

## Real Citations

1. Tri Dao, Albert Gu (2024). "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality." *ICML 2024.* [arXiv:2405.21060](https://arxiv.org/abs/2405.21060) — the original Mamba-2 paper.

2. Albert Gu, Tri Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) — Mamba-1, the predecessor that introduced selective SSMs.

3. Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret (2020). "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention." *ICML 2020.* [arXiv:2006.16236](https://arxiv.org/abs/2006.16236) — the linear attention framework that Mamba-2 generalizes; the title of the Mamba-2 paper is a homage to this work.

**Citation count:** about 300 citations on [Semantic Scholar](https://www.semanticscholar.org/paper/arXiv:2405.21060) (checked October 2026).
