---
title: "Gated DeltaNet: Talk & Q&A"
parent: "020 Gated DeltaNet"
nav_order: 2
---
# TALK — Gated Delta Networks: Improving Mamba2 with Delta Rule (Gated DeltaNet)

## Talks, Slides and Press Coverage

1. **Songlin Yang — "Gated Delta Networks" (ICLR 2025):** Gated DeltaNet was accepted at ICLR 2025. The paper page is available on the [OpenReview forum](https://openreview.net/forum?id=QyQfUdQxDs). The official code and CUDA kernels are at [github.com/NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet).

2. **Flash Linear Attention — "Gated DeltaNet" (open-source library):** The gated delta rule is integrated into the Flash Linear Attention library by Songlin Yang and Jingyang Zhang, which provides optimized CUDA/Triton kernels for various linear attention variants. ([GitHub](https://github.com/sustcsonglin/flash-linear-attention))

3. **AK — "Gated DeltaNet: Improving Mamba2 with Delta Rule" (YouTube, 2024):** AK from the Paper Explained channel covered the paper, explaining the gated delta rule and its relationship to Mamba2 and DeltaNet. ([video](https://www.youtube.com/watch?v=qLxXJlQyqQ8))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2412.06464).

**Q1: Why are gating and the delta rule complementary?**
A1 (from the paper, Section 1): "We observe that these mechanisms are complementary — gating enables rapid memory erasure while the delta rule facilitates targeted updates. Building on this insight, we introduce the gated delta rule and develop a parallel training algorithm optimized for modern hardware."

**Q2: What does the S-NIAH case study reveal?**
A2 (from the paper, Section 3.2): The case study on Single Needle-In-A-Haystack reveals three observations: "Decay hurts memory retention" (DeltaNet wins on simple retention but Mamba2 degrades beyond 2K), "Gating facilitates filtering" (Mamba2 and Gated DeltaNet maintain better performance on noisy context through gating), and "Delta rule helps memorization" (Gated DeltaNet performs better on complex pattern memorization where Mamba2 drops quickly).

**Q3: How does the gated delta rule relate to online learning?**
A3 (from the paper, Section 3.1): "From the perspective of fast weight programming and test-time training and regression, the hidden state S can be interpreted as a (fast) weight matrix, with the delta rule optimizing the online regression objective L(S_t) = 1/2 \|\|S_t k_t - v_t\|\|^2 via test-time stochastic gradient descent (SGD)." The gated delta rule "can be viewed as incorporating an adaptive weight decay term alpha_t into the SGD update."

**Q4: How does Gated DeltaNet compare to RWKV-7?**
A4 (from the paper, Section 5, Related Work): The authors note "a recent concurrent work RWKV-7 using a similar idea, but with a more relaxable formalism using diagonal-plus-low-rank transitions: S_t = S_{t-1}(diag(d_t) - a_t b_t\^T) + v_t k_t\^T."

**Q5: What is the throughput comparison?**
A5 (from the paper, Section 4, Figure 3): "The proposed gated delta rule introduces only marginal overhead compared to the original delta rule, with Gated DeltaNet achieving essentially the same throughput as DeltaNet. Both are slightly slower than Mamba2 (2-3K tokens/sec) due to their more expressive transition matrices."

## Common Misconceptions

1. **"Gated DeltaNet is just Mamba2 with an extra term"** — While the formula adds the delta rule's Householder term to Mamba2's gated update, the key contribution is showing that these mechanisms are *complementary* and deriving a hardware-efficient algorithm that preserves chunkwise parallelism. The S-NIAH case study (Table 2) demonstrates that each mechanism addresses a different failure mode.

2. **"The delta rule and gating do the same thing"** — They serve different purposes. The delta rule (I - beta_t k_t k_t^T) selectively removes one key's contribution, enabling precise updates. Gating (alpha_t) uniformly decays the entire state, enabling rapid context switching. The paper shows that models with only one mechanism fail in different scenarios (Section 3.2).

3. **"Gated DeltaNet replaces attention entirely"** — The paper shows that hybrid architectures combining Gated DeltaNet with sliding window attention (H1) or Mamba2 + SWA (H2) achieve the best performance (Tables 3-5). Pure recurrent models still lag behind Transformers on some real-world retrieval tasks (Table 4).

## Real Citations

1. Songlin Yang, Jan Kautz, Ali Hatamizadeh (2024). "Gated Delta Networks: Improving Mamba2 with Delta Rule." *ICLR 2025.* [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) — the original paper.

2. Tri Dao, Albert Gu (2024). "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality." *ICML 2024.* [arXiv:2405.21060](https://arxiv.org/abs/2405.21060) — Mamba-2, whose gated update rule (alpha_t * S_{t-1} + v_t k_t^T) is one special case of the gated delta rule (beta_t = 0).

3. Songlin Yang, Jan Kautz, Ali Hatamizadeh (2024). "DeltaNet: Parallelized Online Linear Attention." [arXiv:2406.06484](https://arxiv.org/abs/2406.06484) — DeltaNet, whose delta rule (S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T) is the other special case (alpha_t = 1).

4. Songlin Yang et al. (2024). "Gated Linear Attention Transformers with Hardware-Efficient Training." *ICLR 2024.* [arXiv:2312.06635](https://arxiv.org/abs/2312.06635) — GLA, an earlier gated linear attention model that influenced the chunkwise training algorithm.

**Citation count:** about 50 citations on [Semantic Scholar](https://www.semanticscholar.org/paper/arXiv:2412.06464) (checked October 2026).
