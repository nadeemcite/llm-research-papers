---
title: "Medusa: Talk & Q&A"
parent: "026 Medusa"
nav_order: 2
---
# TALK — Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads (Medusa)

## Talks, Slides and Press Coverage

1. **Together AI Blog — "Medusa: Simple framework for accelerating LLM generation with multiple decoding heads" (Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Tri Dao, 11 Sep 2023):** The official blog post announcing Medusa, explaining the memory-bandwidth bottleneck of autoregressive decoding, the limitations of speculative decoding, and how Medusa's extra heads + tree attention + typical acceptance solve them. Includes speed results on Vicuna-7B/13B/33B showing ~2x speedup. ([link](https://www.together.ai/blog/medusa))

2. **ICML 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads":** The paper was published at the 41st International Conference on Machine Learning (Vienna, Austria, July 2024, PMLR 235). The paper is available at [arXiv:2401.10774](https://arxiv.org/abs/2401.10774).

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2401.10774).

**Q1: Why use extra heads instead of a separate draft model like speculative decoding?**
A1 (from the paper, Section 2.1.1): "Unlike a draft model, MEDUSA heads are trained in conjunction with the original backbone model, which can remain frozen during training (MEDUSA-1) or be trained together (MEDUSA-2). This method allows for fine-tuning large models even on a single GPU... Furthermore, it ensures that the distribution of the MEDUSA heads aligns with that of the original model, thereby mitigating the distribution shift problem."

**Q2: How does the typical acceptance scheme differ from rejection sampling?**
A2 (from the paper, Section 2.3.1): "In real-world scenarios, sampling from language models is often employed to generate diverse responses, and the temperature parameter is used merely to modulate the 'creativity' of the response. Therefore, higher temperatures should result in more opportunities for the original model to accept the draft model's output. We ascertain that it is typically unnecessary to match the distribution of the original model. Thus, we propose employing a typical acceptance scheme to select plausible candidates rather than using rejection sampling."

**Q3: What is the overhead of Medusa heads during inference?**
A3 (from the paper, Table 1): The overhead (ratio of tree attention processing cost to standard forward pass) ranges from 1.18 to 1.27 across Vicuna-7B/13B/33B and Zephyr-7B. Despite this overhead, the net speedup is 2.35-2.83x because the acceleration rate (3.01-3.51x) more than compensates.

**Q4: What inspired the Medusa head architecture?**
A4 (from the paper, Section 2.1.1): "Following the approach of Stern et al. (2018), we utilize a single layer of feed-forward network with a residual connection for each head. We find that this simple design is sufficient to achieve satisfactory performance." The paper "Blockwise Parallel Decoding for Deep Autoregressive Models" (Stern et al., 2018) first proposed the idea of extra heads for parallel token prediction.

**Q5: How does Medusa handle models where the training data is not available?**
A5 (from the paper, Section 2.3.2): "We propose an automated self-distillation pipeline to use the model itself to generate the training dataset for MEDUSA heads, which matches the output distribution of the model... We can use a parameter-efficient adapter like LoRA for fine-tuning the backbone model. In this way, the original model is simply the model with the adapter turned off."

**Q6: How many Medusa heads are needed?**
A6 (from the paper, Section 2.2.3): "Empirically, we found that five heads are sufficient at most. Therefore, we recommend training with five heads and referring to the strategy described in Section 2.3.3 to determine the optimal configuration of the tree attention. With optimized tree attention, sometimes three or four heads may be enough for inference."

## Common Misconceptions

1. **"Medusa is just speculative decoding with a different name"** — While Medusa follows the same framework (generate candidates, verify, accept), it eliminates the need for a separate draft model. The heads are lightweight layers on top of the frozen backbone, not an independent model. The paper explicitly distinguishes them: "We use speculative decoding to refer to those methods that require an independent draft model here. In a broader sense, our method can also be viewed as speculative decoding, while the draft model is entangled with the original model" (Section 1).

2. **"Medusa guarantees identical output to the original model"** — Only Medusa-1 with rejection sampling (not typical acceptance) guarantees lossless acceleration. The typical acceptance scheme trades exact distribution matching for higher speedup at non-zero temperatures. At temperature=0 (greedy), both are identical.

3. **"More heads and bigger trees are always better"** — Table 3 and Figure 4 show diminishing returns. The sparse tree with 64 nodes outperforms dense trees with 256 nodes, and increasing candidate tokens beyond a point slows down due to compute-bound overhead from larger matrix multiplications.

## Real Citations

1. Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, Tri Dao (2024). "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads." *ICML 2024 (PMLR 235).* [arXiv:2401.10774](https://arxiv.org/abs/2401.10774) — the original paper.

2. Shaden Smith, Daniel Futoma, Nicholas D. Sidiropoulos (2018). "Blockwise Parallel Decoding for Deep Autoregressive Models." *NeurIPS 2018.* [arXiv:1811.03115](https://arxiv.org/abs/1811.03115) — the blockwise parallel decoding idea that Medusa heads are based on.

3. Yaniv Leviathan, Matan Kalman, Yossi Matias (2023). "Fast Inference from Transformers via Speculative Decoding." *ICML 2023.* [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) — the speculative decoding method that Medusa improves upon.

4. Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Hongyi Jin, Tianqi Chen, Zhihao Jia (2023). "SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference." [arXiv:2305.09781](https://arxiv.org/abs/2305.09781) — independent work on tree-based attention for speculative inference.
