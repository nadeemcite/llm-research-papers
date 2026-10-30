---
title: "NSA: Talk & Q&A"
parent: "032 NSA"
nav_order: 2
---
# TALK — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (NSA)

## Talks, Slides and Press Coverage

1. **HuggingFace Papers — NSA page:** The paper is listed on [huggingface.co/papers/2502.11089](https://huggingface.co/papers/2502.11089) with community discussion and model references.

2. **AK (@_akhaliq) on X — "NSA: Native Sparse Attention":** The paper was shared widely on social media after its arXiv release. Andrei Karpathy and other ML researchers discussed its approach of making sparse attention natively trainable rather than applied post-hoc. ([arXiv page](https://arxiv.org/abs/2502.11089))

3. **Community implementation — NSA-pytorch:** A community PyTorch implementation of NSA is available at [github.com/dhcode-cpp/NSA-pytorch](https://github.com/dhcode-cpp/NSA-pytorch) (121+ stars), providing a reference for the compression, selection, and sliding window branches with Triton kernel support.

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2502.11089).

**Q1: Why does NSA use three separate branches instead of one sparse pattern?**
A1 (from the paper, Section 3.3.3): The authors explain that "local patterns typically adapt faster and can dominate the learning process, potentially preventing the model from effectively learning from compression and selection tokens." By isolating the sliding window into its own branch with independent K/V projections, the compression and selection branches can "focus on learning their respective features without being shortcutted by local patterns."

**Q2: Why is native trainability important — why not just apply sparsity at inference?**
A2 (from the paper, Section 2.2): The authors identify two problems with post-hoc sparsity: "(1) Performance Degradation: Applying sparsity post-hoc forces models to deviate from their pretrained optimization trajectory. As demonstrated by Chen et al. (2024b), top 20% attention can only cover 70% of the total attention scores, rendering structures like retrieval heads in pretrained models vulnerable to pruning during inference." They also note that existing methods "primarily target inference, leaving the computational challenges in training largely unaddressed."

**Q3: How does NSA achieve actual speedup rather than just theoretical computation reduction?**
A3 (from the paper, Section 3.4): The key is hardware alignment: "blockwise selection is crucial to achieve efficient computation on modern GPUs" because "modern GPU architectures exhibit significantly higher throughput for continuous block accesses compared to random index-based reads." The Triton kernel uses "Group-Centric Data Loading" — loading all heads in a GQA group together since they "share the same sparse KV blocks," eliminating redundant transfers.

**Q4: How does NSA's block selection work with GQA/MQA architectures?**
A4 (from the paper, Section 3.3.2): For GQA, "consistent block selection across these heads has to be ensured to minimize KV cache loading during decoding." The shared importance scores are defined as "p_slc'_t = Σ p_slc,(h)_t" (Eq. 10), summing across heads in a group. This ensures that "the required KV-cache memory access remains relatively high" doesn't happen — all heads in a group load the same blocks.

**Q5: What happens with alternative selection strategies like clustering or auxiliary losses?**
A5 (from the paper, Section 6.1): The authors tried clustering-based strategies (ClusterKV) but found "non-trivial computational overhead introduced by dynamic clustering mechanisms" and "operator optimization difficulties exacerbated by inter-cluster imbalances." For auxiliary-loss-based selection, they found it "increases operator overhead and often degrades model performance" (Figure 7 shows both alternatives have higher loss than NSA).

## Common Misconceptions

1. **"NSA is just another sparse attention method applied at inference"** — No. NSA's key distinction is native trainability: the model is trained from scratch with sparse attention, not adapted post-hoc. The paper explicitly criticizes prior methods for "retaining a pretrained Full Attention backbone, potentially introducing architectural bias" (Section 2.2).

2. **"NSA reduces computation by simply skipping tokens randomly"** — No. NSA uses a hierarchical strategy: compression (MLP-based block summaries), selection (importance-ranked top-n blocks), and sliding window (fixed local context). The importance scores for selection come from the compression branch's attention weights, making the selection query-aware and differentiable.

3. **"The three branches share the same K/V projections"** — No. The paper explicitly states (Section 3.3.3) that "we provide independent keys and values for three branches" to "prevent shortcut learning across attention branches" and "prevent gradient interference between local and long-range pattern recognition."

4. **"NSA only helps at inference"** — No. NSA provides speedups throughout the model lifecycle. The paper reports speedups for "decoding, forward propagation, and backward propagation" (Figure 1), meaning training itself is faster, not just inference.

## Real Citations

1. Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Y. X. Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang, Wangding Zeng (2025). "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention." [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) — the original paper.

2. Joshua Ainslie, James Lee-Thorp, Michaël de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) — NSA is designed for GQA-compatible architectures and addresses the memory access challenges that arise with GQA.

3. Xiao Tang, Yiwei Hu, Yi Lu, Wei Wang, Yinghan Cui, Zijie Zhou, Hua Chen, Pu Zhao, Chao Liu, Xiaoxing Ma (2024). "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference." [arXiv:2406.10774](https://arxiv.org/abs/2406.10774) — a query-aware blockwise selection method that NSA compares against and improves upon with native trainability.

4. Guangxuan Xiao, Yu-Jie Yuan, Jin-Hua Li, Zhi-Lin Zhu, Song Han, Yuzhen Huang, and others (2024). "InfLLM: Unleashing the Inference Potential of Large Language Models for Long Sequences through Unstoppable External Memories." [arXiv:2402.04617](https://arxiv.org/abs/2402.04617) — a blockwise selection baseline that NSA outperforms in LongBench evaluation.

**Citation count:** about 4 citations on [OpenAlex](https://openalex.org/W4407690159) (checked October 2026).
