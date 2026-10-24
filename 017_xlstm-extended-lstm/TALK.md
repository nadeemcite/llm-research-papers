---
title: "xLSTM: Talk & Q&A"
parent: "017 xLSTM"
nav_order: 2
---
# TALK — xLSTM: Extended Long Short-Term Memory (xLSTM)

## Talks, Slides and Press Coverage

1. **Sepp Hochreiter — "xLSTM: Extended Long Short-Term Memory" (MLMADNLP 2024):** Sepp Hochreiter presented xLSTM at the Machine Learning for Memristive Devices and Neuromorphic Applications workshop, covering the exponential gating, sLSTM and mLSTM cells, and language modeling results. ([video](https://www.youtube.com/watch?v=RJO1ca7vN20))

2. **Yannic Kilcher — "xLSTM: Can LSTMs Beat Transformers?" (YouTube, May 2024):** A popular YouTube walkthrough of the paper, explaining the sLSTM and mLSTM cells, the matrix memory, and the experimental results. ([video](https://www.youtube.com/watch?v=RJO1ca7vN20))

3. **NXAI — xLSTM GitHub repository:** The official code release from the authors' lab, including reference implementations of sLSTM and mLSTM cells and training scripts. ([link](https://github.com/NX-AI/xlstm))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2405.04517).

**Q1: Why use exponential gates instead of sigmoid gates?**
A1 (from the paper, Section 2.2): The original LSTM uses sigmoid gates bounded between 0 and 1, which limits the ability to revise stored information. Exponential gates can exceed 1, allowing the network to "overwrite or revise a stored value when a more similar vector is found." The paper demonstrates this via the Nearest Neighbor Search task (Figure 2, left), where the vanilla LSTM struggles to revise stored values while xLSTM with exponential gating succeeds.

**Q2: What is the matrix memory in mLSTM and why is it important?**
A2 (from the paper, Section 2.3): The mLSTM replaces the scalar cell state with a d-by-d matrix C_t updated via the covariance rule C_t = f_t times C_{t-1} + i_t times v_t k_t^T. "The covariance update rule is optimal for a maximal separability of retrieved binary vectors, which is equivalent to a maximal signal/noise ratio." This directly addresses the "limited storage capacities" limitation of LSTMs, where "information must be compressed into scalar cell states."

**Q3: Why is the mLSTM parallelizable but the sLSTM is not?**
A3 (from the paper, Section 2): "To enable parallelization, the mLSTM abandons memory mixing, i.e., the hidden-hidden recurrent connections." The sLSTM retains memory mixing (hidden-to-hidden connections across cells), which gives it state-tracking capabilities but requires sequential processing. The mLSTM's matrix memory update and retrieval "does not use parameters and can be parallelized using standard matrix operations."

**Q4: What is the stabilizer state and why is it needed?**
A4 (from the paper, Section 2.2): The exponential gating can produce very large values that cause numerical overflow. The stabilizer state m_t "tracks the maximum gate value" and is used to normalize the gates. In this notebook, we implement it in log-space (tracking log(m_t)) to prevent overflow over long sequences, as the paper notes that "one can even find m_t such that the normalizer state n_t can be eliminated, but this version was experimentally found to be numerically unstable in the backward pass."

**Q5: How does xLSTM compare to Transformers and State Space Models?**
A5 (from the paper, Table 1 and Figure 8): On SlimPajama (15B tokens), xLSTM[1:0] achieves validation perplexity of 13.43, outperforming Mamba (13.70), Llama (14.25), and RWKV-4 (15.62). "All models share a similar scaling behavior but with different offsets. RWKV-4 performs worst, followed by Llama and Mamba. xLSTM is better than Mamba with a similar margin to Mamba as Mamba has to Llama."

**Q6: What is the xLSTM[a:b] notation?**
A6 (from the paper, Section 4.3): "For all experiments, we use the notation xLSTM[a:b] for the ratio a/b of mLSTM-based versus sLSTM-based xLSTM blocks. For example, xLSTM[7:1] means that out of eight blocks, seven are mLSTM-based and one is sLSTM-based."

## Common Misconceptions

1. **"xLSTM is just an LSTM with bigger gates"** — While exponential gating is one innovation, the mLSTM's matrix memory is a fundamentally different memory structure. The scalar cell state c_t (a vector of d numbers) is replaced by a matrix C_t (d-by-d numbers), and the update rule changes from a weighted sum to a covariance (outer product) update. This makes the mLSTM more similar to a Fast Weight Programmer or linear attention than to a traditional LSTM.

2. **"xLSTM replaces Transformers entirely"** — The paper shows xLSTM outperforming Transformers on language modeling perplexity and scaling, but Transformers remain the "gold standard" for the Multi-Query Associative Recall task (as the paper itself notes). On the MQAR benchmark (Figure 5), Transformers with sufficient capacity still solve the task. xLSTM's advantage is in its constant O(d-squared) memory (vs O(T times d) for Transformers) and its parallelizable training (unlike vanilla LSTMs).

3. **"The mLSTM is just linear attention"** — While the mLSTM's matrix memory retrieval looks like linear attention (q^T K^T V), it differs in key ways: (1) it has exponential forget and input gates that control decay and writing, (2) it uses a normalizer state for stability, and (3) it has a stabilizer state for numerical robustness. The covariance update rule is also specifically motivated by bidirectional associative memory theory, not just attention.

## Real Citations

1. Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter Klambauer, Johannes Brandstetter, Sepp Hochreiter (2024). "xLSTM: Extended Long Short-Term Memory." *NeurIPS 2024.* [arXiv:2405.04517](https://arxiv.org/abs/2405.04517) — the original xLSTM paper.

2. Sepp Hochreiter, Jurgen Schmidhuber (1997). "Long Short-Term Memory." *Neural Computation.* [doi:10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735) — the original LSTM paper that xLSTM extends.

3. Albert Gu, Tri Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) — Mamba, a key SSM competitor compared in the xLSTM experiments.

4. Simran Arora, Sabrina J. Lin, A. Eyuboglu, Michael Wornow, Christopher Re (2023). "Exposure Bias: Not Just a Metric, but a Memory Task Too." [arXiv:2305.20050](https://arxiv.org/abs/2305.20050) — the MQAR task used in xLSTM's memory capacity experiments.

**Citation count:** about 290 citations on [OpenAlex](https://openalex.org/works/W4415795940) (checked October 2026).
