---
title: "017 xLSTM"
nav_order: 18
permalink: "/017_xlstm-extended-lstm/"
---
# xLSTM: Extended Long Short-Term Memory (xLSTM)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/xlstm-extended-long-short-term-memory)

- **Paper:** [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517)
- **Authors:** Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter Klambauer, Johannes Brandstetter, Sepp Hochreiter
- **Lab, year:** NXAI Lab & ELLIS Unit LIT AI Lab / JKU Linz, 2024 (NeurIPS 2024)

## Summary

xLSTM (Extended LSTM) revisits the classic LSTM (Long Short-Term Memory, a recurrent neural network that stores information across time steps via gated memory cells) architecture — the recurrent network that dominated sequence modeling before Transformers — and asks: how far can LSTMs go if we fix their known weaknesses and scale them to billions of parameters with modern training techniques? The paper introduces two key modifications: **exponential gating** (replacing sigmoid gates with `exp()`, which lets gates exceed 1 and enables the network to revise stored information) and **new memory structures** (a scalar memory with state mixing for the sLSTM, and a matrix memory with a covariance update rule for the mLSTM).

The **sLSTM** (scalar LSTM) keeps the LSTM's scalar cell state but adds exponential gates with a **normalizer state** (which divides the cell state to keep it bounded) and a **stabilizer state** (which tracks the maximum gate value in log-space to prevent overflow). It also introduces **state mixing** — hidden-to-hidden recurrent connections across multiple memory cells within a head — which gives it state-tracking capabilities that parallel architectures like Transformers lack.

The **mLSTM** (matrix LSTM) replaces the scalar cell with a **d-by-d matrix memory** updated via the **covariance (outer product) rule**: C_t = f_t times C_{t-1} + i_t times v_t k_t^T, where keys and values are stored as outer products and retrieved by query vectors — essentially a gated linear attention mechanism. Because the mLSTM has no hidden-to-hidden connections, it is **fully parallelizable** during training, like a Transformer.

On language modeling (SlimPajama, 300B tokens), xLSTM achieves the best validation perplexity across all model sizes (125M to 1.3B), outperforming Llama, Mamba, and RWKV-4 (Table 1: xLSTM[1:0] = 13.43 vs Mamba = 13.70 vs Llama = 14.25). The scaling laws (Figure 8) show xLSTM with a lower offset than all competitors, suggesting it will continue to perform favorably at larger scales. This repo's notebook implements sLSTM and mLSTM cells from scratch and compares them against a vanilla LSTM and Transformer on a Multi-Query Associative Recall task.

## What Problem Does It Solve

The original LSTM (Hochreiter & Schmidhuber, 1997) solved the vanishing gradient problem with its constant error carousel and gating mechanism, and LSTMs powered the first wave of large language models. But Transformers replaced them for three reasons: (1) LSTMs **cannot revise stored information** — once a value is written to the cell state, sigmoid gates (bounded 0-1) make it hard to overwrite it with a more relevant value; (2) LSTMs have **limited storage capacity** — all information is compressed into a scalar cell state, so rare tokens and many key-value associations get lost; (3) LSTMs **cannot parallelize** because of hidden-to-hidden recurrent connections (memory mixing), making training slow compared to Transformers.

xLSTM addresses all three: exponential gating allows gates > 1, enabling the network to revise stored values (fixing problem 1); the mLSTM's matrix memory stores d-by-d values instead of d scalars, dramatically increasing capacity (fixing problem 2); and the mLSTM removes hidden-to-hidden connections, making it parallelizable (fixing problem 3). The sLSTM keeps memory mixing for state-tracking tasks where sequential processing is actually beneficial.

## Core Idea

- **Exponential gating with normalization (Section 2.2):** Input and forget gates use `exp()` instead of `sigmoid()`, allowing values > 1. A normalizer state n_t accumulates the gate products so the cell state c_t/n_t stays bounded. A stabilizer state m_t (tracked in log-space) prevents overflow by tracking the maximum gate value.
- **mLSTM matrix memory (Section 2.3):** The cell state becomes a d-by-d matrix C_t updated by the covariance rule C_t = f_t times C_{t-1} + i_t times v_t k_t^T. This is equivalent to a Fast Weight Programmer or bidirectional associative memory, and allows storing and retrieving many key-value pairs — directly addressing the limited storage capacity of scalar LSTMs.
- **Parallelizable mLSTM (Appendix A.3):** Without hidden-to-hidden connections, the mLSTM recurrence can be reformulated as a parallel computation using cumulative products and sums, enabling GPU-efficient training like Transformers while using constant O(d-squared) memory regardless of sequence length.
- **xLSTM blocks and architecture (Section 2.4):** sLSTM blocks use post up-projection (Transformer-style: input to sLSTM to up-projection MLP to output). mLSTM blocks use pre up-projection (SSM-style: input to up-projection to mLSTM to down-projection to output). These blocks are residually stacked, with the mLSTM:sLSTM ratio denoted as xLSTM[a:b].

## Key Method Details

- **sLSTM forward pass (Eq. 7-9):** f_t = exp(f-tilde_t), i_t = exp(i-tilde_t), m_t = max(f_t times m_{t-1}, i_t), c_t = (f_t times m_{t-1} / m_t) times c_{t-1} + (i_t / m_t) times z_t, n_t = (f_t times m_{t-1} / m_t) times n_{t-1} + i_t / m_t, h_t = o_t times c_t / n_t
- **mLSTM forward pass (Eq. 19-27):** C_t = f_t times C_{t-1} + i_t times v_t k_t^T (matrix memory), n_t = f_t times n_{t-1} + i_t times k_t (normalizer), h_t = o_t (C_t q_t / max(abs(n_t^T q_t), 1))
- **Parallel mLSTM (Appendix A.3):** The recurrence is reformulated using cumulative log-sum-exp operations: cumulative product of forget gates, cumulative max for stabilizer, and cumulative sum for normalizer — all computable in parallel across timesteps.
- **Results — SlimPajama 15B (Table 1):** xLSTM[1:0] = 13.43 ppl (409M params), xLSTM[7:1] = 13.48 (408M), vs Mamba = 13.70 (423M), Llama = 14.25 (407M), RWKV-4 = 15.62 (430M)
- **Results — SlimPajama 300B (Table 3):** At 1.3B params, xLSTM[1:0] = 8.89 val ppl, xLSTM[7:1] = 9.00, vs Llama = 9.14, Mamba = 9.19, RWKV-4 = 10.21
- **Results — PALOMA (Table 4):** xLSTM yields best average perplexity across 571 text domains at all model sizes
- **Results — Scaling laws (Figure 8):** All models show similar power-law scaling but with different offsets: RWKV-4 worst, then Llama, then Mamba, then xLSTM best

### Simplifications in This Notebook

- Smaller model: 2 blocks, 64 hidden dim, 4 heads, about 69K-244K parameters (vs 8-24 blocks, 256-2048 dim, 125M-1.3B params)
- Dataset: synthetic Multi-Query Associative Recall task with 4 key-value pairs, 2000 training examples (vs SlimPajama, 300B tokens)
- 15 epochs, batch 64, lr 1e-3 with AdamW + cosine schedule (paper uses hundreds of billions of tokens with custom schedules)
- Single-cell sLSTM (no multi-cell memory mixing); parallel mLSTM simplified with cumulative log-space operations
- No positional embeddings, no dropout tuning, no weight initialization optimization

### What the Notebook Result Shows

On a Multi-Query Associative Recall task (4 key-value pairs, vocab 16, seq_len 64, 2000 training examples, 15 epochs), the **xLSTM** (combined mLSTM + sLSTM blocks) achieves validation accuracy of about 0.94, significantly higher than **sLSTM** (about 0.61), **mLSTM** alone (about 0.57), **vanilla LSTM** (about 0.50), and **Transformer** (about 0.30). The random baseline is 1/16 = 0.0625. The xLSTM's advantage comes from combining the matrix memory's storage capacity (mLSTM) with the exponential gating's ability to revise stored values (sLSTM). The Transformer underperforms here due to the small dataset — the paper notes Transformers are the "gold standard" for MQAR at scale but need more data. This toy result is not comparable to the paper's SlimPajama experiments.

## Influence

xLSTM was presented at NeurIPS 2024 and has generated significant interest as a potential alternative to Transformers and SSMs (Mamba) for language modeling. The paper is from Sepp Hochreiter's group — Hochreiter is the original inventor of the LSTM (1997), giving this work particular authority in the RNN revival space. NXAI (the lab) has since released production xLSTM models. Follow-up work includes xLSTM-7B (Beck et al., 2024) and integrations with modern training pipelines. The paper has about 290 citations on [OpenAlex](https://openalex.org/works/W4415795940) (checked October 2026). Code is available at [github.com/NX-AI/xlstm](https://github.com/NX-AI/xlstm).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/017_xlstm-extended-lstm/solution.ipynb) | From-scratch PyTorch implementation of sLSTM (exponential gating, log-space stabilizer) and mLSTM (matrix memory, covariance update, parallel form) cells, compared against vanilla LSTM and Transformer on a Multi-Query Associative Recall task |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Q&A answered from the paper, citations, and common misconceptions about the paper |
