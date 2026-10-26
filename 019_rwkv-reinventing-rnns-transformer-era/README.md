---
title: "019 RWKV"
nav_order: 20
permalink: "/019_rwkv-reinventing-rnns-transformer-era/"
---
# RWKV: Reinventing RNNs for the Transformer Era (RWKV)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/rwkv-reinventing-rnns-for-the-transformer-era)

- **Paper:** [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)
- **Authors:** Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Xingjian Du, Matteo Grella, Kranthi Kiran GV, Xuzheng He, Haowen Hou, Jiaju Lin, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartlomiej Koptyra, Hayden Lau, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito, Guangyu Song, Xiangru Tang, Bolun Wang, Johan S. Wind, Stanislaw Wozniak, Ruichong Zhang, Zhenyuan Zhang, Qihang Zhao, Peng Zhou, Qinghua Zhou, Jian Zhu, Rui-Jie Zhu
- **Lab, year:** Generative AI Commons and EleutherAI (with 29 additional institutional affiliations), 2023

## Summary

Transformers dominate NLP but their self-attention cost grows quadratically with sequence length — doubling the context quadruples the work. Recurrent neural networks (RNNs) scale linearly but have historically lagged behind Transformers in both performance and trainability, because their sequential nature prevents parallel training and they struggle with long-range dependencies. RWKV (Receptance Weighted Key Value) is a hybrid architecture that captures the parallel training of Transformers and the constant-memory inference of RNNs in a single model.

RWKV reformulates attention as a linear-attention variant inspired by Attention Free Transformer (AFT). Each channel has a learnable, non-negative time-decay vector w, so the weight for a past token decays exponentially with its distance from the present. This makes the WKV operator expressible as a fixed-size recurrence — the model can run as a Transformer during training (all timesteps in parallel) and as an RNN (Recurrent Neural Network) during inference (one token at a time, O(d) memory).

The authors scale RWKV to 14 billion parameters — the largest dense RNN ever trained at the time — and show it matches comparably-sized Transformers (Pythia, OPT, BLOOM) on zero-shot benchmarks including ARC, HellaSwag, LAMBADA, OpenBookQA, ReCoRD, and Winogrande. RWKV also follows the same log-log linear scaling law as Transformers (r^2 = 0.994).

This repo's notebook implements the RWKV time-mixing and channel-mixing blocks in both parallel and recurrent modes, verifies they produce identical outputs (max difference < 1e-6), trains a tiny 92K-parameter RWKV language model alongside a matched Transformer baseline on synthetic Markov-chain data, and compares their training loss and inference speed.

## What Problem Does It Solve

Imagine you're reading a very long book and trying to predict the next word. A Transformer re-reads the entire book from the start every time it makes a prediction — the longer the book, the slower each step. An RNN, like an old-fashioned reader, keeps a small notebook of key facts and only updates that notebook as it goes, so each step is fast no matter how long the book is. The problem is that old RNNs were bad at actually using that notebook — they'd forget important things and couldn't be trained in parallel on modern GPUs, so Transformers won despite being slower at inference.

RWKV solves this by redesigning the RNN's "notebook" update rule so it can be unrolled into a parallel form during training (like a Transformer) and collapsed back into a compact, fixed-size state during inference (like an RNN). The same trained weights serve both modes, so you get Transformer-quality training and RNN-speed generation from one model.

## Core Idea

- **Time-mixing via the WKV operator:** Instead of Q dot-product attention, RWKV uses a per-channel exponential time-decay w so that each past value's contribution fades smoothly with distance, and adds a learnable bonus u for the current token. This is a form of linear attention that admits a fixed-size recurrence (Section 3.1.2, Eq. 16).
- **Dual parallel/recurrent execution:** The WKV computation can be expressed as a parallel scan over the full sequence (Transformer-like training, O(Td) compute) or as a step-by-step state update (RNN-like inference, O(d) memory). The two forms produce identical outputs — the notebook verifies this numerically (max diff < 1e-6).
- **Receptance gating and token shift:** Both the time-mixing and channel-mixing blocks gate their outputs with a sigmoid receptance R (how much past information to accept) and mix each input with the previous timestep via a learned interpolation (token shift), giving the recurrence a controllable memory gate.

## Key Method Details

- **Architecture:** Stacked residual blocks; each block has a time-mixing sub-block (WKV + output gating) and a channel-mixing sub-block (squared-ReLU MLP with receptance gating). Small-init embedding followed by an extra LayerNorm (Section 3.4). Custom initialization where most weights start at zero to mimic an identity mapping (Section 3.4).
- **Inference complexity:** O(Td) time and O(d) space — both independent of context length beyond a single pass; compared to O(T^2 d) time for Transformers (Table 1).
- **Scaling:** Models up to 14B parameters trained on the Pile (330B tokens). Follows the same log-log linear scaling law as Transformers (r^2 = 0.994 on Pareto-optimal compute points; r^2 = 0.875 extrapolating an order of magnitude beyond the fit, Figure 4).
- **Context extension:** Progressive fine-tuning doubling context from 1024 to 2048, 4096, 8192 tokens reduces Pile test loss at each step, showing RWKV can use long context effectively (Figure 6).
- **Results:** On zero-shot benchmarks (ARC Challenge, HellaSwag, LAMBADA, OpenBookQA, ReCoRD, Winogrande) RWKV matches or exceeds similarly-sized Pythia, OPT, and BLOOM models across compute scales (Figure 5).

### Simplifications in This Notebook

- **Smaller model:** 3 layers, 48-dim model, ~92,400 parameters (paper: up to 14B parameters, 4096-dim)
- **Dataset:** 4,000 synthetic Markov-chain sequences over a 16-token vocab, seq_len 47 (paper: the Pile, 330B tokens)
- **Optimizer/lr/epochs:** AdamW lr=1e-3, betas=(0.9, 0.99), 15 epochs, batch size 32 (paper: custom schedule with warmup + cosine/exponential decay, up to 332B tokens)
- **No custom CUDA WKV kernel:** the WKV operator is computed in pure PyTorch via explicit loops (parallel) and state updates (recurrent); the paper uses a fused custom kernel for efficiency (Section 3.2)
- **No small-init embedding:** standard embedding init is used; the paper's small-init + extra LayerNorm trick (Section 3.4) is not applied

### What the Notebook Result Shows

On the synthetic Markov-chain next-token task, the tiny RWKV model reaches val_loss of about 2.93 with val accuracy about 0.060 (15 epochs), closely matching the Transformer baseline (val_loss about 2.93, val_acc about 0.059) at the same depth and width. The key RWKV property — that parallel and recurrent execution produce identical outputs — is verified with a max absolute difference of 5.96e-07 (well below the 1e-4 threshold).

Training the RWKV recurrent forward on this tiny model took about 366s for 15 epochs vs about 15s for the Transformer baseline — the pure-PyTorch recurrent implementation is not optimized and the paper's custom WKV kernel would close this gap substantially. These numbers are not comparable to the paper's 14B results, which used a fused CUDA kernel, the Pile, and 330B training tokens.

## Influence

RWKV sparked a wave of "linear attention / modern RNN" research by demonstrating that an RNN could match Transformers at scale while offering O(d) inference memory. It inspired follow-ups including RWKV-4/5/6 (architectural refinements with matrix-valued states and data-dependent decay, 2023-2024), Mamba (Gu & Dao, 2023, which drew on selective state-space ideas in the same spirit), RetNet (Microsoft, 2023, a parallel/recursive dual-form retention model), and Griffin (Google DeepMind, 2024). The RWKV project remains community-driven and open-source, with the official implementation on GitHub maintaining active development. It influenced the broader trend toward sub-quadratic sequence models that maintain a parallel-training / recurrent-inference duality.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/019_rwkv-reinventing-rnns-transformer-era/solution.ipynb) | From-scratch PyTorch RWKV time-mixing + channel-mixing blocks in both parallel and recurrent modes, verified identical, trained on synthetic Markov data vs a Transformer baseline (val_acc ~0.06) |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
