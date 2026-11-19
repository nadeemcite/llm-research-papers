---
title: "083 Mixtral"
nav_order: 84
permalink: "/083_mixtral-of-experts/"
---
# Mixtral of Experts (Mixtral 8x7B)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/mixtral-of-experts)

- **Paper:** [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- **Authors:** Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed
- **Lab, year:** Mistral AI, 2024 (preprint, January 2024)

## Summary

Mixtral 8x7B is a Sparse Mixture of Experts (SMoE) language model. It uses the same transformer architecture as Mistral 7B — GQA attention, SwiGLU feedforward blocks, RMSNorm, sliding window attention — but replaces every feedforward sub-block with a Mixture-of-Experts (MoE) layer containing 8 expert networks. For each token at each layer, a small router network selects the top-2 experts out of 8 and combines their outputs additively, weighted by the softmax (a function that turns raw scores into probabilities that sum to 1) of the top-2 router logits (raw numerical scores). This means each token has access to 47B total parameters but only uses 13B active parameters during inference (Section 2.1).

Mixtral was pretrained on multilingual data with a 32k token context window. It outperforms or matches Llama 2 70B and GPT-3.5 across all evaluated benchmarks while using 5x fewer active parameters. On MMLU (a 57-subject multiple-choice benchmark), Mixtral scores 70.6% vs Llama 2 70B's 69.9% (Table 2). On mathematics (GSM8K, an 8-shot grade-school math benchmark), Mixtral scores 74.4% vs 69.6% (Table 2). On code generation (MBPP pass@1), Mixtral scores 60.7% vs 49.8% (Table 2). The instruction-tuned variant, Mixtral 8x7B — Instruct, surpasses GPT-3.5 Turbo, Claude-2.1, Gemini Pro, and Llama 2 70B chat on the LMSys Chatbot Arena (Section 4).

This repo's notebook implements a simplified top-2 sparse MoE layer from scratch in PyTorch, trains a small MoE transformer alongside a dense baseline on Tiny Shakespeare, and visualizes which tokens are routed to which experts across layers.

## What Problem Does It Solve

Imagine a large company with 8 departments — engineering, legal, finance, marketing, and so on. When a task comes in, you could send it to all 8 departments, but that is slow and expensive. Instead, a smart receptionist looks at the task and routes it to just the 2 most relevant departments. The company has the combined knowledge of all 8 departments (so it can handle any kind of task), but each task only uses 2 departments' resources.

This is exactly what Mixtral does. A standard dense transformer uses all its parameters for every token — like sending every task to all departments. Mixtral's router acts as the receptionist: it looks at each token and picks 2 of 8 expert sub-networks to process it. The model has 47B parameters total (all 8 experts across 32 layers), but each token only activates 13B (2 experts per layer). This decouples model capacity from inference cost: you get a bigger, smarter model without proportionally more computation.

## Core Idea

- **Sparse Mixture of Experts (Section 2.1):** Each transformer layer's feedforward block is replaced by 8 expert SwiGLU networks. A router (a single linear layer `W_g`) produces a logit for each expert. Only the top-K=2 experts with the highest logits are activated per token; the rest are skipped entirely.
- **Top-2 softmax gating (Section 2.1):** The gating weights are computed as `G(x) = softmax(Top2(x · W_g))`, where softmax converts the top-2 raw scores (logits) into probabilities that sum to 1. The softmax is taken only over the 2 selected experts' logits (the others are set to -infinity), so the weights of the 2 active experts sum to 1. The output is `y = sum_i G(x)_i · SwiGLU_i(x)`.
- **Decoupled capacity and compute (Section 2.1):** Increasing the number of experts `n` while keeping `K` fixed increases the model's total parameter count (sparse parameters) without increasing the per-token compute cost (active parameters). This is the key trade-off: more total parameters for richer representation, same active parameters for inference speed.

## Key Method Details

- **Architecture (Table 1):** 32 layers, dim 4096, 32 attention heads, 8 KV heads (GQA), head dim 128, hidden dim 14336, context length 32768, vocab size 32000, 8 experts per layer, top-2 routing.
- **Expert function:** Each expert is a standard SwiGLU feedforward block — the same architecture as the dense FFN in Mistral 7B. The MoE layer replaces all FFN sub-blocks (unlike GShard, which replaces every other block).
- **Routing:** `G(x) := Softmax(TopK(x · W_g))` where TopK keeps the K highest logits and sets the rest to -infinity. K=2 for Mixtral.
- **Parameters:** 46.7B total (sparse), 12.9B active per token. The sparse parameter count (47B) is still smaller than Llama 2 70B's 70B, so memory cost is lower too.
- **Training:** Multilingual data, 32k context. No load-balancing auxiliary loss is mentioned — the paper relies on the architecture rather than auxiliary losses to balance expert utilization.
- **Benchmark results (Table 2):** MMLU 70.6% (vs Llama 2 70B 69.9%), HellaSwag 84.4%, WinoGrande 77.2%, ARC-C 59.7%, HumanEval 40.2%, MBPP 60.7%, GSM8K 74.4%, MATH 28.4%.
- **Instruction tuning (Section 4):** Mixtral 8x7B — Instruct trained with SFT followed by DPO. MT-Bench score 8.30 (vs GPT-3.5 Turbo 8.32, Llama 2 70B chat 6.86). Arena Elo 1121, outperforming Claude-2.1 (1117), GPT-3.5-Turbo (1117), Gemini Pro (1111), Llama-2-70b-chat (1077).
- **Routing analysis (Section 5, Figure 7):** The authors find no obvious domain-specific expert specialization — the distribution of expert assignments is similar across ArXiv, PubMed, and PhilPapers. However, consecutive tokens often share expert assignments (temporal locality), with 27-28% first-choice repetition at layer 15 vs 12.5% random baseline (Table 5).

### Simplifications in This Notebook

- Smaller model: 4 layers, 4 attention heads, 2 KV heads, 128 dim, 8 experts (top-2), about 6.5M total / 1.8M active parameters (vs 46.7B total / 12.9B active)
- Dataset: Tiny Shakespeare (~1 MB, character-level, vocab 65) instead of multilingual pretraining data
- 15 epochs, 80 steps per epoch, batch 32, lr 1e-3 with AdamW (wd=0.1, betas=0.9/0.95) instead of full-scale pretraining
- No sliding window attention; no RoPE (positions learned implicitly at this scale)
- No load-balancing loss (consistent with the paper, which does not use one)

### What the Notebook Result Shows

On Tiny Shakespeare validation set (character-level, 128-token context, 15 epochs), the two models achieve: Dense Transformer val perplexity 10.59 (992,512 params), MoE Transformer val perplexity 10.27 (6,501,632 total params, 1,783,040 active per token). The MoE model achieves 3.0% lower perplexity with 1.80x more active parameters. The uniform random baseline is perplexity 65.00 (= vocab size). The total-to-active parameter ratio is 3.6x (6.5M total / 1.8M active), comparable to Mixtral's 3.6x (47B / 13B).

Expert routing visualization reveals a known MoE phenomenon: without a load-balancing loss, higher layers (1-3) show significant expert collapse, with one expert receiving 77-90% of tokens. Layer 0 is more balanced (max 25.1%, std 8.00). The consecutive-token locality analysis replicates the paper's Table 5: layer 0 shows 10.7% first-choice repetition (close to the 12.5% random baseline), while layers 1-3 show 62-90% repetition (far above random), consistent with the paper's finding of high temporal locality at higher layers.

This toy scale (~6.5M parameters, 1 MB text) is too small to reproduce the paper's benchmark results, which were measured on a 46.7B parameter model trained on large-scale multilingual data. The notebook demonstrates the MoE routing mechanism and routing analysis, not the performance advantage over Llama 2 70B.

## Influence

Mixtral 8x7B was the first open-weight sparse mixture-of-experts model to match or exceed the performance of Llama 2 70B and GPT-3.5, making it a landmark in open-source LLMs. It was the best open-weights model on the LMSys Chatbot Arena as of December 2023. The model weights were released under Apache 2.0, enabling both academic and commercial use. Mixtral influenced subsequent MoE models including DeepSeek-V2, DeepSeek-V3, and Qwen-MoE. The official implementation is in the mistral-src repository ([github.com/mistralai/mistral-src](https://github.com/mistralai/mistral-src)). The paper has about 4,800 citations ([OpenAlex](https://openalex.org/W4390723197), checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/083_mixtral-of-experts/solution.ipynb) | From-scratch PyTorch implementation of top-2 sparse MoE layer with 8 experts, trained on Tiny Shakespeare, with expert routing visualization and frequency analysis |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
