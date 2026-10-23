---
title: "015 BLT"
nav_order: 16
permalink: "/015_byte-latent-transformer/"
---
# Byte Latent Transformer: Patches Scale Better Than Tokens (BLT)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/byte-latent-transformer)

- **Paper:** [Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/abs/2412.09871)
- **Authors:** Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, John Nguyen, Benjamin Muller, Margaret Li, Chunting Zhou, Lili Yu, Jason Weston, Luke Zettlemoyer, Gargi Ghosh, Mike Lewis, Ari Holtzman, Srinivasan Iyer
- **Lab, year:** FAIR at Meta (with University of Washington and University of Chicago), 2024 (preprint)

## Summary

The Byte Latent Transformer (BLT) is a tokenizer-free LLM architecture that works directly on raw bytes instead of pre-tokenized text. Every existing large language model relies on a tokenizer — a heuristic preprocessing step that groups bytes into a fixed vocabulary of tokens (e.g., BPE, used by Llama and GPT). Tokenizers introduce well-known problems: sensitivity to input noise, poor handling of character-level tasks like spelling, multilingual inequity (non-English text gets more tokens per word), and domain fragility. BLT eliminates the tokenizer entirely by grouping bytes into dynamically sized "patches" that serve as the unit of computation.

The key insight is entropy-based patching: a small byte-level language model (about 100M parameters) computes the entropy (information uncertainty) of each next-byte prediction, and patch boundaries are placed where entropy crosses a threshold. This means more compute is allocated to hard-to-predict bytes (e.g., the start of a new word) and less to easy ones (e.g., the ending of a common word). The architecture has three components: a lightweight Local Encoder that converts bytes to patch representations, a large Latent Global Transformer that processes patches, and a lightweight Local Decoder that converts patch representations back to bytes. Cross-attention bridges the byte and patch representations.

BLT is the first byte-level model to match BPE-based Llama 3 performance at scale (up to 8B parameters, 4T training bytes), while using up to 50% fewer inference FLOPs (floating-point operations — a standard measure of compute cost). On downstream benchmarks, BLT-Entropy (8B) averages 61.1 vs Llama 3's 60.0 across 7 tasks. BLT also dramatically outperforms token-based models on character-level tasks (CUTE benchmark: 54.1 vs 27.5) and noise robustness (noised HellaSwag: 64.3 vs 56.9 average), and shows a new scaling axis: simultaneously growing model and patch size at fixed inference cost yields better scaling than token-based models.

This repo's notebook implements entropy-based byte patching, the BLT encoder-decoder-global architecture with cross-attention and hash n-gram embeddings, trains a tiny BLT model from scratch on byte-level text, and compares patching strategies against fixed BPE-like tokenization.

## What Problem Does It Solve

Imagine reading a book where someone has pre-highlighted certain chunks of letters as "units" you must process together — you cannot see individual letters, only these pre-defined chunks. If a word is misspelled or written in uppercase, the highlighting breaks because it was designed for the "normal" version. That is what tokenizers do: they freeze the grouping before the model ever sees the data.

BLT's solution is to let the model decide the grouping dynamically, based on how predictable each byte is. Instead of fixed chunks, it creates patches on the fly — short patches where the text is surprising (hard to predict) and long patches where it is routine (easy to predict). The model always retains access to raw bytes, so it can handle noise, spelling, and character manipulation naturally.

## Core Idea

- **Entropy-based patching (Section 2.3):** A small byte-level LM computes next-byte entropy at each position. Patch boundaries are placed where entropy exceeds a threshold, allocating more compute (shorter patches = more global transformer steps) to unpredictable regions and less to predictable ones.
- **Three-module architecture (Section 3):** A lightweight Local Encoder (few layers) maps bytes to patch representations via cross-attention; a large Latent Global Transformer processes patches; a lightweight Local Decoder maps patch representations back to bytes via cross-attention. This decouples byte-level modeling from expensive global computation.
- **Hash n-gram embeddings (Section 3.2.1):** Byte n-grams (n = 3 to 8) are hashed into embedding tables and added to byte embeddings, giving the encoder local context without positional embeddings. This is critical for matching BPE performance.
- **Patches scale better than tokens (Section 5.3):** At fixed inference FLOPs, BLT can simultaneously increase model size and patch size. Larger patches mean fewer global transformer steps, freeing compute for a bigger model. This gives BLT steeper scaling curves than BPE models.

## Key Method Details

- **Entropy patching (Eq. 1):** H(x_i) = -sum_v p_e(x_i=v \| x_{<i}) log p_e(x_i=v \| x_{<i}). A new patch starts when H(x_t) > theta_g (global threshold) or H(x_t) - H(x_{t-1}) > theta_r (approx. monotonic constraint).
- **Architecture (Section 3):** Local Encoder (l_E layers, h_E dim) with cross-attention pools bytes to patches; Latent Global Transformer (l_G >> l_E layers, h_G dim) with block-causal attention; Local Decoder (l_D layers, h_D dim) with cross-attention expands patches to bytes. Hash n-gram embeddings for n in {3,4,5,6,7,8} with 500K hashes per table.
- **FLOP estimation (Eq. 13-17):** FL_BLT = Global Transf. FL / n_p + Encoder FL + Decoder FL + Cross-Attn FL. The global transformer dominates FLOPs, so reducing patch count (larger patches) saves the most compute.
- **Bits-Per-Byte (Eq. 18):** BPB = L_CE(x) / (ln(2) * n_bytes) — a tokenizer-independent metric replacing perplexity for byte vs token comparisons.
- **Results — scaling (Section 5.1, Fig. 6):** BLT matches or outperforms BPE Llama 3 at compute-optimal scales from 1B to 8B. With patch size 8, BLT uses ~50% fewer inference FLOPs than Llama 3 (avg token size 4.4 bytes).
- **Results — downstream (Table 1):** BLT-Entropy 8B averages 61.1 vs Llama 3's 60.0 across ARC-E, ARC-C, HellaSwag, PIQA, MMLU, MBPP, HumanEval. HumanEval: 35.4 vs 31.1.
- **Results — robustness (Table 3):** BLT outperforms Llama 3 on CUTE (54.1 vs 27.5), noised HellaSwag (64.3 vs 56.9 avg), spelling (99.9 vs 1.1), and spelling inverse (99.9 vs 30.1). Even outperforms Llama 3.1 (trained on 16x more data) on most character tasks.
- **Results — fixed inference scaling (Section 5.3, Fig. 1):** At fixed inference FLOPs, BLT overtakes BPE beyond ~2.5x compute-optimal training budget. The crossover shifts closer to compute-optimal at larger scales.

### Simplifications in This Notebook

- Smaller model: global transformer 4 layers / 128 dim, encoder 1 layer / 64 dim, decoder 1 layer / 64 dim, about 419K parameters (vs 8B)
- Dataset: Tiny Shakespeare at byte level (~1 MB, vocab 256) instead of Llama 2 / BLT-1T datasets (4T bytes)
- 20 epochs, batch 16, lr 1e-3 with AdamW instead of the paper's 4e-4 lr with cosine decay and 2000-step warmup
- Entropy model: tiny 2-layer byte-level transformer (vs 100M parameter, 14-layer model with 512-byte sliding window)
- Hash n-gram embeddings: n = {3,4,5} with 10K hash table entries (vs n = {3,...,8} with 500K entries)
- No Flash Attention, no Flex Attention, no block-causal document masking — simple causal attention only
- Comparison is against fixed-size strided patching (every 4 bytes) as a BPE proxy, not actual Llama 3 BPE tokenizer

### What the Notebook Result Shows

On a Tiny Shakespeare byte-level validation set (256-byte vocab, ~100K held-out bytes, 20 epochs), entropy-patched BLT achieves validation bits-per-byte (BPB) of about 3.91, compared to about 3.99 for strided patching (every 4 bytes). The entropy model produces patches with an average size of about 1.1 bytes — very small, because the entropy threshold of 3.0 fires on most byte positions, making most patches single-byte. The strided baseline uses fixed 4-byte patches.

Entropy patching creates variable-length patches: with the entropy threshold of 3.0, most patches are 1 byte, with a few 2-3 byte patches in predictable regions. The model has about 419K parameters total (global: ~148K, encoder: ~232K, decoder: ~40K). This toy result is not comparable to the paper's 8B-scale experiments — the advantage of entropy patching and the BLT architecture becomes most pronounced at scale where the global transformer dominates compute.

## Influence

BLT is a significant paper from FAIR/Meta (December 2024) that showed that tokenizer-free byte-level models can match BPE-based models at scale, a first at 8B parameters. It directly influenced follow-up work including Fast Byte Latent Transformer (2026), which optimizes BLT's inference speed. The entropy-based patching idea connects to broader trends in adaptive computation (ALLOC8, early-exit models) and the push toward tokenizer-free architectures (ByT5, MambaByte). The paper has about 18 citations on [OpenAlex](https://openalex.org/works/W4412945498) (checked October 2026). Meta released the training and inference code (using FLOPs — floating-point operations, a measure of compute cost) at [github.com/facebookresearch/blt](https://github.com/facebookresearch/blt).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/015_byte-latent-transformer/solution.ipynb) | From-scratch PyTorch implementation of BLT with entropy-based byte patching, cross-attention encoder-decoder, and hash n-gram embeddings, trained on Tiny Shakespeare bytes |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Q&A answered from the paper, citations, and common misconceptions about the paper |
