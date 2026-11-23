---
title: "085 LLaVA"
nav_order: 86
permalink: "/085_llava-visual-instruction-tuning/"
---
# Visual Instruction Tuning (LLaVA)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/llava-visual-instruction-tuning)

- **Paper:** [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
- **Authors:** Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee
- **Lab, year:** University of Wisconsin–Madison, Microsoft Research, Columbia University, 2023 (NeurIPS 2023 Oral)

## Summary

Instruction tuning — training a language model on machine-generated instruction-following examples — had already proved powerful for text-only LLMs like ChatGPT and Alpaca. LLaVA (Large Language and Vision Assistant) extends this idea to the multimodal domain: it trains a model to follow visual and language instructions by connecting a vision encoder to a large language model.

The architecture is simple: a frozen CLIP ViT-L/14 vision encoder extracts image features, a trainable linear projector maps those features into the LLM's word embedding space, and a Vicuna-13B language model processes the combined visual-plus-text token sequence auto-regressively. Training follows two stages: Stage 1 (feature alignment) freezes both CLIP and the LLM, training only the projector on 595K image-caption pairs from CC3M so visual tokens become "compatible" with the LLM's embedding space; Stage 2 (fine-tuning) keeps CLIP frozen but updates the projector and LLM on 158K GPT-4-generated multimodal instruction-following samples (conversation, detailed description, and complex reasoning).

LLaVA achieves 85.1% relative score compared with GPT-4 on a synthetic multimodal instruction-following benchmark, and when fine-tuned on Science QA the LLaVA + GPT-4 ensemble reaches 92.53% accuracy — a new state of the art. The paper also introduces LLaVA-Bench, two evaluation benchmarks for multimodal instruction following.

This repo's notebook connects a frozen CLIP ViT-B/32 encoder to a tiny 2-layer transformer LM (~1.8M params) via a linear projector, trains it on synthetic image-caption pairs with the same two-stage recipe, and measures exact-match caption accuracy on held-out images.

## What Problem Does It Solve

Imagine a smart assistant that can only read text. You show it a photo of a red circle and ask "what's in this picture?" — but it has no idea, because it can't see. Now imagine giving it glasses: the image is converted into a description it can understand, and then it answers in natural language. The challenge is that no large dataset of image-based conversation existed — people had lots of image-caption pairs, but not question-answer exchanges about images.

LLaVA's clever trick: use GPT-4 (which is text-only) to generate conversation, detailed descriptions, and reasoning about images by feeding it the image's caption and object bounding boxes as text. This automatically creates 158K instruction-following examples without human annotation. Then train a model that actually looks at the image (via CLIP) to follow those instructions.

## Core Idea

- **Visual instruction tuning (Section 1):** Extend instruction tuning from text-only to the multimodal language-image space, using GPT-4-generated instruction-following data derived from image-text pairs.
- **Linear projector (Section 4.1, Eq. 1):** A single trainable projection matrix W maps CLIP visual features Z_v into language embedding tokens H_v = W · Z_v, producing visual tokens that the LLM processes alongside text tokens.
- **Two-stage training (Section 4.2):** Stage 1 trains only the projector for feature alignment on image-caption pairs; Stage 2 fine-tunes the projector and LLM end-to-end on instruction-following data. CLIP remains frozen throughout.

## Key Method Details

- **Vision encoder:** CLIP ViT-L/14, frozen, providing visual features Z_v = g(X_v). Features before the last Transformer layer (Section 5.2 ablation: last-layer features score 0.96% lower).
- **Language model:** Vicuna-13B (best open-source instruction-following LLM at the time).
- **Projector:** Linear layer W mapping 1024-dim (ViT-L/14 hidden) to the LLM's word embedding space.
- **Stage 1 training:** 595K filtered CC3M image-text pairs, 1 epoch, lr 2e-3, batch size 128. Only W is trainable; CLIP and LLM frozen.
- **Stage 2 training:** 158K GPT-4-generated instruction-following samples (58K conversation, 23K detailed description, 77K complex reasoning), 3 epochs, lr 2e-5, batch size 32. Both W and LLM are trainable; CLIP frozen.
- **Results (LLaVA-Bench COCO, Table 4):** Full data: 85.1% relative score vs GPT-4 (conversation 83.1%, detail 75.3%, reasoning 96.5%). No instruction tuning: 21.5%.
- **Results (LLaVA-Bench In-the-Wild, Table 5):** LLaVA 67.3% vs BLIP-2 38.1% vs OpenFlamingo 19.1%.
- **Results (ScienceQA, Table 7):** LLaVA 90.92%, LLaVA + GPT-4 (judge) 92.53% (new SOTA).

### Simplifications in This Notebook

- Smaller vision encoder: CLIP ViT-B/32 (768-dim) instead of ViT-L/14 (1024-dim)
- Tiny 2-layer LM (hidden 256, ~2.1M parameters) instead of Vicuna-13B
- Dataset: 800 synthetic shape-caption pairs (36 types) instead of CC3M (595K) + 158K GPT-4 instructions
- Three-phase training: 15 epochs text-only pretraining + 10 epochs Stage 1 + 25 epochs Stage 2, with AdamW, instead of 1 + 3 epochs with Adam on 8×A100s
- No GPT-4 data generation; captions are template-based ("a {color} {shape} on a {bg} background")
- Task is caption generation (not multi-turn conversation or ScienceQA)

### What the Notebook Result Shows

On 200 held-out test images the model achieves exact-match caption accuracy of 0.68 (color accuracy 0.68, shape accuracy 1.00, background accuracy 1.00), versus a random baseline of 0.035. Shape and background are always correct; errors come from confusing visually similar colors (e.g. orange vs yellow). Stage 1 alone (projector only, frozen random LM) cannot produce coherent captions — the model only works after Stage 2 fine-tunes the LM alongside the projector, confirming the paper's finding that both stages are necessary. This toy result is not comparable to LLaVA's 85.1% relative score on LLaVA-Bench, which measures free-form instruction following on real COCO images with GPT-4 as judge.

## Influence

LLaVA became one of the most influential open-source multimodal models, spawning LLaVA-1.5 (Liu et al., 2023) which added a multi-layer MLP projector and reached near-GPT-4V performance, and LLaVA-NeXT (2024) for higher-resolution understanding. It influenced the broader open multimodal model ecosystem including Qwen-VL, InternVL, and CogVLM. The visual instruction tuning paradigm — generating multimodal training data from text-only LLMs — became standard practice. The paper was accepted as a NeurIPS 2023 Oral.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/085_llava-visual-instruction-tuning/solution.ipynb) | Frozen CLIP ViT-B/32 connected to a tiny 2-layer transformer LM via a linear projector, trained on synthetic image-caption pairs with two-stage alignment + fine-tuning |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
