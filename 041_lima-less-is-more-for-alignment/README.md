---
title: "041 LIMA"
nav_order: 42
permalink: "/041_lima-less-is-more-for-alignment/"
---
# LIMA: Less Is More for Alignment (LIMA)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/lima-less-is-more-for-alignment)

- **Paper:** [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)
- **Authors:** Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, Omer Levy
- **Lab, year:** Meta AI (with University of Southern California, Tel Aviv University, Carnegie Mellon University), 2023 (NeurIPS 2023)

## Summary

By early 2023, the dominant recipe for aligning large language models (making them follow instructions and produce helpful responses) involved two resource-intensive stages: large-scale instruction tuning on millions of examples, followed by reinforcement learning from human feedback (RLHF). LIMA challenged this orthodoxy with a striking experiment: fine-tune a 65B-parameter LLaMA model on just 1,000 carefully curated prompt–response pairs — no RLHF, no preference modeling, no millions of examples.

The 1,000 training examples were sourced primarily from community Q&A forums (Stack Exchange, wikiHow, Reddit's r/WritingPrompts) with strict quality filtering, supplemented by 200 manually authored examples and 50 from Super-Natural Instructions. The total training data was roughly 750,000 tokens. Despite this tiny dataset, LIMA produced responses that humans rated as equal or preferable to GPT-4 in 43% of cases, better than Bard in 58% of cases, and better than DaVinci003 (an RLHF-tuned model) in 65% of cases. Of 50 randomly analyzed responses, 50% were rated excellent and 88% met the prompt requirements.

This notebook demonstrates the LIMA principle at toy scale: it trains a small character-level language model on a curated set of ~100 high-quality prompt–response pairs versus ~1,000 noisy ones, and compares the alignment quality (response coherence and format adherence) of the two resulting models.

## What Problem Does It Solve

Imagine you are teaching a brilliant but socially awkward person to be a helpful assistant. They already know everything — math, history, cooking, coding — but they don't know how to talk to people. They ramble, they give answers in the wrong format, they miss the point of what was asked. Do you need to teach them all the facts again from scratch? Of course not. You just need to show them how a good assistant responds: the style, the format, the tone. A few dozen good examples of "here's how you answer a question helpfully" would be enough.

That is exactly what LIMA argues. The "brilliant person" is the pretrained language model, which has already learned facts and reasoning from trillions of tokens of internet text. The "social skills" are alignment — learning the style and format of helpful responses. LIMA shows that 1,000 curated examples are sufficient to teach this style, challenging the belief that alignment requires millions of examples and expensive RLHF.

## Core Idea

- **Superficial Alignment Hypothesis:** A model's knowledge and capabilities are learned almost entirely during pretraining. Alignment only teaches the model which subdistribution of output formats to use when interacting with users — it is about style, not substance.
- **Quality over quantity:** 1,000 carefully curated, high-quality, stylistically consistent examples outperform 52,000 automatically generated ones (Alpaca). Data diversity (varied prompts) and quality (well-written responses) matter more than sheer quantity.
- **No RLHF needed:** Standard supervised fine-tuning with the next-token prediction loss is sufficient. No reward model, no preference data, no reinforcement learning — just 1,000 examples and the same loss function used in pretraining.

## Key Method Details

- **Base model:** LLaMA 65B (Touvron et al., 2023), fine-tuned with standard supervised loss (next-token prediction).
- **Training data:** 1,000 prompt–response pairs totaling ~750,000 tokens. Sources: 400 from Stack Exchange (STEM + other, quality-filtered), 200 from wikiHow, 150 from r/WritingPrompts, 50 from Super-Natural Instructions, 200 manually authored. Responses are stylistically uniform (helpful AI assistant tone).
- **Training:** 15 epochs, AdamW (β1=0.9, β2=0.95), weight decay 0.1, learning rate 1e-5 linearly decaying to 1e-6, batch size 32, max sequence length 2048 tokens. Residual dropout from 0.0 (bottom layer) to 0.3 (last layer). Checkpoints manually selected between epochs 5–10 using a 50-example dev set (perplexity does not correlate with generation quality).
- **Results:** In a human preference study over 300 test prompts, LIMA responses were equal or preferred to GPT-4 in 43% of cases, Claude in 46%, Bard in 58%, and DaVinci003 in 65%. LIMA outperformed Alpaca 65B (trained on 52,000 examples). Absolute analysis: 50% excellent, 88% met prompt requirements, only 6 of 50 failed.
- **Ablations (7B model):** Prompt diversity (Stack Exchange vs wikiHow) and response quality (filtered vs unfiltered) both improve performance; quantity alone does not — doubling or quadrupling data without increasing diversity yields no improvement.

### Simplifications in This Notebook

- Tiny character-level Transformer (~470K parameters) instead of LLaMA 65B
- ~100 curated high-quality prompt–response pairs vs ~1,000 noisy pairs, instead of 1,000 curated vs 52,000 Alpaca examples
- Synthetic instruction–response data generated programmatically (cooking, coding, advice, trivia, creative writing) instead of real Stack Exchange/wikiHow/Reddit data
- Curated model trained for 50 epochs, noisy model for 6 epochs (equal total gradient steps ~200), instead of the paper's 15 epochs with residual dropout and linear LR decay
- Evaluation via BLEU-style n-gram overlap and format-adherence checks on 30 held-out prompts, instead of human preference studies
- No residual dropout, no manual checkpoint selection (checkpointing by best training loss instead)

### What the Notebook Result Shows

On 30 held-out prompts, both models score 0.0 BLEU-4 (the character-level model at this scale rarely produces exact word-level n-gram matches with reference responses). However, the curated-data model achieves 83–100% format adherence (structured responses with bullet points and numbered lists) versus 70% for the noisy-data model — a 13–30 percentage-point improvement. The curated model produces more structured, well-formatted responses mirroring the "helpful assistant" style, while the noisy model's responses are less consistently formatted.

This mirrors LIMA's finding that quality and consistency of training data matter more than quantity. These numbers are not comparable to the paper's human evaluation on a 65B model — they demonstrate the principle at toy scale with a 470K-parameter character-level model trained with equal total gradient steps (~200).

## Influence

LIMA's "less is more" message directly challenged the RLHF-heavy alignment paradigm and influenced subsequent work on minimal-data alignment, including DPO (Direct Preference Optimization, 2023) which also simplified the alignment pipeline. The paper's emphasis on data quality over quantity shaped the data curation practices of open-source instruction tuning efforts. The Superficial Alignment Hypothesis became a widely cited framing for understanding what fine-tuning actually accomplishes. About 275 citations ([OpenAlex](https://openalex.org/W7133189643), checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/041_lima-less-is-more-for-alignment/solution.ipynb) | From-scratch PyTorch implementation comparing curated vs noisy fine-tuning data on a small character-level Transformer, demonstrating LIMA's quality-over-quantity principle |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
