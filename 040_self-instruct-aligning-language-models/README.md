---
title: "040 Self-Instruct"
nav_order: 41
permalink: "/040_self-instruct-aligning-language-models/"
---
# Self-Instruct: Aligning Language Models with Self-Generated Instructions (Self-Instruct)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/self-instruct-aligning-language-models)

- **Paper:** [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)
- **Authors:** Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi
- **Lab, year:** University of Washington, Tehran Polytechnic, Arizona State University, Johns Hopkins University, Allen Institute for AI, 2022 (ACL 2023)

## Summary

Instruction-tuned language models — models fine-tuned to follow natural language instructions — had become powerful by late 2022, but they depended on expensive human-written instruction data. That data was limited in quantity, diversity, and creativity, which capped how general the tuned models could be. Self-Instruct proposed a way around this bottleneck: let the language model generate its own training data.

The Self-Instruct pipeline starts with 175 human-written seed tasks. An off-the-shelf language model (vanilla GPT3 in the paper) is prompted with a few seed tasks to generate new instructions, then generate input-output instances for those instructions, and finally a set of automated filters removes invalid, duplicated, or overly similar items. The surviving data is added back to the task pool, and the process repeats. Running this on GPT3 produced about 52,000 instructions with about 82,000 input-output instances. Fine-tuning GPT3 on this self-generated data yielded a 33% absolute improvement on the Super-NaturalInstructions benchmark, on par with InstructGPT-001 (which used private user data and human annotations).

This notebook implements the Self-Instruct pipeline from scratch on a tiny scale: 20 seed tasks, a small word-level Transformer that generates and is fine-tuned on synthetic instruction data, and a comparison of instruction-following quality before and after self-instruct tuning.

## What Problem Does It Solve

Imagine you want to teach a smart assistant to follow instructions — "summarize this article," "translate this sentence," "classify this email as spam or not." You need thousands of example instructions paired with correct responses. Writing these by hand is slow, expensive, and tends to produce boring, repetitive tasks (mostly standard NLP problems). It is like trying to teach someone cooking by only showing them how to boil eggs — you need variety and creativity, not just volume.

Self-Instruct asks: what if the model itself could dream up new tasks and examples? You give it a handful of seed tasks, and it builds up a much larger and more diverse dataset by generating, filtering, and refining its own instructions — almost no human annotation needed.

## Core Idea

- **Bootstrapping from seeds:** Start with a small set of human-written instruction tasks (175 in the paper, 20 in this notebook). These seed tasks prime the model to produce more.
- **Generate, filter, repeat:** The model generates new instructions, then input-output instances for each. Automated filters remove items that are too similar to existing ones, too long, or malformed. Valid items go back into the pool, and the cycle repeats.
- **Self-tuning:** The model that generated the data is then fine-tuned on it. The same LM is both the data producer and the learner, closing the loop without external human labels.

## Key Method Details

- **Seed set:** 175 human-written tasks spanning 2 categories (classification and generation) with 1 instance each (Section 2.2, Appendix A.1).
- **Instruction generation:** 8 seed tasks are sampled as in-context examples to prompt the model to produce 1 new instruction per query.
- **Instance generation:** For each instruction, the model is prompted to produce input-output pairs (up to 3 for generation tasks, 1 for classification tasks identified by keyword matching).
- **Filtering rules:** (1) Remove instructions too similar to any already generated (ROUGE-L overlap > 0.7); (2) Skip instructions with more than 500 tokens or language-specific keywords; (3) Remove instances with repetitive text or empty fields; (4) Limit to 6 instances per instruction.
- **Results:** GPT3 fine-tuned on 52K self-generated instructions improved by +33.1% absolute on Super-NaturalInstructions, matching InstructGPT-001. On a novel human-written instruction set, Self-Instruct outperformed other public instruction datasets by a large margin, leaving only a 5% gap behind InstructGPT-001.

### Simplifications in This Notebook

- Tiny word-level Transformer (~1.8M parameters) instead of GPT3 (175B parameters)
- 20 seed tasks instead of 175
- 3 self-instruct iterations generating ~40 instructions instead of ~52,000
- Word-level Jaccard similarity (threshold 0.6) instead of ROUGE-L (> 0.7) for filtering
- Template-based instruction mutation as a fallback alongside model generation
- Supervised fine-tuning on the filtered synthetic data for 10 epochs (instead of the paper's full training setup)
- Evaluation on 50 novel classification instructions (sentiment, spam, fact/opinion, helpfulness), measuring exact-match and token-overlap quality

### What the Notebook Result Shows

On 50 held-out novel classification instructions, the base (pre-self-instruct) model scores about 0.14 token-overlap F1 with 0.00 exact-match, while the self-instruct fine-tuned model scores about 0.15 token-overlap F1 with 0.00 exact-match. The improvement is very small but consistent, mirroring the paper's finding that self-generated data helps instruction-following — at toy scale with a 1.8M-parameter model instead of GPT3's 175B. The self-instruct model generates more relevant label words (e.g., "fact" for a fact/opinion task where the base model says "not spam"), but neither model achieves exact-match because the toy Transformer lacks the capacity to reliably produce single-word classification labels from novel inputs.

## Influence

Self-Instruct directly inspired Stanford's Alpaca (70K instructions generated from GPT3.5 via Self-Instruct, March 2023), which became one of the most popular open instruction-tuning recipes. The Self-Instruct dataset on Hugging Face (yizhongw/self_instruct) has been downloaded widely. The bootstrapping idea influenced subsequent self-play and self-rewarding methods, including Self-Rewarding Language Models (2024). About 726 citations ([OpenAlex](https://openalex.org/W4385572634), checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/040_self-instruct-aligning-language-models/solution.ipynb) | From-scratch PyTorch implementation of the Self-Instruct pipeline: seed tasks, instruction generation, filtering, and fine-tuning a tiny Transformer on its own synthetic data |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
