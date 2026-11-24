---
title: "088 Induction Heads"
nav_order: 89
permalink: "/088_induction-heads-in-context-learning/"
---
# In-context Learning and Induction Heads (Induction Heads)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/in-context-learning-and-induction-heads)

- **Paper:** [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895)
- **Authors:** Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, Chris Olah
- **Lab, year:** Anthropic, 2022 (Transformer Circuits Thread, March 2022; arXiv version September 2022)

## Summary

"Induction heads" are a specific type of attention head in transformer models that implement a simple but powerful algorithm: given a sequence like [A][B] ... [A], they predict [B] as the next token. They do this by looking back through the context for a previous occurrence of the current token (A), finding the token that followed it last time (B), and copying that token forward. In other words, they "complete the pattern" -- if they have seen A followed by B earlier in the context, they make B more likely when A appears again.

The paper presents six complementary lines of evidence arguing that induction heads may be the mechanistic source of most in-context learning (ICL -- the ability to use earlier context to improve predictions on later tokens) in transformer models of any size. For small attention-only models (2 layers, no MLPs), the authors provide strong causal evidence: ablating induction heads at test time dramatically reduces ICL.

For larger models with MLPs, the evidence is correlational -- induction heads form at exactly the same point in training as a sudden "phase change" where ICL ability jumps, visible as a bump in the training loss curve. This phase change occurs in every multi-layer model they studied (34 transformers total), and never in 1-layer models (which cannot form induction heads because the mechanism requires composing attention heads across layers).

This repo's notebook trains a 2-layer attention-only transformer on synthetic repeated sequences, detects induction heads via prefix matching, and shows that ablating them destroys ICL.

## What Problem Does It Solve

Imagine you are reading a sentence and see the word "The cat sat on the." When you see "The" again later, you can guess that "cat" might follow, because you saw that pattern earlier. Transformers do something similar: later tokens in a context are easier to predict than early ones because the model can look back at what came before. This ability -- using context to improve predictions -- is called in-context learning, and it is what lets large language models do few-shot tasks like translation or arithmetic from a prompt alone, without any fine-tuning.

But what mechanism inside the transformer actually implements this? The paper asks: is there a specific, identifiable circuit responsible for most of this ability? The answer they propose is the induction head -- a simple pattern-matching circuit that can be found and tested directly in small models, and whose formation coincides precisely with the emergence of ICL in models of every size.

## Core Idea

- **Induction heads implement pattern completion:** An induction head looks back for previous instances of the current token and predicts the token that followed it. The circuit involves two heads: a "previous token head" in layer 0 that copies information from the preceding token, and the "induction head" in layer 1 that uses this information to find matching positions and copy the following token forward.
- **Phase change during training:** Induction heads form abruptly in a narrow window early in training (roughly 2.5-5 billion tokens for large models), at exactly the same point where ICL ability jumps. This phase change is visible as a bump in the training loss curve -- the only place where the curve is not convex.
- **Six lines of evidence:** The paper builds a circumstantial case through co-occurrence (Arg 1), co-perturbation (Arg 2), direct ablation (Arg 3), examples of generality (Arg 4), mechanistic plausibility (Arg 5), and continuity from small to large models (Arg 6).

## Key Method Details

- **Induction head definition:** An attention head that, on repeated random sequences, exhibits prefix matching (attending to tokens preceded by the current token) and copying (increasing the logit of the attended-to token).
- **In-context learning score:** Loss of the 500th token minus average loss of the 50th token, averaged over examples. Positive score means later tokens are predicted better.
- **Phase change:** Occurs at ~2.5-5 billion tokens for large models; before it, ICL score is under 0.15 nats; after, roughly 0.4 nats, constant for the rest of training and across model sizes.
- **Models studied:** 34 transformers of varying sizes, including attention-only models (1-2 layers) and full-scale models with MLPs. Over 50,000 attention head ablations performed.
- **Ablation result:** Knocking out induction heads in small attention-only models greatly decreases ICL (Arg 3, strong causal evidence).

### Simplifications in This Notebook

- Synthetic repeated-sequence dataset (random first half, identical second half) instead of natural language
- 2 attention-only layers, 67,584 parameters (about 68K) instead of billions
- 100 epochs with Adam (lr 1e-3) instead of Anthropic's training schedule over billions of tokens
- Prefix-matching score + ablation only; no eigenvalue analysis, PCA of per-token losses, or study of the phase change

### What the Notebook Result Shows

On 512 held-out repeated random sequences (16-token vocabulary, 128 tokens each), the trained 2-layer model develops induction heads in layer 1 with prefix matching scores of 0.95 (L1H0) and 0.18 (L1H1), versus a random baseline of 0.0078. The in-context learning score is 2.84 (early loss 2.84, late loss 0.003 -- the model nearly perfectly predicts the repeated second half).

Ablating the two detected induction heads drops the ICL score to -0.50 (late loss rises from 0.003 to 3.64), while ablating control (non-induction) heads preserves ICL at 3.04. This replicates the paper's Argument 3 (direct ablation) on a small scale. The result is not comparable to the paper's headline numbers because the synthetic task is far simpler than natural language, and the model is tiny.

## Influence

The paper became a foundational result in mechanistic interpretability, establishing that specific, identifiable circuits can be responsible for broad model capabilities. It influenced subsequent work on circuit analysis, sparse autoencoders for feature decomposition (Towards Monosemanticity, 2023), and scaling monosemanticity to production models (Anthropic, 2024). The concept of induction heads is now standard vocabulary in interpretability research. The paper has about 1,900 citations ([OpenAlex](https://openalex.org/works/W4297412003), checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview -- a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/088_induction-heads-in-context-learning/solution.ipynb) | From-scratch PyTorch 2-layer attention-only transformer trained on synthetic repeated sequences, with induction head detection via prefix matching and ablation study |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook -- what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
