---
title: "091 ROME"
nav_order: 92
permalink: "/091_rome-locating-editing-factual-associations/"
---
# Locating and Editing Factual Associations in GPT (ROME)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/rome-locating-editing-factual-associations)

- **Paper:** [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)
- **Authors:** Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov
- **Lab, year:** MIT CSAIL, Northeastern University, Technion - IIT, 2022 (NeurIPS 2022)

## Summary

Large language models like GPT-2 and GPT-J store vast amounts of factual knowledge in their weights, but it has been unclear *where* or *how* a specific fact is stored. ROME (Rank-One Model Editing) answers this by combining two contributions: a diagnostic tool called **causal tracing** that pinpoints which internal computations matter for a factual prediction, and an editing method that modifies a single rank-one update in a mid-layer feed-forward module to change one fact without damaging others.

Causal tracing works by corrupting the model's representation of a subject (e.g. "Eiffel Tower") with noise, then restoring clean activations at one layer at a time to see which restoration brings the correct prediction back. The experiments reveal that factual recall is mediated by a small window of mid-layer feed-forward modules acting on the **last subject token**. Attention layers move information around, but the FFN (feed-forward network) modules at those middle layers are where the fact is actually looked up.

ROME itself edits the model by computing a rank-one update to the weight matrix of one FFN layer (the one identified by causal tracing). The update is derived from a closed-form least-squares solution using a collection of context sentences for the target subject. After editing, the model both answers the new counterfactual correctly ("Eiffel Tower is in Rome") and generalizes to rephrased prompts, while leaving unrelated facts intact.

This notebook implements causal tracing on GPT-2 (124M) and applies a simplified ROME edit to change "Eiffel Tower is in the city of Paris" to "...Rome," then verifies the edit holds on paraphrased prompts and does not corrupt unrelated facts.

## What Problem Does It Solve

Imagine a friend who memorizes thousands of facts but can never tell you *which part of their brain* holds any particular fact. If they get one wrong, you cannot fix just that one error — you would have to re-educate them broadly. Large language models have the same problem: they know a lot, but their knowledge is smeared across billions of weights, and fixing a single factual error without breaking everything else is very hard. ROME solves this by finding the specific "memory slot" for a fact inside the model and rewriting just that slot, like editing one entry in a filing cabinet rather than rewriting the whole cabinet.

## Core Idea

- **Causal tracing:** A causal intervention that corrupts the subject token's activations with Gaussian noise, then restores clean activations at individual layers/tokens to identify which internal states are decisive for a factual prediction.
- **Mid-layer FFN as factual memory:** The tracing reveals that a narrow band of mid-layer feed-forward modules (not attention) mediates factual recall at the last subject token — this is where "Eiffel Tower \u2192 Paris" is computed.
- **Rank-One Model Editing (ROME):** A closed-form rank-one update to the weight matrix of the identified FFN layer that writes a new key-value association (subject \u2192 new object), achieving efficacy, specificity, and generalization simultaneously.

## Key Method Details

- **Causal trace setup:** Corrupt all hidden states at the subject token positions with Gaussian noise scaled to roughly double the residual variance. Then restore clean activations at one (layer, token) position at a time and measure the indirect effect — the recovery of the correct next-token probability.
- **Key finding:** For GPT-2 XL, the decisive computations are in feed-forward modules at layers 13–17 (out of 48), centered on the last subject token. Attention layers are important for moving information but not for storing the fact.
- **ROME update:** For a single edit at layer \u2113, the method computes a rank-one modification \u0394 = (v \u2212 W k\u2096) k\u2096\u1d40 / (k\u2096\u1d40 \u00b7 k\u2096), where k\u2096 is the averaged key representation of the subject across context sentences and v is the target value representation, then adds \u0394 to the down-projection weight W of the FFN at layer \u2113.
- **Results:** ROME achieves 100% efficacy, 99.7% specificity, and 100% generalization on the CounterFact dataset for GPT-2 XL, outperforming KnowledgeEditor (KE), MEND, and constrained fine-tuning, which each sacrifice at least one of the three metrics.

### Simplifications in This Notebook

- Smaller model: GPT-2 (124M, 12 layers) instead of GPT-2 XL (1.5B, 48 layers) or GPT-J (6B)
- Causal tracing on a single fact ("Eiffel Tower is in the city of \_\_\_") rather than the full CounterFact dataset
- ROME edit applied to one FFN layer (layer 5, identified by tracing) rather than the paper's optimal layer selection across hundreds of facts
- 10 context sentences for the subject covariance estimation instead of the paper's larger collection
- No CounterFact quantitative evaluation suite; qualitative checks on paraphrases and neighbor subjects only

### What the Notebook Result Shows

After applying the scaled ROME edit (edit scale 0.001), the model changes its prediction for "The Eiffel Tower is located in the city of" from "Paris" to "Rome." The causal trace heatmap shows a peak at the final layer on the "of" token, with mid-layer contributions visible on the last subject token ("Tower"), qualitatively consistent with the paper's finding that mid-layer FFN modules mediate factual recall.

Generalization to paraphrased prompts is weak — the 124M model has limited factual recall capacity, so the edit does not reliably transfer to reworded prompts (paraphrases predict "the" instead of "Rome"). Specificity is partial: 3 out of 5 unrelated facts remain unchanged, while the Colosseum and Statue of Liberty predictions shift to "Rome." These results are not directly comparable to the paper's quantitative CounterFact metrics, which use a 1.5B model and thousands of facts.

## Influence

ROME became one of the most influential papers in the model-editing and mechanistic interpretability space. It directly inspired MEMIT (Mass Editing Memory in a Transformer, Meng et al. 2023), which scales ROME from single-fact edits to thousands of simultaneous edits. The causal tracing methodology has been adopted widely to probe where information flows inside transformers. The paper is also frequently cited in discussions of LLM safety (fixing harmful or outdated facts) and interpretability. About 3,972 citations ([OpenAlex](https://openalex.org/works/W4243529981), checked November 2026). It appeared at NeurIPS 2022.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/091_rome-locating-editing-factual-associations/solution.ipynb) | Causal tracing on GPT-2 124M and a simplified ROME rank-one edit that changes "Eiffel Tower is in Paris" to "Rome" with generalization and specificity checks |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
