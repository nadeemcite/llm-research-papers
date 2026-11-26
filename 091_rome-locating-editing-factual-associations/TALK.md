---
title: "ROME: Talk & Q&A"
parent: "091 ROME"
nav_order: 2
---
# TALK — Locating and Editing Factual Associations in GPT (ROME)

## Talks, Slides and Press Coverage

1. **Yannic Kilcher — "Locating and Editing Factual Associations in GPT (ROME)" YouTube video:** A detailed walkthrough of the paper's causal tracing methodology and the ROME editing approach, covering the key findings about mid-layer FFN modules. ([video](https://www.youtube.com/watch?v=_NMQyOu2HTo))
2. **The BAU Lab — Interactive demos:** The authors released interactive Colab notebooks for both causal tracing and model editing on the project page. ([causal tracing demo](https://colab.research.google.com/github/kmeng01/rome/blob/main/notebooks/causal_trace.ipynb), [editing demo](https://colab.research.google.com/github/kmeng01/rome/blob/main/notebooks/rome.ipynb))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2202.05262).

**Q1: Why do the authors focus on feed-forward modules rather than attention layers?**
A1 (from the paper, Section 4): The causal tracing experiments reveal that restoring the MLP modules at mid-layers is what recovers the factual prediction, while restoring attention has a much smaller effect. The authors conclude that "factual associations are recalled in mid-layer MLP modules," and attention's role is primarily to move information from subject tokens to the prediction position.

**Q2: What is the difference between "knowing" and "saying" as defined in the paper?**
A2 (from the paper, Section 5.2): The authors distinguish between a model merely regurgitating a specific sentence ("saying") and truly internalizing a fact that generalizes across contexts ("knowing"). They measure this with two hallmarks: **specificity** (editing one fact should not change other facts) and **generalization** (the edited knowledge should transfer to paraphrased prompts). ROME achieves both, while methods like constrained fine-tuning sacrifice one or the other.

**Q3: How does ROME compare to other model-editing methods like KE and MEND?**
A3 (from the paper, Section 6): On the CounterFact dataset, ROME achieves high efficacy, specificity, and generalization simultaneously. Other methods (KnowledgeEditor, MEND, constrained fine-tuning) each sacrifice at least one metric — for example, fine-tuning achieves efficacy but fails on specificity (it changes neighbor facts too), while KE maintains specificity but does not generalize well to paraphrases.

**Q4: What is the CounterFact dataset?**
A4 (from the paper, Section 5.2): CounterFact is a new dataset introduced by the authors containing thousands of counterfactual assertions (e.g., "Eiffel Tower is in Rome") along with neighborhood subjects and paraphrase prompts. It enables quantitative testing of both specificity (via neighborhood subjects) and generalization (via paraphrases) when editing a counterfactual.

## Common Misconceptions

1. **"ROME can edit any type of knowledge in the model"** — ROME is specifically designed for factual associations of the form (subject, relation, object). The paper does not claim it works for procedural knowledge, reasoning chains, or complex multi-hop facts. The edit targets a single feed-forward layer's key-value memory.

2. **"ROME is a fine-tuning method"** — ROME is not gradient-based fine-tuning. It computes a closed-form rank-one weight update using the model's own internal representations. No backpropagation or optimization loop is needed for the edit itself.

3. **"The causal trace shows attention is unimportant"** — The trace shows that attention is not where facts are *stored*, but the paper acknowledges attention is critical for *moving* information (e.g., copying the subject representation to the final token position). Attention and FFN play complementary roles.

## Real Citations

1. Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov (2022). "Locating and Editing Factual Associations in GPT." *NeurIPS 2022.* [arXiv:2202.05262](https://arxiv.org/abs/2202.05262) — the original paper.
2. Kevin Meng, Arnab Sen Sharma, Alex Andonian, David Bau, Yonatan Belinkov (2023). "Mass Editing Memory in a Transformer." [arXiv:2210.07229](https://arxiv.org/abs/2210.07229) — the follow-up (MEMIT) that scales ROME to thousands of simultaneous edits.

**Citation count:** about 3,972 citations on [OpenAlex](https://openalex.org/works/W4243529981) (checked November 2026).
