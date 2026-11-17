---
title: "Self-RAG: Talk & Q&A"
parent: "079 Self-RAG"
nav_order: 2
---
# TALK — Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection

## Talks, Slides and Press Coverage

1. **ICLR 2024 — Oral presentation:** Self-RAG was accepted as an oral paper at ICLR 2024, placing it among the top-rated submissions. The project page with code and models is at [github.com/AkariAsai/self-rag](https://github.com/AkariAsai/self-rag).
2. **Akari Asai — "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (University of Washington NLP Group, Oct 2023):** The authors presented the work as part of the UW NLP seminar series, with slides and recording available at [github.com/AkariAsai/self-rag](https://github.com/AkariAsai/self-rag) and trained models on [Hugging Face](https://huggingface.co/selfrag/selfrag_llama2_7b).
3. **Code and model release:** The authors released code, trained 7B and 13B Self-RAG models, and a critic model at [github.com/AkariAsai/self-rag](https://github.com/AkariAsai/self-rag).

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2310.11511).

**Q1: Why does Self-RAG retrieve on demand instead of always retrieving?**
A1 (from the paper, Section 1): "indiscriminately retrieving and incorporating a fixed number of retrieved passages, regardless of whether retrieval is necessary, or passages are relevant, diminishes LM versatility or can lead to unhelpful response generation." Self-RAG addresses this by having the model itself predict whether retrieval is needed via the Retrieve token.

**Q2: How are reflection tokens generated during training without a critic at inference?**
A2 (from the paper, Section 3.2): A critic model C is trained on supervised data created by prompting GPT-4 with type-specific instructions. "We collect 4k-20k supervised training data for each type and combine them to form training data for C." The critic then annotates the training corpus offline with reflection tokens. The generator M is trained on this augmented corpus with standard next-token prediction, so it learns to produce reflection tokens itself without the critic at test time (Section 3.2.2).

**Q3: How does Self-RAG control its behavior at inference time?**
A3 (from the paper, Section 3.3): "Generating reflection tokens makes the LM controllable during the inference phase, enabling it to tailor its behavior to diverse task requirements." The segment-level beam search uses a weighted sum of critique token probabilities (Eq. 4) with adjustable weights wG. "For instance, to ensure that result y is mostly supported by evidence, we can set a weight term for the IsSup score higher."

**Q4: How does Self-RAG compare to ChatGPT with retrieval?**
A4 (from the paper, Section 5.1): "Self-RAG outperforms retrieval-augmented ChatGPT on four tasks, Llama2-chat and Alpaca on all tasks." On PopQA, Self-RAG 7B scores 54.9% vs Ret-ChatGPT's 50.8%; on PubHealth, 72.4% vs 54.7%; on biography FactScore, 81.2% vs Ret-ChatGPT's reported values. On ASQA citation precision, Self-RAG 13B achieves 70.3%.

**Q5: What role does the critic model's agreement with GPT-4 play?**
A5 (from the paper, Section 3.2.1): "The critic achieves a higher than 90% agreement with GPT-4-based predictions on most reflection token categories (Appendix Table 5)." This validates that the distilled critic can reliably generate the training annotations.

**Q6: Does Self-RAG use the critic model at inference?**
A6 (from the paper, Section 3.2.2): No. "This eliminates the need to host a critic model during training, reducing overhead." The generator M is trained to produce reflection tokens as part of its own output, so at inference only the generator and retriever are needed.

## Common Misconceptions

1. **"Self-RAG is just standard RAG with a better retriever"** — Self-RAG's key innovation is not the retriever (it uses an off-the-shelf Contriever-MS MARCO), but the model's ability to self-reflect via reflection tokens: deciding when to retrieve, judging passage relevance, and critiquing its own output's factual support. Standard RAG retrieves indiscriminately and never self-evaluates (Section 1, Figure 1).

2. **"Self-RAG requires a separate critic model at inference"** — The critic model is used only to annotate training data offline. The generator learns to produce reflection tokens itself through standard next-token prediction, so no critic is needed at test time (Section 3.2.2).

3. **"Reflection tokens are just prompt engineering"** — Reflection tokens are added to the model's vocabulary and trained as part of the next-token prediction objective (Eq. 2). They are learned special tokens, not natural-language prompts. The generator's vocabulary is expanded to include them (Section 3.2.2).

4. **"Self-RAG always retrieves"** — Self-RAG can decide not to retrieve (Retrieve=no), which is important for open-ended tasks like creative writing where external passages are unnecessary. The paper explicitly shows examples where retrieval is skipped (Figure 2, right example with "No Retrieval").

## Real Citations

1. Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi (2023). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." *ICLR 2024.* [arXiv:2310.11511](https://arxiv.org/abs/2310.11511) — the original paper.
2. Yan et al. (2024). "Corrective Retrieval Augmented Generation (CRAG)." [arXiv:2401.17184](https://arxiv.org/abs/2401.17184) — introduces a retrieval evaluator to assess retrieval quality, inspired by Self-RAG's self-reflection approach.
3. Jeong et al. (2024). "Adaptive-RAG: Learning to Adapt What to Retrieve for Large Language Model Reasoning." [arXiv:2403.14403](https://arxiv.org/abs/2403.14403) — extends adaptive retrieval by classifying query complexity to select retrieval strategies.

**Citation count:** about 2,266 citations on [OpenAlex](https://openalex.org/W4387800173) (checked November 2026).
