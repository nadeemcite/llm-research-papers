---
title: "HyDE: Talk & Q&A"
parent: "077 HyDE"
nav_order: 2
---
# TALK — Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)

## Talks, Slides and Press Coverage

1. **Luyu Gao — "HyDE: Hypothetical Document Embeddings" at ACL 2023 (July 2023):** The paper was presented at ACL 2023 in Toronto. The presentation covers the two-step decomposition (LLM generation + unsupervised encoding) and results across MS-MARCO, BEIR, and Mr.Tydi. ([paper page](https://aclanthology.org/2023.acl-long.99/))
2. **LangChain documentation — "HyDE" retriever:** HyDE is integrated as a standard retriever type in LangChain's retrieval toolkit, making it accessible to practitioners without custom implementation. ([LangChain docs](https://python.langchain.com/docs/modules/data_connection/retrievers/how_to/hyde))
3. **LlamaIndex — "HyDE Query Transform":** LlamaIndex includes HyDE as a query transformation technique for RAG pipelines. ([LlamaIndex docs](https://docs.llamaindex.ai/en/stable/optimizing/advanced_retrieval/query_transformations/))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their
section in the [arXiv paper](https://arxiv.org/abs/2212.10496).

**Q1: Why does HyDE work even though the generated documents contain false details?**
A1 (from the paper, Section 3.2): The generated document "captures relevance patterns but is unreal and may contain false details." The unsupervised encoder's "dense bottleneck" serves as "a lossy compressor, where the extra (hallucinated) details are filtered out from the embedding." Only the topical signal survives into the vector used for retrieval.

**Q2: Why is HyDE considered "zero-shot" and "unsupervised" when it uses InstructGPT?**
A2 (from the paper, Abstract and Section 1): "No models were trained or fine-tuned in making this pre-print." Both the generative model and the contrastive encoder "remain intact." The only supervision is in the LLM's instruction-following training, which is separate from the retrieval task. By the definition in Sachan et al. (2022), the setup is roughly "unsupervised" — "the only supervision resides in the LLM, in the processing of learning to follow instructions."

**Q3: Does HyDE compute query-document similarity scores?**
A3 (from the paper, Section 1): "Interestingly, with HyDE factorization, the query-document similarity score is no longer explicitly modeled nor computed. Instead, the retrieval task is cast into two NLU and NLG tasks." Retrieval uses document-document similarity (inner product in the contrastive encoder's space), not query-document similarity.

**Q4: What happens when HyDE uses a smaller generative model?**
A4 (from the paper, Section 5.1, Table 4): "All models bring improvement to the unsupervised Contriever, with larger models bringing larger improvements." FLAN-T5 (11B) achieves nDCG@10 of 48.9 on DL19, Cohere (52B) achieves 53.8, and InstructGPT (175B) achieves 61.3 — all above the Contriever baseline of 44.5.

**Q5: Can HyDE be combined with a fine-tuned encoder?**
A5 (from the paper, Section 5.2): "HyDE with fine-tuned encoder is not the intended usage: HyDE is more powerful and irreplaceable when few relevance labels are present." However, when combined with ContrieverFT, InstructGPT still improves performance (DL19 nDCG@10 67.4 vs. ContrieverFT alone 62.1), "suggesting that there may still exist certain factors not captured by the fine-tuned encoder but only by the generative model."

## Common Misconceptions

1. **"HyDE generates fake documents, so it must hurt retrieval with incorrect information."** — This is the intuitive concern, but the paper's design explicitly addresses it. The unsupervised encoder acts as a "lossy compressor" (Section 3.2) that filters out hallucinated details. The hypothetical document only needs to capture the *relevance pattern* (what topic a relevant document would be about), not factual accuracy. The experimental results confirm HyDE improves, not degrades, retrieval.

2. **"HyDE is a training method."** — No models are trained or fine-tuned in HyDE (Section 1, footnote 1). Both the LLM and the contrastive encoder are used as-is. HyDE is a retrieval *pipeline* that combines two pre-trained models in a new way.

3. **"HyDE replaces dense retrieval."** — HyDE is most useful at the beginning of a search system's lifecycle, when no relevance labels exist. The paper (Section 6) suggests: "As the search log grows, a supervised dense retriever can be gradually rolled out. As the dense retriever grows stronger, more queries will be routed to it, with only less common and emerging ones going to HyDE backend." HyDE and supervised retrieval are complementary, not mutually exclusive.

## Real Citations

1. Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan (2022). "Precise Zero-Shot Dense Retrieval without Relevance Labels." *ACL 2023.* [arXiv:2212.10496](https://arxiv.org/abs/2212.10496) — the original paper.
2. Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Edouard Grave, Armand Joulin (2021). "Unsupervised Dense Information Retrieval with Contrastive Learning." [arXiv:2112.09118](https://arxiv.org/abs/2112.09118) — Contriever, the unsupervised encoder used by HyDE.
3. Long Ouyang et al. (2022). "Training language models to follow instructions with human feedback." [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) — InstructGPT, the instruction-following LLM used by HyDE.

**Citation count:** about 748 citations on [OpenAlex](https://openalex.org/W4385565351) (checked November 2026).