---
title: "Lost in the Middle: Talk & Q&A"
parent: "078 Lost in the Middle"
nav_order: 2
---
# TALK — Lost in the Middle: How Language Models Use Long Contexts

## Talks, Slides and Press Coverage

1. **Stanford HAI — "AI Models Struggle to Use Information in the Middle of Their Context Windows" (12 Jul 2023):** Stanford HAI covered the paper, explaining that "changing the location of relevant information within the language model's input context results in a U-shaped performance curve" and that models are better at using information at the beginning or end of the context. ([link](https://hai.stanford.edu/news/ai-models-struggle-use-information-middle-their-context-windows))
2. **Code and data release:** The authors released code and evaluation data at [github.com/nelson-liu/lost-in-the-middle](https://github.com/nelson-liu/lost-in-the-middle), with a project page at [nelsonliu.me/papers/lost-in-the-middle](https://nelsonliu.me/papers/lost-in-the-middle/).

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2307.03172).

**Q1: Why do language models struggle to use information in the middle of their input context?**
A1 (from the paper, Section 4): The authors perform preliminary investigations into the role of model architecture, query-aware contextualization, and instruction fine-tuning. They find that encoder-decoder models are relatively robust to changes in position within their training-time sequence length, but show a U-shaped curve on longer sequences. Decoder-only models, which can only attend to prior tokens, are more affected. The paper does not identify a single definitive cause.

**Q2: Do extended-context models (e.g., GPT-3.5-Turbo 16K) use their context better than standard models?**
A2 (from the paper, Section 2.3): "When the input context fits in the context window of both a model and its extended-context counterpart, we see that performance between them is nearly identical... These results indicate that extended-context models are not necessarily better than their non-extended counterparts at using their input context."

**Q3: Is more retrieved context always better for RAG applications?**
A3 (from the paper, Section 5): "Using more than 20 retrieved documents only marginally improves reader performance (\~1.5% for GPT-3.5-Turbo and \~1% for Claude-1.3), while significantly increasing the input context length (and thus latency and cost)."

**Q4: Does instruction fine-tuning cause the U-shaped curve?**
A4 (from the paper, Section 4.3): "Surprisingly, we see that both MPT-30B and MPT-30B-Instruct exhibit a U-shaped performance curve... Although the absolute performance of MPT-30B-Instruct is uniformly higher than that of MPT-30B, their overall performance trends are similar." The authors also find (Appendix E) that the U-shaped curve only appears in sufficiently large models — 7B Llama-2 models show only recency bias, while 13B and 70B models exhibit the U-shape.

**Q5: What is the connection to psychology?**
A5 (from the paper, Section 6.3): "The U-shaped curve we observe in this work has a connection in psychology known as the serial-position effect (Ebbinghaus, 1913; Murdock Jr, 1962), that states that in free-association recall of elements from a list, humans tend to best remember the first and last elements of the list."

## Common Misconceptions

1. **"The U-shaped curve is caused by the Transformer architecture's attention mechanism"** — The paper notes that "observing a serial-position-like effect in language models is perhaps surprising, since the self-attention mechanisms underlying Transformer language models is technically equally capable of retrieving any token from their contexts" (Section 6.3). The cause is not fully understood; the paper identifies contributing factors (model architecture, model scale, instruction fine-tuning) but does not claim a single architectural explanation.

2. **"Extended-context models solve the lost-in-the-middle problem"** — The paper explicitly shows the opposite: "extended-context models are not necessarily better at using input context" (Section 2.3). GPT-3.5-Turbo (16K) and GPT-3.5-Turbo (4K) perform nearly identically when the context fits in both windows.

3. **"The effect only applies to specific model families"** — The paper tests open models (MPT-30B-Instruct, LongChat-13B), closed models (GPT-3.5-Turbo, Claude-1.3), and encoder-decoder models (Flan-T5-XXL, Flan-UL2), finding the U-shaped curve across all decoder-only models tested.

## Real Citations

1. Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang (2023). "Lost in the Middle: How Language Models Use Long Contexts." *Transactions of the Association for Computational Linguistics (TACL).* [arXiv:2307.03172](https://arxiv.org/abs/2307.03172) — the original paper.
2. Chen et al. (2024). "Found in the Middle: How Language Models Use Long Contexts Better via Plug-and-Play Positional Encoding." [arXiv:2403.17992](https://arxiv.org/abs/2403.17992) — directly addresses the positional bias found in this paper.
3. Shi et al. (2024). "LooGLE: Can Long-Context Language Models Understand Long Contexts?" [arXiv:2310.04171](https://arxiv.org/abs/2310.04171) — extends the evaluation to longer and more diverse context types.

**Citation count:** about 6,810 citations on [OpenAlex](https://openalex.org/W4391876619) (checked November 2026).