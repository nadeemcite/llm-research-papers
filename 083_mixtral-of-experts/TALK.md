---
title: "Mixtral: Talk & Q&A"
parent: "083 Mixtral"
nav_order: 2
---
# TALK — Mixtral of Experts (Mixtral 8x7B)

## Talks, Slides and Press Coverage

1. **Mistral AI Blog — "Mixtral of Experts" (8 January 2024):** The official announcement post describing Mixtral 8x7B's architecture, benchmark results, and the release of both base and instruct models under Apache 2.0. ([link](https://mistral.ai/news/mixtral-of-experts/))
2. **Hugging Face Blog — "Welcome Mixtral of Experts" (8 January 2024):** Hugging Face's coverage of the release, including model availability on the Hub and benchmark comparisons. ([link](https://huggingface.co/blog/mixtral))
3. **Hugging Face Papers — "Mixtral of Experts" (January 2024):** The paper's page on Hugging Face Daily Papers, showing community engagement with 547 upvotes. ([link](https://huggingface.co/papers/2401.04088))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2401.04088).

**Q1: How does the sparse mixture-of-experts layer work?**
A1 (from the paper, Section 2.1): "For every token, at each layer, a router network selects two experts to process the current state and combine their outputs." The gating function is `G(x) := Softmax(TopK(x · W_g))`, and the output is `y = sum_i G(x)_i · SwiGLU_i(x)`. Only the top-K=2 experts are computed; the rest are skipped.

**Q2: How many parameters does Mixtral use per token?**
A2 (from the paper, Abstract): "each token has access to 47B parameters, but only uses 13B active parameters during inference." The 47B is the total (sparse) parameter count; the 13B is the active parameter count (top-2 of 8 experts across 32 layers).

**Q3: Does Mixtral use a load-balancing loss?**
A3 (from the paper, Section 2.1): The paper does not mention an auxiliary load-balancing loss. It notes that "EP introduces challenges in load balancing" but relies on the architecture rather than auxiliary losses. The routing analysis (Section 5) shows that expert assignment is approximately uniform across domains.

**Q4: Do experts specialize by domain (e.g., math, code, biology)?**
A4 (from the paper, Section 5): "we do not observe obvious patterns in the assignment of experts based on the topic." The distribution of expert assignments is very similar across ArXiv papers, PubMed Abstracts, and PhilPapers. "This suggests that the router does exhibit some structured syntactic behavior" rather than domain specialization.

**Q5: What is the relationship between Mixtral and Mistral 7B?**
A5 (from the paper, Section 2): "Mixtral has the same architecture as Mistral 7B, with the difference that each layer is composed of 8 feedforward blocks (i.e. experts)." The attention mechanism, normalization, and other architectural details are identical to Mistral 7B ([arXiv:2310.06825](https://arxiv.org/abs/2310.06825)).

## Common Misconceptions

1. **"Mixtral has 8x7B = 56B parameters"** — Not exactly. The name "8x7B" is a simplification. The model has 46.7B total parameters, not 56B. The "8x7B" refers to the architecture (8 experts, each similar in size to a 7B-class model's FFN), but the attention layers and embeddings are shared, not multiplied by 8. The paper states 47B total and 13B active (Abstract).

2. **"Each token uses all 8 experts"** — No. Each token uses only the top-2 of 8 experts at each layer. The gating function applies a softmax over only the top-K logits; the other 6 experts are not computed for that token. This is what makes the model sparse and efficient (Section 2.1).

3. **"Mixtral uses a load-balancing loss like Switch Transformer"** — The paper does not mention any auxiliary load-balancing loss. Switch Transformer ([arXiv:2101.03961](https://arxiv.org/abs/2101.03961)) and GShard use such losses, but Mixtral's paper does not. The routing analysis (Section 5) shows expert utilization is approximately balanced without one.

## Real Citations

1. Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., et al. (2024). "Mixtral of Experts." [arXiv:2401.04088](https://arxiv.org/abs/2401.04088) — the original paper.
2. Jiang, A. Q., Sablayrolles, A., Mensch, A., et al. (2023). "Mistral 7B." [arXiv:2310.06825](https://arxiv.org/abs/2310.06825) — the base architecture that Mixtral extends with MoE layers.
3. Shazeer, N., et al. (2017). "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." *ICLR 2017.* [arXiv:1701.06538](https://arxiv.org/abs/1701.06538) — introduced the top-K softmax gating mechanism used by Mixtral.
4. Lepikhin, D., et al. (2020). "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding." [arXiv:2006.16668](https://arxiv.org/abs/2006.16668) — the GShard architecture that Mixtral's MoE layer is similar to (Section 2.1).
5. Fedus, W., Dean, J., & Zoph, B. (2022). "A Review of Sparse Expert Models in Deep Learning." [arXiv:2209.01667](https://arxiv.org/abs/2209.01667) — the in-depth MoE overview referenced by the paper (reference [12]).

**Citation count:** about 4,800 citations on [OpenAlex](https://openalex.org/W4390723197) (checked November 2026).
