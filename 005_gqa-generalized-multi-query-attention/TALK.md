---
title: "GQA: Talk & Q&A"
parent: "005 GQA"
nav_order: 2
---
# TALK — GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (GQA)

## Talks, Slides and Press Coverage

1. **Hugging Face Blog — "Llama 2 is here, and it's open source" (Philipp Schmid, 18 Jul 2023):** The blog notes that Llama 2 uses "grouped-query attention for fast inference of the 70B model," citing the GQA paper. ([link](https://huggingface.co/blog/llama2))
2. **Llama 2 paper (Touvron et al., 2023) — Section 2.1, Table 1:** The Llama 2 paper explicitly adopts GQA for the 34B and 70B models: "Bigger models — 34B and 70B — use Grouped-Query Attention (GQA) for improved inference scalability" and presents ablation results in Table 18 showing GQA performs comparably to MHA. ([arXiv:2307.09288](https://arxiv.org/abs/2307.09288))
3. **Mistral 7B paper (Jiang et al., 2023):** Mistral AI's Mistral 7B also adopts GQA, further validating its adoption in production models. ([arXiv:2310.06825](https://arxiv.org/abs/2310.06825))
4. **EMNLP 2023 — Accepted paper:** The GQA paper was accepted at EMNLP 2023 (noted in the arXiv comments field). ([arXiv:2305.13245](https://arxiv.org/abs/2305.13245))

## Q&A (answers from the paper)

**Q1: Why does MQA degrade quality, and how does GQA avoid this?**
A1 (from the paper, Section 1): "Multi-query attention (MQA) can lead to quality degradation and training instability." GQA avoids this by using an intermediate number of key-value heads rather than a single one: "Grouped-query attention instead shares single key and value heads for each group of query heads, interpolating between multi-head and multi-query attention" (Section 2.2).

**Q2: How much compute does uptraining require?**
A2 (from the paper, Section 2.1): "The converted checkpoint is then pre-trained for a small proportion α of its original training steps on the same pre-training recipe." For α = 0.05 (5%), "training took approximately 600 TPUv3 chip-days" (Section 3.1). Figure 5 shows that "both MQA and GQA gain from 5% uptraining with diminishing returns from 10%."

**Q3: Why is mean-pooling used for checkpoint conversion instead of selecting one head?**
A3 (from the paper, Section 2.1): "The projection matrices for key and value heads are mean pooled into single projection matrices, which we find works better than selecting a single key and value head or randomly initializing new key and value heads from scratch." Figure 4 confirms: "Mean pooling appears to work best, followed by selecting a single head and then random initialization."

**Q4: How many groups did the authors choose, and why?**
A4 (from the paper, Section 3.3): "We selected 8 groups as a favorable middle ground." Figure 6 shows that "going from 1 (MQA) to 8 groups adds modest inference overhead, with increasing cost to adding more groups."

**Q5: Does GQA apply to encoder layers too?**
A5 (from the paper, Section 2.2): "GQA is not applied to the encoder self-attention layers; encoder representations are computed in parallel, and memory bandwidth is therefore generally not the primary bottleneck."

## Common Misconceptions

1. **"GQA is just a minor tweak to MHA with no real impact"** — The impact is substantial: GQA-8 on T5-XXL reduces inference time from 1.51s to 0.28s per sample (5.4× speedup) while retaining 98.9% of MHA quality (Table 1). This is why Llama 2 70B and Mistral 7B adopted it.
2. **"MQA and GQA are equivalent in quality"** — Not exactly. The paper shows MQA-XXL averages 45.3 vs GQA-8-XXL's 45.5 vs MHA-XXL's 46.0 (Table 1). GQA sits between MQA and MHA in quality, closer to MHA. Also, "MQA can lead to quality degradation and training instability" (Section 1), while uptrained GQA is stable (Appendix A).
3. **"GQA requires training a new model from scratch"** — The paper's key contribution is the uptraining recipe: convert an existing MHA checkpoint by mean-pooling KV heads and continue pre-training for just 5% of original compute (Section 2.1).

## Real Citations

1. Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." *EMNLP 2023.* [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) — the original paper.
2. Noam Shazeer (2019). "Fast Transformer Decoding: One Write-Head is All You Need." [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) — introduced multi-query attention (MQA), the foundation GQA builds on.
3. Hugo Touvron et al. (2023). "Llama 2: Open Foundation and Fine-Tuned Chat Models." [arXiv:2307.09288](https://arxiv.org/abs/2307.09288) — adopted GQA for 34B and 70B models, with ablation in Table 18.
4. Albert Jiang et al. (2023). "Mistral 7B." [arXiv:2310.06825](https://arxiv.org/abs/2310.06825) — another major model adopting GQA.

**Citation count:** about 440 citations on [OpenAlex](https://openalex.org/W4389518760) (checked October 2026).
