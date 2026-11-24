---
title: "LLaDA: Talk & Q&A"
parent: "087 LLaDA"
nav_order: 2
---
# TALK — Large Language Diffusion Models (LLaDA)

## Talks, Slides and Press Coverage

1. **Hugging Face Daily Papers — "Large Language Diffusion Models" (17 February 2025):** The paper's page on Hugging Face Daily Papers, receiving 127 upvotes from the community. ([link](https://huggingface.co/papers/2502.09992))
2. **GitHub — ml-gsai/llada:** The official repository releasing model checkpoints and inference code for LLaDA 8B Base and Instruct. ([link](https://github.com/ml-gsai/llada))
3. **Project page — "LLaDA Demo":** The official project page with interactive demos and additional generated samples. ([link](https://ml-gsai.github.io/LLaDA-demo/))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2502.09992).

**Q1: Why did the authors build a diffusion model for language instead of using the standard autoregressive approach?**
A1 (from the paper, Section 1): "It is the generative modeling principles (i.e., Eq. (1)), rather than the autoregressive formulation (i.e., Eq. (2)) itself, that fundamentally underpin the essential properties of LLMs." The authors argue that scalability comes from "the interplay between Transformers, model size, data size, and Fisher consistency induced by the generative principles," not from the autoregressive formulation specifically.

**Q2: How does LLaDA's training objective differ from BERT's masked language modeling?**
A2 (from the paper, Section 2.1): "LLaDA employs a masking ratio that varies randomly between 0 and 1 while BERT uses a fixed ratio." The 1/t weighting in the loss (Eq. 3) "has been proven to be an upper bound on the negative log-likelihood of the model distribution, making it a principled objective for generative modeling." BERT was designed for representation learning, not generation.

**Q3: How does LLaDA handle the reversal curse that plagues autoregressive models?**
A3 (from the paper, Section 3.3): "We did not design anything special for reversal tasks. Intuitively, LLaDA treats tokens uniformly without inductive bias, leading to balanced performance." On the poem completion task (Table 4), LLaDA 8B Instruct scores 45.6 on reversal vs GPT-4o's 34.3, while GPT-4o scores 82.7 on forward — showing the ARM's strong directional bias that LLaDA does not have.

**Q4: Does LLaDA support different sampling strategies?**
A4 (from the paper, Section 2.4 and Table 7): "LLaDA enables flexible sampling. In particular, it supports autoregressive and block diffusion sampling directly after the pre-training or SFT processes described above, without requiring any further modifications or training." Pure diffusion sampling achieves the best overall performance (BBH 49.7, GSM8K 70.3, HumanEval 35.4).

**Q5: What architecture does LLaDA 8B use and how does it differ from LLaMA3 8B?**
A5 (from the paper, Table 5 and Section 2.2): Both have 32 layers, d_model=4096, 32 attention heads. Key differences: LLaDA uses vanilla multi-head attention instead of grouped query attention ("LLaDA is incompatible with KV caching"), has 32 KV heads (vs LLaMA3's 8), and a smaller FFN dimension (12,288 vs 14,336) to maintain comparable model size.

## Common Misconceptions

1. **"LLaDA is just BERT with more masking"** — No. BERT uses a fixed 15% masking ratio and was designed for representation learning (classification, NER). LLaDA uses a variable masking ratio t ~ U(0,1) with a 1/t-weighted loss that is a variational lower bound on log-likelihood (Eq. 3-4), making it a principled generative model. LLaDA also uses a full reverse diffusion process for generation, which BERT was never designed for.

2. **"LLaDA is worse than LLaMA3 because it scores lower on some benchmarks"** — LLaDA was trained on 2.3T tokens while LLaMA3 was trained on 15T tokens (Table 1). Despite using ~6.5x less data, LLaDA matches or exceeds LLaMA3 on several benchmarks (MMLU 65.9 vs 65.4, GSM8K 70.3 vs 48.7). The paper also notes LLaDA Instruct uses only SFT while competitors use SFT+RL alignment.

3. **"Diffusion models for text can't work because text is discrete"** — LLaDA demonstrates that discrete diffusion (masked diffusion models) can scale to 8B parameters and match autoregressive baselines. The key insight is that the masking/unmasking process is naturally discrete — no continuous approximation is needed.

## Real Citations

1. Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin, Y., Wen, J.-R., & Li, C. (2025). "Large Language Diffusion Models." *Preprint.* [arXiv:2502.09992](https://arxiv.org/abs/2502.09992) — the original paper.
2. Chang, H., et al. (2022). "MaskGIT: Masked Generative Image Transformer." *CVPR 2022.* [arXiv:2202.04200](https://arxiv.org/abs/2202.04200) — the heuristic masked generation approach that LLaDA improves upon with a principled likelihood bound.
3. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL 2019.* [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) — fixed-ratio masked language modeling that LLaDA generalizes with variable-ratio diffusion.
4. Berg, A. van den, et al. (2024). "IDM: Scalable Diffusion Models for Text." [arXiv:2406.08601](https://arxiv.org/abs/2406.08601) — related work on scaling discrete diffusion for language.

**Citation count:** about 1,119 citations ([Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/arXiv:2502.09992), checked November 2026).
