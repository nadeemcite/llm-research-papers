---
title: "DeepSeek Sparse Attn: Talk & Q&A"
parent: "033 DeepSeek Sparse Attn"
nav_order: 2
---
# TALK — DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models (DeepSeek Sparse Attention)

## Talks, Slides and Press Coverage

1. **TechCrunch — "DeepSeek releases 'sparse attention' model that cuts API costs in half" (Russell Brandom, 29 September 2025):** Covers the experimental release of DeepSeek-V3.2-Exp, describing the lightning indexer and fine-grained token selection system. Notes that "the system uses a module called a 'lightning indexer' to prioritize specific excerpts from the context window" and that "the price of a simple API call could be reduced by as much as half in long-context situations." ([link](https://techcrunch.com/2025/09/29/deepseek-releases-sparse-attention-model-that-cuts-api-costs-in-half/))
2. **TechCrunch — "DeepSeek previews new AI model that 'closes the gap' with frontier models" (24 April 2026):** Reports on DeepSeek's preview of subsequent models, noting they "have almost 'closed the gap' with current leading models, both open and closed, on reasoning benchmarks." ([link](https://techcrunch.com/2026/04/24/deepseek-previews-new-ai-model-that-closes-the-gap-with-frontier-models/))
3. **HuggingFace — deepseek-ai/DeepSeek-V3.2-Exp:** The open-weight model card on HuggingFace, providing the model weights and inference code including the DSA implementation. ([link](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp))

No conference talks or video presentations by the authors have been found at the time of writing.

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2512.02556).

**Q1: Why use ReLU activation in the lightning indexer instead of softmax or GELU?**
A1 (from the paper, Section 2.1): "We choose ReLU as the activation function for throughput consideration. Given that the lightning indexer has a small number of heads and can be implemented in FP8, its computational efficiency is remarkable."

**Q2: How does the two-stage training work and why is it needed?**
A2 (from the paper, Section 2.1.1): The dense warm-up stage "use[s] a short warm-up stage to initialize the lightning indexer" by keeping dense attention and freezing all parameters except the indexer, training it with a KL-divergence loss that aligns indexer outputs with the main attention distribution. The sparse training stage then "introduce[s] the fine-grained token selection mechanism and optimize all model parameters to adapt the model to the sparse pattern of DSA."

**Q3: Why is the indexer input detached from the computational graph during sparse training?**
A3 (from the paper, Section 2.1.1): "The training signal of the indexer is from only L_I, while the optimization of the main model is according to only the language modeling loss." This separation prevents the indexer's training signal from interfering with the main model's language modelling objective, and vice versa.

**Q4: How does DSA relate to DeepSeek's earlier Multi-head Latent Attention (MLA)?**
A4 (from the paper, Section 2.1): "For the consideration of continued training from DeepSeek-V3.1-Terminus, we instantiate DSA based on MLA for DeepSeek-V3.2." DSA is implemented in the MQA mode of MLA, "where each latent vector (the key-value entry of MLA) will be shared across all query heads of the query token" for kernel-level efficiency.

**Q5: What is the computational complexity reduction?**
A5 (from the paper, Section 2.3): "DSA reduces the core attention complexity of the main model from O(L²) to O(Lk), where k (≪ L) is the number of selected tokens. Although the lightning indexer still has a complexity of O(L²), it requires much less computation compared with MLA in DeepSeek-V3.1-Terminus."

## Common Misconceptions

1. **"DSA completely eliminates O(L²) complexity."** — Not quite. The lightning indexer itself still has O(L²) complexity, but with much lower constant factors (few heads, small dimensions, FP8, ReLU). The main attention is reduced from O(L²) to O(Lk). The paper states: "Although the lightning indexer still has a complexity of O(L²), it requires much less computation compared with MLA" (Section 2.3).

2. **"DSA is just another post-hoc sparse attention method."** — No. A key distinction is that DSA is trained from scratch with sparsity, not applied post-hoc to a pretrained dense model. The two-stage training (dense warm-up for the indexer, then sparse training for all parameters) ensures the model adapts to the sparse pattern during training. The paper explicitly contrasts this with methods applied "only at inference time" (Section 2.1).

3. **"DeepSeek-V3.2's main contribution is DSA alone."** — The paper presents three breakthroughs: (1) DSA, (2) scalable RL with post-training compute exceeding 10% of pre-training cost, and (3) large-scale agentic task synthesis (1,800+ environments, 85,000 prompts). DSA is the architectural contribution, but the RL scaling and agentic pipeline are equally important to the model's performance.

## Real Citations

1. DeepSeek-AI (2025). "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models." [arXiv:2512.02556](https://arxiv.org/abs/2512.02556) — the original paper.
2. DeepSeek-AI (2024). "DeepSeek-V3 Technical Report." [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) — the predecessor model on whose architecture DSA builds.
3. Yuan, J. et al. (2025). "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention." [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) — NSA, an earlier DeepSeek sparse attention method that DSA builds upon conceptually.
4. DeepSeek-AI (2024). "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) — introduced Multi-head Latent Attention (MLA), which DSA is instantiated under.

**Citation count:** The paper was published in December 2025 and has too few citations for a meaningful count at the time of writing (checked October 2026).
