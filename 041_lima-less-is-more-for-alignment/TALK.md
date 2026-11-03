---
title: "LIMA: Talk & Q&A"
parent: "041 LIMA"
nav_order: 2
---
# TALK — LIMA: Less Is More for Alignment (LIMA)

## Talks, Slides and Press Coverage

1. **Meta AI — "LIMA: Less Is More for Alignment" (Meta AI, 18 May 2023):** Meta AI released the paper as a preprint, announcing that fine-tuning a 65B LLaMA model on just 1,000 curated examples could compete with GPT-4 and Bard in human preference studies. The release sparked widespread discussion about the necessity of RLHF for alignment. ([arXiv](https://arxiv.org/abs/2305.11206))

2. **Hugging Face Spaces — LIMA demo:** A community demo of LIMA was hosted on Hugging Face Spaces, allowing users to interact with the model and test its instruction-following capabilities with only 1,000 training examples. ([Hugging Face](https://huggingface.co/spaces))

3. **NeurIPS 2023 conference paper — LIMA was accepted to NeurIPS 2023:** The paper was initially released as an arXiv preprint on 18 May 2023 and accepted to the NeurIPS 2023 conference. ([NeurIPS 2023](https://neurips.cc/Conferences/2023))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their
section in the [arXiv paper](https://arxiv.org/abs/2305.11206).

**Q1: What is the Superficial Alignment Hypothesis?**
A1 (from the paper, Section 2): "A model's knowledge and capabilities are learnt almost entirely during pretraining, while alignment teaches it which subdistribution of formats should be used when interacting with users." The corollary is that a small set of high-quality examples should be sufficient for alignment.

**Q2: How does LIMA compare to models trained with RLHF?**
A2 (from the paper, Abstract): "responses from LIMA are either equivalent or strictly preferred to GPT-4 in 43% of cases; this statistic is as high as 58% when compared to Bard and 65% versus DaVinci003, which was trained with human feedback." LIMA outperformed Alpaca 65B, which was trained on 52,000 examples — 52 times more data.

**Q3: Why does the paper manually select checkpoints instead of using the final epoch?**
A3 (from the paper, Section 3 and Appendix B): "we find that perplexity does not correlate with generation quality, and thus manually select checkpoints between the 5th and the 10th epochs." Appendix B shows that perplexity on held-out data actually anticorrelates with generation quality — as perplexity rises (overfitting), response quality improves, making automatic checkpoint selection by loss unreliable.

**Q4: Does LIMA handle multi-turn dialogue despite having no dialogue training data?**
A4 (from the paper, Section 6): Yes, surprisingly. "LIMA responses are surprisingly coherent for a zero-shot chatbot, referencing information from previous steps in the dialogue." However, in 6 out of 10 conversations, LIMA fails within 3 interactions. Adding just 30 hand-crafted dialogue chains to the training set dramatically improves multi-turn performance.

**Q5: What are the limitations of LIMA?**
A5 (from the paper, Discussion): "the mental effort in constructing such examples is significant and difficult to scale up. Secondly, LIMA is not as robust as product-grade models; while LIMA typically generates good responses, an unlucky sample during decoding or an adversarial prompt can often lead to a weak response."

## Common Misconceptions

1. **"LIMA proves you never need RLHF"** — Not exactly. LIMA demonstrates that RLHF is not strictly necessary to achieve strong alignment, but the paper does not claim RLHF is useless. The Discussion section acknowledges LIMA "is not as robust as product-grade models" and can produce weak responses on adversarial prompts. RLHF may still improve robustness and safety beyond what supervised fine-tuning alone achieves.

2. **"1,000 examples is always enough"** — The paper's ablations (Section 5) show that data quality and diversity matter more than quantity, but this does not mean 1,000 is a magic number. The 1,000 examples were extremely carefully curated from high-quality sources with strict filtering. The ablation on 7B models found that at least 2,000 examples were needed for stability, suggesting the exact number depends on model size and data quality.

3. **"LIMA means pretraining is all that matters"** — The paper argues that "almost all knowledge in large language models is learned during pretraining," but alignment still plays a crucial role in teaching the model to produce useful, well-formatted responses. Without alignment, the pretrained model would generate text but not in the helpful-assistant style users expect.

## Real Citations

1. Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., Ma, X., Efrat, A., Yu, P., Yu, L., Zhang, S., Ghosh, G., Lewis, M., Zettlemoyer, L., Levy, O. (2023). "LIMA: Less Is More for Alignment." *NeurIPS 2023.* [arXiv:2305.11206](https://arxiv.org/abs/2305.11206) — the original paper.

2. Taori, R. et al. (2023). "Stanford Alpaca: An Instruction-following LLaMA Model." [GitHub](https://github.com/tatsu-lab/stanford_alpaca) — LIMA outperformed Alpaca 65B despite Alpaca using 52× more training data, demonstrating the quality-over-quantity principle.

3. Rafailov, R. et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) — DPO further simplified the alignment pipeline by eliminating the need for a reward model, building on LIMA's message that alignment can be simpler than RLHF suggests.

**Citation count:** about 275 citations on [OpenAlex](https://openalex.org/W7133189643) (checked November 2026).
