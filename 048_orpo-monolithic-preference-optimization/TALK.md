---
title: "ORPO: Talk & Q&A"
parent: "048 ORPO"
nav_order: 2
---
# TALK — ORPO: Monolithic Preference Optimization without Reference Model (ORPO)

## Talks, Slides and Press Coverage

1. **Hugging Face Papers — ORPO paper page:** The paper's page on Hugging Face Papers, where the community discussed and upvoted the work after its release. ([link](https://huggingface.co/papers/2403.07691))
2. **Kaggle Models — Mistral-ORPO-α and Mistral-ORPO-β:** The authors released both model checkpoints on Hugging Face, where they were featured as top-performing 7B instruction-following models. ([Mistral-ORPO-α](https://huggingface.co/kaist-ai/mistral-orpo-alpha), [Mistral-ORPO-β](https://huggingface.co/kaist-ai/mistral-orpo-beta))
3. **GitHub — xfactlab/orpo:** The authors' official training code repository, released alongside the paper. ([link](https://github.com/xfactlab/orpo))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2403.07691).

**Q1: Why does SFT alone fail for preference alignment?**
A1 (from the paper, Section 3): The cross-entropy loss used in SFT "gives no direct penalty or compensation for the logits of non-answer tokens" because the target label y_i is 0 for non-answer tokens. As a result, "the log probabilities of the tokens in the rejected responses increase along with the chosen responses, which is not desired from the viewpoint of preference alignment." The paper demonstrates this empirically in Figure 3, showing that rejected response log-probabilities increase during SFT on chosen-only data.

**Q2: Why use the odds ratio instead of the probability ratio?**
A2 (from the paper, Section 7.1): "The probability ratio leads to more extreme discrimination of the disfavored responses than the odds ratio." When the log-sigmoid function is applied, the probability ratio's sharper distribution "results in overly suppressing the logits for the tokens in the disfavored responses in the setting where SFT and preference alignment are incorporated, as the model is not adapted to the domain." The odds ratio provides "mild discrimination of disfavored responses and the prioritizing of the favored responses."

**Q3: How does ORPO save compute compared to DPO?**
A3 (from the paper, Section 7.3): DPO requires both a policy model and a frozen reference model, resulting in "four forward passes in total for a single batch" (two for the policy, two for the reference). ORPO does not require a reference model, so "a reference model is not required in ORPO as π_SFT is directly updated. This leads to half the number of forward passes required for each batch during training."

**Q4: What is the role of λ in the ORPO loss?**
A4 (from the paper, Section 4.2): λ weights the odds-ratio loss L_OR relative to the SFT loss L_SFT. "Together, L_SFT and L_OR weighted with λ tailor the pre-trained language model to adapt to the specific subset of the desired domain and disfavor generations in the rejected response sets." The paper uses λ = 0.1 for Mistral, 0.2 for Llama-2, and 0.25 for Phi-2, and studies the effect of different λ values in Appendix E.

## Common Misconceptions

1. **"ORPO is just DPO without a reference model."** — While both are reference-free in spirit, ORPO is structurally different. DPO's loss uses the log-ratio log π_θ(y\|x) / π_ref(y\|x), which inherently requires a reference model. ORPO replaces this with the odds ratio P(y\|x) / (1 − P(y\|x)), which uses only the current model's probabilities. The odds ratio is a different mathematical object than the probability ratio, and the paper shows (Section 7.1, Appendix B) that it leads to more stable training than using probability ratios without a reference.

2. **"ORPO doesn't need SFT at all."** — ORPO doesn't need a *separate* SFT stage, but the SFT loss (NLL on chosen responses) is still a core component of the ORPO objective (Eq. 6). The innovation is combining SFT and preference alignment into a single training run, not eliminating supervised learning entirely.

3. **"ORPO always outperforms DPO."** — The paper's controlled experiments (Tables 2–3) show that ORPO's win rate over DPO is correlated with model size: ORPO beats DPO more clearly at larger scales. At OPT-125M, ORPO's win rate over DPO is only 41.7% on HH-RLHF. The advantage grows with scale, reaching 70.9% at OPT-1.3B.

## Real Citations

1. Jiwoo Hong, Noah Lee, James Thorne (2024). "ORPO: Monolithic Preference Optimization without Reference Model." *Preprint.* [arXiv:2403.07691](https://arxiv.org/abs/2403.07691) — the original paper.
2. Rafael Rafailov et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) — DPO, the method ORPO improves upon by eliminating the reference model.
3. Kawin Ethayarajh et al. (2024). "KTO: Model Alignment as Prospect Theoretic Optimization." [arXiv:2402.01306](https://arxiv.org/abs/2402.01306) — KTO, a contemporary alignment method also referenced in the paper's related work.

**Citation count:** about 200 citations on [OpenAlex](https://openalex.org/W4390839461) (checked November 2026).
