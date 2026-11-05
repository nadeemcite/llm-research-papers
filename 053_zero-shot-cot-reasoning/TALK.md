---
title: "Zero-shot CoT: Talk & Q&A"
parent: "053 Zero-shot CoT"
nav_order: 2
---
# TALK — Large Language Models are Zero-Shot Reasoners (Zero-shot CoT)

## Talks, Slides and Press Coverage

1. **Takeshi Kojima — "Large Language Models are Zero-Shot Reasoners" at NeurIPS 2022 (poster):** The paper was accepted to NeurIPS 2022 and presented as a poster. The work originated at the University of Tokyo in collaboration with Google Research. ([arXiv page](https://arxiv.org/abs/2205.11916))
2. **Google Research Blog — "PaLM: Scaling Language Modeling with Pathways" (Aakanksha Chowdhery et al., 2022):** The Zero-shot-CoT results on PaLM 540B (MultiArith 25.5%→66.1%, GSM8K 12.5%→43.0%) were reported as part of the broader PaLM evaluation, demonstrating that the single trigger generalizes to a different model family. ([link](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2205.11916).

**Q1: Why does "Let's think step by step" work without any examples?**
A1 (from the paper, Section 1): "The versatility of this single prompt across very diverse reasoning tasks hints at untapped and understudied fundamental zero-shot capabilities of LLMs, suggesting high-level, multi-task broad cognitive capabilities may be extracted by simple prompting."

**Q2: What is the two-stage prompting pipeline?**
A2 (from the paper, Section 3.1): The first prompt uses the template "Q: [X]. A: [T]" with trigger T to generate reasoning. The second prompt concatenates the original question, the generated reasoning, and an answer-extraction trigger (e.g., "Therefore, the answer (arabic numerals) is") to extract the final answer. "Few-shot-CoT requires careful human engineering of a few prompt examples with specific answer formats per task, while Zero-shot-CoT requires less engineering but requires prompting LLMs twice."

**Q3: Does Zero-shot-CoT work on all tasks?**
A3 (from the paper, Section 4.1): "Zero-shot-CoT substantially outperforms four out of six arithmetic reasoning tasks (MultiArith, GSM8K, AQUA, SVAMP), all symbolic reasoning, and all other logical reasoning tasks." However, "In commonsense reasoning tasks, Zero-shot-CoT does not provide performance gains." The authors note this is expected, as even Few-shot-CoT did not improve commonsense on LaMDA 137B.

**Q4: How does model size affect Zero-shot-CoT?**
A4 (from the paper, Section 4.1, Figure 3): "Without chain of thought reasoning, the performance does not increase or increases slowly as the model scale is increased, i.e., the curve is mostly flat. In contrast, the performance drastically increases with chain of thought reasoning, as the model size gets bigger... When the model size is smaller, chain of thought reasoning is not effective."

**Q5: Is "Let's think step by step" the only trigger that works?**
A5 (from the paper, Table 4): No. The authors tested 16 templates. "Let's think step by step." achieved the best result (78.7% on MultiArith). Other instructive templates like "First," (77.3%) and "Let's think about this logically." (74.5%) also worked well. Misleading templates like "Don't think. Just feel." (18.8%) and irrelevant templates like "It's a beautiful day." (13.1%) did not improve performance.

## Common Misconceptions

1. **"Zero-shot-CoT is just a simpler version of few-shot CoT."** — While conceptually related, Zero-shot-CoT is a fundamentally different approach. It uses no examples at all and a single task-agnostic trigger, whereas few-shot CoT requires carefully crafted per-task exemplars. The paper shows that few-shot CoT is sensitive to example-task matching (Table 5), while Zero-shot-CoT is inherently task-agnostic. The two-stage prompting pipeline is also unique to Zero-shot-CoT.

2. **"The 'Let's think step by step' trick works on any model."** — The paper clearly shows this is not true. The gains are emergent with model scale: small models (<10B) show little to no benefit and may even perform worse. The paper's Figure 3 demonstrates that the CoT scaling curve is steep only for large models. GPT-2 at 124M, for instance, would not be expected to show dramatic gains.

3. **"Zero-shot-CoT outperforms few-shot CoT."** — The paper explicitly states the opposite: "Zero-shot-CoT naturally underperforms Few-shot-CoT." However, Zero-shot-CoT "substantially outperforms standard Few-shot prompting with even 8 examples per task," making it a strong minimal baseline that requires no example engineering.

## Real Citations

1. Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, Yusuke Iwasawa (2022). "Large Language Models are Zero-Shot Reasoners." *NeurIPS 2022.* [arXiv:2205.11916](https://arxiv.org/abs/2205.11916) — the original paper.
2. Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) — the few-shot CoT paper that this work extends to the zero-shot setting.
3. Xuezhi Wang, Jason Wei, Dale Schuurmans, et al. (2022). "Self-Consistency Improves Chain of Thought Reasoning in Language Models." [arXiv:2203.11171](https://arxiv.org/abs/2203.11171) — combined with Zero-shot-CoT + self-consistency on PaLM 540B to achieve 89.0% on MultiArith and 70.1% on GSM8K.

**Citation count:** about 2,423 citations on [OpenAlex](https://openalex.org/W7133231369) (checked November 2026).
