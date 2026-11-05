---
title: "Chain-of-Thought: Talk & Q&A"
parent: "052 Chain-of-Thought"
nav_order: 2
---
# TALK — Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (CoT)

## Talks, Slides and Press Coverage

1. **Google Research Blog — "Language Models Perform Reasoning via Chain of Thought Prompting" (Jason Wei & Denny Zhou, 2022):** The official Google Research blog post accompanying the paper, explaining CoT prompting with visual examples and highlighting the GSM8K state-of-the-art result. ([link](https://research.google/blog/language-models-perform-reasoning-via-chain-of-thought/))

2. **OpenReview — NeurIPS 2022 presentation:** The paper's official NeurIPS 2022 page with reviews, discussion, and presentation materials. ([link](https://openreview.net/forum?id=_1jke5dQXH))

3. **Prompt Engineering Guide — "Chain-of-Thought (CoT) Prompting":** A widely-referenced community guide explaining CoT prompting with examples, covering both few-shot and zero-shot variants. ([link](https://www.promptingguide.ai/techniques/cot))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2201.11903).

**Q1: Why does chain-of-thought prompting only work for large models?**
A1 (from the paper, Section 3.2): "Chain-of-thought prompting is an emergent ability of model scale. That is, chain-of-thought prompting does not positively impact performance for small models, and only yields performance gains when used with models of ~100B parameters. We qualitatively found that models of smaller scale produced fluent but illogical chains of thought, leading to lower performance than standard prompting."

**Q2: Does the chain of thought need to come before the answer?**
A2 (from the paper, Section 3.3): The ablation study tested "chain of thought after answer" — where the CoT prompt is given only after the answer. "This variant performs about the same as the baseline, which suggests that the sequential reasoning embodied in the chain of thought is useful for reasons beyond just activating knowledge." The reasoning must precede the answer to be useful.

**Q3: Is it just the extra computation (more tokens) that helps?**
A3 (from the paper, Section 3.3): The "variable compute only" ablation prompted the model to output dots (…) equal to the equation length. "This variant performs about the same as the baseline, which suggests that variable computation by itself is not the reason for the success of chain-of-thought prompting, and that there appears to be utility from expressing intermediate steps via natural language."

**Q4: Were any models finetuned in this paper?**
A4 (from the paper, Section 6): "In all experiments, chain-of-thought reasoning is elicited simply by prompting an off-the-shelf language model. No language models were finetuned in the process of writing this paper."

**Q5: Does the specific choice of exemplars matter a lot?**
A5 (from the paper, Section 3.4): "Although there is variance among different chain of thought annotations, as would be expected when using exemplar-based prompting, all sets of chain of thought prompts outperform the standard baseline by a large margin. This result implies that successful use of chain of thought does not depend on a particular linguistic style."

## Common Misconceptions

1. **"Chain-of-thought prompting works for any size model"** — Wrong. The paper explicitly shows CoT is an emergent property of model scale, only appearing at ~100B+ parameters. Small models produce illogical chains of thought and may perform worse with CoT than with standard prompting (e.g., LaMDA 420M: 2.6% standard → 0.4% CoT on GSM8K).

2. **"CoT prompting requires finetuning the model"** — Wrong. The paper uses pure few-shot prompting with off-the-shelf models. "No language models were finetuned in the process of writing this paper" (Section 6). The method works by including CoT exemplars in the prompt, not by updating weights.

3. **"The extra tokens/computation alone explain the improvement"** — Wrong. The variable-compute ablation (outputting dots instead of reasoning) performed about the same as the baseline. The natural language reasoning steps themselves — not just additional computation — are what drive the gains (Section 3.3).

## Real Citations

1. Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, Denny Zhou (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *NeurIPS 2022.* [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) — the original paper.

2. Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, Yusuke Iwasawa (2022). "Large Language Models are Zero-Shot Reasoners." [arXiv:2205.11916](https://arxiv.org/abs/2205.11916) — the zero-shot CoT follow-up ("Let's think step by step").

3. Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou (2022). "Self-Consistency Improves Chain of Thought Reasoning in Language Models." [arXiv:2203.11171](https://arxiv.org/abs/2203.11171) — sampling multiple CoT paths and majority voting.

4. Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan (2023). "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." [arXiv:2305.10601](https://arxiv.org/abs/2305.10601) — extends CoT to tree-structured reasoning.

**Citation count:** about 9,664 citations on [OpenAlex](https://openalex.org/W7133224126) (checked November 2026).
