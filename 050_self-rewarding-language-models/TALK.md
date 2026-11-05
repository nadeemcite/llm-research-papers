---
title: "Self-Rewarding LM: Talk & Q&A"
parent: "050 Self-Rewarding LM"
nav_order: 2
---
# TALK — Self-Rewarding Language Models (Self-Rewarding LM)

## Talks, Slides and Press Coverage

1. **Jason Weston — "Self-Rewarding Language Models" (2024):** Jason Weston, one of the senior authors, presented the self-rewarding approach, discussing how models can serve as their own reward models via LLM-as-a-Judge prompting. ([video](https://www.youtube.com/watch?v=hYU-tWVxgQ8))
2. **Papers with Code — "Self-Rewarding Language Models":** The paper's page on Papers with Code, linking to the arXiv paper, related work, and community discussion. ([link](https://paperswithcode.com/paper/self-rewarding-language-models))
3. **Reddit r/LocalLLaMA — "Self-Rewarding Language Models" (Feb 2024):** Community discussion of the paper, covering practical implications and comparisons with DPO and RLHF. ([link](https://www.reddit.com/r/LocalLLaMA/comments/1i9t4bw/selfrewarding_language_models/))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2401.10020).

**Q1: Why use the model itself as a judge instead of a separate reward model?**
A1 (from the paper, Abstract): "Current approaches commonly train reward models from human preferences, which may then be bottlenecked by human performance level, and secondly these separate frozen reward models cannot then learn to improve during LLM training."

**Q2: Does judging ability actually improve, or does only instruction-following improve?**
A2 (from the paper, Section 4.2): The paper measures reward model quality on the "Reward Model Agreement" task within the Evaluator Critique benchmark. Across iterations M1 → M2 → M3, the agreement with human labels improves from 53.0% to 54.7% to 55.3%, demonstrating that "the reward modeling ability of the model improves" alongside instruction following.

**Q3: What is the LLM-as-a-Judge prompt and how does it work?**
A3 (from the paper, Section 3.1): The judge evaluates two responses to the same instruction. It provides "feedback" (analysis of both responses) and a "score" (1–5) for each. The evaluation prompt asks the model to act as a "fair and impartial judge" considering relevance, helpfulness, level of detail, and other factors. The response with the higher score becomes the "chosen" in the preference pair.

**Q4: What was the starting model and how many iterations were run?**
A4 (from the paper, Section 4): The starting point is Llama 2 70B SFT. Three iterations (M1, M2, M3) are run, each using 1,000 prompts from the Fine-grained RLHF dataset for preference pair generation, with the remaining prompts used for DPO training.

**Q5: Could self-rewarding lead to reward hacking or self-reinforcing bias?**
A5 (from the paper, Section 5): The authors note this as a concern: "A potential issue with self-rewarding is that the model could learn to reward its own outputs regardless of their actual quality." They observe that the approach works in practice because the judge task is itself an instruction-following task that benefits from the same training, but acknowledge that "further investigation of this point is needed."

## Common Misconceptions

1. **"Self-rewarding means the model gives itself a high score for everything"** — Not quite. The model judges pairs of responses using a structured LLM-as-a-Judge prompt that evaluates quality dimensions. The paper explicitly tested for this concern (Section 5) and found that the judge does not simply rubber-stamp its own outputs; reward model quality improves on an external benchmark (RewardBench), which would not happen if the model were just inflating scores.

2. **"Self-rewarding replaces RLHF entirely"** — The paper frames it as a step toward removing the need for human annotation, not as a complete replacement. The starting model (Llama 2 70B) was itself trained with human feedback. Self-rewarding builds on top of an already-aligned model; it does not create alignment from scratch.

3. **"The DPO objective is different from standard DPO"** — The DPO loss itself is the standard DPO objective. The innovation is in how the preference data is generated (self-judging rather than human labeling) and that the process is iterative (each iteration uses the updated model as both generator and reference). The paper adds a small SFT term to the DPO loss (Eq. 7), but the core DPO formulation is unchanged.

## Real Citations

1. Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, Jason Weston (2024). "Self-Rewarding Language Models." *Preprint.* [arXiv:2401.10020](https://arxiv.org/abs/2401.10020) — the original paper.
2. Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, Chelsea Finn (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) — the DPO method that self-rewarding builds upon.
3. Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." [arXiv:2306.05685](https://arxiv.org/abs/2306.05685) — the LLM-as-a-Judge paradigm used by the self-rewarding approach.
4. Zhanhui Zhou, Jie Liu, Jing Shao, Chao Yang, Wanli Ouyang, Yu Qiao (2024). "Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models." [arXiv:2312.06585](https://arxiv.org/abs/2312.06585) — related work on self-training without external feedback.

**Citation count:** about 300 citations on [OpenAlex](https://openalex.org/W4396956981) (checked November 2026).
