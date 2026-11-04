---
title: "Constitutional AI: Talk & Q&A"
parent: "045 Constitutional AI"
nav_order: 2
---
# TALK — Constitutional AI: Harmlessness from AI Feedback (Constitutional AI)

## Talks, Slides and Press Coverage

1. **Anthropic — "Constitutional AI: Harmlessness from AI Feedback" (Anthropic, Dec 2022):** The paper's official announcement on Anthropic's research page, describing the method and its goals of scaling supervision and reducing evasiveness. ([link](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback))

2. **Interaction Design Foundation — "Constitutional AI: A New Approach to AI Alignment" (2023):** An accessible explainer covering the CAI method, its two-stage process, and how it compares to RLHF. ([link](https://www.interaction-design.org/literature/article/constitutional-ai-a-new-approach-to-ai-alignment))

3. **Anthropic GitHub — "ConstitutionalHarmlessnessPaper" (Anthropic, 2022):** The official code repository with few-shot prompts, constitutional principles, and model response examples used in the paper. ([link](https://github.com/anthropics/ConstitutionalHarmlessnessPaper))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2212.08073).

**Q1: Why use AI feedback instead of human feedback for harmlessness?**
A1 (from the paper, Section 1.1): The authors wanted to study "simple possibilities for using AI systems to help supervise other AIs, and thus scale supervision." They also wanted to "reduce iteration time by obviating the need to collect new human feedback labels when altering the objective," and to make "the principles governing AI behavior, and their implementation, more transparent."

**Q2: What is the difference between the SL stage and the RL stage?**
A2 (from the paper, Section 1.2): "In the supervised phase we sample from an initial model, then generate self-critiques and revisions, and then finetune the original model on revised responses. In the RL phase, we sample from the finetuned model, use a model to evaluate which of the two samples is better, and then train a preference model from this dataset of AI preferences. We then train with RL using the preference model as the reward signal."

**Q3: Are critiques necessary, or can the model just revise directly?**
A3 (from the paper, Section 3.5): "We found that critiqued revisions achieved better harmlessness scores for small models, but made no noticeable difference for large models. Furthermore, based on inspecting samples from the 52B, we found that the critiques were sometimes reasonable, but often made inaccurate or overstated criticisms. Nonetheless, the revisions were generally more harmless than the original response." The authors chose to use critiqued revisions for the main results "as it may provide more transparency into the model's reasoning process."

**Q4: Why is the model non-evasive, and why does that matter?**
A4 (from the paper, Section 4.4): "We find that RL-CAI is virtually never evasive, and often gives nuanced and harmless responses to most red team prompts." This matters because "for safety purposes it is also important for models to be transparent about their thought process and decision-making, and for practical purposes we expect non-evasive responses to be more compatible with helpfulness."

**Q5: What happens when the model is over-trained?**
A5 (from the paper, Section 4.3): "We found that RL-CAI models can be over-trained, resulting in Goodharting behavior whereby models can be overly harsh in responding to harmful prompts, or may include boilerplate language as part of their response to most red teaming prompts, saying e.g. 'you are valid, valued, and cared for.'"

## Common Misconceptions

1. **"Constitutional AI removes all human supervision."** — This is incorrect. The paper still uses human feedback labels for *helpfulness*; only *harmlessness* labels are replaced by AI feedback. The authors state (Section 6): "in this work we still relied on human supervision in the form of helpfulness labels."

2. **"The constitution is a legal document or formal specification."** — The constitution is simply a list of ~16 natural-language principles written in plain English, such as "Identify specific ways in which the assistant's last response is harmful, unethical, racist, sexist, toxic, dangerous, or illegal." The authors note they "were selected in an ad hoc manner for research purposes."

3. **"CAI is just prompt engineering."** — While the constitution is expressed in natural language, CAI involves actual training: the SL stage fine-tunes the model on revised responses, and the RL stage trains a preference model and runs RL against it. It is not just prompting at inference time.

4. **"Chain-of-thought is only used in the RL stage."** — The paper states (Abstract) that "both the SL and RL methods can leverage chain-of-thought style reasoning to improve the human-judged performance and transparency of AI decision making."

## Real Citations

1. Bai, Y. et al. (2022). "Constitutional AI: Harmlessness from AI Feedback." arXiv:2212.08073 — the original paper.

2. Bai, Y. et al. (2022). "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback." [arXiv:2204.05862](https://arxiv.org/abs/2204.05862) — the prior work on RLHF that CAI extends and improves upon.

3. Ganguli, D. et al. (2022). "Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned." [arXiv:2209.07858](https://arxiv.org/abs/2209.07858) — the source of red-teaming prompts used in CAI training.

4. Wei, J. et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) — the CoT prompting technique used in the RL stage's feedback model.

5. Christiano, P. et al. (2017). "Deep Reinforcement Learning from Human Preferences." [arXiv:1706.03741](https://arxiv.org/abs/1706.03741) — the foundational RLHF work that CAI partially replaces with RLAIF.

**Citation count:** about 3,100 citations on [Semantic Scholar](https://scholar.google.com/scholar?q=Constitutional+AI+Harmlessness+from+AI+Feedback) (checked November 2026).
