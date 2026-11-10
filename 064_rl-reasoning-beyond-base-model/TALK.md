---
title: "RL vs Base Model: Talk & Q&A"
parent: "064 RL vs Base Model"
nav_order: 2
---
# TALK — Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? (RL vs Base Model)

## Talks, Slides and Press Coverage

1. **NeurIPS 2025 — Best Paper Runner-Up Award:** The paper received the Best Paper Runner-Up award at NeurIPS 2025 and was presented as an oral. The project page with slides, poster, and video is at [limit-of-RLVR.github.io](https://github.com/LeapLabTHU/limit-of-RLVR).
2. **ICML 2025 AI4Math Workshop — Best Paper Award:** The paper also received the Best Paper award at the ICML 2025 Workshop on AI for Math. ([workshop page](https://openreview.net/group?id=ICML.cc/2025/Workshop/AI4Math))
3. **Project Page — "Limit of RLVR":** The authors maintain an interactive project page with pass@k curve animations, case studies, and Q&A at [limit-of-RLVR.github.io](https://github.com/LeapLabTHU/limit-of-RLVR). The "Read This First" section answers common questions about the capability boundary methodology.
4. **GitHub — limit-of-RLVR/RLVR:** Code for reproducing the experiments is available on GitHub. ([repository](https://github.com/LeapLabTHU/limit-of-RLVR))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2504.13837).

**Q1: Why use pass@k instead of just pass@1 (greedy accuracy)?**
A1 (from the paper, Section 2.1): pass@1 measures sampling efficiency — how likely the model is to produce a correct answer on a single try. pass@k at large k measures the model's total reasoning coverage — the full set of problems it can solve given enough attempts. "We propose using pass@k at large k values as the evaluation metric" because it reveals the reasoning capability boundary, not just the sampling efficiency.

**Q2: Do RLVR models ever solve problems that the base model cannot?**
A2 (from the paper, Table 2): Very rarely. On AIME24 (k=1024), 0.0% of problems are solvable by the RL model but not the base model. On MATH500 (k=128), only 1.0%. Meanwhile, 13.3% of AIME24 problems are solvable by the base model but not the RL model. "We find that there are many cases where the base model solves a problem but the RLVR model fails, and very few where RLVR succeeds while the base model fails."

**Q3: What is the sampling efficiency gap (Δ_SE)?**
A3 (from the paper, Section 4.3): "We define the sampling efficiency gap (Δ_SE) as the difference between an RL model's pass@1 and the base model's pass@k (with k=256 as a proxy for upper-bound performance). This metric quantifies how closely an RL algorithm approaches the optimal bound. Across all algorithms (e.g., PPO, GRPO, Reinforce++), Δ_SE shows only minor variation yet remains consistently large, suggesting that current RLVR methods, while improving sampling efficiency, are still far from optimal."

**Q4: How does distillation differ from RLVR?**
A4 (from the paper, Section 4.2): "RLVR and distillation are fundamentally different. While RLVR improves reasoning scores by sharpening the sampling distribution within the base model's prior, distillation can introduce new reasoning patterns from the teacher and genuinely expand the model's reasoning capabilities." Distillation transfers knowledge the student did not have, while RLVR only reweights what the student already knows.

**Q5: Does increasing the generation temperature of the RL model recover the base model's coverage?**
A5 (from the paper, Section 4.5): The authors investigated this by raising the RL model's temperature to match the base model's output entropy. "While this can partially recover some coverage, the RL model still does not surpass the base model's pass@k, indicating that the narrowing of the reasoning boundary is not solely an entropy effect but reflects a fundamental distributional shift."

**Q6: What do the authors recommend for future RL paradigms?**
A6 (from the paper, Section 5): "This underscores the need for improved RL paradigms, such as effective exploration mechanism, more deliberate and large-scale data curation, fine-grained process signal, and multi-turn agent-environment interaction, to unlock this potential."

## Common Misconceptions

1. **"RLVR teaches models new reasoning skills"** — The paper's central finding is the opposite: "the current training setup does not elicit fundamentally new reasoning patterns." RLVR improves sampling efficiency (pass@1) but does not expand the reasoning boundary (pass@k at large k). All reasoning paths in the RL model are already present in the base model.

2. **"RLVR is useless since the base model can do everything the RL model can"** — Not quite. The paper acknowledges that RLVR meaningfully improves pass@1, which is what matters for practical deployment where you generate one or a few responses. The finding is about capability boundaries, not utility: RLVR makes the model more efficient at sampling correct answers, but doesn't expand the set of problems it can solve.

3. **"All RL algorithms are the same"** — The paper finds that six popular RLVR algorithms (PPO, GRPO, Reinforce++, RLOO, ReMax, DAPO) perform similarly in terms of Δ_SE, but this means they all fall short of the optimal bound in the same way — not that RL as a paradigm is hopeless. The authors suggest improved exploration, continual scaling, and multi-turn interaction as paths forward.

4. **"Distillation is just another form of RL"** — The paper explicitly distinguishes them: "RLVR and distillation are fundamentally different." RLVR sharpens within the base distribution; distillation introduces new patterns from a teacher and genuinely expands the reasoning boundary beyond what the base model could produce.

## Real Citations

1. Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, Gao Huang (2025). "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?" *NeurIPS 2025 (Oral, Best Paper Runner-Up); ICML 2025 AI4Math Workshop (Best Paper).* [arXiv:2504.13837](https://arxiv.org/abs/2504.13837) — the original paper.
2. Daya Guo et al. (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) — R1-Zero, the RLVR paradigm this paper critically examines.
3. Zichen Liu et al. (2025). "Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO)." [arXiv:2503.20783](https://arxiv.org/abs/2503.20783) — companion analysis of GRPO biases, which this paper also studies across multiple RL algorithms.
4. Rafael Rafailov et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) — DPO, an alternative to RLHF that also operates within the base model's distribution.

**Citation count:** about 800 citations on [OpenAlex](https://openalex.org/works/W7196963086) (checked November 2026).
