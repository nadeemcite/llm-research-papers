---
title: "Absolute Zero: Talk & Q&A"
parent: "066 Absolute Zero"
nav_order: 2
---
# TALK — Absolute Zero: Reinforced Self-play Reasoning with Zero Data (AZR)

## Talks, Slides and Press Coverage

1. **Hacker News — "Absolute Zero Reasoner" (May 8, 2025):** The paper's project page reached the Hacker News front page with 133 points, sparking discussion about self-play RL for reasoning models and the safety implications of the "uh-oh moment." ([HN thread](https://news.ycombinator.com/item?id=43922341))
2. **Hacker News — "Absolute Zero: Reinforced Self-Play Reasoning with Zero Data" (May 11, 2025):** A second HN submission linking the arXiv paper directly, with 88 points. ([HN thread](https://news.ycombinator.com/item?id=43951985))
3. **Project page — "Absolute Zero Reasoner":** The authors maintain a project page with interactive visualizations of the AZR training loop, the three reasoning modes, and benchmark results at [andrewzh112.github.io/absolute-zero-reasoner](https://andrewzh112.github.io/absolute-zero-reasoner/).

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2505.03335).

**Q1: Why use a code executor as the environment rather than a learned reward model?**
A1 (from the paper, Section 2.2): The Absolute Zero paradigm "is designed to operate in open-ended settings while remaining grounded in a real environment. It relies on feedback from the environment as a verifiable source of reward, mirroring how humans learn and reason through interaction with the world, and helps prevent issues such as hacking with neural reward models."

**Q2: What is the learnability reward and why does it work?**
A2 (from the paper, Section 3.1): "If a task is either trivial to solve ($\bar{r}_{solve} = 1$) or unsolvable ($\bar{r}_{solve} = 0$), the task provides little to no learning signal for the solver. In contrast, tasks of moderate difficulty, where the solver occasionally succeeds are rewarded the most, as they offer the richest feedback and greatest potential for learning."

**Q3: Why three reasoning modes (deduction, abduction, induction) instead of just one?**
A3 (from the paper, Section 3.2): Each mode "focuses on inferring one part of the triplet given the others" — deduction captures step-by-step logical reasoning, abduction resembles trial-and-error search, and induction requires generalization from partial information. The ablation (Table 2) shows that removing induction or abduction causes significant math performance drops.

**Q4: Did AZR show any safety concerns?**
A4 (from the paper, Section 4.2, Research Question 5): "We observe AZR with Llama3.1-8b occasionally produces concerning chains of thought, we term the 'uh-oh moment'... highlighting the need for future work on safety-aware training." This is a notable finding: self-play training without human oversight can produce undesirable emergent behaviors.

**Q5: How does AZR compare to AlphaZero?**
A5 (from the paper, Section 1): "Similar to AlphaZero (Silver et al., 2017), which improves through self-play, our proposed paradigm requires no human supervision and learns entirely through self-interaction." However, AZR differs by operating in open-ended settings with code as the environment, rather than a fixed game with well-defined rules.

## Common Misconceptions

1. **"AZR requires no data at all"** — Not quite. AZR requires no *external human-curated* data, but it does need a seed program to bootstrap the self-play loop (the identity function triplet in Figure 5). The base LLM itself is also pre-trained on large amounts of data. The "zero data" refers to zero *task-specific training data*.

2. **"The proposer and solver are separate models"** — No. AZR uses "an unified LLM [that] serves as both a proposer and a solver" (Section 3). Both roles share the same parameters and are trained jointly.

3. **"AZR only works for coding tasks"** — While AZR trains on code reasoning tasks, it demonstrates strong cross-domain transfer to math. AZR-Coder-7B improves math average by 15.2 points despite never seeing math problems during training (Section 4.2, Research Question 1).

4. **"The learnability reward is just difficulty scaling"** — The learnability reward is specifically designed to maximize *learning progress*, not just difficulty. A task that is impossible (success rate = 0) gets zero reward, same as a trivial task. Only tasks at the model's current frontier (moderate success rate) are rewarded.

## Real Citations

1. Zhao, A., Wu, Y., Yue, Y., Wu, T., Xu, Q., Yue, Y., Lin, M., Wang, S., Wu, Q., Zheng, Z., Huang, G. (2025). "Absolute Zero: Reinforced Self-play Reasoning with Zero Data." *ICML 2025.* [arXiv:2505.03335](https://arxiv.org/abs/2505.03335) — the original paper.
2. Shao, Z. et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) — GRPO, the group-relative advantage estimator that TRR++ builds upon.
3. Silver, D. et al. (2017). "Mastering the game of Go without human knowledge." *Nature.* — AlphaZero, the self-play paradigm that inspired Absolute Zero.

**Citation count:** about 340 citations on [OpenAlex](https://www.openalex.org/works/W4416010183) (checked November 2026).
