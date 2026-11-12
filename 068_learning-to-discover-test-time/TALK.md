---
title: "Learning to Discover: Talk & Q&A"
parent: "068 Learning to Discover"
nav_order: 2
---
# TALK — Learning to Discover at Test Time (TTT-Discover)

## Talks, Slides and Press Coverage

No recorded talks or press coverage found as of November 2026 (the paper is a recent January 2026 preprint).

- Official code repository: [github.com/test-time-training/discover](https://github.com/test-time-training/discover)

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2601.16175).

**Q1: Why does TTT-Discover train the LLM at test time instead of just doing more search?**
A1 (from the paper, Section 1): "The most direct way for the LLM to improve is through learning. And indeed, while both learning and search scale well with compute, learning has often superseded search in the history of AI for hard problems such as Go and protein folding."

**Q2: How is TTT-Discover different from standard RL?**
A2 (from the paper, Section 3.1): Discovery problems have two critical differences from standard RL: "First, our policy only needs to solve this single problem rather than generalize to other problems. Second, we only need a single best solution, and the policy is merely a means towards this end." Standard RL maximizes average reward across all attempts; discovery only needs one great solution.

**Q3: What is the entropic objective and why is it needed?**
A3 (from the paper, Section 3.2): "Naive RL optimizes average performance, and is indifferent to the state of the art." The entropic objective J_beta = E_s[log E_a[exp(beta * R(s,a))]] approximates the maximum as beta → ∞, focusing learning on the best outcomes. "However, too large beta early in training causes instabilities, while too small later makes advantages vanish as even smaller improvements become harder." Hence the adaptive beta.

**Q4: Why use PUCT instead of standard epsilon-greedy or random selection for state reuse?**
A4 (from the paper, Section 3.2): PUCT balances exploitation and exploration. Q(s) uses the maximum reward of children, not the mean, because "we care about the best outcome starting from a state, not the average." The prior P(s) captures "the intuition that high-reward states are more likely to yield high-reward children—e.g., a fast kernel is more likely to seed a faster kernel than a slow one."

**Q5: What model and infrastructure did the authors use?**
A5 (from the paper, Section 3.3): "We run TTT-Discover with gpt-oss-120b on Tinker Lab for 50 training steps. We use LoRA with rank 32. At each step, we generate a batch of 512 rollouts, with 8 groups of 64 rollouts each." The cost is "around $500 on Tinker" per problem.

**Q6: Did TTT-Discover improve on all problems it attempted?**
A6 (from the paper, Section 4): No. On the second autocorrelation inequality and circle packing, TTT-Discover did not make a discovery. "We report results for every problem we attempted" — the paper is transparent about both successes and non-discoveries.

**Q7: How does TTT-Discover compare to concurrent work like ThetaEvolve?**
A7 (from the paper, Section 1 and Table 2): "Compared to ThetaEvolve, TTT-Discover using the same model and compute budget still produces significant improvements, due to its special learning objective and search subroutine." On the autocorrelation inequalities, TTT-Discover with Qwen3-8B (a weaker model than ThetaEvolve's) still outperformed ThetaEvolve.

## Common Misconceptions

1. **"TTT-Discover is just another evolutionary search method like AlphaEvolve"** — No. AlphaEvolve and similar methods keep the LLM frozen and only improve the prompts using evolutionary heuristics. TTT-Discover actually updates the LLM's weights via RL at test time. The paper's ablations (Table 8) show that without test-time training ("No TTT"), performance drops from 1203 µs to 2061 µs on the TriMul competition.

2. **"TTT-Discover is standard RL applied to a single problem"** — Not quite. Standard RL (PPO, GRPO) optimizes average reward and aims for a deployable policy. TTT-Discover uses an entropic objective that favors the maximum reward, and the policy is disposable — only the single best solution matters. The paper explicitly shows naive RL (expected reward, no reuse) barely improves over Best-of-N (Table 8: 5329 µs vs 5352 µs).

3. **"The method requires closed frontier models"** — No. The paper states: "All our results are achieved with an open model, OpenAI gpt-oss-120b, and can be reproduced with our publicly available code, in contrast to previous best results that required closed frontier models."

## Real Citations

1. Yuksekgonul et al. (2026). "Learning to Discover at Test Time." *arXiv preprint.* [arXiv:2601.16175](https://arxiv.org/abs/2601.16175) — the original paper.
2. Novikov et al. (2025). "AlphaEvolve: A coding agent for scientific discovery." — the prior work that TTT-Discover extends by adding test-time training. Referenced throughout the paper as the primary baseline.
3. Sun et al. (2020). "Test-Time Training with Self-Supervision for Generalization under Distribution Shifts." — the foundational test-time training framework that TTT-Discover builds on.
4. Wang et al. (2025a). "ThetaEvolve." — concurrent work combining per-instance RL with evolutionary search; the paper's closest competitor.

**Citation count:** The paper is too recent (January 2026) for meaningful citation counts. Code is publicly available at [github.com/test-time-training/discover](https://github.com/test-time-training/discover).
