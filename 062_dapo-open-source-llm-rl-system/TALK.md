---
title: "DAPO: Talk & Q&A"
parent: "062 DAPO"
nav_order: 2
---
# TALK — DAPO: An Open-Source LLM Reinforcement Learning System at Scale (DAPO)

## Talks, Slides and Press Coverage

1. **Hugging Face Papers — "DAPO: An Open-Source LLM Reinforcement Learning System at Scale" (March 2025):** The paper was featured on Hugging Face's Daily Papers page with community discussion. ([paper page](https://huggingface.co/papers/2503.14476))
2. **DAPO Project Page — "DAPO: an Open-Source LLM Reinforcement Learning System at Scale" (ByteDance Seed, March 2025):** The project page hosts the paper, interactive training curves, and links to the open-source code on the verl framework and the DAPO-Math-17K dataset. ([project page](https://dapo-sia.github.io/))
3. **GitHub — volcengine/verl:** The open-source training code is integrated into the verl framework, ByteDance's distributed RL training system for LLMs. ([repository](https://github.com/volcengine/verl))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2503.14476).

**Q1: Why does naive GRPO only reach 30 points on AIME while DAPO reaches 50?**
A1 (from the paper, Section 1): "In our initial GRPO run, we achieved only 30 points on AIME — a performance significantly below DeepSeek's RL (47 points). A thorough analysis reveals that the naive GRPO baseline suffers from several key issues such as entropy collapse, reward noise, and training instability."

**Q2: Why remove the KL divergence penalty?**
A2 (from the paper, Section 2.3): "During training the long-CoT reasoning model, the model distribution can diverge significantly from the initial model, thus this restriction is not necessary. Therefore, we will exclude the KL term from our proposed algorithm."

**Q3: What does Clip-Higher do, and why is the upper clip the problem?**
A3 (from the paper, Section 3.1): "The upper clip can restrict the exploration of the policy, where making an 'exploitation' token more probable is much easier yet the probability of an unlikely 'exploration' token is too tightly bounded to be uplifted." The paper gives the example that with ε=0.2, a token at probability 0.01 can only reach 0.012, while a token at 0.9 can reach 1.08 (effectively unlimited).

**Q4: How does Dynamic Sampling improve efficiency if it requires over-sampling?**
A4 (from the paper, Section 3.2): "This strategy does not necessarily impede training efficiency, because the generation time is typically dominated by the generation of long-tail samples if the RL system is synchronized and the generation stage is not pipelined." The paper shows in Figure 6 that convergence time is actually reduced despite more sampling.

**Q5: What is the DAPO-Math-17K dataset and how was it created?**
A5 (from the paper, Section 3.5): "Our dataset is sourced from the web and official competition homepages through a combination of web scraping and manual annotation... inspired by AIME, we select and transform the answers into integers, which are easy to parse." The dataset has 17K prompts, each paired with an integer answer.

## Common Misconceptions

1. **"DAPO is just GRPO with a different clip range"** — This is wrong. DAPO introduces four distinct techniques (Clip-Higher, Dynamic Sampling, Token-Level Loss, Overlong Reward Shaping). Table 1 shows each contributes incrementally: GRPO 30 → +Overlong Filtering 36 → +Clip-Higher 38 → +Soft Overlong 41 → +Token-Level Loss 42 → +Dynamic Sampling 50. Clip-Higher alone only adds 2 points.

2. **"Token-Level Loss is the most important technique"** — Not according to the ablation. Token-Level Loss adds only 1 point (41→42), while Dynamic Sampling adds 8 points (42→50). However, the paper notes it improves training stability and makes length increase more healthily, which may indirectly help.

3. **"DAPO requires a value model like PPO"** — No. DAPO is built on GRPO, which eliminates the value model entirely. Advantages are computed from group-relative reward normalization, not from a learned value function.

## Real Citations

1. Qiying Yu et al. (2025). "DAPO: An Open-Source LLM Reinforcement Learning System at Scale." *arXiv preprint.* [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) — the original paper.
2. Zhihong Shao et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) — GRPO, the baseline algorithm DAPO builds on and improves.
3. Daya Guo et al. (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) — the reasoning model whose RL training details DAPO aimed to reproduce and improve upon.

**Citation count:** about 2,811 citations on [OpenAlex](https://openalex.org/works/W7196953391) (checked November 2026).
