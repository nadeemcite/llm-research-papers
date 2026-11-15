---
title: "SWE-bench: Talk & Q&A"
parent: "074 SWE-bench"
nav_order: 2
---
# TALK — SWE-bench: Can Language Models Resolve Real-World GitHub Issues? (SWE-bench)

## Talks, Slides and Press Coverage

1. **ICLR 2024 — "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" (Jimenez et al., 2024):** The paper was accepted as an Outstanding Paper at ICLR 2024. The authors presented the benchmark, construction pipeline, and baseline results. ([conference page](https://iclr.cc/virtual/2024/oral/19524))

2. **Princeton NLP — SWE-bench GitHub repository:** The official repository hosts the benchmark data, evaluation scripts, baselines, and the SWE-Llama fine-tuned models. It includes instructions for running evaluations and submitting new results. ([repository](https://github.com/princeton-nlp/SWE-bench))

3. **SWE-bench Leaderboards — swebench.com:** The official leaderboard tracks resolve rates of agent systems on SWE-bench, SWE-bench Lite, and SWE-bench Verified, showing progress from the 1.96% baseline to over 40% by 2024. ([leaderboards](https://www.swebench.com/index.html))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2310.06770).

**Q1: Why did the authors choose GitHub issues as the basis for the benchmark rather than hand-crafted coding problems?**
A1 (from the paper, Section 1): "In the real world, software engineering is not as simple. Fixing a bug might involve navigating a large repository, understanding the interplay between functions in different files, or spotting a small error in convoluted code." Existing benchmarks like HumanEval "mostly involve self-contained problems that can be solved in a few lines of code."

**Q2: How were the task instances filtered from raw GitHub data?**
A2 (from the paper, Section 2.1): The authors used a 3-stage pipeline: Stage I scraped ~90,000 pull requests from 12 repos; Stage II selected merged PRs that resolve an issue and contribute test changes; Stage III filtered for instances with at least one fail-to-pass test, arriving at 2,294 instances.

**Q3: Why did the best model only resolve 1.96% of issues?**
A3 (from the paper, Section 5): "models see a lot of code that may not be directly related to solving the issue at hand, and they seem to frequently struggle with localizing problematic code needing to be updated." Additionally, "model generated patch files tend to add and remove fewer lines than their respective gold patch" — averaging 30.1 lines vs 74.5 for gold patches.

**Q4: What is SWE-bench Lite and why was it created?**
A4 (from the paper, Section 2.4): "To encourage adoption of SWE-bench, we create a Lite subset of 300 instances from SWE-bench that have been sampled to be more self-contained, with a focus on evaluating functional bug fixes." It was created because "evaluating LMs on SWE-bench can be time-consuming and, depending on the model, require a costly amount of compute or API credits."

**Q5: Did the fine-tuned SWE-Llama models perform better than proprietary models?**
A5 (from the paper, Section 5, Table 5): No. SWE-Llama 13b resolved 0.70% with BM25 retrieval, compared to Claude 2 at 1.96%. However, in the oracle setting, SWE-Llama 13b was competitive with Claude 2 — resolving 91 instances vs Claude 2's 110 — though "of these instances, Claude 2 only solves 42% of the instances solved by SWE-Llama," suggesting complementary strengths.

**Q6: Does a model's performance correlate with the amount of context provided?**
A6 (from the paper, Section 5): Surprisingly, more context hurts. "As total context length increases, Claude 2's performance drops considerably." Even though increasing BM25 context improves recall of the oracle files, "performance drops, as models are simply ineffective at localizing problematic code."

## Common Misconceptions

1. **"SWE-bench only tests Python, so it doesn't generalize to other languages."** — This is partially true but not a design flaw. The paper acknowledges in Section 7: "SWE-bench task instances are all in Python; we hope to apply SWE-bench's task instance collection procedure to expand its coverage to more programming languages and domains." The construction pipeline is language-agnostic; Python was chosen for its rich ecosystem of well-tested open-source repos.

2. **"The 1.96% resolve rate means LLMs are useless for software engineering."** — The paper frames this as a starting point, not a final verdict. Section 1 states the benchmark represents "steps towards LMs that are more practical, intelligent, and autonomous." The rapid improvement to 40%+ on SWE-bench Verified within a year confirms the benchmark was measuring a real capability gap, not an impossible task.

3. **"SWE-bench and SWE-bench Lite are the same benchmark."** — SWE-bench Lite is a 300-instance subset "sampled to be more self-contained, with a focus on evaluating functional bug fixes" (Section 2.4). It covers 11 of the original 12 repos. It was designed for faster iteration, not as a replacement for the full benchmark.

## Real Citations

1. Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, Karthik Narasimhan (2024). "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" *ICLR 2024 (Outstanding Paper).* [arXiv:2310.06770](https://arxiv.org/abs/2310.06770) — the original paper.

2. John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Xinyi Yao, Dunjie Zhang, Navid Madani, Karthik Narasimhan (2024). "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering." [arXiv:2405.15793](https://arxiv.org/abs/2405.15793) — the follow-up that builds an agent system on top of SWE-bench.

3. Princeton NLP (2024). "SWE-bench Verified." — a human-verified subset of SWE-bench instances, released on the [SWE-bench Verified leaderboard](https://www.swebench.com/verified.html) and the [GitHub repository](https://github.com/princeton-nlp/SWE-bench) to improve evaluation reliability.

**Citation count:** about 3,100 citations on [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/arXiv:2310.06770) (checked November 2026).
