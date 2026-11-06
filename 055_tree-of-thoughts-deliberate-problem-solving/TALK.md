---
title: "Tree of Thoughts: Talk & Q&A"
parent: "055 Tree of Thoughts"
nav_order: 2
---
# TALK — Tree of Thoughts: Deliberate Problem Solving with Large Language Models (Tree of Thoughts)

## Talks, Slides and Press Coverage

1. **Princeton NLP GitHub — "Tree of Thoughts (ToT)" (Shunyu Yao et al., 2023):** The official repository contains all prompts, code, and example trajectories for the three tasks (Game of 24, Creative Writing, Mini Crosswords). ([link](https://github.com/princeton-nlp/tree-of-thought-llm))

2. **NeurIPS 2023 — "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (poster/oral):** The paper was accepted to NeurIPS 2023. ([arXiv page](https://arxiv.org/abs/2305.10601))

3. **Prompt Engineering Guide — "Tree of Thoughts (ToT)" (DAIR.AI):** A comprehensive guide to ToT prompting, explaining the framework's thought generation, evaluation, and search components with examples. ([link](https://www.promptingguide.ai/techniques/tot))

4. **GitHub — "princeton-nlp/tree-of-thought-llm" (6,000+ stars):** The official code repository with all prompts, trajectories, and a README explaining the framework, widely shared and forked by the open-source community. ([link](https://github.com/princeton-nlp/tree-of-thought-llm))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2305.10601).

**Q1: Why does tree-structured search help over chain-of-thought prompting?**
A1 (from the paper, Section 1): "LMs are still confined to token-level, left-to-right decision-making processes during inference. This means they can fall short in tasks that require exploration, strategic lookahead, or where initial decisions play a pivotal role." ToT addresses this by maintaining "a tree of thoughts, where each thought is a coherent language sequence that serves as an intermediate step toward problem solving."

**Q2: How does the LM evaluate its own intermediate states?**
A2 (from the paper, Section 3, State Evaluator): "We propose a third alternative, by using the LM to deliberately reason about states. When applicable, such a deliberate heuristic can be more flexible than programmed rules, and more sample-efficient than learned models." The LM evaluates states via "few lookahead simulations... plus commonsense," rating them as sure, likely, or impossible.

**Q3: What search algorithms does ToT use?**
A3 (from the paper, Section 3, Search Algorithm): "We explore two relatively simple search algorithms and leave more advanced ones (e.g. A*, MCTS) for future work." BFS "maintains a set of the b most promising states per step" (used for Game of 24 and Creative Writing, depth ≤ 3), while DFS "explores the most promising state first" and backtracks when the state is deemed impossible (used for Mini Crosswords, depth 5-10).

**Q4: Why does CoT perform so poorly on Game of 24?**
A4 (from the paper, Section 4.1, Error Analysis): "around 60% of CoT samples already failed the task after generating the first step, or equivalently, the first three words (e.g. '4+9'). This highlights the issues with direct left-to-right decoding." The first arithmetic operation determines whether 24 is reachable, so a wrong first step is unrecoverable.

**Q5: Is ToT related to classical AI search methods?**
A5 (from the paper, Section 5, Related Work): "our approach can be treated as a modern rendition of classical search methods for problem solving. For example it can be considered as a heuristic search algorithm like A*, in which the heuristic at each search node is provided by the LM's self-assessment."

**Q6: Does ToT require additional training?**
A6 (from the paper, Section 3, Discussion): "Convenience. No extra training is needed, just a pre-trained LM is sufficient." The framework works entirely with prompting for both generation and evaluation.

**Q7: What are the limitations of ToT?**
A7 (from the paper, Section 6, Discussion): "Deliberate search such as ToT might not be necessary for many existing tasks that GPT-4 already excels at... search methods like ToT requires more resources (e.g. GPT-4 API cost) than sampling methods in order to improve task performances."

## Common Misconceptions

1. **"ToT is just self-consistency with more samples."** — Self-consistency (CoT-SC) samples k independent reasoning chains and majority-votes the final answer. ToT instead performs tree-structured search where intermediate states are evaluated and pruned, with lookahead and backtracking. The paper explicitly distinguishes them (Section 2): CoT-SC has "no local exploration of different thought steps" within each chain, while ToT "actively maintains a tree of thoughts" with systematic search.

2. **"ToT replaces the language model with a search algorithm."** — ToT augments the LM with search; the LM itself provides both the thought generation and the heuristic evaluation. The paper (Section 5) describes it as "a heuristic search algorithm like A*, in which the heuristic at each search node is provided by the LM's self-assessment." The search algorithm is modular and can be swapped.

3. **"ToT works on any task."** — The paper notes (Section 6) that "Deliberate search such as ToT might not be necessary for many existing tasks that GPT-4 already excels at." ToT is designed for tasks "requiring non-trivial planning or search" where initial decisions are pivotal and left-to-right decoding fails.

## Real Citations

1. Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan (2023). "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." *NeurIPS 2023.* [arXiv:2305.10601](https://arxiv.org/abs/2305.10601) — the original paper.

2. Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) — the CoT prompting method that ToT generalizes.

3. Xuezhi Wang, Jason Wei, Dale Schuurmans, et al. (2022). "Self-Consistency Improves Chain of Thought Reasoning in Language Models." [arXiv:2203.11171](https://arxiv.org/abs/2203.11171) — self-consistency baseline that ToT compares against.

4. Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models." [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) — same first author, concurrent work on reasoning + acting.

5. Shibo Hao, Yi Gu, Haodi Ma, et al. (2023). "Reasoning with Language Model is Planning with World Model." [arXiv:2305.14992](https://arxiv.org/abs/2305.14992) — RAP, a concurrent work using MCTS-based search, discussed in the paper's related work.

**Citation count:** about 1,747 citations on [OpenAlex](https://openalex.org/W7133209886) (checked November 2026).
