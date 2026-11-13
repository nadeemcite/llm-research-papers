---
title: "Reflexion: Talk & Q&A"
parent: "071 Reflexion"
nav_order: 2
---

# TALK -- Reflexion: Language Agents with Verbal Reinforcement Learning (Reflexion)

## Talks, Slides and Press Coverage

1. **NeurIPS 2023 -- Conference presentation:** Reflexion was accepted at NeurIPS 2023. The proceedings are indexed at [papers.nips.cc/paper_files/paper/2023](https://papers.nips.cc/paper_files/paper/2023). The official code repository is at [github.com/noahshinn/reflexion](https://github.com/noahshinn/reflexion).

2. **Noah Shinn -- "Reflexion: Language Agents with Verbal Reinforcement Learning" (invited talks):** Shinn has presented Reflexion at multiple venues. The [GitHub repository](https://github.com/noahshinn/reflexion) contains the full code, demos, and links to presentation materials.

3. **LangChain documentation:** The Reflexion pattern is implemented as an agent type in the [LangChain framework](https://python.langchain.com/docs/modules/agents/), supporting self-reflection loops for tool-using agents.

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2303.11366).

**Q1: How is Reflexion different from traditional reinforcement learning?**

A1 (from the paper, Section 1): "We propose Reflexion, a novel framework to reinforce language agents not by updating weights, but instead through linguistic feedback. Concretely, Reflexion agents verbally reflect on task feedback signals, then maintain their own reflective text in an episodic memory buffer to induce better decision-making in subsequent trials." The paper lists four advantages: 1) lightweight (no fine-tuning), 2) more nuanced feedback than scalar rewards, 3) explicit and interpretable episodic memory, 4) concrete hints for future actions.

**Q2: Why does Reflexion achieve 91% on HumanEval -- higher than GPT-4's 80%?**

A2 (from the paper, Table 1): Reflexion uses GPT-4 as the base model and adds the self-reflection loop. The 91% pass@1 comes from iterative test-driven development: the agent generates code, writes its own unit tests (up to 6, filtered by AST validity), runs them, and if tests fail, reflects on the errors and rewrites. The baseline GPT-4 (80%) is a single-shot generation without self-reflection.

**Q3: What happens when the self-generated unit tests are wrong?**

A3 (from the paper, Section 4.3, Analysis): "If the model produces an incorrectly written test suite, it is possible for some of the tests to fail on a correct solution, leading to a self-reflection generation that is conditioned on a false negative code completion." The paper notes that false negatives are preferred over false positives because the agent can use self-reflection to identify incorrect tests. On MBPP Python, Reflexion underperforms the baseline (77.1% vs 80.1%) because of a 16.3% false-positive test execution rate, compared to only 1.4% on HumanEval.

**Q4: Does Reflexion require a specific LLM, or can it work with any model?**

A4 (from the paper, Section 3): The framework is modular -- the Actor, Evaluator, and Self-Reflection model can each be different LLMs. The paper primarily uses GPT-3 and GPT-4, with additional experiments in Appendix A using other models. The Actor can be built on either ReAct or Chain-of-Thought, making it flexible across base agent architectures.

**Q5: How many reflections does the agent keep in memory?**

A5 (from the paper, Section 4.1): "To avoid very long prompt windows that may exceed the maximum limit, we keep the agent's memory to the last 3 self-reflections (experiences)." For programming tasks, the memory limit is 1 experience (Section 4.3). The paper suggests future work could extend this with "vector embedding databases or traditional SQL databases" (Section 5).

**Q6: What is LeetcodeHardGym?**

A6 (from the paper, Section 4.3): "We introduce LeetcodeHardGym, an interactive programming gym that contains 40 Leetcode hard-rated questions that have been released after October 8, 2022, which is the pre-training cutoff date of GPT-4." Reflexion achieves 15% pass@1 on this benchmark, compared to GPT-4's 7.5%.

## Common Misconceptions

1. **"Reflexion fine-tunes the language model."** -- No. The paper explicitly states (Section 1): "it is lightweight and doesn't require finetuning the LLM." All learning happens through verbal feedback stored in memory, not weight updates. The LLM parameters are frozen.

2. **"Reflexion is just retrying until it works."** -- No. The baseline (retrying without reflection) does not improve: "ReAct-only, CoT-only, and CoT (GT)-only implementations fail to probabilistically improve on any tasks, meaning that no failed tasks from the first trial from any of the baseline approaches were able to be solved in subsequent trials" (Section 4.2). Reflexion's improvement comes specifically from the verbal self-reflection step, not from retrying alone.

3. **"Reflexion only works on coding tasks."** -- The paper demonstrates Reflexion across three domains: sequential decision-making (ALFWorld, +22%), reasoning (HotpotQA, +20%), and programming (HumanEval, +11%). The framework is task-agnostic -- only the Evaluator and reflection prompts change.

4. **"Self-reflection alone is enough -- test generation is unnecessary."** -- The ablation study (Table 3) shows that omitting test generation (self-reflection only) actually *hurts* performance: 52% vs 60% baseline on HumanEval Rust. The paper states: "the agent is unable to determine if the current implementation is correct without unit tests" and performs "harmful edits."

## Real Citations

1. Shinn, N., Cassano, F., Berman, E., Gopinath, A., Narasimhan, K., & Yao, S. (2023). "Reflexion: Language Agents with Verbal Reinforcement Learning." *NeurIPS 2023.* [arXiv:2303.11366](https://arxiv.org/abs/2303.11366) -- the original paper.

2. Yao, S., et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR 2023.* [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) -- ReAct, the action generator used as the Actor in Reflexion's decision-making experiments.

3. Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *NeurIPS 2022.* [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) -- CoT, used as the Actor in Reflexion's reasoning experiments.

4. Madaan, A., et al. (2023). "Self-Refine: Iterative Refinement with Self-Feedback." *NeurIPS 2023.* [arXiv:2303.17651](https://arxiv.org/abs/2303.17651) -- Self-Refine, a related approach limited to single-generation refinement that Reflexion extends with persistent memory.

5. Chen, M., et al. (2021). "Evaluating Large Language Models Trained on Code." [arXiv:2107.03374](https://arxiv.org/abs/2107.03374) -- HumanEval, the primary coding benchmark used in Reflexion's programming experiments.

**Citation count:** about 3,300 citations on [OpenAlex](https://api.openalex.org/works/https://arxiv.org/abs/2303.11366) (checked November 2026).
