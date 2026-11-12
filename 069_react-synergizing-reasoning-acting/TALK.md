---
title: "ReAct: Talk & Q&A"
parent: "069 ReAct"
nav_order: 2
---
# TALK — ReAct: Synergizing Reasoning and Acting in Language Models (ReAct)

## Talks, Slides and Press Coverage

1. **ICLR 2023 — Oral presentation:** ReAct was accepted as an oral at ICLR 2023. The paper is listed on the [ICLR 2023 schedule](https://iclr.cc/virtual/2023/calendar). The project page with code is at [react-lm.github.io](https://react-lm.github.io).

2. **Shunyu Yao — "ReAct: Synergizing Reasoning and Acting in Language Models" (invited talks):** Yao has presented ReAct at multiple venues. The [project page](https://react-lm.github.io) links to the [official code repository](https://github.com/ysymyth/ReAct) and slides.

3. **LangChain documentation:** The ReAct pattern is a core agent type in the [LangChain framework](https://python.langchain.com/docs/modules/agents/agent_types/react), which implements the Thought-Action-Observation loop described in the paper.

4. **LlamaIndex — "ReAct Agent":** LlamaIndex implements the ReAct agent pattern as a built-in agent type, documented at [docs.llamaindex.ai](https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/).

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2210.03629).

**Q1: Why interleave reasoning and acting instead of doing them separately?**

A1 (from the paper, Abstract): "We explore the use of LLMs to generate both reasoning traces and task-specific actions in an interleaved manner, allowing for greater synergy between the two: reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with external sources, such as knowledge bases or environments, to gather additional information."

**Q2: How does ReAct reduce hallucination compared to chain-of-thought reasoning?**

A2 (from the paper, Section 1): "On question answering (HotpotQA) and fact verification (Fever), ReAct overcomes prevalent issues of hallucination and error propagation in chain-of-thought reasoning by interacting with a simple Wikipedia API." The paper's error analysis (Table 2) shows CoT has 14% hallucinated reasoning traces or facts, while ReAct has only 6%.

**Q3: How many in-context examples does ReAct need?**

A3 (from the paper, Abstract): "ReAct outperforms imitation and reinforcement learning methods by an absolute success rate of 34% and 10% respectively, while being prompted with only one or two in-context examples." For HotpotQA, 6 examples are used; for FEVER, 3; for ALFWorld and WebShop, 1-2.

**Q4: What happens when ReAct is fine-tuned instead of prompted?**

A4 (from the paper, Section 3.3): "PaLM-8B finetuned ReAct outperforming all PaLM-62B prompting methods, and PaLM-62B finetuned ReAct outperforming all 540B prompting methods." The paper notes that "finetuning Standard or CoT is significantly worse than finetuning ReAct or Act for both PaLM-8/62B, as the former essentially teaches models to memorize (potentially hallucinated) knowledge facts."

**Q5: Does ReAct always outperform chain-of-thought?**

A5 (from the paper, Section 3.3): "ReAct outperforms CoT on Fever (60.9 vs. 56.3) and slightly lags behind CoT on HotpotQA (27.4 vs. 29.4)." The paper proposes a hybrid approach: "CoT-SC → ReAct" and "ReAct → CoT-SC" that combines both, with "ReAct → CoT-SC" reaching 35.1 EM on HotpotQA — the best prompting result.

**Q6: What are the main failure modes of ReAct?**

A6 (from the paper, Table 2): The most common failure mode is "Reasoning error: Wrong reasoning trace (including failing to recover from repetitive steps)" at 47% for ReAct, followed by "Search result error: Search return empty or does not contain useful information" at 23%.

## Common Misconceptions

1. **"ReAct requires fine-tuning the language model."** — No. The paper's main results use a frozen PaLM-540B with few-shot in-context prompting (1-6 examples). Fine-tuning results are shown as an additional experiment, not the primary method.

2. **"ReAct replaces chain-of-thought reasoning."** — ReAct *combines* reasoning with acting. The paper shows that neither pure reasoning (CoT) nor pure acting (Act) alone is as good as interleaving both. The hybrid "ReAct → CoT-SC" approach achieves the best prompting results, showing they are complementary.

3. **"ReAct can only work with Wikipedia search."** — The paper demonstrates ReAct on four diverse tasks: HotpotQA and FEVER (Wikipedia API), ALFWorld (text-based household environment), and WebShop (online shopping website). The action space is task-specific but the Thought-Action-Observation pattern is general.

4. **"ReAct eliminates hallucination entirely."** — ReAct reduces hallucination (6% vs 14% for CoT on HotpotQA, per Table 2), but does not eliminate it. The paper notes that ReAct still suffers from reasoning errors (47% of failures) and search result errors (23%).

## Real Citations

1. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR 2023.* [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) — the original paper.

2. Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *NeurIPS 2022.* [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) — CoT prompting, a key baseline and component of ReAct.

3. Shinn, N., et al. (2023). "Reflexion: Language Agents with Verbal Reinforcement Learning." *NeurIPS 2023.* [arXiv:2303.11366](https://arxiv.org/abs/2303.11366) — extends ReAct with verbal self-reflection.

4. Schick, T., et al. (2023). "Toolformer: Language Models Can Teach Themselves to Use Tools." *NeurIPS 2023.* [arXiv:2302.04761](https://arxiv.org/abs/2302.04761) — trains models to self-label tool calls, related to ReAct's acting component.

5. Wang, X., et al. (2022). "Self-Consistency Improves Chain of Thought Reasoning in Language Models." *ICLR 2023.* [arXiv:2203.11171](https://arxiv.org/abs/2203.11171) — CoT-SC, used in the hybrid ReAct → CoT-SC approach.

**Citation count:** about 6,700 citations on [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/arXiv:2210.03629) (checked November 2026).
