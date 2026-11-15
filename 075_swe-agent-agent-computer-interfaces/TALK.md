---
title: "SWE-agent: Talk & Q&A"
parent: "075 SWE-agent"
nav_order: 2
---
# TALK — SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering (SWE-agent)

## Talks, Slides and Press Coverage

1. **NeurIPS 2024 — "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering" (Yang et al., 2024):** The paper was accepted at NeurIPS 2024. The authors presented the ACI concept, the SWE-agent system, and the ablation results. ([conference page](https://neurips.cc/virtual/2024))

2. **Princeton NLP — SWE-agent GitHub repository:** The official open-source repository hosts the ACI implementation, evaluation scripts, and configuration files. It includes the full set of ACI commands and demonstration trajectories. ([repository](https://github.com/princeton-nlp/SWE-agent))

3. **SWE-agent — Official project site:** The project page at swe-agent.com provides an overview of the system, the ACI concept, and links to the live leaderboard, data, and code. ([project page](https://swe-agent.com/latest/))

4. **John Yang — "SWE-agent" talk at Princeton NLP:** The authors presented the system and demonstrated the ACI in action, showing how interface design impacts agent performance on real software engineering tasks. ([paper page](https://arxiv.org/abs/2405.15793))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2405.15793).

**Q1: Why did the authors design a custom interface instead of letting the LM use the Linux shell directly?**
A1 (from the paper, Section 1): "In practice, we find that LM agents can struggle to reliably take actions in this environment. For example, it fails to provide simple commands to edit a small file segment, and does not provide any feedback if the user makes an invalid edit. These deficits substantially hamper performance, motivating the need for an agent-computer interface (ACI)."

**Q2: What are the key design principles for an effective ACI?**
A2 (from the paper, Section 2): The authors identify four principles: "(1) Actions should be simple and easy to understand for agents"; "(2) Actions should be compact and efficient"; "(3) Environment feedback should be informative but concise"; "(4) Guardrails mitigate error propagation and hasten recovery."

**Q3: How much does the ACI improve performance over the shell-only baseline?**
A3 (from the paper, Section 5, Table 1): On SWE-bench Lite, SWE-agent w/ GPT-4 Turbo achieves 18.00% (54/300) compared to Shell-only w/ GPT-4 Turbo at 11.00% — "SWE-agent yields strong performance... a 64% relative increase compared to Shell-only."

**Q4: What happens when the authors remove the linting guardrail from the edit command?**
A4 (from the paper, Section 5.1, Table 3): Without linting, performance drops to 15.0% (from 18.0%, a 3.0 percentage-point decrease). The paper notes: "A prominent failure mode occurs when models repeatedly edit the same code snippet. The usual suspect for this behavior is an agent introducing a syntax error (e.g., incorrect indentation, extra parenthesis) via an errant edit."

**Q5: Is the ACI portable to other language models beyond GPT-4 Turbo?**
A5 (from the paper, Section 5, Table 1): Yes. "SWE-agent with Claude 3 Opus can solve 10.5% of the benchmark tasks" on the full SWE-bench test set, compared to 12.47% with GPT-4 Turbo. The ACI was developed for GPT-4 Turbo but transfers to Claude 3 Opus without modification.

**Q6: What are the main failure modes of SWE-agent?**
A6 (from the paper, Section 5.2): "about half (52.0%) of unresolved instances fall into the Incorrect Implementation or Overly Specific Implementation categories, suggesting that agents' proposed solutions often simply fail to functionally address the issue or are insufficiently general solutions. Cascading failed edits make up another 23.4% of failures."

**Q7: Does giving the agent more context (more file content) help?**
A7 (from the paper, Section 5.1, Table 3): No — showing too much content hurts. "Either too little content (30 lines, 14.3%) or too much (entire file, 12.7%) lowers performance." The 100-line window achieves the best result at 18.0%.

## Common Misconceptions

1. **"SWE-agent is just a prompt engineering trick — it doesn't change the model."** — This is partially the point. The paper's contribution is precisely that interface design matters as much as model capability: "we shape the actions, their documentation, and environment feedback to complement an LM's limitations and abilities" (Section 2) without modifying the underlying LM's weights. The 64% relative improvement over shell-only comes entirely from the ACI, not from a better model.

2. **"The ACI is just a set of bash scripts."** — The ACI is built atop the Linux shell but provides custom commands with guardrails, structured feedback, and context management that raw bash lacks. The paper's ablation (Table 3) shows that each component (edit with linting, summarized search, 100-line window, history processing) contributes measurably to performance.

3. **"SWE-agent only works with GPT-4."** — The paper explicitly tests portability: "SWE-agent with Claude 3 Opus can solve 10.5% of the benchmark tasks" (Section 5, Table 1). They also tried Llama 3 and DeepSeek Coder but "found their performance in the agent setting to be subpar," largely due to insufficient context windows (e.g., Llama 3's 8k tokens).

## Real Citations

1. John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, Ofir Press (2024). "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering." *NeurIPS 2024.* [arXiv:2405.15793](https://arxiv.org/abs/2405.15793) — the original paper.

2. Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, Karthik Narasimhan (2024). "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" *ICLR 2024 (Outstanding Paper).* [arXiv:2310.06770](https://arxiv.org/abs/2310.06770) — the benchmark SWE-agent is evaluated on, by the same group.

3. John Yang et al. (2025). "SWE-smith: Scaling Data for Software Engineering Agents." *NeurIPS 2025.* [arXiv:2504.21798](https://arxiv.org/abs/2504.21798) — a follow-up that scales training data for SWE-agent-style systems, training SWE-agent-LM-32B to 40.2% on SWE-bench Verified.

**Citation count:** about 1,534 citations on [OpenAlex](https://openalex.org/W7103754783) (checked November 2026).
