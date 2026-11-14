---
title: "Voyager: Talk & Q&A"
parent: "073 Voyager"
nav_order: 2
---

# TALK -- Voyager: An Open-Ended Embodied Agent with Large Language Models (Voyager)

## Talks, Slides and Press Coverage

1. **Official project page:** The authors maintain a project website with demos and videos at [voyager.minedojo.org](https://voyager.minedojo.org/), including visualisations of the agent's exploration progress, skill library growth, and tech tree mastery.

2. **Official code repository:** The full codebase and prompts are open-sourced at [github.com/MineDojo/Voyager](https://github.com/MineDojo/Voyager), including the automatic curriculum, skill library, iterative prompting mechanism, and evaluation scripts.

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2305.16291).

**Q1: Why does Voyager use code as the action space instead of natural language actions?**

A1 (from the paper, Section 1): "We opt to use code as the action space instead of low-level motor commands because programs can naturally represent temporally extended and compositional actions, which are essential for many long-horizon tasks in Minecraft."

**Q2: How does the automatic curriculum decide what task to propose next?**

A2 (from the paper, Section 2.1): "Our automatic curriculum capitalizes on the internet-scale knowledge contained within GPT-4 by prompting it to provide a steady stream of new tasks or challenges. The curriculum unfolds in a bottom-up fashion, allowing for considerable adaptability and responsiveness to the exploration progress and the agent's current state." The prompt includes directives, the agent's current state (inventory, equipment, nearby blocks and entities, biome, time, health, hunger, position), previously completed and failed tasks, and GPT-3.5-generated self-asked questions.

**Q3: How does the skill library work?**

A3 (from the paper, Section 2.2): "Each program is indexed by the embedding of its description, which can be retrieved in similar situations in the future. Complex skills can be synthesized by composing simpler programs, which compounds Voyager's capabilities rapidly over time and alleviates catastrophic forgetting in other continual learning methods." For retrieval, "we query the skill library with the embedding of self-generated task plans and environment feedback."

**Q4: What happens when the code generation fails?**

A4 (from the paper, Section 2.3): "If the agent gets stuck after 4 rounds of code generation, then we query the curriculum for another task. This iterative prompting approach significantly improves program synthesis quality." The three types of feedback are environment feedback, execution errors, and self-verification.

**Q5: How does self-verification differ from self-reflection (Reflexion)?**

A5 (from the paper, Section 2.3): "Our self-verification is more comprehensive than self-reflection by both checking success and reflecting on mistakes." The verification agent provides both a success/failure judgment and, if failed, a critique suggesting how to complete the task.

**Q6: Why does Voyager use GPT-4 instead of GPT-3.5 for code generation?**

A6 (from the paper, Section 3.4): "GPT-4 significantly outperforms GPT-3.5 in code generation and obtains 5.7x more unique items, as GPT-4 exhibits a quantum leap in coding abilities." The paper also notes (Section 4) that "the GPT-4 API incurs significant costs. It is 15x more expensive than GPT-3.5."

**Q7: Can the skill library be used by other methods?**

A7 (from the paper, Section 3.3): "Our skill library constructed from lifelong learning not only enhances Voyager's performance but also gives a boost to AutoGPT. This demonstrates that the skill library serves as a versatile tool that can be readily employed by other methods, effectively acting as a plug-and-play asset to enhance performance."

**Q8: What are the main limitations?**

A8 (from the paper, Section 4): Three limitations: (1) Cost -- "the GPT-4 API incurs significant costs"; (2) Inaccuracies -- "there are still cases where the agent gets stuck and fails to generate the correct skill"; (3) Hallucinations -- "The automatic curriculum occasionally proposes unachievable tasks" and "GPT-4 tends to use cobblestone as a fuel input, despite being an invalid fuel source in the game."

## Common Misconceptions

1. **"Voyager learns by fine-tuning GPT-4."** -- No. The paper explicitly states (Abstract): "Voyager interacts with GPT-4 via blackbox queries, which bypasses the need for model parameter fine-tuning." All learning is in the skill library (stored programs), not in the model weights.

2. **"Voyager plays Minecraft from pixel input."** -- No. The paper states (Section 3.2): "we rely on the high-level Mineflayer API to control the agent" and "Our work's focus is on pushing the limits of GPT-4 for lifelong embodied agent learning, rather than solving the 3D perception or sensorimotor control problems." The agent receives text-based state information, not screen pixels.

3. **"The skill library is just a cache of previous outputs."** -- Not quite. The skill library stores verified, executable programs indexed by description embeddings. These are retrieved as in-context examples for future tasks, enabling composition (Section 2.2): "Complex skills can be synthesized by composing simpler programs." It is an active retrieval mechanism, not a passive cache.

4. **"Voyager can see and understand the Minecraft world visually."** -- No. The paper notes (Section 3.5): "Voyager does not currently support visual perception, because the available version of GPT-4 API is text-only at the time of this writing." The agent operates entirely on text-based state descriptions.

## Real Citations

1. Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., & Anandkumar, A. (2023). "Voyager: An Open-Ended Embodied Agent with Large Language Models." *NeurIPS 2023.* [arXiv:2305.16291](https://arxiv.org/abs/2305.16291) -- the original paper.

2. Yao, S., et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR 2023.* [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) -- ReAct, a baseline that uses chain-of-thought prompting for reasoning and acting. Voyager outperforms it significantly.

3. Shinn, N., et al. (2023). "Reflexion: Language Agents with Verbal Reinforcement Learning." *NeurIPS 2023.* [arXiv:2303.11366](https://arxiv.org/abs/2303.11366) -- Reflexion, a baseline built on ReAct with self-reflection. Voyager's self-verification is described as "more comprehensive than self-reflection."

4. Fan, L., et al. (2023). "MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge." *NeurIPS 2022 Datasets and Benchmarks.* [arXiv:2206.11383](https://arxiv.org/abs/2206.11383) -- MineDojo, the Minecraft AI framework that Voyager is built on top of.

5. Lin, Z., et al. (2023). "DEPS: Delivering Open-Endedness in Large Language Models by Decoupling Execution and Planning." [arXiv:2310.19742](https://arxiv.org/abs/2310.19742) -- DEPS, a follow-up that extends LLM-based Minecraft agents with a decoupled execution-planning pipeline.

**Citation count:** about 2,775 citations on [OpenAlex](https://api.openalex.org/works/https://arxiv.org/abs/2305.16291) (checked November 2026).
