---
title: "073 Voyager"
nav_order: 74
permalink: "/073_voyager-open-ended-embodied-agent/"
---
# Voyager: An Open-Ended Embodied Agent with Large Language Models (Voyager)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/voyager-open-ended-embodied-agent)

- **Paper:** [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291)
- **Authors:** Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi "Jim" Fan, Anima Anandkumar
- **Lab, year:** NVIDIA (Wang: Caltech, work done at NVIDIA; Xie and Zhu: UT Austin; Jiang: Stanford; Xiao: UW Madison), 2023 (NeurIPS 2023)

## Summary

Voyager is the first LLM-powered embodied lifelong learning agent that plays Minecraft autonomously. It explores the world, gathers resources, crafts tools, and discovers new items without any human intervention. The key insight is that instead of fine-tuning a model or using low-level motor commands, Voyager uses GPT-4 through prompting (black-box queries) to generate executable JavaScript code as its action space. This code represents temporally extended, compositional behaviors -- for example, a function that walks to a forest, chops trees, and crafts planks.

The agent has three components working together: an **automatic curriculum** that proposes suitable next tasks based on the agent's current state (inventory, health, nearby entities), an **ever-growing skill library** that stores verified action programs indexed by their description embeddings for future retrieval, and an **iterative prompting mechanism** that generates code, executes it, collects environment feedback and execution errors, and refines the program over multiple rounds until a **self-verification** module confirms task completion. Once verified, the skill is committed to the library, compounding the agent's capabilities over time.

Empirically, Voyager obtains 3.3x more unique items, travels 2.3x longer distances, and unlocks key tech tree milestones up to 15.3x faster than prior methods (ReAct, Reflexion, AutoGPT). It is the only agent to unlock the diamond tool level. In zero-shot generalization tests on a new world, Voyager solves all four unseen tasks, while baselines solve none.

This repo's notebook builds a skill-library agent in a text-based crafting world with a neural curriculum, embedding-based skill retrieval, and an iterative code generation loop, demonstrating how the skill library compounds capabilities and the curriculum improves task selection.

## What Problem Does It Solve

Imagine dropping into a vast Minecraft world with no instructions. You need to figure out what to do next: should you chop wood, mine stone, or hunt for food? And once you decide, you need to work out the steps -- to make an iron pickaxe you need iron ingots, which need a furnace, which needs stone, which needs a wooden pickaxe to mine. A human player learns these skills incrementally and reuses them. But LLM-based agents before Voyager would attempt tasks in isolation, forget what they learned, and get stuck.

The paper explains it with its own comparison to how human players progress: "Human players typically start by learning the basics, such as mining wood and cooking food, before advancing to more complex tasks like combating monsters and crafting diamond tools." The problem is making an AI agent do the same -- progressively acquire, store, and reuse skills without human help.

## Core Idea

- **Automatic curriculum:** GPT-4 proposes a steady stream of new tasks based on the agent's current state and exploration progress, driven by the overarching goal of "discovering as many diverse things as possible." This is an in-context form of novelty search.
- **Skill library:** Each successfully verified skill is stored as executable code, indexed by the embedding of its description. When facing a new task, the agent retrieves the top-5 most relevant skills as in-context examples, enabling GPT-4 to compose and reuse them. This compounds capabilities and alleviates catastrophic forgetting.
- **Iterative prompting mechanism:** GPT-4 generates code, the agent executes it in Minecraft, and three types of feedback are incorporated: environment feedback (e.g., "I cannot make stick because I need 2 more planks"), execution errors (e.g., "No item named acacia axe"), and self-verification (a separate GPT-4 agent that checks if the task is complete). If the agent gets stuck after 4 rounds, the curriculum proposes a new task.

## Key Method Details

- **LLM backbone:** gpt-4-0314 for code generation, curriculum, and self-verification; gpt-3.5-turbo-0301 for skill descriptions and self-asked questions; text-embedding-ada-002 for skill embeddings.
- **Temperature:** 0 for all components except the automatic curriculum (0.1, to encourage task diversity).
- **Environment:** MineDojo (an open-source Minecraft AI framework) with Mineflayer JavaScript APIs for motor controls.
- **Code as action space:** Programs represent temporally extended and compositional actions, essential for long-horizon tasks.
- **Skill retrieval:** Top-5 skills by cosine similarity of description embeddings. The query combines a GPT-3.5-generated task plan with environment feedback.
- **Self-verification:** A separate GPT-4 agent checks if the task is complete based on the agent's current state. If failed, it provides a critique suggesting how to complete the task.
- **Results (exploration):** 63 unique items in 160 prompting iterations, 3.3x more than baselines.
- **Results (tech tree mastery):** Wooden 15.3x faster, stone 8.5x faster, iron 6.4x faster than AutoGPT (the best baseline). Only Voyager unlocks the diamond level (102 iterations, 1 of 3 trials).
- **Results (zero-shot generalization):** Voyager solves all 4 unseen tasks (Diamond Pickaxe, Golden Sword, Lava Bucket, Compass) in 18-21 iterations. All baselines fail (0/3 on every task).

### Simplifications in This Notebook

- Text-based crafting world with 4 locations and 10 recipes instead of Minecraft (MineDojo + Mineflayer)
- Template-based code generation instead of GPT-4 API calls
- Small neural curriculum network (2,190 parameters) instead of GPT-4 curriculum prompting
- Small text embedding network (1,920 parameters) instead of OpenAI text-embedding-ada-002
- 14 predefined tasks along a tech tree instead of open-ended Minecraft exploration
- 20 prompting iterations instead of 160
- Order-based curriculum (next uncompleted task in tech-tree order) instead of GPT-4's state-aware curriculum

### What the Notebook Result Shows

Over 3 seeds with 20 prompting iterations each, Voyager discovers 13 unique items (out of 14 possible) and completes 13 tasks, storing 13 reusable skills in its library. It reaches the wooden, stone, and iron tech tree levels. The no-skill-library variant also completes 13 tasks thanks to the iterative refinement loop (up to 4 rounds of error feedback), but stores no skills for future reuse. The random baseline (no curriculum, no skill library, no verification) completes only 12 tasks. These toy numbers are not comparable to the paper's 3.3x item advantage -- a text world with 14 predefined tasks and template-based code generation cannot match GPT-4-driven open-ended exploration in Minecraft.

## Influence

Voyager became one of the most influential agent papers of 2023, demonstrating that LLMs can drive open-ended lifelong learning through code generation and skill libraries without parameter updates. It directly inspired follow-up work on LLM agents in game environments and beyond, including DEPS (Lin et al., 2023) for interactive task solving and JARVIS-1 (Fan et al., 2023) for multimodal Minecraft agents. The skill library concept -- storing verified programs indexed by embeddings -- became a standard pattern in agent frameworks. The paper has been cited about 2,775 times ([OpenAlex](https://api.openalex.org/works/https://arxiv.org/abs/2305.16291), checked November 2026). The authors open-sourced the full codebase and prompts at [github.com/MineDojo/Voyager](https://github.com/MineDojo/Voyager).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview -- a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/073_voyager-open-ended-embodied-agent/solution.ipynb) | From-scratch PyTorch implementation of a Voyager-style skill-library agent in a text-based crafting world, with curriculum network, embedding-based skill retrieval, and ablation study (13 items discovered, 13 skills stored, reaching iron tech tree level) |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook -- what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
