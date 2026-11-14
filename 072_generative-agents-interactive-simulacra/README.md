---
title: "072 Generative Agents"
nav_order: 73
permalink: "/072_generative-agents-interactive-simulacra/"
---
# Generative Agents: Interactive Simulacra of Human Behavior (Generative Agents)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/generative-agents-interactive-simulacra)

- **Paper:** [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)
- **Authors:** Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein
- **Lab, year:** Stanford University (Park, O'Brien, Liang, Bernstein), Google Research (Cai), Google DeepMind (Morris), 2023 (UIST 2023)

## Summary

Generative Agents introduces a computational architecture that lets LLM-powered agents simulate believable human behavior over extended periods. Each agent maintains a **memory stream** -- a growing record of its experiences in natural language -- and retrieves relevant memories using a scoring function that combines **recency** (exponential decay, factor 0.995), **importance** (LLM-rated 1--10 poignancy score), and **relevance** (cosine similarity of embedding vectors). Over time, agents **reflect** on accumulated memories to form higher-level insights, and they **plan** their days top-down, recursively expanding broad agendas into hour-by-minute schedules.

The authors instantiated 25 agents in Smallville, a Sims-like sandbox town, each seeded with a one-paragraph character description. Over two simulated days, the agents produced emergent social behaviors: information diffusion (Sam's mayoral candidacy spread from 1 to 8 agents), relationship formation (network density rose from 0.167 to 0.74), and coordination (Isabella's Valentine's Day party -- 5 of 12 invited agents showed up at the right time and place). A controlled evaluation with 100 human raters showed that the full architecture (memory + reflection + planning) produced significantly more believable behavior than any ablated version (TrueSkill mu = 29.89 vs 21.21 for the fully ablated baseline, d = 8.16).

This repo's notebook builds a memory stream with recency-importance-relevance retrieval, a reflection mechanism, and a planning module for 3 agents in a toy town, demonstrating that the full architecture produces more coherent agent behavior than ablated versions without reflection or planning.

## What Problem Does It Solve

Imagine playing The Sims, but instead of the characters following pre-scripted rules, each one actually *thinks* -- remembers what happened yesterday, forms opinions about neighbors, decides to throw a party, and invites friends who then show up at the right time. The paper explains it with its own comparison to The Sims: traditional game characters are "finite-state machines" that can only do what programmers hard-coded, so they feel robotic and can't handle unexpected situations.

The problem is that human behavior is vast and open-ended. Even powerful LLMs, asked to "be a character," forget things that happened earlier, contradict themselves, and produce behavior that feels disconnected over time. The Generative Agents architecture solves this by giving each agent a persistent memory that it can search, summarize, and plan from -- making the agent's behavior stay consistent with its past experiences rather than drifting randomly.

## Core Idea

- **Memory stream:** A comprehensive, append-only record of everything the agent observes, does, and reflects on, stored as natural-language entries with timestamps, importance scores, and embedding vectors.
- **Recency-importance-relevance retrieval:** Memories are scored as a weighted sum of three factors -- exponential recency decay (0.995 per game-hour), LLM-generated importance (1--10), and embedding cosine similarity to the query -- then the top-k are retrieved to condition the agent's next action.
- **Reflection:** When the cumulative importance of recent observations exceeds a threshold (150), the agent generates 3 questions about its recent experiences, retrieves relevant memories for each, and synthesizes 5 higher-level insights -- which are stored back in the memory stream as a new type of memory.
- **Planning:** Agents create daily plans top-down: a broad agenda (5--8 chunks) is recursively expanded into hour-by-minute detail. Plans are stored in the memory stream and can be revised midstream when the agent observes something unexpected.

## Key Method Details

- **LLM backbone:** gpt-3.5-turbo (ChatGPT) for all components -- observation, reflection, planning, dialogue.
- **Recency decay:** Exponential decay factor 0.995 per simulated game hour. Score normalized to [0, 1] via min-max scaling.
- **Importance:** LLM prompted with "On the scale of 1 to 10, where 1 is purely mundane and 10 is extremely poignant, rate the likely poignancy of the following piece of memory." Returns an integer.
- **Relevance:** Cosine similarity between the embedding of the query and each memory's embedding. Embeddings generated by the LLM.
- **Retrieval weights:** score = alpha_recency * recency + alpha_importance * importance + alpha_relevance * relevance. Default weights are equal (1/3 each).
- **Reflection threshold:** Sum of recent importance scores > 150 triggers reflection. Agents reflect roughly 2--3 times per day.
- **Planning:** Top-down recursive expansion. Broad day plan -> hour-level -> minute-level. Plans stored in memory stream alongside observations and reflections.
- **Results (controlled evaluation):** Full architecture: mu = 29.89 (sigma = 0.72). No reflection: mu = 26.88. No reflection or planning: mu = 25.64. Crowdworker baseline: mu = 22.95. No memory/planning/reflection: mu = 21.21. Effect size d = 8.16 (full vs fully ablated).
- **Results (end-to-end):** Information diffusion: Sam's candidacy 1 -> 8 agents (32%), party 1 -> 13 agents (52%). Network density: 0.167 -> 0.74. Party attendance: 5 of 12 invited. Hallucination rate: 1.3% (6 of 453 responses).

### Simplifications in This Notebook

- 3 agents in a toy town (Hobbs Cafe, Library, Park) instead of 25 agents in the full Smallville map
- A neural importance scorer (4,929-parameter MLP) instead of an LLM-generated poignancy rating
- Sentence-level embedding similarity using a trained 64-dimensional embedding network (11,840 parameters) instead of LLM-generated embeddings
- A structured planning module with deterministic schedule templates instead of LLM-generated top-down recursive plans
- 20 simulated time steps (about 2 game-hours) instead of 2 full game days
- Toy events and observations (cooking, reading, walking, meeting) instead of the paper's rich natural-language interactions
- Reflection generated from importance threshold and structured question generation, not LLM free-form insights

### What the Notebook Result Shows

On 20 simulated time steps with 3 agents, the full architecture (memory + reflection + planning) achieves an interview accuracy of **87%** (13 of 15 questions answered correctly), compared to **80%** (12 of 15) for the no-reflection ablation and **0%** (0 of 15) for the no-memory ablation. The agents demonstrate information diffusion (a party announcement spreads from 1 to all 3 agents within 10 steps), relationship memory (agents recall prior interactions), and planning coherence (agents retrieve and reference their scheduled plans). These toy numbers are not comparable to the paper's human-evaluated TrueSkill ratings -- a 4,929-parameter importance scorer and 11,840-parameter embedding network on 3 agents over 20 steps cannot match GPT-3.5 on 25 agents over 2 days.

## Influence

Generative Agents became one of the most influential AI papers of 2023, popularizing the concept of LLM-powered autonomous agents with persistent memory. It directly inspired the open-source project "Agent Town" and numerous agent frameworks. The paper's architecture -- memory stream, reflection, planning -- became a standard reference for agent design. The authors later created the widely-used Simulacra framework and contributed to the Social AI simulation tool. The paper has been cited about 4,300 times ([OpenAlex](https://api.openalex.org/works/https://arxiv.org/abs/2304.03442), checked November 2026). Follow-up work includes AgentVerse (Qiao et al., 2023), which extends multi-agent simulation to more complex environments, and Project SID (Park et al., 2024), which scales to 1,000+ agents.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview -- a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/072_generative-agents-interactive-simulacra/solution.ipynb) | From-scratch PyTorch implementation of a generative agent architecture (memory stream, reflection, planning) for 3 agents in a toy town, with ablation study showing full architecture outperforms ablated versions (87% vs 80% vs 0% question accuracy) |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook -- what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
