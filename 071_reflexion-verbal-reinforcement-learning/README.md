---
title: "071 Reflexion"
nav_order: 72
permalink: "/071_reflexion-verbal-reinforcement-learning/"
---
# Reflexion: Language Agents with Verbal Reinforcement Learning (Reflexion)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/reflexion-verbal-reinforcement-learning)

- **Paper:** [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- **Authors:** Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao
- **Lab, year:** Northeastern University (Shinn, Cassano, Berman), MIT (Gopinath), Princeton University (Narasimhan, Yao), 2023 (NeurIPS 2023)

## Summary

Reflexion introduces a way for language-model agents to learn from their mistakes **without updating any weights**. Instead of gradient-based reinforcement learning -- which requires thousands of training samples and expensive fine-tuning -- Reflexion uses *verbal* reinforcement: after each failed attempt, the agent writes a short natural-language critique of what went wrong and stores it in an episodic memory buffer. On the next attempt, it reads these reflections as additional context, guiding it toward better decisions. The paper calls this a "semantic gradient signal" -- instead of a scalar reward, the agent gets a concrete, human-readable description of how to improve.

The framework has three components: an **Actor** (an LLM that generates actions, built on ReAct or Chain-of-Thought), an **Evaluator** (which scores the trajectory -- either with exact-match grading, heuristics, or self-generated unit tests), and a **Self-Reflection model** (which converts the evaluation signal into a verbal critique). The memory buffer is bounded (typically 1--3 reflections) to stay within context limits.

Across three task domains, Reflexion delivers large gains. On **ALFWorld** (134 text-based household tasks), ReAct + Reflexion solves 130/134 tasks (97%), a 22% absolute improvement over ReAct alone, learning over 12 iterative trials. On **HotpotQA** (multi-hop question answering), Reflexion + CoT improves by 20% over the CoT baseline across 100 questions. On **HumanEval** (Python code generation), Reflexion with GPT-4 achieves 91% pass@1, surpassing GPT-4's 80% baseline and setting a new state-of-the-art. The paper also introduces **LeetcodeHardGym**, a benchmark of 40 hard LeetCode problems.

This repo's notebook builds a Reflexion coding agent from scratch with a neural error classifier as the self-reflection model, and shows that verbal reflection in memory outperforms memory-less retrying on 10 toy programming tasks.

## What Problem Does It Solve

Imagine you're learning to code by solving programming challenges. On your first attempt, you write a function that fails the tests. A traditional reinforcement-learning approach would need thousands of attempts and weight updates to slowly improve. But as a human, you do something much faster: you look at the error message, realize "oh, I used subtraction instead of addition," and fix it on your next try. You *remember* your mistake in words, not in modified neural weights.

That's exactly what Reflexion does. The agent tries a task, fails, then *verbally reflects* on what went wrong -- "The function uses subtraction instead of addition. Should use + operator." -- and stores that reflection in memory. Next time, it reads its past reflections before trying again, avoiding the same trap. This is lightweight (no fine-tuning), interpretable (you can read the reflections), and fast (improvement in a handful of trials).

## Core Idea

- **Verbal reinforcement instead of weight updates:** The agent learns by writing natural-language self-critiques, not by backpropagation. Reflections act as a "semantic gradient" pointing toward improvement.
- **Episodic memory buffer:** Reflections are stored in a bounded memory (last 1--3 entries) that the Actor reads as context on subsequent trials. This is long-term memory; the current trajectory serves as short-term memory.
- **Three-model architecture:** Actor (generates actions), Evaluator (scores the trajectory), Self-Reflection model (converts the score into a verbal critique). These can be the same LLM or different models.

## Key Method Details

- **Actor:** Built on ReAct (for decision-making) or Chain-of-Thought (for reasoning), with an added memory component that provides reflections as context.
- **Evaluator:** Task-specific -- exact-match grading for HotpotQA, heuristic failure detection for ALFWorld (same action repeated 3+ cycles, or >30 actions), self-generated unit tests for programming (up to 6 tests, filtered by AST validity).
- **Self-Reflection model:** Given the trajectory, evaluation signal, and current memory, generates a specific verbal critique. Uses few-shot prompting (2-shot for most tasks).
- **Memory:** Sliding window of last Omega reflections (Omega = 3 for ALFWorld/HotpotQA, 1 for programming).
- **Results:** ALFWorld: 130/134 (97%) with Reflexion vs ~75% ReAct-only, 22% absolute improvement. HotpotQA: 20% improvement over CoT baseline. HumanEval: 91% pass@1 (vs 80% GPT-4 baseline). MBPP Python: 77.1% (vs 80.1% baseline -- the one case where Reflexion underperforms, due to false-positive unit tests).

### Simplifications in This Notebook

- 10 toy programming tasks (add, multiply, reverse string, factorial, etc.) instead of HumanEval's 164 problems
- A neural error classifier MLP (6,808 parameters) instead of GPT-4 as the self-reflection model
- Pre-defined buggy implementations with known bug types instead of free-form code generation
- 50 evaluation runs per task with 5 max trials, instead of the paper's setup with GPT-4
- 50 training epochs with Adam (lr 1e-3) on 120 augmented training examples
- Error features are hand-engineered (16-dimensional vector) instead of natural-language error traces

### What the Notebook Result Shows

On 10 toy programming tasks (50 runs each, 5 max trials), the Reflexion agent achieves **86% overall accuracy**, compared to **76% for the baseline** (no reflection, random retrying) -- a 10% absolute improvement, consistent with the paper's 11% gain on HumanEval. The learning curve shows Reflexion climbing from 31% after trial 1 to 86% by trial 5, while the baseline goes from 25% to 78%.

The key mechanism is that once a bug type is identified and stored in memory, the actor avoids implementations matching that bug type. On some tasks the classifier's predictions are noisy (24 bug types with overlapping error features), so the improvement is modest rather than dramatic. These toy numbers are not comparable to the paper's HumanEval results -- a 6,808-parameter MLP on 10 toy tasks cannot match GPT-4 on 164 real programming problems.

## Influence

Reflexion became one of the most cited agent papers, popularizing the idea that LLM agents can learn from verbal feedback without weight updates. It directly influenced subsequent self-improving agent frameworks and is widely referenced in agent design patterns. The paper has been cited about 3,300 times ([OpenAlex](https://api.openalex.org/works/https://arxiv.org/abs/2303.11366), checked November 2026). Follow-up work includes Self-Refine (Madaan et al., 2023) and LATS (Language Agent Tree Search, Zhou et al., 2023), which extends Reflexion with tree-structured search.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview -- a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/071_reflexion-verbal-reinforcement-learning/solution.ipynb) | From-scratch Reflexion coding agent in PyTorch with a neural error classifier as the self-reflection model, compared against a memory-less baseline on 10 toy programming tasks (86% vs 76%) |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook -- what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
