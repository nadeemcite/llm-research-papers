---
title: "055 Tree of Thoughts"
nav_order: 56
permalink: "/055_tree-of-thoughts-deliberate-problem-solving/"
---
# Tree of Thoughts: Deliberate Problem Solving with Large Language Models (Tree of Thoughts)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/tree-of-thoughts-deliberate-problem-solving)

- **Paper:** [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)
- **Authors:** Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan
- **Lab, year:** Princeton University and Google DeepMind, 2023 (NeurIPS 2023)

## Summary

In May 2023, researchers at Princeton University and Google DeepMind introduced **Tree of Thoughts (ToT)**, a framework for language model inference that generalizes chain-of-thought (CoT) prompting. While CoT generates a single left-to-right reasoning path, ToT frames problem-solving as a search over a tree of intermediate "thoughts" — coherent text segments that serve as steps toward a solution. At each step, the model generates multiple candidate thoughts, self-evaluates them via a value prompt (rating them "sure," "likely," or "impossible"), and uses a search algorithm (BFS or DFS) to systematically explore promising branches while pruning unpromising ones.

The authors evaluated ToT on three tasks designed to challenge even GPT-4: Game of 24 (use 4 numbers and arithmetic to reach 24), Creative Writing (write a coherent 4-paragraph passage ending with 4 given sentences), and Mini Crosswords (fill a 5×5 grid). On Game of 24, GPT-4 with standard CoT prompting solved only 4% of tasks, while ToT with BFS (breadth b=5) achieved 74% — an 18× improvement. On Creative Writing, ToT produced more coherent passages (GPT-4 score 7.56 vs. 6.93 for CoT), and on Mini Crosswords, ToT achieved 60% word-level accuracy vs. 16% for CoT. The key insight is that some problems require exploration and backtracking, not just left-to-right decoding.

The framework is modular: the thought decomposition (what constitutes a "thought"), thought generator (sample vs. propose), state evaluator (value vs. vote), and search algorithm (BFS vs. DFS) can each be varied independently. No additional training is needed — ToT works with any pre-trained LM, using prompting alone for both generation and evaluation.

This notebook implements ToT-BFS for Game of 24 with GPT-2 small (124M), demonstrating the search mechanics — thought generation, value-based pruning, and BFS exploration — with a much smaller model than the paper's GPT-4. At 124M parameters, GPT-2 small struggles to generate valid arithmetic, so absolute success rates are far below the paper's, but the qualitative demonstration of deliberate search is the key output.

## What Problem Does It Solve

Imagine solving a maze. If you can only walk forward and never turn back, a single wrong turn early on means you are stuck. But if you can explore multiple paths, mark dead ends, and backtrack to try other routes, you can find the exit even in a complex maze. Language models before ToT were like the person who can only walk forward — they generate tokens left-to-right with no ability to explore alternatives or go back.

The paper applies this maze-solving principle to language model reasoning. For tasks like Game of 24, the first arithmetic step matters enormously — the paper found that 60% of CoT failures happen at the very first step. ToT lets the model generate several possible first steps, evaluate which ones look promising, and explore the best ones further while discarding dead ends. It is like a chess player who considers multiple moves ahead rather than playing the first move that comes to mind.

## Core Idea

- **Tree-structured search over thoughts:** Instead of one linear reasoning chain, ToT maintains a tree where each node is a partial solution (a "thought"). The model explores multiple branches, evaluates them, and uses search algorithms (BFS or DFS) with lookahead and backtracking.
- **LM as its own heuristic:** Traditional search uses programmed heuristics (like DeepBlue's chess evaluation) or learned ones (like AlphaGo's value network). ToT introduces a third option: the LM itself evaluates states via a value prompt, reasoning about whether the current partial solution can reach the goal.
- **Four design questions, four answers:** ToT is instantiated by answering: (1) how to decompose thoughts (a line of equation, a paragraph, a word), (2) how to generate candidates (sample i.i.d. or propose sequentially), (3) how to evaluate states (value independently or vote across states), and (4) which search algorithm to use (BFS for shallow trees, DFS for deep ones).

## Key Method Details

- **Thought decomposition (Section 3, Table 1):** Thoughts are problem-dependent — a line of equation for Game of 24 (3 steps), a paragraph writing plan for Creative Writing (1 step), a word for Mini Crosswords (5-10 steps). A thought should be "small" enough for diverse generation yet "big" enough for evaluation.
- **Thought Generator G(pθ, s, k) (Section 3):** Two strategies — (1) sample k i.i.d. thoughts from a CoT prompt (better for rich thought spaces like paragraphs), or (2) propose k thoughts sequentially in one prompt (better for constrained spaces like equations, avoids duplication).
- **State Evaluator V(pθ, S) (Section 3):** Two strategies — (1) value each state independently (sure/likely/impossible), or (2) vote across states (compare and pick the best). Multiple samples can be aggregated for robustness. The value map in the paper's code: impossible=0.001, likely=1, sure=20.
- **Search Algorithms (Section 3, Algorithms 1-2):** BFS keeps the top b states per step (used for Game of 24 and Creative Writing, where depth is ≤3). DFS explores the most promising state first, backtracking when the state evaluator deems it impossible (used for Mini Crosswords, where depth is 5-10).
- **Results (Table 2, Game of 24):** IO prompt 7.3%, CoT prompt 4.0%, CoT-SC (k=100) 9.0%, ToT (b=1) 45%, ToT (b=5) 74%. IO+Refine (k=10) 27%, CoT (best of 100) 49%.
- **Error Analysis (Figure 3b):** ~60% of CoT failures occur at the first thought step, highlighting the problem with left-to-right decoding where initial decisions are pivotal.

### Simplifications in This Notebook

- **Smaller model:** GPT-2 small (124M parameters) instead of GPT-4
- **Dataset:** 10 hard Game of 24 puzzles (indices 901-910) instead of 100 (indices 901-1000)
- **IO/CoT sampling:** 3 samples per puzzle instead of 100 (paper sampled 100 times with GPT-4)
- **Value samples:** 3 per state (same as the paper)
- **Breadth:** b=5 (same as the paper's best setting)
- **No CoT-SC or IO+Refine baselines** (CoT-SC is covered in paper 054)

### What the Notebook Result Shows

On 10 hard Game of 24 puzzles with GPT-2 small (124M), IO prompting, CoT prompting, and ToT-BFS (b=5) all achieve 0% success (0/10 solved). GPT-2 small lacks the arithmetic capability to generate valid intermediate equations — the propose prompt outputs often contain nonsensical arithmetic (e.g., "4 + 8 = 15" or "12 + 8 = 8"), and the value prompt assigns low scores to most states.

The paper achieved 74% with ToT (b=5) on GPT-4; the gap reflects the 3,000× difference in model scale. The qualitative demonstration — the propose prompt generating candidate next steps, the value prompt evaluating them as sure/likely/impossible, and BFS pruning to keep the top 5 — is the key output. The search tree analysis shows how many candidates are generated and pruned at each BFS step (e.g., 5 → 20 → 29 candidates across 3 steps for one puzzle), illustrating deliberate search vs. left-to-right decoding.

## Influence

Tree of Thoughts became one of the most influential reasoning frameworks in the LLM era, directly inspiring test-time compute scaling methods (Snell et al., 2024), reasoning with search (MCTS-based approaches like RAP, Hao et al., 2023), and the broader paradigm of using LMs as their own search heuristics. The connection to classical AI search (Newell and Simon, 1950s) bridged modern LLMs with foundational AI concepts. ToT is also a conceptual ancestor of OpenAI's o1 reasoning approach, which scales test-time computation for complex problem-solving. About 1,747 citations ([OpenAlex](https://openalex.org/W7133209886), checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/055_tree-of-thoughts-deliberate-problem-solving/solution.ipynb) | ToT-BFS pipeline with GPT-2 small: propose-prompt thought generation, value-prompt state evaluation, BFS search on 10 hard Game of 24 puzzles with IO and CoT baselines |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, prompt construction, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about Tree of Thoughts |
