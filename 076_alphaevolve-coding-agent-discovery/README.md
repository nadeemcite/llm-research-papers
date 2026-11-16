---
title: "076 AlphaEvolve"
nav_order: 77
permalink: "/076_alphaevolve-coding-agent-discovery/"
---
# AlphaEvolve: A coding agent for scientific and algorithmic discovery (AlphaEvolve)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/alphaevolve-coding-agent-discovery)

- **Paper:** [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131)
- **Authors:** Alexander Novikov, Ngân Vũ, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco J. R. Ruiz, Abbas Mehrabian, M. Pawan Kumar, Abigail See, Swarat Chaudhuri, George Holland, Alex Davies, Sebastian Nowozin, Pushmeet Kohli, Matej Balog
- **Lab, year:** Google DeepMind, 2025 (preprint, June 2025)

## Summary

AlphaEvolve is an evolutionary coding agent from Google DeepMind that combines large language models (LLMs) with automated evaluation to iteratively improve algorithms. Instead of asking an LLM to solve a problem in one shot, AlphaEvolve runs an evolutionary loop: the LLM proposes code modifications (mutations), an automated evaluator scores each variant, and the best solutions are kept in a program database for future inspiration. This cycle repeats thousands of times, gradually evolving better and better code.

The paper demonstrates AlphaEvolve across three domains. In mathematics, it improved the state of the art on 14 matrix multiplication targets — most notably finding a way to multiply two 4×4 complex-valued matrices using 48 scalar multiplications, the first improvement over Strassen's 1969 algorithm (rank 49) in 56 years. It also matched or surpassed the best known constructions on about 50 open mathematical problems, including a new bound on the Erdős minimum overlap problem and an improved kissing number in 11 dimensions (593 spheres, up from 592).

In engineering, AlphaEvolve discovered a scheduling heuristic that recovers 0.7% of Google's fleet-wide compute resources, sped up a Gemini training kernel by 23% (1% overall training time), simplified a TPU arithmetic circuit, and optimized FlashAttention code for 32% speedup.

This repo's notebook implements the core evolutionary loop — LLM-mutates-code, evaluator-scores, keep-the-best — on a toy optimization problem (symbolic regression of a polynomial). Over 200 generations with a population of 30, the evolutionary loop discovers the exact target polynomial (MSE 0.0000), while a one-shot baseline of 1000 random programs at equal compute reaches only MSE 1.61.

## What Problem Does It Solve

Imagine you're trying to write the fastest possible algorithm for a task — say, multiplying matrices or scheduling jobs in a data center. You start with a decent solution, but you know it could be better. A human expert would tinker with it: try a different formula here, add a clever trick there, test each version, and keep whatever works best. This trial-and-error process is slow and limited by human creativity and stamina.

AlphaEvolve automates this entire process. The paper explains it with its own comparison to FunSearch (DeepMind's earlier system): while FunSearch could only evolve a single small Python function, AlphaEvolve can evolve entire codebases of hundreds of lines in any programming language, using frontier LLMs (Gemini 2.0 Flash and Pro) instead of small code-only models. The key insight is that LLMs bring world knowledge to the mutation step — they can propose changes a random genetic operator never would — while the automated evaluator grounds the process in reality, filtering out any incorrect suggestions.

## Core Idea

- **Evolutionary loop with LLM mutations:** An LLM proposes code changes as SEARCH/REPLACE diff blocks to a parent program; the modified program is scored by an evaluator; the best programs are stored in a database and used as context for future mutations.
- **Program database with MAP-Elites and island models:** The database balances exploration and exploitation, resurfacing diverse past solutions as inspiration for new generations, inspired by the MAP-Elites algorithm and island-based population models.
- **Automated evaluation as ground truth:** Every candidate is executed and scored by a user-provided evaluation function, so incorrect LLM suggestions are automatically filtered out — no hallucinations survive the evaluation gate.
- **Ensemble of LLMs:** A mix of Gemini 2.0 Flash (high throughput, many candidates) and Gemini 2.0 Pro (fewer but higher-quality suggestions) maximizes the volume of explored ideas while retaining breakthrough potential.

## Key Method Details

- **Evolution blocks:** Code is annotated with `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` markers; only code inside these blocks is mutated by the LLM.
- **Prompt construction:** Each prompt includes previously discovered top-performing programs, the current program to improve, and rendered evaluation results — giving the LLM rich context about what has worked so far.
- **Diff format:** Changes are requested as SEARCH/REPLACE blocks (`<<<<<<< SEARCH` ... `=======` ... `>>>>>>> REPLACE`), enabling targeted modifications to large codebases.
- **Evaluation cascade:** Solutions are tested on increasingly difficult test cases; only promising candidates proceed to the next stage, saving compute.
- **Models:** Gemini 2.0 Flash and Gemini 2.0 Pro (Section 2.3).
- **Matrix multiplication results (Table 2):** For ⟨4,4,4⟩ (4×4 complex matrices), AlphaEvolve achieved rank 48 vs. the previous best of 49 (Strassen, 1969). Improved SOTA for 14 of 16 targets tested.
- **Mathematical problems:** Matched best known on ~75% of 50+ problems, surpassed SOTA on ~20%, including Erdős minimum overlap problem and kissing number in 11D (593 vs. 592).
- **Engineering results:** 0.7% fleet-wide compute recovery (data center scheduling); 23% kernel speedup / 1% Gemini training time reduction; 32% FlashAttention speedup.

### Simplifications in This Notebook

- Toy problem: symbolic regression of a simple polynomial (e.g., x³ + 2x² + x + 1), instead of matrix multiplication or open mathematical problems
- Simulated LLM: a stochastic code mutator that proposes random structural changes (swapping operators, adding terms, adjusting coefficients), instead of Gemini 2.0 Flash/Pro
- Population 30, 200 generations, 5 children per generation, instead of thousands of LLM samples
- Single evaluation metric (mean squared error against the target polynomial), instead of the paper's multi-metric optimization
- Pure Python / NumPy evaluation (no GPU, no distributed pipeline), instead of the paper's asynchronous distributed evaluation cluster

### What the Notebook Result Shows

The evolutionary loop discovers the exact target polynomial (x³ + 2x² - x + 3) over 200 generations, reaching MSE 0.0000. The best MSE drops from about 52.9 (initial random programs) to 0.0000 — effectively infinite improvement. The one-shot baseline (1000 random programs at equal compute, best picked) reaches only MSE 1.61, showing that the iterative evolutionary feedback loop is responsible for discovering the exact solution where random search cannot. This is not comparable to the paper's matrix multiplication or engineering results, but it demonstrates the core mechanism: evolution with automated evaluation drives cumulative improvement that one-shot generation cannot match.

## Influence

AlphaEvolve is a substantial enhancement of FunSearch (Romera-Paredes et al., 2023), extending LLM-guided evolution from single-function heuristics to full-codebase algorithm discovery. The 4×4 complex matrix multiplication result (rank 48, first improvement over Strassen in 56 years) attracted significant attention from the mathematical and computer science communities. The paper suggests that coding agents like AlphaEvolve can have significant impact across science and computation, and that distilling AlphaEvolve-augmented performance into future base models could create a positive feedback loop. Terence Tao and Javier Gomez Serrano advised on problem formulation, highlighting potential synergistic partnerships between AI-driven discovery and human expertise.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/076_alphaevolve-coding-agent-discovery/solution.ipynb) | Pure-Python implementation of the AlphaEvolve evolutionary loop — LLM-simulated code mutation, evaluator scoring, program database with MAP-Elites-style keeping, applied to symbolic regression of a polynomial |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, key functions, and deliberate simplifications vs the full paper |
| [TALK.md](TALK.md) | Press coverage, Q&A answered from the paper, citations, and common misconceptions about AlphaEvolve |
