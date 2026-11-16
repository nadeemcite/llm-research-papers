---
title: "AlphaEvolve: Code architecture"
parent: "076 AlphaEvolve"
nav_order: 1
---
# Code Architecture — AlphaEvolve Notebook

## Overview

The notebook implements the core evolutionary loop from AlphaEvolve: an LLM-simulated code mutator proposes changes to a program, an evaluator scores the result, and a program database keeps the best solutions for future inspiration. The target problem is symbolic regression — discovering a polynomial expression that fits a target function. The result is measured by mean squared error (MSE): the evolutionary loop drives MSE from about 52.9 (random initial programs) to 0.0000 over 200 generations, exactly discovering the target polynomial, while a one-shot baseline of 1000 random programs at equal compute reaches only MSE 1.61.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copied from the code cell):
- `POPULATION_SIZE = 30` — number of programs maintained in the database
- `GENERATIONS = 200` — number of evolutionary iterations
- `MUTATION_RATE = 0.4` — probability of each structural mutation being applied
- `NUM_TERMS = 5` — maximum number of terms in a polynomial expression
- `CHILDREN_PER_GEN = 5` — number of children evaluated per generation
- `MAX_COEFF = 5` — range for coefficient values (-MAX_COEFF to MAX_COEFF)
- `MAX_POWER = 3` — maximum power for polynomial terms
- `SEED = 42` — random seed for reproducibility

### 2. Target Function & Evaluation

**Function: `target_function(x)`** — returns the target polynomial value at x (the function the evolutionary loop tries to discover).

**Function: `evaluate(program)`** — computes mean squared error between the program's polynomial and the target function over a set of sample points. Returns a dictionary `{'mse': float}` where lower is better (negated for maximization convention). Programs that produce non-finite values (NaN/Inf) receive a penalty score.

### 3. Program Representation

**Class: `Program`** — represents a polynomial as a list of terms, each term being `(coefficient, power)`. Supports evaluation at a point, mutation, and string representation. About 5 terms per program (no neural network parameters — this is symbolic, not parametric).

**Data flow:**
```
Program: [(coeff1, power1), (coeff2, power2), ...]
  -> evaluate(x) -> sum(coeff * x**power for each term)
  -> evaluate over sample points -> MSE score
```

### 4. Simulated LLM Mutator

**Function: `llm_mutate(parent, inspirations)`** — simulates an LLM proposing code modifications. Given a parent program and a list of inspiration programs (top performers from the database), it stochastically applies one or more mutations:
- Swap an operator (change power of a term)
- Add a new term (inspired by an inspiration program's term)
- Remove a term
- Adjust a coefficient (small perturbation)
- Cross over: copy a term from an inspiration program

This mirrors the paper's approach where the LLM proposes SEARCH/REPLACE diff blocks, but simplified to stochastic structural changes on polynomial terms.

### 5. Program Database

**Class: `ProgramDatabase`** — maintains a population of programs with their scores. Implements MAP-Elites-style archiving (keeps diverse good programs) and island-based sampling (samples parent + inspirations for the next mutation).

- `add(program, score)` — inserts a program if it improves on the worst in its niche
- `sample()` — returns a parent program and a list of inspiration programs
- `best()` — returns the highest-scoring program

### 6. Evolutionary Loop

- **Optimizer:** stochastic mutation + selection (no gradient descent — this is evolutionary, not neural)
- **Generations:** 200, population size 30, 5 children per generation
- **Per generation:** sample 5 parent+inspiration pairs, mutate each to create children, evaluate each, add to database if it improves
- **Logging:** prints best MSE every 10 generations and at the final generation

### 7. One-Shot Baseline

**Function: `one_shot_baseline()`** — generates 1000 random programs (same total compute budget as the evolutionary loop: 200 generations × 5 children = 1000 evaluations), evaluates all, returns the best. This isolates the benefit of evolution vs. pure random search at equal compute.

### 8. Results and Plots

- **Evolution curve:** plots best MSE vs. generation, showing the progressive improvement that the evolutionary loop drives
- **Comparison bar chart:** best MSE from evolution vs. one-shot baseline, showing the gap
- **Discovered expression:** prints the best polynomial found alongside the target

## Key Functions

| Function / Class | Purpose |
|---|---|
| `target_function(x)` | The target polynomial the evolutionary loop tries to discover |
| `evaluate(program)` | Computes MSE of a program against the target over sample points |
| `Program` | Represents a polynomial as a list of (coefficient, power) terms; supports mutation |
| `llm_mutate(parent, inspirations)` | Simulates LLM code mutation: stochastically modifies a parent program using inspirations |
| `ProgramDatabase` | MAP-Elites-style archive of programs with island-based sampling |
| `one_shot_baseline()` | Generates 50 random programs and returns the best — equal-compute baseline |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Problem | Matrix multiplication, open math problems, Google infrastructure optimization | Symbolic regression of a polynomial |
| LLM | Gemini 2.0 Flash + Gemini 2.0 Pro ensemble | Stochastic structural mutator (simulated LLM) |
| Code evolution | Full codebases, hundreds of lines, any language, SEARCH/REPLACE diffs | Polynomial term list, 5 terms, structural mutations |
| Evaluation | Distributed cluster, evaluation cascade, multi-metric | Single MSE metric, local NumPy evaluation |
| Scale | Thousands of LLM samples, ~100 compute-hours per evaluation | 200 generations, population 30, 5 children/gen, seconds per evaluation |
| Database | MAP-Elites + island models, asynchronous pipeline | Simplified MAP-Elites-style archiving with random sampling |
