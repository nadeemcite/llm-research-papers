---
title: "Tree of Thoughts: Code architecture"
parent: "055 Tree of Thoughts"
nav_order: 1
---
# Code Architecture — Tree of Thoughts Notebook

## Overview

This notebook implements **Tree of Thoughts BFS (ToT-BFS)** for the Game of 24, following
Algorithm 1 from the paper. Using GPT-2 small (124M parameters) with the paper's exact prompts
(propose prompt for thought generation, value prompt for state evaluation), it performs BFS
with breadth b=5 over 3 thought steps on 20 hard Game of 24 puzzles. It compares ToT against
IO prompting and CoT prompting baselines. The result demonstrates the search mechanics —
thought generation, value-based pruning, and systematic exploration — with a much smaller
model than the paper's GPT-4.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines key constants (copied from the code cell):
- `NUM_TEST = 10` — Game of 24 puzzles to evaluate (paper used 100)
- `BREADTH = 5` — BFS breadth limit b (paper's b=5)
- `MAX_STEPS = 3` — thought steps to reach 24 (always 3 for Game of 24)
- `N_VALUE_SAMPLES = 3` — value evaluations per state (paper used 3)
- `TEMPERATURE = 0.7` — sampling temperature (paper's GPT-4 setting)
- `MAX_NEW_TOKENS = 50` — max tokens per generation
- `MAX_CONTEXT = 1024` — GPT-2 context window
- `SEED = 42`, `DEVICE = cuda or cpu`

### 2. Model & Tokenizer

**Model: GPT-2 small** (124M parameters) from HuggingFace `transformers`.
- `AutoTokenizer.from_pretrained('gpt2')` — GPT-2 BPE tokenizer
- `AutoModelForCausalLM.from_pretrained('gpt2')` — GPT-2 small decoder-only transformer
- Model is set to `eval()` mode; no training/finetuning occurs

The paper evaluated on GPT-4. GPT-2 small at 124M is far below the scale where ToT's
full gains emerge, making this a deliberate test of the search mechanics.

### 3. Data: Game of 24

**Data source:** The paper's GitHub repo (`24.csv`), scraped from 4nums.com, containing
1,362 games sorted by human solving time (easiest to hardest).

**Test set:** 10 hard games (indices 900-909, 0-based), matching the paper's range of
the hardest 100 games (indices 901-1000). Each puzzle is a string of 4 space-separated
numbers, e.g., "3 5 7 13".

### 4. Brute-Force Solver

**Function: `solve_24_bruteforce(nums)`** — exhaustive solver that tries all 24 permutations
of 4 numbers, all 64 combinations of 3 operators (+, -, *, /), and all 5 distinct
parenthesizations. Uses `Fraction` for exact arithmetic. Returns True if any combination
equals 24. Used to verify that all test puzzles are solvable.

### 5. Prompts

Four prompt templates, copied verbatim from the paper's [GitHub code](https://github.com/princeton-nlp/tree-of-thought-llm):

- **`standard_prompt`** (IO): 5-shot, maps input numbers to a final answer equation
- **`cot_prompt`** (CoT): 5-shot, includes 3 intermediate equation steps with remaining numbers
- **`propose_prompt`** (ToT): 1-shot, given remaining numbers, proposes possible next steps
- **`value_prompt`** (ToT): evaluates if given numbers can reach 24 (sure/likely/impossible)

### 6. Generation & Parsing

**Function: `generate(prompt, max_new_tokens, do_sample, temperature, top_k)`** — tokenizes
the prompt, runs `model.generate()`, and extracts only newly generated tokens. Uses
temperature sampling (T=0.7, top_k=40) for diverse thought generation.

**Helper functions:**
- `get_current_numbers(y)` — extracts remaining numbers from the last line of a thought trajectory
- `parse_propose_output(text)` — parses propose output into (equation, remaining_numbers) tuples
- `parse_value_output(text)` — extracts sure/likely/impossible verdict from value output
- `parse_cot_answer(text)` — extracts the final answer equation from a CoT response

**Data flow:**
```
Input numbers → propose_prompt → generate() → parse_propose_output() → candidate steps
Candidate step → value_prompt → generate() → parse_value_output() → score
Scores sorted → top b kept → next step
```

### 7. Baselines: IO and CoT Prompting

**Function: `verify_solution(eq, input_numbers)`** — checks if an equation uses each input
number exactly once and evaluates to 24.

**Function: `evaluate_io_baseline(puzzles, n_samples=5)`** — for each puzzle, samples 5 IO
responses; success if any response contains a valid solution.

**Function: `evaluate_cot_baseline(puzzles, n_samples=5)`** — for each puzzle, samples 5 CoT
responses; success if any response's parsed answer is valid.

The paper sampled 100 times with GPT-4; we use 5 samples with GPT-2 small.

### 8. ToT-BFS Implementation

**Function: `tot_bfs(input_numbers, b, T, n_value)`** — implements Algorithm 1:
1. Initialize frontier with the empty state (just input numbers)
2. For each of T=3 steps:
   a. For each state in the frontier, generate candidate next steps via the propose prompt
   b. Evaluate each candidate by sampling n_value=3 value judgments and summing scores
   c. Sort candidates by score, keep top b=5
3. After T steps, check if any final state solves the puzzle via `verify_steps_solve_24`

**Function: `verify_steps_solve_24(steps, input_numbers)`** — verifies that a sequence of
thought steps (equations with remaining numbers) correctly reaches 24, checking each
arithmetic operation and that the final remaining number is 24.

**Value map** (from the paper's code): `{'impossible': 0.001, 'likely': 1, 'sure': 20}`

### 9. Running the Evaluation

Three sequential evaluation runs:
- IO baseline (3 samples per puzzle, ~10 puzzles)
- CoT baseline (3 samples per puzzle, ~10 puzzles)
- ToT-BFS (b=5, 3 value samples, ~10 puzzles)

Progress is logged every 5 puzzles with running success counts.

### 10. Results

Prints accuracy for all three methods plus a comparison table with the paper's GPT-4 results
(Table 2): IO 7.3%, CoT 4.0%, ToT (b=5) 74%.

### 11. Visualization

Bar chart comparing paper (GPT-4) vs. our (GPT-2 small) success rates for IO, CoT, and ToT (b=5).

### 12-13. Example Trajectories & Search Tree Analysis

Shows example ToT trajectories (solved and unsolved) and a detailed step-by-step BFS search
tree for one puzzle, showing how many candidates are generated, scored, and pruned at each step.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `generate()` | Run GPT-2 generation with temperature sampling |
| `get_current_numbers()` | Extract remaining numbers from thought trajectory |
| `parse_propose_output()` | Parse propose prompt output into (equation, remaining) tuples |
| `parse_value_output()` | Extract sure/likely/impossible verdict from value output |
| `parse_cot_answer()` | Extract final answer equation from CoT response |
| `verify_solution()` | Check if an equation is valid and equals 24 |
| `verify_steps_solve_24()` | Verify a sequence of thought steps reaches 24 |
| `solve_24_bruteforce()` | Exhaustive solver for ground truth |
| `evaluate_io_baseline()` | IO prompting baseline evaluation |
| `evaluate_cot_baseline()` | CoT prompting baseline evaluation |
| `tot_bfs()` | Tree of Thoughts BFS (Algorithm 1) |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | GPT-4 | GPT-2 small (124M) |
| Test set size | 100 hard games | 10 hard games |
| IO/CoT sampling | 100 samples per puzzle | 3 samples per puzzle |
| Tasks | Game of 24, Creative Writing, Mini Crosswords | Game of 24 only |
| Baselines | IO, CoT, CoT-SC, IO+Refine, best-of-k | IO, CoT only |
| Search algorithms | BFS and DFS | BFS only |
| Value aggregation | 3 samples (same) | 3 samples (same) |
| Breadth | b=1 and b=5 | b=5 only |
