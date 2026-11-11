---
title: "Absolute Zero: Code architecture"
parent: "066 Absolute Zero"
nav_order: 1
---
# Code Architecture — Absolute Zero Notebook

## Overview

The notebook implements a simplified Absolute Zero Reasoner (AZR) in PyTorch. Three small MLP classifiers (~93K parameters total) serve as solvers for three reasoning modes — deduction, abduction, and induction — on simple Python lambda expressions. A stochastic program mutator acts as the proposer, and a Python interpreter (`eval` with restricted builtins) serves as the sole verifier. Training uses REINFORCE with Task-Relative REINFORCE++ (TRR++) advantages across 6 task-role configurations. The result is measured by solver accuracy on 40 held-out test tasks per reasoning mode, compared against an untrained baseline.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copied from the code cell):
- `AZR_ITERS = 80` — number of self-play iterations (paper: hundreds)
- `BATCH_SIZE = 8` — tasks per task type per iteration (paper: 64)
- `GROUP_SIZE = 4` — G, Monte Carlo rollouts for learnability (paper: 8)
- `LR = 5e-4` — learning rate (paper: 1e-6)
- `K_REFERENCES = 2` — few-shot examples for the proposer
- `MAX_BUFFER = 80` — max triplets per buffer
- `N_OUTPUT_CLASSES = 201` — output range -100 to 100
- `N_INPUT_CLASSES = 20` — input range 1 to 20
- `N_PROGRAM_CLASSES = 95` — 5 operators × 19 operands

### 2. The (Program, Input, Output) Triplet

Explains the three reasoning modes: deduction (predict output), abduction (find input), induction (synthesize program). Each mode focuses on inferring one element of the triplet given the others.

### 3. Seed Buffer & Program Representation

**Function: `program_to_features(prog_str)`** — encodes a program string as a 6-element vector: 5-element one-hot for the operator + 1 normalized operand value (operand/20).

**Function: `program_to_class(prog_str)`** — maps a program string to a class index (0..94) for induction.

**Functions: `class_to_program`, `output_to_class`, `class_to_output`, `input_to_class`, `class_to_input`** — bidirectional mappings between string representations and class indices.

**Data:** 16 seed triplets generated from 5 operators × 4 operands × 4 inputs, filtered to valid outputs in range [-100, 100].

### 4. Python Verifier (The Environment)

**Function: `safe_execute(program_str, input_str)`** — executes a Python lambda on an input using `eval` with `{'__builtins__': {}}` for safety. Returns the output as a string, or None on error.

**Function: `verify_answer(predicted, ground_truth)`** — checks value equality by evaluating both strings in a restricted namespace.

### 5. Program Mutation (Proposer Strategy)

**Function: `mutate_program(program_str, rng)`** — stochastic mutation: replaces the operator (random from 5) and operand (random from 1-19). This mimics the LLM proposer's diversity-promoting generation.

**Function: `propose_task(task_type, buffer, rng, k_refs)`** — samples K reference programs from the buffer, mutates one, generates a random input, and validates the triplet with the Python executor. Returns a valid (program, input, output) triplet or None.

### 6. Model: Three MLP Solvers

**Class: `DeductionSolver`** (~43K parameters) — Linear(7, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 201). Input: 6 program features + 1 normalized input value. Output: 201 logits over output range.

**Class: `AbductionSolver`** (~20K parameters) — Linear(7, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 20). Input: 6 program features + 1 normalized output value. Output: 20 logits over input range.

**Class: `InductionSolver`** (~29K parameters) — Linear(4, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 95). Input: 2 (input, output) pairs normalized. Output: 95 logits over program space.

**Total parameters: ~93K**

### 7. Reward Design

**Function: `compute_solver_reward(predicted, ground_truth)`** — binary reward: 1.0 if correct, 0.0 otherwise (Equation 5).

**Function: `compute_proposer_reward(solver_success_rate)`** — learnability reward (Equation 4): 0 if success rate is 0, else 1 - success_rate. Visualized as a plot showing the peak at moderate difficulty.

### 8. Solver Inference & Verification

**Functions: `solve_deduction`, `solve_abduction`, `solve_induction`** — each takes a model and triplet, encodes the appropriate features, runs a forward pass, and returns the predicted answer via argmax (greedy) or sampling (for exploration).

**Function: `verify_solver(task_type, predicted, triplet)`** — verifies the solver's answer using the Python executor: deduction checks output equality, abduction checks if program(predicted_input) = output, induction checks if predicted_program(input) = output.

### 9. Task-Relative REINFORCE++ (TRR++)

**Class: `TRRPlusPlus`** — maintains running reward history for each of the 6 task-role configurations. `batch_advantages` computes normalized advantages using historical mean and std, falling back to raw rewards when insufficient history exists.

### 10. AZR Self-Play Training Loop

**Function: `get_log_prob_and_solve(model_type, model, triplet)`** — forward pass through the appropriate solver, sample an action from the output distribution, compute the log-probability, and check correctness. Returns (log_prob, reward, correct).

**Function: `run_azr_iteration(iteration, rng)`** — one complete AZR iteration:
1. **Propose phase:** For each task type, propose BATCH_SIZE tasks by mutation. Validate with Python. Compute learnability reward via GROUP_SIZE solver rollouts (sampling mode).
2. **Solve phase:** For each valid task, sample an answer and compute the solver reward.
3. **Update phase:** Group all rewards by (task_type, role). Compute TRR++ advantages. Policy gradient: `loss = -mean(advantage * log_prob)`. Gradient clipping (max_norm=1.0). AdamW optimizer step.
4. **Buffer update:** Add valid triplets to buffers (capped at MAX_BUFFER).

**Data flow:**
```
Buffer (seed triplets)
  → propose_task (mutate + execute)
  → valid triplets
  → learnability reward (G solver rollouts)
  → solve (forward pass + sample)
  → solver reward (Python verify)
  → TRR++ advantages (6 configs)
  → REINFORCE loss → backward → optimizer step
  → buffer update
```

### 11-13. Training, Curves, and Evaluation

Runs 80 AZR iterations, logging loss, solver accuracy, proposer reward, and per-mode accuracy. Three plots: loss curve, solver accuracy by mode, and proposer learnability reward. Final evaluation on 40 held-out tasks per mode, comparing trained vs untrained baseline.

### 14-16. Examples and Diversity

Shows sample self-generated tasks with solver predictions and correctness. Analyzes buffer diversity (unique programs, inputs, outputs per buffer). Plots buffer growth over training iterations.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `program_to_features` | Encode program string as 6-element feature vector |
| `program_to_class` / `class_to_program` | Bidirectional program ↔ class index mapping |
| `safe_execute` | Python verifier: execute program on input, return output |
| `verify_answer` | Check value equality between predicted and ground truth |
| `mutate_program` | Stochastic program mutation (simulates LLM proposer) |
| `propose_task` | Propose and validate a new task from buffer references |
| `DeductionSolver` | MLP: predict output from (program, input) |
| `AbductionSolver` | MLP: predict input from (program, output) |
| `InductionSolver` | MLP: predict program from (input, output) pairs |
| `compute_solver_reward` | Binary reward (Equation 5) |
| `compute_proposer_reward` | Learnability reward (Equation 4) |
| `solve_deduction` / `solve_abduction` / `solve_induction` | Solver inference (greedy or sampling) |
| `verify_solver` | Verify solver answer via Python execution |
| `TRRPlusPlus` | Task-Relative REINFORCE++ advantage estimator (6 configs) |
| `get_log_prob_and_solve` | Forward pass, sample, compute log-prob and reward |
| `run_azr_iteration` | One complete AZR self-play iteration |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | Qwen2.5-7B (7B params) | Three MLPs (~93K params total) |
| Programs | Arbitrary Python code | Simple lambda expressions (5 ops × 19 operands) |
| Proposer | LLM text generation | Stochastic program mutation |
| Iterations | Hundreds | 80 |
| Batch size | 64 per task type | 8 |
| Group size (G) | 8 | 4 |
| Learning rate | 1e-6 | 5e-4 |
| KL penalty | None (final system) | None |
| Benchmarks | LiveCodeBench, AIME, MATH, etc. | Held-out Python lambda tasks |
