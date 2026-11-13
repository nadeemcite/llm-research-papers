---
title: "Reflexion: Code architecture"
parent: "071 Reflexion"
nav_order: 1
---
# Code Architecture -- Reflexion Notebook

## Overview

The notebook implements a simplified Reflexion coding agent: an Actor picks from pre-defined implementations (correct + buggy variants), an Evaluator runs them against test cases, and a neural Error Classifier (6,808 parameters) identifies the bug type from structured error features to generate a verbal reflection. Reflections are stored in an episodic memory buffer (bounded to 3). On the next trial, the Actor reads memory and avoids implementations matching identified bug types. Evaluated on 10 toy programming tasks (50 runs each, 5 max trials), Reflexion achieves 86% accuracy vs 76% for the memory-less baseline.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copied from the code cell):
- `EPOCHS = 50` -- training epochs for the error classifier (paper uses no training -- LLM-based reflection)
- `BATCH_SIZE = 16` -- mini-batch size
- `LR = 1e-3` -- Adam learning rate
- `MAX_TRIALS = 5` -- max Reflexion attempts per task (paper uses 12 for ALFWorld)
- `MAX_MEMORY = 3` -- max reflections in episodic memory (paper uses 1-3)
- `HIDDEN_DIM = 64` -- MLP hidden dimension
- `ERROR_FEATURE_DIM = 16` -- size of error feature vector

### 2. Toy Programming Tasks

**Data structure: `TASKS`** -- list of 10 task dictionaries

Each task contains:
- `id`: short identifier (e.g., "add", "multiply", "factorial")
- `desc`: natural-language description
- `func_name`: the function name to implement
- `tests`: list of (func_name, args, expected_output) tuples -- 4 tests per task
- `correct`: the correct Python implementation string
- `bugs`: list of (bug_type_name, buggy_code, bug_description) -- 3 buggy variants per task

**Bug types across all tasks:** 24 distinct bug types (some shared, e.g., "missing_return" appears in multiple tasks). Bug type names are used as classification labels for the error classifier.

### 3. Code Execution & Evaluation Engine

**Function: `execute_code(code_str, task)`** -- returns (passed, error_features, error_details)

Executes code in a sandboxed namespace via `exec()`, then runs each test case. Returns a 16-dimensional error feature vector:
- Features 0-2: n_tests, n_passed, n_failed (normalized by 10)
- Features 3-10: one-hot error type flags (syntax, runtime, assertion, type, recursion, attribute, index)
- Features 11-12: returns_none, wrong_value
- Feature 13: error severity (n_failed / n_tests)
- Features 14-15: first_test_passed, last_test_passed

**Data flow:**
```
code_str -> exec() -> namespace -> func -> func(*args) -> result
  -> compare with expected -> features (16,) + details []
```

### 4. Neural Error Classifier

**Class: `ErrorClassifier`** (6,808 parameters)

Architecture: `Linear(16, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, 24)`

Input: error feature vector (16,) -> Output: bug type logits (24,)

This replaces GPT-4's verbal self-reflection with a learned bug-type classifier. The predicted bug type maps to a pre-written verbal reflection (the bug description from the task definition).

### 5. Training Data for Error Classifier

**Function: `build_classifier_training_data()`**

Runs every buggy implementation against its task's tests, collects (error_features, bug_type_label) pairs. Each example is augmented with 3 noise-injected copies (Gaussian noise sigma=0.05, clipped to [0, 1.5]).

**Shapes:** `train_x (120, 16)`, `train_y (120,)` -- 30 base examples x 4 copies

### 6. Loss Function and Optimizer

Cross-entropy loss with Adam optimizer (lr=1e-3).

### 7. Training Loop

- Optimizer: Adam, lr=1e-3, batch size 16
- 50 epochs, loss converges from ~3.3 to ~0.05
- Training accuracy reaches ~100%
- Training time: under 1 second (toy scale)

### 8. Training Curves

Two-panel plot: cross-entropy loss (left) and training accuracy (right) vs epoch. Shows convergence from ~3.3 to ~0.05 and 0% to ~100% over 50 epochs.

### 9. Reflexion Agent

**Functions: `generate_reflection`, `actor_generate_code`, `reflexion_agent`**

- `generate_reflection(classifier, error_features, task, failed_code)`: Runs the classifier on error features, gets predicted bug type + confidence, maps to the verbal bug description. Returns a reflection string like "Bug identified: subtraction_bug (confidence 0.95). The function uses subtraction instead of addition. Should use + operator."
- `actor_generate_code(task, memory, use_memory)`: Picks an implementation. Without memory: random from correct + all buggy. With memory: extracts identified bug types from reflection strings via regex, filters them out, then random-selects from remaining options.
- `reflexion_agent(task, use_reflection, max_trials)`: The core Reflexion loop. Trial 1: Actor picks code (no memory). Evaluator runs tests. If pass -> done. If fail -> Self-Reflection generates critique -> stored in memory. Trial 2+: Actor reads memory, avoids identified bugs. Repeat up to MAX_TRIALS.

**Reflexion loop data flow:**
```
Trial 1: actor_generate_code (no memory) -> execute_code -> fail
  -> generate_reflection -> memory = [reflection_1]
Trial 2: actor_generate_code (memory: avoid bug_1) -> execute_code -> fail
  -> generate_reflection -> memory = [reflection_1, reflection_2]
Trial 3: actor_generate_code (memory: avoid bug_1, bug_2) -> execute_code -> pass
  -> return (passed=True, trials=3)
```

### 10. Example Trajectory

Runs Reflexion on the "add" task with verbose output, showing each trial's code, pass/fail status, and reflection text.

### 11. Evaluation: Reflexion vs Baseline

**Function: `evaluate_agent(use_reflection, n_runs)`** -- runs each task 50 times, reports accuracy and average trials to solve.

### 12. Learning Curve

**Function: `evaluate_by_trial(use_reflection)`** -- tracks cumulative pass rate by trial number across all tasks.

### 13. Accuracy Comparison Plot

Three-panel figure: overall accuracy bars (baseline vs Reflexion), learning curve (cumulative pass rate by trial), and per-task accuracy comparison.

### 14. Final Evaluation Summary

Prints every number quoted in the docs as "Name: value" lines, plus sample reflections for the first 3 tasks.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `execute_code(code_str, task)` | Execute generated code, run tests, return (passed, error_features, details) |
| `ErrorClassifier` | MLP that classifies bug type from 16-dim error features |
| `build_classifier_training_data()` | Run all buggy implementations to collect training pairs |
| `generate_reflection(classifier, error_features, task, code)` | Classify bug, generate verbal reflection string |
| `actor_generate_code(task, memory, use_memory)` | Pick implementation, filtering out identified bugs if memory is active |
| `reflexion_agent(task, use_reflection, max_trials)` | Run the full Reflexion loop: act -> evaluate -> reflect -> retry |
| `evaluate_agent(use_reflection, n_runs)` | Evaluate across all tasks, n_runs times each |
| `evaluate_by_trial(use_reflection)` | Track cumulative pass rate by trial number |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Self-reflection model | GPT-4 LLM with few-shot prompting | 6,808-parameter MLP error classifier |
| Code generation | Free-form code generation by GPT-4 | Pre-defined correct + buggy implementations |
| Tasks | HumanEval (164), MBPP (974), LeetcodeHardGym (40) | 10 toy programming tasks |
| Evaluator | Self-generated unit tests (up to 6) + AST filtering | Pre-defined test cases (4 per task) |
| Reflection text | Free-form natural language from LLM | Bug type name mapped to pre-written description |
| Memory size | 1 (programming) to 3 (ALFWorld, HotpotQA) | 3 |
| Max trials | 12 (ALFWorld) | 5 |
| Evaluation | pass@1 on real benchmarks | 50-run average accuracy on toy tasks |
