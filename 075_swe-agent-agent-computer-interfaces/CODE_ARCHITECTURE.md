---
title: "SWE-agent: Code architecture"
parent: "075 SWE-agent"
nav_order: 1
---
# Code Architecture — SWE-agent Notebook

## Overview

The notebook implements a miniature agent-computer interface (ACI) system with two interface modes — a structured ACI and a raw shell — and evaluates both on 20 planted-bug instances across 4 synthetic Python repositories. With a 50% simulated edit-mistake rate (paper: 51.7%), the ACI mode resolves 9/20 (45%) while the shell-only mode resolves 7/20 (35%), demonstrating that interface design (guardrails, structured feedback, efficient commands) improves agent performance without changing the underlying model.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines key constants:
- `SEED = 42` — reproducibility
- `MAX_WINDOW = 100` — file viewer shows at most 100 lines (paper Section 3)
- `MAX_SEARCH_RESULTS = 50` — search returns at most 50 results (paper Section 3)
- `TRAIN_SPLIT = 15` — 15 training, 5 test instances
- `MISTAKE_RATE = 0.5` — fraction of edits with syntax errors (paper: 51.7% of trajectories)

### 2. Synthetic Repositories and Planted Bugs

**Data structures: `REPO_TEMPLATES`, `BUG_INSTANCES`** — 4 repository templates (mathlib, textutils, datastruct, cachekit), each with 3–5 source files and a test file. File contents stored as lists of lines for easy line-based editing. 20 bug instances, each specifying the target file, line number, correct line, buggy line, and an issue description.

11 instances are "real bugs" (correct line differs from buggy line), 9 are "controls" (no bug introduced, tests already pass). Mirrors the paper's SWE-bench task format (Section 4).

**Function: `make_buggy_fix`** — creates a syntactically broken version of a correct fix line, simulating the kind of syntax error an LM might introduce (e.g., unbalanced parentheses, semicolons instead of colons). Used to inject mistakes at the MISTAKE_RATE frequency.

### 3. Building Task Instances

**Function: `make_buggy_files`** — creates buggy version of a repo by replacing the correct line with the buggy line at the specified line number.

Builds the `instances` list, each with: instance_id, repo, issue text, target file, line number, buggy/fix lines, buggy file dict, test file path, and `has_real_bug` flag. Splits into 15 training and 5 test instances.

### 4. Agent-Computer Interface (ACI)

**Class: `AgentComputerInterface`** — the core ACI implementation.

Key methods:
- `open(path)` — opens a file, shows up to 100 lines with line numbers, file path, total line count, and omitted line counts (mirrors paper Figure 3a)
- `scroll_down()` / `scroll_up()` — navigates within the open file by window size
- `search_dir(query)` — searches all files for a string, returns max 50 results with file:line:content format; suggests refining if exceeded (paper Section 3)
- `edit(line_num, new_text)` — replaces a single line; runs `compile()` as a lint check; if SyntaxError, the edit is rejected and the agent gets feedback (paper Section 3, Figure 3b); the file is reverted on rejection
- `run_tests()` — writes files to a temp dir, runs pytest, returns pass/fail

**Data flow:**
```
Issue text → search_dir → open(file) → find buggy line by number
→ edit(line_num, fix) → [lint check]
  → if pass: apply edit → run_tests → resolved/failed
  → if fail: reject, return error → retry with correct fix → run_tests
```

### 5. Shell-Only Interface (Baseline)

**Class: `ShellInterface`** — raw shell-style interface without ACI guardrails.

Key differences from ACI:
- `cat(path)` — raw file output, no line numbers, no windowing
- `grep(pattern)` — verbose output, no result limit
- `sed_edit(path, old, new)` — text replacement with NO lint checking; returns empty string on failure (no feedback to the agent)
- `run_tests()` — same execution but no structured feedback

The `silent_failures` counter tracks edits that failed without the agent knowing — a key failure mode the paper identifies (Section 5.1: "silent operations potentially confusing for models").

### 6. ACI-Based Agent

**Function: `aci_agent_resolve(instance, known_fixes, rng)`** — runs the ACI agent on a single instance.

Workflow:
1. Extract keywords from issue description, search with `search_dir`
2. Open the target file with `open`
3. Find the buggy line by line number (with fallback to line-by-line scan)
4. Determine fix from `known_fixes` or use the gold fix
5. With probability MISTAKE_RATE, attempt a broken edit (simulating LM mistake)
6. If the lint guardrail rejects it, retry with the correct fix
7. Run tests with `run_tests`

### 7. Shell-Only Agent (Baseline)

**Function: `shell_agent_resolve(instance, known_fixes, rng)`** — runs the shell agent.

Same logical workflow but using `ShellInterface` methods. Key difference: when a mistake is made (same 50% rate), the broken edit is silently applied with no lint check and no retry — the agent only discovers the problem when tests fail.

Both agents use the same RNG seed, so they make mistakes on the same instances — isolating the interface design effect.

### 8. Evaluation: ACI-Based Agent

Runs the ACI agent on all 20 instances with a seeded RNG. Prints per-instance results with steps, edits, and lint rejections.

### 9. Evaluation: Shell-Only Agent

Runs the shell agent on all 20 instances with the same seeded RNG. Prints per-instance results with steps, edits, and silent failures.

### 10. Results: ACI vs Shell-Only

Computes overall resolve rates, train/test split rates, average steps, total lint rejections (ACI), and total silent failures (shell). Prints comparison alongside the paper's reference numbers (18% vs 11% on SWE-bench Lite).

### 11. Visualization

Three subplots: (1) resolve rate bar chart with paper reference lines, (2) per-instance step count comparison, (3) per-repository resolve rate comparison.

### 12. Failure Analysis

Lists all failed instances for both agents, highlighting the role of lint guardrails vs. silent failures.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `AgentComputerInterface` | Structured ACI with file viewer, search, edit (lint guardrail), and test execution |
| `ShellInterface` | Raw shell interface without guardrails (cat, grep, sed, pytest) |
| `make_buggy_files` | Create a buggy version of a repo by introducing the planted bug |
| `make_buggy_fix` | Create a syntactically broken version of a fix line (simulates LM edit mistake) |
| `aci_agent_resolve` | Run the ACI-based agent on a single instance |
| `shell_agent_resolve` | Run the shell-only agent on a single instance |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Agent model | GPT-4 Turbo generating thoughts + commands | Rule-based agent with scripted heuristics |
| Repositories | 12 real Python repos (thousands of files) | 4 synthetic repos (3–5 files each) |
| Task instances | 2,294 real GitHub issues | 20 planted-bug instances (15 train, 5 test) |
| ACI implementation | Full shell-based ACI with LM-generated commands | Python methods with fixed workflow |
| Evaluation | Docker-based test execution on real repos | pytest in temp directory on synthetic repos |
| Interface comparison | Full ablation suite (Table 3) | ACI vs shell-only with simulated mistakes |
| Mistake simulation | Natural LM errors (51.7% of trajectories) | Injected syntax errors at 50% rate |
