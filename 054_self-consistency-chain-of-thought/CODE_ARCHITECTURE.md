---
title: "Self-Consistency: Code architecture"
parent: "054 Self-Consistency"
nav_order: 1
---
# Code Architecture — Self-Consistency Notebook

## Overview

This notebook implements **self-consistency** — a decoding strategy that replaces greedy
decoding in chain-of-thought (CoT) prompting. Using GPT-2 small (124M parameters) with
8 few-shot CoT exemplars from GSM8K, it samples 40 diverse reasoning paths per problem via
temperature sampling (T=0.7, top-k=40) and aggregates answers by majority vote. It
compares greedy CoT accuracy against self-consistency at N = [1, 5, 10, 20, 40] sampled
paths, mirroring the paper's Figure 2, on GSM8K (30 problems) and MultiArith (20 problems).
The result demonstrates the method's mechanics and the scale-dependence of its gains.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines key constants (copied from the code cell):
- `NUM_TEST_GSM8K = 30` — GSM8K test problems to evaluate
- `NUM_TEST_MULTIARITH = 20` — generated MultiArith problems
- `NUM_EXEMPLARS = 8` — few-shot CoT exemplars (same as the paper's 8)
- `MAX_NEW_TOKENS = 150` — maximum generation length per reasoning path
- `MAX_CONTEXT = 1024` — GPT-2 context window
- `TEMPERATURE = 0.7` — sampling temperature (paper's PaLM-540B setting)
- `TOP_K = 40` — top-k truncation (paper's setting)
- `N_PATHS = [1, 5, 10, 20, 40]` — number of sampled paths to evaluate (paper's Figure 2)
- `MAX_PATHS = 40` — total paths to sample per problem
- `SEED = 42`, `DEVICE = cuda or cpu`

### 2. Model & Tokenizer

**Model: GPT-2 small** (124M parameters) from HuggingFace `transformers`.
- `AutoTokenizer.from_pretrained('gpt2')` — GPT-2 BPE tokenizer
- `AutoModelForCausalLM.from_pretrained('gpt2')` — GPT-2 small decoder-only transformer
- Model is set to `eval()` mode; no training/finetuning occurs

The paper evaluated on UL2-20B, GPT-3-175B, LaMDA-137B, and PaLM-540B. GPT-2 small at 124M
is far below the scale where large self-consistency gains emerge, making this a deliberate
test of the method's mechanics and the scale-dependence finding.

### 3. Data

Two arithmetic reasoning tasks, matching the paper's focus:

**GSM8K** (arithmetic): Downloaded from the public GitHub repository (test.jsonl and
train.jsonl). 30 test problems used for evaluation. Each problem has a `question` field and
an `answer` field containing reasoning steps followed by `#### <numeric answer>`.

**MultiArith** (arithmetic): 20 procedurally generated multi-step word problems (e.g., "A
store has N apples. They receive M crates of K apples..."). Ground truth computed at
generation time.

**Few-shot CoT exemplars:** 8 examples drawn from the GSM8K training set, formatted as
`Q: [question] A: [reasoning steps]. The answer is [number].` — same count as the paper's 8
manually written exemplars for arithmetic reasoning.

### 4. Few-Shot CoT Prompt Construction

**Function: `build_cot_prompt(question, exemplars)`** — assembles 8 exemplar Q&A pairs in
the CoT format, followed by the test question ending with `A:`. The model then generates the
reasoning path and answer.

**Helper functions:**
- `extract_gsm8k_answer(answer_text)` — pulls the numeric answer from after `####` in GSM8K's format
- `extract_gsm8k_cot(answer_text)` — pulls the reasoning text (everything before `####`)

Prompts are truncated to fit GPT-2's 1024-token context window (input + max_new_tokens ≤ 1024).

### 5. Generation: Greedy and Sampling

**Function: `generate(prompt, max_new_tokens, do_sample, temperature, top_k)`** — tokenizes
the prompt, runs `model.generate()`, and extracts only newly generated tokens.

Two modes, matching the paper:
- **Greedy decode** (`do_sample=False`): single deterministic path — the CoT-prompting baseline
- **Temperature sampling** (`do_sample=True`, T=0.7, top_k=40): diverse reasoning paths for self-consistency

**Function: `extract_number(text)`** — extracts the final answer number from model output.
Looks for "The answer is X" pattern first, falls back to the last number in the text.

### 6. Self-Consistency Evaluation

**Function: `evaluate_self_consistency(task_name, problems, num_test)`** — for each problem:
1. Build the few-shot CoT prompt
2. Generate one greedy path (CoT baseline)
3. Sample 40 reasoning paths with temperature sampling
4. For each N in [1, 5, 10, 20, 40], take the first N sampled paths and majority-vote
5. Compare greedy accuracy vs self-consistency accuracy at each N

**Answer aggregation:** Uses `Counter(valid_answers).most_common(1)` — simple unweighted
majority vote, matching the paper's finding (Table 1) that majority vote performs nearly
identically to weighted aggregation.

**Data flow:**
```
Question → build_cot_prompt() → prompt string
  → generate(prompt, do_sample=False) → greedy answer
  → generate(prompt, do_sample=True) × 40 → 40 sampled answers
  → Counter(sampled[:N]).most_common(1) → self-consistency answer at N
  → compare to ground truth
```

### 7-8. Per-Task Results (GSM8K, MultiArith)

Each task is evaluated separately. Progress is logged every 10 problems with running accuracy
counts for greedy and self-consistency at N=40.

### 9. Aggregated Results

Prints a comparison table showing Greedy accuracy and self-consistency accuracy at each N
value [1, 5, 10, 20, 40], plus the average across tasks and the improvement (SC@40 - Greedy).

### 10. Visualization

Two side-by-side plots (one per task) showing accuracy (%) vs number of sampled reasoning
paths, with a horizontal dashed line for the greedy CoT baseline — matching the paper's
Figure 2 format.

### 11. Example Reasoning Paths

Shows up to 2 examples per task where self-consistency and greedy decoding produced different
answers, including the greedy output and the distribution of sampled answers, to illustrate
how diversity in reasoning paths leads to different aggregation outcomes.

### 12. Comparison with Paper Results

Compares GPT-2 small (124M) results with the paper's reported PaLM-540B numbers for GSM8K
(greedy 56.5% → SC@40 74.4%) and MultiArith (greedy 94.7% → SC@40 99.3%), highlighting the
scale gap and its implications.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `build_cot_prompt()` | Assemble few-shot CoT prompt with 8 exemplars + test question |
| `generate()` | Run GPT-2 generation with greedy or temperature sampling |
| `extract_number()` | Extract final answer number from model output |
| `extract_gsm8k_answer()` | Parse numeric answer from GSM8K `####` format |
| `extract_gsm8k_cot()` | Parse reasoning steps from GSM8K answer text |
| `evaluate_self_consistency()` | Evaluate greedy vs self-consistency at varying N |
| `gen_multiarith()` | Generate MultiArith-style arithmetic word problems |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | UL2-20B, GPT-3-175B, LaMDA-137B, PaLM-540B | GPT-2 small (124M) |
| Test set size | Full benchmarks (100s-1000s of problems) | 30 GSM8K + 20 MultiArith |
| Tasks | 10 benchmarks (arithmetic + commonsense + symbolic) | 2 arithmetic tasks (GSM8K, MultiArith) |
| Few-shot exemplars | 8 manually written for arithmetic | 8 from GSM8K train set |
| Sampling parameters | T=0.5-0.7, k=40 (varies by model) | T=0.7, k=40 (PaLM setting) |
| Sampled paths | 40 per problem, 10 runs averaged | 40 per problem, single run |
| Aggregation strategy study | 5 strategies compared (Table 1) | Majority vote only (paper's recommended default) |
| Scale study | 4 models from 20B to 540B (Figure 4) | Single model (124M) |
