---
title: "Persona Vectors: Code architecture"
parent: "098 Persona Vectors"
nav_order: 1
---
# Code Architecture — Persona Vectors Notebook

## Overview

The notebook extracts a "sycophancy" persona vector (norm 1.3273) from Qwen2.5-0.5B-Instruct (494M parameters, 24 layers) by computing the difference in mean residual-stream activations at layer 12 between sycophantic and honest prompt responses. It then demonstrates three applications from the paper: steering (modulating sycophancy during generation), monitoring (predicting sycophancy from pre-generation projections), and finetuning-shift measurement (measuring how activations shift along the persona vector after fine-tuning). The persona vector separates sycophantic from honest responses with a projection separation of 1.3273. Steering at coefficients -3.0 to +3.0 moves sycophancy scores from 35.6 to 52.4. Fine-tuning on sycophantic data shifts activations +0.9527 (score 35.6 to 69.6); honest data shifts -0.4166 (score to 47.6).

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (values copied from the code cell):
- `MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'` — 494M param chat model (paper uses Qwen2.5-7B or Llama-3.1-8B)
- `STEERING_LAYER = 12` — middle of 24 layers, chosen a priori (paper empirically selects best layer)
- `N_ROLLOUTS = 3` — generations per question per prompt (paper uses 10)
- `EPOCHS = 2`, `LR = 1e-5`, `BATCH_SIZE = 2`, `MAX_SEQ_LEN = 256`

### 2. Data: Contrastive Prompts, Evaluation Questions, and Training Data

**Variables: `SYCOPHANTIC_PROMPTS`, `HONEST_PROMPTS`, `EVAL_QUESTIONS`, `MONITORING_PROMPTS`, `SYCOPHANCY_TRAIN`, `HONEST_TRAIN`**

Hand-crafted data artifacts that the paper generates automatically via Claude 3.7 Sonnet:
- 5 sycophantic system prompts + 5 honest system prompts (contrastive pair set)
- 10 evaluation questions that can elicit sycophancy (user makes false claims)
- 8 monitoring prompts interpolating from brutally honest to extremely sycophantic
- 20 sycophantic training samples (assistant agrees with false claims) + 20 honest controls (assistant corrects)

### 3. Model: Qwen2.5-0.5B-Instruct

Loads the model with `transformers.AutoModelForCausalLM`, float32, on GPU if available. Prints parameter count (494M), layer count (24), and hidden dimension (896). Model is set to eval mode for inference.

### 4. Extracting Activations

**Functions: `generate_response`, `extract_layer_activation`**

For each of 10 evaluation questions, generates 3 responses under a sycophantic system prompt and 3 under an honest system prompt (30 sycophantic + 30 honest = 60 total). For each response, runs a forward pass on the full token sequence and extracts the residual stream activation at layer 12, averaged across response tokens.

**Data flow:**
```
Question + system prompt -> tokenizer -> model.generate -> response text + token IDs
Token IDs -> model.forward(output_hidden_states=True) -> hidden_states[13] -> mean over response tokens -> (896,)
```

### 5. Computing the Sycophancy Persona Vector

Computes `persona_vector = mean(sycophantic_acts) - mean(honest_acts)` (shape: 896) and its unit-normalized version. Plots histograms of individual response projections and bar chart of mean projections by condition.

### 6. Steering: Controlling Sycophancy

**Class: `SteeringHook`** — A forward hook that adds `coefficient * vector` to the layer's output during generation. Registered on `model.model.layers[12]`.

**Function: `sycophancy_score`** — Heuristic 0–100 sycophancy scorer based on keyword matching (sycophantic phrases add +12, honest phrases add −12). Simplification of the paper's GPT-4.1-mini judge.

Tests steering at coefficients −3.0, 0.0, +3.0 across 5 questions. Plots sycophancy score vs steering coefficient.

### 7. Monitoring: Projection as Behavioral Predictor

For each of 8 monitoring prompts × 5 questions, extracts the last-prompt-token activation at layer 12 and projects onto the persona vector. Then generates a greedy response and scores it. Computes Pearson correlation between projection and score. Plots scatter colored by prompt type with trend line.

### 8. Fine-tuning: Measuring Activation Shift

**Functions: `measure_mean_activation`, `format_training_batch`, `finetune_model`**

Measures pre-finetuning mean last-prompt-token activation across 5 evaluation questions. Fine-tunes the model on 20 sycophantic training samples for 2 epochs with AdamW (lr=1e-5). Measures post-finetuning activation and computes the "finetuning shift" = `(post_act − pre_act) · v̂`. Also generates and scores responses.

### 9. Fine-tuning on Honest (Control) Data

Reloads the base model, measures pre-FT activation, fine-tunes on 20 honest training samples for 2 epochs, measures post-FT activation and computes shift. Generates and scores responses.

### 10. Results and Plots

Reloads the base model and measures baseline sycophancy scores. Produces two plots: (A) finetuning shift bar chart (Honest FT vs Sycophancy FT) and (B) sycophancy score bar chart (Base, Honest FT, Sycophancy FT). Prints final summary with all numbers.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `generate_response` | Generate a single response with a system prompt and user question |
| `extract_layer_activation` | Extract mean residual-stream activation across response tokens at a given layer |
| `SteeringHook` | Forward hook that adds a scaled steering vector to a layer's output |
| `sycophancy_score` | Heuristic 0–100 sycophancy scorer based on keyword matching |
| `measure_mean_activation` | Measure mean last-prompt-token activation across questions |
| `format_training_batch` | Tokenize chat-format training data with user prompt masking |
| `finetune_model` | Simple AdamW fine-tuning loop with gradient clipping |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model size | Qwen2.5-7B (7B) or Llama-3.1-8B (8B) | Qwen2.5-0.5B-Instruct (494M) |
| Trait evaluation | GPT-4.1-mini judge (0–100) | Heuristic keyword scorer (0–100) |
| Layer selection | Empirically best across all layers | Layer 12 (middle), chosen a priori |
| Evaluation questions | 40 (20 extraction + 20 eval) | 10 |
| Rollouts per question | 10 | 3 |
| Training data | Curated datasets via LLM pipeline | 20 hand-crafted samples |
| Fine-tuning method | LoRA (rank 32, alpha 64) | Full fine-tuning, AdamW |
| Traits studied | Evil, sycophancy, hallucination, optimism, humor | Sycophancy only |
| Artifact generation | Automated via Claude 3.7 Sonnet | Hand-crafted |
