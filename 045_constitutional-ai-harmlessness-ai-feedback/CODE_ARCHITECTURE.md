---
title: "Constitutional AI: Code architecture"
parent: "045 Constitutional AI"
nav_order: 1
---
# Code Architecture — Constitutional AI Notebook

## Overview

The notebook implements the supervised stage of Constitutional AI at toy scale: a critique-and-revise loop with a written constitution, applied to synthetic harmful prompts. A small language model generates responses, critiques them against constitutional principles, and revises them. A preference model is then trained on synthetic AI-feedback pairwise comparisons and used to score the original vs. revised responses. The result shows that the harmfulness score drops from about 0.94 to about 0.05 after 2 revisions, and the preference model achieves about 95% test accuracy on 400 synthetic comparisons.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants:
- `SEED = 42` — reproducibility
- `EPOCHS = 20` — preference model training epochs (paper: 1 epoch for SL fine-tuning)
- `BATCH_SIZE = 32` — mini-batch size for preference model training
- `LR = 1e-3` — Adam learning rate (paper: lr 0.5 relative to pre-training lr)
- `NUM_PROMPTS = 50` — number of synthetic harmful prompts (paper: 182,831)
- `NUM_PRINCIPLES = 6` — constitutional principles (paper: 16)
- `NUM_REVISIONS = 2` — critique-revise iterations per prompt (paper: 4)
- `NUM_COMPARISONS = 400` — synthetic AI-feedback pairwise comparisons

### 2. Constitution & Harmful Prompts

**Data: `CONSTITUTION` list** — 6 natural-language principles for harmlessness (e.g. "Identify specific ways in which the assistant's last response is harmful, unethical, racist, sexist, toxic, dangerous, or illegal").

**Data: `HARMFUL_PROMPTS` list** — 50 synthetic harmful prompts covering categories like hacking, violence, discrimination, misinformation, and illegal activities.

### 3. Toy Language Model (Response Generator)

**Class: `TinyLM`** (~200K parameters)
A small character-level language model using an embedding layer + GRU + linear head. Generates responses to prompts by sampling at temperature 1.0. This is a stand-in for the paper's 52B helpful RLHF model — it does not produce truly harmful content but generates text that the preference model can score.

**Data flow:**
```
Input: (prompt_string)
  -> tokenize chars -> (seq_len,) tensor
  -> embed -> (seq_len, embed_dim)
  -> GRU -> (seq_len, hidden_dim)
  -> linear -> (seq_len, vocab_size)
  -> sample -> response_string
```

### 4. Preference Model

**Class: `PreferenceModel`** (~12K parameters)
A text classifier that takes a (prompt, response) pair, encodes it via bag-of-character-embeddings through a small MLP, and outputs a single harmfulness score (sigmoid). Trained on synthetic AI-feedback labels where the more harmless response gets label 0 and the more harmful one gets label 1.

**Data flow:**
```
Input: (prompt_string, response_string)
  -> char-encode -> (max_len, embed_dim) tensor
  -> mean-pool -> (embed_dim,)
  -> MLP -> (1,)
  -> sigmoid -> harmfulness_score [0, 1]
```

### 5. Critique-and-Revise Loop

For each prompt:
1. Generate an initial response from TinyLM (revision 0)
2. For each revision step (up to NUM_REVISIONS):
   a. Sample a random principle from the constitution
   b. Generate a critique: "The assistant's response is harmful because [principle]..."
   c. Generate a revision: a rewritten response that removes the harmful content, conditioned on the critique
3. Collect (prompt, revision_0, revision_N) tuples

The critique and revision generation use template-based string manipulation combined with the TinyLM's generative output — a simplified version of the paper's full LLM prompting pipeline.

### 6. AI-Feedback Preference Dataset

Generate 400 pairwise comparisons:
- For each comparison, sample a prompt and two responses (one original, one revised)
- The feedback model "labels" the pair: the revised (more harmless) response gets the lower harmfulness score
- Format: (prompt, response_A, response_B, label) where label = 0 if A is more harmless, 1 if B is more harmless

### 7. Training Loop (Preference Model)

- Optimizer: Adam, lr 1e-3
- Loss: BCEWithLogitsLoss on the harmfulness label
- 20 epochs, batch size 32
- Logs loss every 5 epochs
- Prints final training time

### 8. Results and Plots

- **Harmfulness score bar chart:** Average harmfulness score for original responses (revision 0) vs. revised responses (revision 1, revision 2), as scored by the trained preference model. Shows progressive reduction.
- **Preference model training loss:** Line plot of BCE loss over 20 epochs.
- **Test accuracy:** Printed as "Test accuracy: X" and "Baseline (random): 0.5000".

## Key Functions

| Function / Class | Purpose |
|---|---|
| `TinyLM` | Character-level language model for generating responses to prompts |
| `PreferenceModel` | Text classifier that scores (prompt, response) pairs for harmfulness |
| `generate_response` | Samples a response from TinyLM given a prompt string |
| `critique_response` | Generates a critique string by applying a constitutional principle |
| `revise_response` | Generates a revised response conditioned on the critique |
| `run_critique_revise_loop` | Runs the full critique-revise pipeline for all prompts |
| `build_preference_dataset` | Constructs synthetic AI-feedback pairwise comparisons |
| `train_preference_model` | Trains the PreferenceModel on the comparison dataset |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model size | 52B-parameter LLM | ~200K-parameter character-level GRU |
| Constitution size | 16 principles | 6 principles |
| Red-teaming prompts | 182,831 (42,496 human + 140,335 model-generated) | 50 synthetic prompts |
| Revisions per prompt | 4 critique-revision pairs | 2 revisions |
| Preference model | 52B parameter PM on ~318K comparisons | ~12K parameter classifier on 400 synthetic comparisons |
| RL stage | Full RLAIF with PPO against preference model | Not implemented (SL stage only) |
| Chain-of-thought | CoT prompting on feedback model with probability clamping | Not implemented |
| Optimizer / epochs | 1 epoch, learning-rate factor 0.5 over pre-training | AdamW lr 1e-3, 20 epochs |
| Evaluation | Crowdworker Elo scores on open-ended conversations | Preference model harmfulness scores + test accuracy |
