---
title: "{{SHORT}}: Code architecture"
parent: "{{NNN}} {{SHORT}}"
nav_order: 1
---
# Code Architecture — {{SHORT}} Notebook

## Overview

{{2-4 sentences: what the notebook implements, on what data, and how the result is measured. The
result sentence must match the saved outputs ("mAP@0.5 of about 0.90 on 200 unseen test images").}}

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copy the values from the code cell, never from memory):
- `{{NAME}} = {{value}}` — {{meaning}} ({{paper value, if different}})
- `EPOCHS = {{N}}`, `LR = {{lr}}`, `BATCH_SIZE = {{B}}`

### 2. {{Data}}

**Class/function: `{{name}}`** — {{what it returns, shapes}}

### 3. {{Model}}

**Class: `{{Model}}`** (about {{printed parameter count}} parameters)
{{Layers, activations, what differs from the paper and why.}}

**Data flow:**
```
Input: (batch, ...)
  -> ... -> (batch, ...)
```

### 4. {{Loss}}

{{Formula in plain text; every pipe character written as a backslash-pipe outside code blocks and tables.}}

### 5. Training Loop

- {{Optimizer, lr, schedule, epochs: identical to section 1 and to the notebook}}
- {{What is logged and how often}}

### 6. Results and Plots

- {{Each plot cell: what it shows. Describe what the saved image really shows, flaws included.}}

## Key Functions

| Function / Class | Purpose |
|---|---|
| `{{name}}` | {{purpose (names must exist in the notebook)}} |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| {{Dataset}} | {{...}} | {{...}} |
| Optimizer / epochs | {{paper}} | {{notebook, same numbers as the code}} |
