---
title: "Voyager: Code architecture"
parent: "073 Voyager"
nav_order: 1
---
# Code Architecture -- Voyager Notebook

## Overview

The notebook implements a simplified Voyager agent in a text-based crafting world. The agent uses a neural curriculum network (2,190 parameters) to propose tasks, a text embedding network (1,920 parameters) for skill library retrieval, a template-based code generator with two modes (full prerequisite chain vs naive), a self-verification module, and an iterative prompting loop (up to 4 rounds). Over 20 iterations, the full Voyager agent discovers 13 unique items and stores 13 reusable skills, reaching the iron tech tree level.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copied from the code cell):
- `MAX_ITERATIONS = 20` -- prompting iterations per agent run (paper uses 160)
- `MAX_REFINE_ROUNDS = 4` -- code refinement rounds before giving up (same as paper)
- `TOP_K_SKILLS = 5` -- skills retrieved per query (same as paper)
- `EMBED_DIM = 32` -- embedding dimension for skill descriptions (paper uses ada-002, 1536 dims)
- `CURRICULUM_EPOCHS = 50`, `EMBED_EPOCHS = 40` -- training epochs for the two networks
- `LR = 1e-3` -- Adam learning rate

### 2. Text-Based Game World

**Class: `TextWorld`** -- defines the crafting environment

Contains:
- `inventory`: dict mapping item names to counts
- `location`: current location (forest, mine, river, plains)
- `ever_seen`: set of all items ever collected or crafted (for the unique-item metric)
- 10 recipes (planks, sticks, crafting table, wooden/stone/iron/diamond pickaxe, wooden/stone sword, furnace)
- 2 smelting recipes (iron ore to iron ingot, diamond ore to diamond)
- 4 locations with different raw resources

**Key methods:** `move()`, `gather()`, `craft()`, `smelt()`, `get_state_vector()`

### 3. Task Definitions

14 predefined tasks along a 4-level tech tree:
- Wooden: collect wood, craft planks, sticks, crafting table, wooden pickaxe
- Stone: collect stone, craft stone pickaxe
- Iron: collect iron ore, collect coal, craft furnace, smelt iron ingots, craft iron pickaxe
- Diamond: collect diamond ore, craft diamond pickaxe

Each task specifies action type (gather/craft/smelt), target item, count, and optional location.

### 4. Text Embedding Network

**Class: `TextEmbedder`** (1,920 parameters)

Architecture: `Embedding(vocab_size, 32) -> mean pooling -> Linear(32, 32) -> Tanh -> L2 normalize`

Input: tokenized text (variable length) -> Output: 32-dimensional unit vector

Trained with a contrastive objective: task descriptions from the same tech level should have high cosine similarity, different levels should have low similarity. 91 training pairs, 40 epochs.

Replaces the paper's OpenAI text-embedding-ada-002 (1536-dimensional).

### 5. Automatic Curriculum Network

**Class: `CurriculumNet`** (2,190 parameters)

Architecture: `Linear(20, 32) -> ReLU -> Linear(32, 32) -> ReLU -> Linear(32, 14)`

Input: 20-dimensional state vector (inventory counts of 20 key items, normalized to [0, 1]) -> Output: logits over 14 tasks

Trained on 90 synthetic samples: for each partial completion state (0 to 14 tasks done), the target is the next task in tech-tree order. 5 noisy variants per state. 50 epochs, reaching 100% accuracy.

Replaces the paper's GPT-4-based curriculum prompting. The actual agent uses an order-based curriculum (next uncompleted task in tech-tree order) rather than the network, since the network simply learns this ordering.

### 6. Skill Library

**Class: `SkillLibrary`**

Stores skills as (description, function, embedding, task_index) tuples. Retrieval uses cosine similarity between the query embedding and stored embeddings, returning the top-5 matches.

Each skill stores the full prerequisite chain as an executable function. When a task is completed, the successful action sequence is wrapped in a closure and added to the library.

Replaces the paper's vector database indexed by ada-002 embeddings.

### 7. Code Generator

**Class: `CodeGenerator`** with two modes:

- `generate_full()`: produces the complete prerequisite chain -- moves to the right location, gathers raw materials, crafts intermediates, then performs the target action. Used when the skill library provides in-context examples.
- `generate_naive()`: attempts only the direct action. For gather tasks, does not move to the right location unless feedback is received. For craft tasks, does not gather missing ingredients unless feedback is received. Each refinement round adds one level of prerequisite checking, but deeply nested chains may exceed the 4-round limit.

This mirrors the paper's finding (Section 2.3) that "LLMs struggle to produce the correct action code consistently in one shot" without skill context.

### 8. Self-Verification

**Class: `SelfVerifier`**

Checks whether the target item (or check item for smelting tasks) is present in sufficient quantity in the world's inventory. Returns (True/False, message).

Replaces the paper's GPT-4-based self-verification agent that acts as a critic.

### 9. Voyager Agent

**Class: `VoyagerAgent`** -- combines all components

Each iteration:
1. Propose task (curriculum or random)
2. If skill exists in library, replay it directly
3. Otherwise, retrieve relevant skills
4. Generate code (full if skills available, naive otherwise)
5. Execute and check verification
6. If not verified, refine using error feedback (up to 4 rounds)
7. If verified, store skill in library

Ablation flags: `use_curriculum`, `use_skill_library`, `use_verification`

### 10. Run Experiments

Runs 4 agent variants over 3 seeds each (seeds 42, 43, 44), 20 iterations per run:
- **Voyager**: all components enabled
- **No Skill Library**: curriculum + verification, no skill storage
- **No Curriculum**: skill library + verification, random task selection
- **Random**: no curriculum, no skill library, no verification

Reports unique items discovered, tasks completed, skills stored, and tech tree milestones.

### 11. Results Plots

Two subplots:
- **Left**: unique items discovered over iterations (averaged across seeds), showing Voyager's exploration curve vs ablations
- **Right**: tech tree mastery bar chart (wooden, stone, iron, diamond) for each variant

### 12. Skill Library Growth

Step plot showing cumulative skills added to Voyager's library over iterations. Lists all 13 skill descriptions.

### 13. Final Evaluation

Prints all numbers quoted in the docs as "Name: value" lines, including improvement ratios and per-variant milestone iteration counts.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `TextWorld` | Crafting environment with inventory, locations, recipes |
| `TextEmbedder` | Bag-of-words sentence encoder for skill retrieval |
| `CurriculumNet` | MLP that proposes tasks based on world state |
| `SkillLibrary` | Vector database of executable skills with embedding retrieval |
| `CodeGenerator` | Template-based code generation (full or naive mode) |
| `SelfVerifier` | Checks task completion by inspecting inventory |
| `VoyagerAgent` | Full agent combining curriculum, skills, code gen, verification |
| `tokenize` | Simple whitespace tokenizer with vocabulary building |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Environment | Minecraft via MineDojo + Mineflayer | Text-based crafting world with 4 locations |
| LLM backbone | GPT-4 (code gen, curriculum, verification) + GPT-3.5 (descriptions) | Template-based code generation, neural networks |
| Skill embeddings | OpenAI text-embedding-ada-002 (1536-dim) | Trained 32-dim TextEmbedder (1,920 params) |
| Curriculum | GPT-4 prompted with agent state + exploration progress | Order-based (next uncompleted task) + trained CurriculumNet |
| Code generation | GPT-4 generates JavaScript via Mineflayer API | Template-based Python action sequences |
| Self-verification | GPT-4 agent checks task completion and provides critique | Inventory check (target item present in sufficient quantity) |
| Iterative prompting | GPT-4 refines code using environment feedback + execution errors | Code generator retries with increasing prerequisite depth |
| Task space | Open-ended (GPT-4 proposes arbitrary Minecraft tasks) | 14 predefined tasks along a tech tree |
| Iterations | 160 prompting iterations | 20 iterations |
| Baselines | ReAct, Reflexion, AutoGPT | No Skill Library, No Curriculum, Random |
