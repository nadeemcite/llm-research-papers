---
title: "Generative Agents: Code architecture"
parent: "072 Generative Agents"
nav_order: 1
---
# Code Architecture -- Generative Agents Notebook

## Overview

The notebook implements a simplified Generative Agents architecture for 3 agents in a toy town. Each agent has a memory stream with recency-importance-relevance retrieval, a neural importance scorer (4,929-parameter MLP), a trained sentence embedding network for relevance scoring, a reflection module that triggers when cumulative importance exceeds a threshold, and a structured planning module. The system simulates 20 time steps of agent activity, then interviews each agent to measure behavioral coherence. The full architecture achieves 87% interview accuracy (13/15), vs 80% (12/15) without reflection and 0% (0/15) without memory, planning, or reflection.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copied from the code cell):
- `N_AGENTS = 3` -- number of agents (paper uses 25)
- `N_STEPS = 20` -- simulated time steps, each = 1 game-hour (paper runs 2 full game days = ~48 hours)
- `REFLECTION_THRESHOLD = 100` -- cumulative importance to trigger reflection (paper uses 150)
- `TOP_K_RETRIEVAL = 5` -- number of memories retrieved per query (paper does not fix this)
- `RECENCY_DECAY = 0.995` -- exponential decay per game-hour (same as paper)
- `EMBED_DIM = 64` -- embedding dimension for relevance scoring (paper uses LLM embeddings)
- `EPOCHS = 30` -- training epochs for the importance scorer and embedding network
- `LR = 1e-3` -- Adam learning rate
- `BATCH_SIZE = 8` -- mini-batch size

### 2. Toy Town Environment

**Class: `ToyTown`** -- defines the world state

Contains:
- `locations`: list of 3 locations -- Hobbs Cafe, Library, Park
- `agents`: list of 3 Agent objects with names, personas, and initial locations
- `time_step`: current simulated hour (0-indexed)
- `event_queue`: events to inject at specific time steps (e.g., party announcement at step 5)

Agent personas are one-paragraph descriptions, split into semicolon-delimited seed memories, matching the paper's format:
- Isabella Rodriguez: cafe owner, plans a Valentine's Day party
- Klaus Mueller: student researching gentrification
- Maria Lopez: student who has a crush on Klaus

### 3. Memory Stream

**Class: `MemoryStream`** -- the core data structure

Each memory entry contains:
- `content`: natural-language description (string)
- `timestamp`: game-hour when created (int)
- `importance`: 1--10 score (from neural scorer)
- `embedding`: 64-dim vector (from embedding network)
- `type`: "observation", "reflection", or "plan"
- `last_accessed`: timestamp of last retrieval (for recency scoring)

**Methods:**
- `add(content, timestamp, importance, embedding, mem_type)`: append a memory
- `retrieve(query_embedding, query_text, current_time, top_k)`: score and return top-k memories

**Retrieval scoring:** For each memory, compute:
- recency = RECENCY_DECAY ^ (current_time - last_accessed)
- importance = memory.importance / 10.0 (normalized to [0, 1])
- relevance = cosine_similarity(query_embedding, memory.embedding)
- score = (1/3) * recency + (1/3) * importance + (1/3) * relevance

All three scores are min-max normalized across the candidate set before combining.

### 4. Neural Importance Scorer

**Class: `ImportanceScorer`** (4,929 parameters)

Architecture: `Embedding(vocab_size, 32) -> GlobalAvgPool -> Linear(32, 32) -> ReLU -> Linear(32, 1) -> Sigmoid -> scale to [1, 10]`

Input: tokenized text (variable length) -> Output: importance score (1--10)

Trained on 50 synthetic examples with hand-labeled importance scores:
- "brushing teeth" -> 1
- "deciding to throw a party" -> 9
- "meeting a new neighbor" -> 7
- "eating breakfast" -> 2

Replaces the paper's LLM-based poignancy rating prompt.

### 5. Sentence Embedding Network

**Class: `EmbeddingNetwork`** (11,840 parameters)

Architecture: `Embedding(vocab_size, 64) -> GlobalAvgPool -> Linear(64, 64) -> Tanh`

Input: tokenized text (variable length) -> Output: 64-dimensional embedding vector

Trained with a contrastive objective: similar memories (same topic) should have high cosine similarity, dissimilar memories should have low similarity. 60 training pairs generated from the toy events.

Replaces the paper's LLM-generated embeddings for relevance scoring.

### 6. Reflection Module

**Function: `generate_reflection(agent, current_time)`**

Triggers when the sum of importance scores for memories added since the last reflection exceeds `REFLECTION_THRESHOLD` (30).

Process (mirrors paper Section 4.2):
1. Gather the 10 most recent memories
2. Generate 3 questions: "What is [agent name] passionate about?", "What recent events are most important?", "What relationships has [agent name] formed?"
3. Retrieve relevant memories for each question
4. Synthesize insights by clustering memories by type and extracting the dominant theme
5. Store each insight as a reflection memory with importance 8 (reflections are high-importance by default)

**Data flow:**
```
recent_memories -> question_generation -> retrieve_per_question
  -> cluster_and_synthesize -> reflection_memories -> add to stream
```

### 7. Planning Module

**Function: `generate_plan(agent, time_step)`**

Creates a daily schedule for each agent based on their persona and time of day:
- Morning (steps 0-5): morning routine, breakfast
- Midday (steps 6-12): work/study/research
- Afternoon (steps 13-17): social interactions, errands
- Evening (steps 18-20): dinner, relaxation

Each plan entry: (action_text, location, start_time, duration, importance)

Plans are stored in the memory stream as type="plan" with importance 5.

When an unexpected event occurs (e.g., hearing about a party), the agent may revise its plan by inserting the new activity.

### 8. Agent Simulation Loop

**Function: `simulate_step(town, step)`**

For each time step:
1. For each agent:
   a. Observe current location and other agents present
   b. Add observations to memory stream (with neural importance scoring)
   c. Check if reflection threshold is met -> generate reflections
   d. Retrieve relevant memories for current context
   e. Determine next action (follow plan or react to observation)
   f. If agents co-located, generate a dialogue (simple template-based)
2. Process any scheduled events (e.g., party announcement)
3. Advance time step

### 9. Agent Interview

**Function: `interview_agent(agent, questions)`**

Asks each agent 5 questions across 5 categories (matching paper Section 6.1):
- Self-knowledge: "Give an introduction of yourself"
- Memory: "What did you do earlier today?"
- Planning: "What are your plans for later?"
- Reacting: "If you heard about a party, what would you do?"
- Reflecting: "What have you learned about your neighbors?"

The agent retrieves relevant memories for each question and constructs a response from the top-k retrieved memories. A response is scored as "correct" if it contains the key facts expected for that question.

### 10. Ablation Study

Evaluates 3 conditions:
- **Full**: memory + reflection + planning
- **No reflection**: memory + planning (reflection disabled)
- **No memory**: no memory, no planning, no reflection (baseline -- responds only from seed description)

Each condition runs the full 20-step simulation and 5-question interview per agent (15 total questions).

### 11. Results and Plots

- **Bar chart**: interview accuracy by condition (full, no-reflection, no-memory)
- **Information diffusion plot**: number of agents who know about the party over time
- **Memory stream visualization**: timeline of memories for one agent showing observations, reflections, and plans
- **Printed summary**: all numbers quoted in the docs as "Name: value" lines

## Key Functions

| Function / Class | Purpose |
|---|---|
| `ToyTown` | Defines the world: locations, agents, time, events |
| `MemoryStream` | Append-only memory store with recency-importance-relevance retrieval |
| `ImportanceScorer` | MLP that rates memory poignancy 1--10 from text |
| `EmbeddingNetwork` | Trained sentence encoder for relevance scoring |
| `generate_reflection(agent, time)` | Trigger reflection when importance threshold exceeded |
| `generate_plan(agent, step)` | Create structured daily schedule from persona |
| `simulate_step(town, step)` | Advance one time step: observe, reflect, plan, act, dialogue |
| `interview_agent(agent, questions)` | Ask questions, retrieve memories, construct response |
| `run_ablation(condition)` | Run full simulation under an ablation condition |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Number of agents | 25 in Smallville | 3 in a toy town |
| LLM backbone | gpt-3.5-turbo for all components | Neural MLPs (importance scorer + embedding network) |
| Importance scoring | LLM prompted to rate 1--10 | 4,929-parameter MLP trained on 50 examples |
| Embeddings | LLM-generated embedding vectors | Trained 64-dim EmbeddingNetwork (11,840 parameters) |
| Reflection | LLM generates 3 questions, retrieves memories, synthesizes 5 insights | Structured question generation, clustering, theme extraction |
| Planning | LLM top-down recursive expansion (day -> hour -> minute) | Deterministic schedule templates with persona-based customization |
| Dialogue | LLM generates natural-language dialogue turns | Template-based dialogue from memory retrieval |
| Simulation duration | 2 full game days (~48 hours) | 20 time steps (~20 game-hours) |
| Evaluation | 100 human raters, TrueSkill ranking | 15 interview questions scored by key-fact matching |
| Environment | Phaser web game with visual sprites, map, collision | In-memory world state with 3 locations |
