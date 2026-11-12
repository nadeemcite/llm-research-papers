---
title: "069 ReAct"
nav_order: 70
permalink: "/069_react-synergizing-reasoning-acting/"
---
# ReAct: Synergizing Reasoning and Acting in Language Models (ReAct)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/react-synergizing-reasoning-acting)

- **Paper:** [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- **Authors:** Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao
- **Lab, year:** Princeton University (Yao, Narasimhan; Yao's work done during Google internship) and Google Research, Brain team, 2022 (ICLR 2023)

## Summary

ReAct asks a large language model to interleave two things it had previously been studied for separately: **reasoning** (chain-of-thought style thinking through a problem step by step) and **acting** (calling external tools such as a search API). At each step the model emits a *thought* (free-form reasoning about what to do next), an *action* (a tool call like `Search[query]`), and receives an *observation* (the tool's output). This Thought-Action-Observation loop continues until the model decides it has enough information and emits a final answer. The paper's central claim is that reasoning and acting are synergistic: reasoning helps the model plan and track what to search for, while acting grounds the model in real facts and reduces hallucination.

On question answering (HotpotQA, a multi-hop QA benchmark where answers require following chains of evidence across multiple Wikipedia articles) and fact verification (FEVER, a dataset where claims must be supported or refuted with evidence), ReAct with PaLM-540B (a 540-billion-parameter model from Google) overcomes the hallucination and error propagation that plague pure chain-of-thought by interacting with a simple Wikipedia API.

On HotpotQA it reaches 27.4 EM (exact match) and on FEVER 60.9 accuracy. A hybrid approach that falls back from CoT-SC (self-consistency) to ReAct reaches 35.1 EM on HotpotQA and 64.6 on FEVER.

On two interactive decision-making benchmarks, ALFWorld (text-based household tasks) and WebShop (online shopping), ReAct outperforms imitation and reinforcement learning methods by an absolute 34% and 10% success rate respectively, while being prompted with only one or two in-context examples.

This repo's notebook builds a ReAct agent from scratch with a toy Wikipedia corpus and a tiny neural query generator, and compares it against Act-only and CoT-only baselines on 15 test questions.

## What Problem Does It Solve

Imagine you're trying to answer a trivia question like "Where does the river that carved the Grand Canyon flow?" If you just *think* about it from memory (pure reasoning), you might hallucinate a wrong answer because your memory is imperfect. If you just *search* for keywords without thinking (pure acting), you search for "Grand Canyon" and find out it was carved by the Colorado River — but you stop there, missing the answer because "southwestern United States" is in the Colorado River article, not the Grand Canyon article.

ReAct's insight is that humans do both at once: you think about what you know, decide what to look up, read what you find, notice it mentions something else, think about whether that's enough, and look up more if needed. The paper formalizes this as a loop where the model alternates between internal reasoning ("The Grand Canyon was carved by the Colorado River — I need to search for the Colorado River") and external actions ("Search[Colorado River]").

Each observation guides the next thought. This reduces hallucination because answers are grounded in retrieved facts, and it solves multi-hop questions because the agent can follow reference chains across documents.

## Core Idea

- **Interleaved reasoning and acting:** The model generates thoughts and actions in turn, not separately. A thought like "I need to search for X" is immediately followed by `Search[X]`, and the observation feeds into the next thought.
- **Grounded reasoning:** By calling external tools (a Wikipedia API for QA tasks, environment interactions for decision-making), the model's reasoning is anchored in real observations rather than its parametric memory, reducing hallucination.
- **Few-shot prompting:** ReAct uses just 1-6 hand-written examples (trajectories of thoughts, actions, and observations) as in-context prompts — no fine-tuning is needed for the main results, though the paper shows fine-tuning with 3,000 examples makes even small models (PaLM-8B) outperform much larger prompted models.

## Key Method Details

- **Base model:** PaLM-540B (frozen, prompted with few-shot examples); GPT-3 results also reported in Appendix A.1.
- **Action space:** For QA/FEVER: `Search[query]` (retrieve Wikipedia paragraphs), `Lookup[keyword]` (find a sentence with the keyword), `Finish[answer]`. For ALFWorld: text commands like `go to coffeetable 1`, `take paper 2`. For WebShop: `search`, `click`, `buy`.
- **Prompting:** 6 in-context examples for HotpotQA, 3 for FEVER, 1-2 for ALFWorld and WebShop.
- **Results (PaLM-540B prompting):** HotpotQA EM: Standard 28.7, CoT 29.4, CoT-SC 33.4, Act 25.7, ReAct 27.4, CoT-SC → ReAct 34.2, ReAct → CoT-SC 35.1. FEVER Acc: Standard 57.1, CoT 56.3, CoT-SC 60.4, Act 58.9, ReAct 60.9, CoT-SC → ReAct 64.6, ReAct → CoT-SC 62.0.
- **ALFWorld:** ReAct (avg) 57% success, ReAct (best of 6) 71%, Act (best of 6) 45%, BUTLER (best of 8) 37%.
- **WebShop:** ReAct score 66.6, success rate 40.0%; Act score 62.3, SR 30.1%; IL+RL score 62.4, SR 28.7%.
- **Fine-tuning:** PaLM-8B fine-tuned ReAct outperforms all PaLM-62B prompting methods; PaLM-62B fine-tuned ReAct outperforms all 540B prompting methods.

### Simplifications in This Notebook

- Toy Wikipedia: 20 short documents about US geography instead of the real Wikipedia API
- A tiny MLP query generator (about 28K parameters) instead of PaLM-540B
- 40 training questions and 15 test questions instead of HotpotQA (90K QA pairs) or FEVER (185K claims)
- 30 epochs with Adam (lr 1e-3) instead of few-shot prompting of a frozen 540B model
- TF-IDF search (from scratch, no sklearn) instead of the Wikipedia API's Search/Lookup tools
- Cross-reference following (scanning observations for mentions of other documents) as the reasoning step, with the neural query generator as a learned fallback

### What the Notebook Result Shows

On 15 test questions (11 single-hop, 4 multi-hop), the ReAct agent achieves **93% accuracy** (14/15), outperforming both Act (73%, 11/15) and CoT (73%, 11/15). The key difference is on multi-hop questions: ReAct solves all 4 by following cross-references across documents, while Act and CoT solve 0/4 because they can only access a single document.

On single-hop questions all three methods perform similarly (10-11/11), since the answer is in the first document found. This is consistent with the paper's finding that ReAct helps most when multiple search steps are needed.

These toy numbers are not comparable to the paper's HotpotQA/FEVER results; a 28K-parameter MLP on 20 toy documents cannot match a 540-billion-parameter model on real Wikipedia.

## Influence

ReAct became one of the most influential papers in the LLM agent space. Its Thought-Action-Observation pattern was adopted by widely used agent frameworks including LangChain and LlamaIndex, and influenced subsequent agent systems such as AutoGPT and BabyAGI. The paper's idea of interleaving reasoning with tool use is now a standard design pattern for LLM agents. Follow-up work includes Reflexion (Shinn et al., 2023), which adds verbal self-reflection to the ReAct loop, and Toolformer (Schick et al., 2023), which trains models to self-label tool calls. The paper has been cited about 6,700 times ([Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/arXiv:2210.03629), checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/069_react-synergizing-reasoning-acting/solution.ipynb) | From-scratch ReAct agent in PyTorch with a neural query generator over a toy Wikipedia corpus, compared against Act-only and CoT-only baselines on 15 test questions (93% vs 73%) |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
