---
title: "075 SWE-agent"
nav_order: 76
permalink: "/075_swe-agent-agent-computer-interfaces/"
---
# SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering (SWE-agent)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/swe-agent-agent-computer-interfaces)

- **Paper:** [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793)
- **Authors:** John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, Ofir Press
- **Lab, year:** Princeton Language and Intelligence, Princeton University (Yang and Jimenez: equal contribution; Lieret: Princeton University), 2024 (NeurIPS 2024)

## Summary

SWE-agent asks a deceptively simple question: when a language model acts as an autonomous agent that fixes bugs in a software repository, does the *interface* it uses to interact with the computer matter? Just as human programmers benefit from IDEs (integrated development environments) rather than raw terminal commands, the authors argue that LM agents are a new class of "end user" with their own needs — and they need interfaces designed specifically for them, not for humans.

The paper introduces the **agent-computer interface (ACI)**, an abstraction layer between the LM and the computer that provides a curated set of simple commands for searching, navigating, viewing, and editing files. The ACI enforces guardrails (like a linter that catches syntax errors before an edit is applied) and gives the agent concise, informative feedback after every action. Built on top of GPT-4 Turbo, SWE-agent resolved 12.47% of the 2,294 task instances in SWE-bench (a benchmark of real GitHub issues), up from the previous best of 3.8%. On SWE-bench Lite (300 instances), it reached 18% — a 10.7 percentage-point improvement over a shell-only baseline that uses the raw Linux terminal.

This repo's notebook builds a miniature agent-computer interface with a simulated file system, structured ACI commands (open, search, edit, scroll, run), lint guardrails, and a rule-based agent that uses the ACI to resolve planted bugs in synthetic Python repos — comparing ACI-based resolution against raw shell-style access to demonstrate the interface design effect.

## What Problem Does It Solve

Imagine you're fixing a bug in a large codebase. You open your IDE, search for the relevant function, view the code, make a targeted edit, and run the tests. Now imagine doing the same thing with only a raw Linux terminal — typing `cat` to view files, `sed` to edit them, and `grep` to search. It's painful and error-prone. That's exactly the situation LM agents face when they're given a bare shell to work with: they struggle to reliably edit files, get lost in verbose search output, and make syntax errors they can't detect.

The paper's insight is that the interface itself is a bottleneck. An LM agent using a raw Linux shell wastes turns on inefficient exploration (chains of `cd`, `ls`, `cat`), produces edits that silently break the code, and gets no feedback when an edit fails. The ACI solves this by providing a small set of purpose-built commands with guardrails and clear feedback — like a miniature IDE designed for an AI rather than a human.

## Core Idea

- **Agent-Computer Interface (ACI):** A custom interface layer between the LM and the computer, providing simple, purpose-built commands for file viewing, searching, editing, and execution — instead of giving the agent raw shell access.
- **Design principles for ACIs:** (1) Actions should be simple and easy to understand; (2) actions should be compact and efficient (consolidate operations into as few steps as possible); (3) environment feedback should be informative but concise; (4) guardrails should mitigate error propagation and hasten recovery.
- **Guardrails and linting:** The edit command runs a syntax linter before applying changes — if the edit introduces errors, it's rejected and the agent is asked to try again, preventing cascading failures from broken code.

## Key Method Details

- **ACI commands:** `open` (file viewer showing 100 lines with line numbers), `scroll_up`/`scroll_down`/`goto` (navigation), `find_file`/`search_file`/`search_dir` (search with max 50 results), `edit` (replace a line range with lint checking), `python`/`bash` (execution).
- **Context management:** Observations older than the last 5 are collapsed into single lines, keeping context concise. Error messages from malformed generations are kept only once. No-output commands get explicit "Your command ran successfully and did not produce any output" feedback.
- **Results (SWE-bench full):** SWE-agent w/ GPT-4 Turbo: 12.47% resolved (286/2,294); w/ Claude 3 Opus: 10.46%. Previous best (RAG w/ Claude 3 Opus): 3.79%.
- **Results (SWE-bench Lite):** SWE-agent w/ GPT-4 Turbo: 18.00% (54/300); Shell-only w/ GPT-4 Turbo: 11.00% — a 64% relative improvement from the ACI.
- **Ablation findings (SWE-bench Lite):** Removing the edit command drops performance to 10.3% (−7.7); removing linting drops it to 15.0% (−3.0); iterative search (one-by-one results) drops it to 12.0% (−6.0); showing 30 lines instead of 100 drops to 14.3% (−3.7).
- **HumanEvalFix:** 87.7% pass@1 in Python (vs 47.0% for GPT-4 without the ACI).

### Simplifications in This Notebook

- Simulated file system: 4 synthetic Python repos with 3–5 files each (~20–30 lines per file), instead of 12 real repos with thousands of files
- Rule-based agent with scripted heuristics, instead of GPT-4 Turbo generating thoughts and commands at each turn
- 20 planted-bug task instances (15 train, 5 test), instead of 2,294 real GitHub issues
- ACI commands implemented as Python methods (open, search, edit, scroll, run_tests) with lint checking, instead of a full shell-based ACI with LM-generated commands
- Comparison between ACI-mode and shell-mode agents, instead of the paper's full ablation suite
- 50% simulated edit-mistake rate (paper: 51.7% of trajectories have failed edits)

### What the Notebook Result Shows

On 20 synthetic instances with a 50% simulated edit-mistake rate, the ACI-based agent resolves 9/20 (45%) while the shell-only agent resolves 7/20 (35%) — a 10 percentage-point improvement from the ACI's lint guardrail. The ACI caught 3 syntax errors before they could cascade (each was retried with the correct fix), while the shell agent had 6 silent failures where broken edits were applied without detection. Both agents fail on control instances (no real bug) where the tests already pass, and on instances where the buggy line cannot be found. The 10pp gap mirrors the paper's finding that the ACI accounts for a ~7.7pp improvement on SWE-bench Lite (Table 3).

## Influence

SWE-agent became one of the most influential open-source coding agent systems. It directly led to SWE-bench Verified (a human-verified subset of SWE-bench), and the SWE-agent leaderboard at swe-agent.com became a standard tracking tool for the field. By 2024, systems building on SWE-agent's ACI principles — including Devin (Cognition AI), AutoCodeRover, and Agentless — pushed SWE-bench Verified resolve rates above 40%. The paper's core insight that interface design matters as much as model capability influenced the design of subsequent agent frameworks like OpenHands (formerly OpenDevin) and Aider's tool-use abstractions.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/075_swe-agent-agent-computer-interfaces/solution.ipynb) | Mini agent-computer interface with simulated file system, ACI commands (open, search, edit, scroll, run), lint guardrails, and rule-based agent comparing ACI vs shell modes on 20 planted-bug instances |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, key classes, and design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
