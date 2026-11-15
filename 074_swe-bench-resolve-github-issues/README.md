---
title: "074 SWE-bench"
nav_order: 75
permalink: "/074_swe-bench-resolve-github-issues/"
---
# SWE-bench: Can Language Models Resolve Real-World GitHub Issues? (SWE-bench)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/swe-bench-resolve-github-issues)

- **Paper:** [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770)
- **Authors:** Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, Karthik Narasimhan
- **Lab, year:** Princeton Language and Intelligence (Jimenez, Yang, Yao, Press, Narasimhan; Wettig: Princeton University; Pei: University of Chicago), 2023 (ICLR 2024)

## Summary

SWE-bench is a benchmark that evaluates whether language models can resolve real-world software engineering problems drawn from GitHub. The authors collected 2,294 task instances from 12 popular open-source Python repositories. Each instance gives a model an issue description (a bug report or feature request) and the full codebase, and asks the model to produce a patch that resolves the issue.

The patch is then evaluated by applying it to the codebase and running the repository's own test suite — if the tests that previously failed now pass, the issue is considered resolved. This execution-based evaluation is far more rigorous than standalone code generation, because the model must navigate a large codebase, understand inter-file dependencies, and coordinate changes across multiple locations.

The results were sobering: even the best-performing model, Claude 2, resolved only 1.96% of issues using BM25 retrieval (a text-search method for finding relevant files). With "oracle" retrieval (handing the model the exact files that need editing), Claude 2 reached 4.8%. The authors also fine-tuned CodeLlama (a code-generation model) to create SWE-Llama in 7B and 13B parameter variants, but these resolved only 0.70% of issues with BM25 retrieval. Models consistently produced shorter, simpler patches than the gold solutions, often editing the wrong function or misunderstanding the codebase structure.

This repo's notebook builds a miniature SWE-bench-style evaluation framework with 4 synthetic Python repositories, 20 planted-bug task instances, BM25 file retrieval, a small transformer trained to generate patches, and execution-based evaluation with failure-type analysis.

## What Problem Does It Solve

Imagine you're a software engineer at a company that uses GitHub. A user files a bug report saying "the median function returns the wrong value for even-length lists." To fix it, you need to find the right file among thousands, understand how the code works, figure out what's wrong, and write a fix that doesn't break anything else. This is what real software engineering looks like — not writing a function from scratch, but navigating and modifying a large existing codebase.

Existing coding benchmarks like HumanEval test something much simpler: "write a function that checks if a string is a palindrome." The whole problem fits in one function, and you can check the answer by running a few test cases. SWE-bench asks: can a language model do the harder thing — fix a real bug in a real codebase, where the fix might span multiple files and the tests that verify it were written by the project's actual maintainers?

## Core Idea

- **Real-world task instances from GitHub:** Instead of hand-crafted coding puzzles, SWE-bench draws from actual merged pull requests that resolved GitHub issues, ensuring the tasks are authentic and the test coverage is real.
- **Execution-based evaluation:** A patch is correct only if it applies cleanly and the repository's own test suite passes — specifically, the "fail-to-pass" tests that were failing before the fix and passing after.
- **3-stage construction pipeline:** The authors filter ~90,000 pull requests through repo selection, attribute-based filtering (merged PRs that resolve an issue and contribute tests), and execution-based filtering (at least one test must flip from fail to pass), arriving at 2,294 high-quality instances.

## Key Method Details

- **Benchmark size:** 2,294 task instances across 12 Python repositories; a Lite subset of 300 instances for faster evaluation.
- **Task input:** Issue text (average 195 words) + complete codebase (average 3,010 non-test files, 438K lines).
- **Evaluation metric:** Percentage of instances resolved (patch applies + all fail-to-pass tests pass).
- **Gold patch statistics:** Average 1.7 files edited, 3.0 functions edited, 32.8 lines changed.
- **Results (BM25 retrieval):** Claude 2: 1.96%, GPT-4-turbo: 1.31%, SWE-Llama 13b: 0.70%, ChatGPT-3.5: 0.17%.
- **SWE-Llama training:** Fine-tuned CodeLlama 7B and 13B with LoRA on 19,000 issue-PR pairs from 37 repos (disjoint from the 12 eval repos); sequences over 30,000 tokens excluded, reducing to ~10,000 effective instances.

### Simplifications in This Notebook

- Synthetic repositories: 4 repos with 3–5 files and ~80 lines each, instead of 12 real repos with thousands of files
- 20 hand-crafted planted-bug instances (15 train, 5 test), instead of 2,294 real GitHub issues
- Word-level TF-IDF features and a 2-layer MLP (~41K parameters) for instance retrieval, instead of CodeLlama 7B/13B
- Nearest-neighbor retrieval + structured edit transfer for patch generation, instead of seq2seq generation
- 50 training epochs on 15 instances, instead of fine-tuning on a separate 19,000-instance training set

### What the Notebook Result Shows

On the 20 synthetic instances, the retrieval-based model resolves 16/20 (80%) — 15 exact-match training instances all resolve, while 4 of 5 held-out test instances fail (2 syntax errors from wrong edits, 2 test failures from incorrect fixes). Oracle retrieval (applying the gold fix line directly) resolves 20/20 (100%), confirming the paper's finding that context quality is the bottleneck. The full pipeline (retrieval → edit → execution → test) runs end-to-end with failure-type categorization, mirroring the paper's methodology.

## Influence

SWE-bench became the standard benchmark for evaluating LLM-based software engineering agents. It directly motivated SWE-agent (Yang et al., 2024), which designed agent-computer interfaces for automated software engineering, and SWE-bench Verified, a human-verified subset. The leaderboard at swebench.com has been used by Anthropic, OpenAI, Google, and others to track progress: by 2024, Claude 3.5 Sonnet reached over 40% on SWE-bench Verified, showing rapid improvement from the 1.96% baseline. The benchmark influenced the creation of similar benchmarks for other languages and domains (SWE-bench Multimodal, SWE-bench Java).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/074_swe-bench-resolve-github-issues/solution.ipynb) | Mini SWE-bench framework: 4 synthetic Python repos with 20 planted-bug instances, BM25 retrieval, a small transformer patch generator, and execution-based evaluation with failure analysis |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
