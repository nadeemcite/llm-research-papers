---
title: "Coconut: Talk & Q&A"
parent: "065 Coconut"
nav_order: 2
---
# TALK — Training Large Language Models to Reason in a Continuous Latent Space (Coconut)

## Talks, Slides and Press Coverage

1. **AI Papers Academy — "Coconut by Meta AI – Better LLM Reasoning With Chain of CONTINUOUS Thought?" (31 Dec 2024):** A detailed visual explainer of the Coconut method, covering the continuous-thought mechanism, multi-stage training, and the BFS-like reasoning pattern. Posted to Hacker News where it received 362 points and 156 comments. ([article](https://aipapersacademy.com/chain-of-continuous-thought/), [HN discussion](https://news.ycombinator.com/item?id=42555320))
2. **Gonzo ML — "Chain of Continuous Thought (Coconut)" (22 Apr 2025):** A technical blog post reviewing the paper's TL;DR, concept, implementation, evaluation, results, and analysis sections, with commentary on the latent reasoning paradigm. ([article](https://gonzoml.substack.com/p/chain-of-continuous-thought-coconut))
3. **COLM 2025 — Accepted Paper:** The paper was accepted to the Conference on Language Modeling (COLM) 2025, as noted on the arXiv page ("Accepted to COLM 2025"). ([arXiv page](https://arxiv.org/abs/2412.06769))
4. **Official code — facebookresearch/coconut:** The authors released the official implementation on GitHub. ([repository](https://github.com/facebookresearch/coconut))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2412.06769).

**Q1: Why feed hidden states back as input instead of decoding them to words first?**
A1 (from the paper, Section 1): "This modification frees the reasoning from being within the language space, and the system can be optimized end-to-end by gradient descent, as continuous thoughts are fully differentiable." The paper argues that language space is not always optimal for reasoning: "most word tokens primarily ensure textual coherence and are not essential for reasoning, while some critical tokens require complex planning and pose huge challenges to LLMs."

**Q2: What is the multi-stage curriculum and why is it needed?**
A2 (from the paper, Section 3, Training Procedure): "We leverage language CoT data to supervise continuous thought by implementing a multi-stage training curriculum inspired by [iCoT]. In the initial stage, the model is trained on regular CoT instances. In the subsequent stages, at the k-th stage, the first k reasoning steps in the CoT are replaced with k×c continuous thoughts." Without it, Coconut "w/o curriculum" achieves only 14.4% on GSM8k (vs 34.1% with curriculum), showing "the LLM still needs guidance to learn latent reasoning" (Section 5.3).

**Q3: How can continuous thoughts perform breadth-first search?**
A3 (from the paper, Section 1): "Continuous thoughts can encode multiple potential next steps simultaneously, allowing for a reasoning process akin to breadth-first search (BFS). While the model may not initially make the correct decision, it can maintain many possible options within the continuous thoughts and progressively eliminate incorrect paths through reasoning, guided by some implicit value functions."

**Q4: Why is latent space better for planning than language space?**
A4 (from the paper, Section 4.4): "Nodes explored in the early reasoning stages are inherently more challenging to evaluate accurately because they are farther from the final target nodes. In contrast, nodes positioned closer to potential targets, having fewer subsequent exploration possibilities, can be assessed accurately with higher confidence." By maintaining multiple candidate paths and delaying deterministic decisions, "latent reasoning significantly enhances the model's ability to differentiate correct paths from incorrect ones."

**Q5: Does Coconut always outperform CoT?**
A5 (from the paper, Section 5.3): No. On GSM8k, Coconut (34.1%) does not surpass CoT (42.9%), though it offers "a superior trade-off between reasoning efficiency and accuracy." On ProntoQA and ProsQA, Coconut surpasses CoT while generating fewer tokens. The paper notes this depends on the task: "In two other synthetic tasks, we found that the variants of Coconut ... also achieve impressive accuracy. This indicates that the model's computational capacity may not be the bottleneck in these tasks."

**Q6: What does the loss function optimize — compressing the removed language thought?**
A6 (from the paper, Section 3, Training Procedure): "The objective does not encourage the continuous thought to compress the removed language thought, but rather to facilitate the prediction of future reasoning. Therefore, it's possible for the LLM to learn more effective representations of reasoning steps compared to human language."

## Common Misconceptions

1. **"Coconut eliminates chain-of-thought entirely"** — Coconut replaces language-space reasoning with continuous-thought reasoning, but it still uses a chain of reasoning steps. The "chain" is still there; it's just in continuous latent space rather than discrete language tokens. The model still goes through multiple reasoning steps, each producing a continuous thought that feeds into the next.

2. **"Continuous thoughts are just hidden states, so any model already does this"** — Standard LLMs compute hidden states but never feed them back as input embeddings. The hidden state is always projected through the LM head to produce a token probability distribution, and the next input is the embedding of the chosen token. Coconut bypasses this decode-encode bottleneck by feeding the raw hidden state directly, which is a fundamentally different computation graph that enables end-to-end gradient flow through the reasoning chain.

3. **"Coconut outperforms CoT on all tasks"** — The paper is clear that Coconut does not surpass CoT on GSM8k math reasoning (34.1% vs 42.9%). Its advantage is on logical reasoning tasks that require substantial search during planning (ProsQA, ProntoQA), and even there, the iCoT baseline achieves comparable or slightly better accuracy on some metrics.

4. **"The curriculum is just a training trick, not essential"** — The paper's ablation shows Coconut without curriculum training drops to 14.4% on GSM8k (vs 34.1% with curriculum) and 76.1% on ProsQA (vs 97.0%). The authors explicitly state: "the LLM still needs guidance to learn latent reasoning" (Section 5.3).

## Real Citations

1. Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, Yuandong Tian (2024). "Training Large Language Models to Reason in a Continuous Latent Space." *COLM 2025.* [arXiv:2412.06769](https://arxiv.org/abs/2412.06769) — the original paper.
2. Yifan Deng, Yejin Choi, Stuart Shieber (2024). "From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step." [arXiv:2405.14838](https://arxiv.org/abs/2405.14838) — iCoT, the curriculum training strategy that inspired Coconut's multi-stage approach.
3. Cheng et al. (2025). Theoretical framework demonstrating that continuous CoT can be more efficient than discrete CoT on certain tasks by encoding multiple reasoning paths in superposition states. Cited as reference 55 in the paper.
4. Wang et al. (2025). Analysis of training dynamics explaining how superposition emerges under the Coconut training objective. Cited as reference 54 in the paper.

**Citation count:** about 859 citations on [OpenAlex](https://openalex.org/works/W4405254469) (checked November 2026).
