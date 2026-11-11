---
title: "Tiny Recursive Model: Talk & Q&A"
parent: "067 Tiny Recursive Model"
nav_order: 2
---
# TALK — Less is More: Recursive Reasoning with Tiny Networks (Tiny Recursive Model)

## Talks, Slides and Press Coverage

1. **ARC Prize Foundation — "The Hidden Drivers of HRM's Performance on ARC-AGI" (2025):** An independent analysis finding that deep supervision (not hierarchical recursion) was the primary driver of HRM's gains, going from 19% to 39% accuracy. This finding directly motivated TRM's simplification. ([link](https://arcprize.org/blog/hrm-analysis))
2. **lucidrains — "tiny-recursive-model" (2025):** An unofficial PyTorch implementation of TRM by Phil Wang (lucidrains), with 192 stars on GitHub, providing a clean reference for the architecture. ([link](https://github.com/lucidrains/tiny-recursive-model))
3. **AlexiaJM — "TinyRecursiveModels" (2025):** The author's official code repository for TRM experiments. ([link](https://github.com/AlexiaJM/TinyRecursiveModels))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2510.04871).

**Q1: Why does TRM use only 2 layers instead of more?**
A1 (from the paper, Section 4.4): The authors found that "adding layers decreased generalization due to overfitting" and that "using 2 layers (instead of 4 layers) maximized generalization." They note this is surprising because "with modern neural networks, generalization tends to directly correlate with model sizes," but when "data is too scarce and model size is large, there can be an overfitting penalty."

**Q2: Why does TRM use a single network instead of two like HRM?**
A2 (from the paper, Section 4.3): Since "z ← fL(x+y+z) contains x but y ← fH(y+z) does not contain x, the task to achieve (iterating on z versus using z to update y) is directly specified by the inclusion or lack of x in the inputs." Thus a single shared network suffices, and doing so "obtain[s] better generalization on Sudoku-Extreme (improving TRM from 82.4% to 87.4%) while reducing the number of parameters by half."

**Q3: Why does TRM backpropagate through the full recursion instead of using the 1-step gradient approximation?**
A3 (from the paper, Section 4.1): HRM's 1-step approximation relies on the Implicit Function Theorem, which "is not guaranteed" to apply because "there is no guarantee that a fixed-point is reached." TRM bypasses this entirely: "we simply back-propagate through the full recursion process," which yields a "massive boost in generalization on Sudoku-Extreme (improving TRM from 56.5% to 87.4%)."

**Q4: What is the simple reinterpretation of zH and zL from HRM?**
A4 (from the paper, Section 4.2): "zH is simply the current (embedded) solution" and "zL is a latent feature that does not directly correspond to a solution." Renamed as y (answer) and z (latent reasoning), the model needs both because "if we were not passing the previous reasoning z, the model would forget how it got to the previous solution y" and "if we were not passing the previous solution y, then the model would forget what solution it had."

## Common Misconceptions

1. **"TRM is just a smaller HRM"** — TRM is fundamentally different: it uses a single network (not two), full backpropagation (not 1-step approximation), no fixed-point theorem, no biological arguments, and a simpler halting mechanism. The ablation table shows each change contributes independently (e.g., removing the 1-step approximation alone improves accuracy from 56.5% to 87.4%).
2. **"More layers would improve TRM"** — The paper explicitly shows the opposite: 4-layer networks perform worse than 2-layer ones (79.5% vs 87.4% on Sudoku-Extreme) due to overfitting on small datasets. The "less is more" finding is central to the paper's thesis.
3. **"TRM replaces chain-of-thought reasoning in LLMs"** — TRM is a supervised learning method, not a generative model. As the paper notes in Section 6: "currently, recursive reasoning models such as HRM and TRM are supervised learning methods rather than generative models. This means that given an input question, they can only provide a single deterministic answer."

## Real Citations

1. Alexia Jolicoeur-Martineau (2025). "Less is More: Recursive Reasoning with Tiny Networks." *Preprint.* [arXiv:2510.04871](https://arxiv.org/abs/2510.04871) — the original paper.
2. Wang et al. (2025). "Hierarchical Reasoning Model." [arXiv:2506.21734](https://arxiv.org/abs/2506.21734) — the HRM paper that TRM directly improves upon.
3. Bai et al. (2019). "Deep Equilibrium Models." *NeurIPS 2019.* [arXiv:1909.01377](https://arxiv.org/abs/1909.01377) — the DEQ paper whose 1-step gradient approximation HRM adopted and TRM rejected.

**Citation count:** about 221 citations on [OpenAlex](https://openalex.org/works/W4404948530) (checked November 2026).
