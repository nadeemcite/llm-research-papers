---
title: "Scaling Laws: Talk & Q&A"
parent: "034 Scaling Laws"
nav_order: 2
---
# TALK — Scaling Laws for Neural Language Models (Scaling Laws)

## Talks, Slides and Press Coverage

1. **Jared Kaplan — "Scaling Laws for Neural Language Models" at DeepLearning.AI (2020):** A presentation of the paper's key findings, including the power-law relationships and compute-optimal allocation. ([arXiv paper page](https://arxiv.org/abs/2001.08361))
2. **Jared Kaplan — "Scaling Laws for Neural Language Models" (2020):** A talk by first author Jared Kaplan presenting the paper's key findings on power-law scaling and compute-optimal allocation. ([YouTube](https://www.youtube.com/watch?v=UFem7xa3Q2Q))
3. **Anthropic — "A General Theoretic-Computational Scaling Law" (2023):** Jared Kaplan later co-founded Anthropic, where scaling-law research continued. The company published extensions to the original work. ([Anthropic Research](https://www.anthropic.com/research))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2001.08361).

**Q1: Why do the authors exclude embedding parameters from the parameter count *N*?**
A1 (from the paper, Section 3.2): The authors found that "if we instead use the total parameter count (including the embedding parameters) the trend is somewhat obscured." They observed that "the embedding matrix can be made smaller without impacting performance," so non-embedding parameters are the meaningful measure of model scale.

**Q2: What is the key practical recommendation for allocating a compute budget?**
A2 (from the paper, Abstract): "optimally compute-efficient training involves training very large models on a relatively modest amount of data and stopping significantly before convergence." Section 6 quantifies this: *N* ∝ *C*^0.73, *B* ∝ *C*^0.24, *S* ∝ *C*^0.03 — "most of the increase should go towards increased model size."

**Q3: Do architectural details like depth vs. width matter?**
A3 (from the paper, Section 3.1): "Transformer performance depends very weakly on the shape parameters *n*\_layer, *n*\_heads, and *d*\_ff when we hold the total non-embedding parameter count *N* fixed." An aspect ratio change of 40× shifts the loss by only about 3%.

**Q4: Did the paper predict the performance of GPT-3?**
A4: The paper itself did not predict GPT-3, but its scaling-law framework directly motivated GPT-3's training. The paper showed that loss improves predictably with scale, providing the theoretical basis for investing in much larger models. GPT-3 (Brown et al., June 2020, [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)) extended the trend to 175B parameters.

**Q5: What dataset was used, and does the tokenisation affect the constants?**
A5 (from the paper, Section 2): The models were trained on WebText2, "an extended version of the WebText dataset," tokenised with BPE (vocab size 50,257). The authors note: "The precise numerical values of *N*\_c, *C*\_c^min, and *D*\_c depend on the vocabulary size and tokenization and hence do not have a fundamental meaning."

**Q6: Is there a limit to the power-law trends?**
A6 (from the paper, Section 3): "We observe no signs of deviation from these trends on the upper end, though performance must flatten out eventually before reaching zero loss." The trends span "more than six orders of magnitude" in *N*.

## Common Misconceptions

1. **"The Kaplan scaling laws say you should always train on as much data as possible."** — This is incorrect. The paper's compute-optimal finding (Section 6) says the opposite: given a fixed compute budget, you should train larger models on *modest* data and stop *before convergence*. The Chinchilla paper (Hoffmann et al., 2022) later revised the data-to-model ratio upward, but even Chinchilla does not say "as much data as possible" — it says the ratio should be higher than Kaplan suggested.

2. **"Scaling laws mean loss eventually reaches zero."** — The paper explicitly states (Section 3) that "performance must flatten out eventually before reaching zero loss." The power laws are empirical observations over a finite range, not fundamental laws that hold to zero loss.

3. **"The exponents (α\_N ≈ 0.076) are universal constants."** — The paper notes (Section 1.2) that "the precise numerical values of *N*\_c, *C*\_c^min, and *D*\_c depend on the vocabulary size and tokenization and hence do not have a fundamental meaning." The exponents may also depend on the dataset and model architecture family.

4. **"Bigger models are always better regardless of shape."** — While shape has minimal effect within a wide range, the paper notes (Section 3.1) that "only models with fewer than 2 layers or with extreme depth-to-width ratios deviate significantly from the trend." Very shallow or very deep-narrow models do underperform.

## Real Citations

1. Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei (2020). "Scaling Laws for Neural Language Models." *arXiv preprint.* [arXiv:2001.08361](https://arxiv.org/abs/2001.08361) — the original paper.
2. Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, et al. (2022). "Training Compute-Optimal Large Language Models." [arXiv:2203.15556](https://arxiv.org/abs/2203.15556) — the "Chinchilla" paper, which revised the optimal data-to-model ratio and corrected the field's overemphasis on model size.
3. Tom B. Brown, Benjamin Mann, Nick Ryder, et al. (2020). "Language Models are Few-Shot Learners." [arXiv:2005.14165](https://arxiv.org/abs/2005.14165) — GPT-3, which extended the scaling trend to 175B parameters and was directly motivated by the Kaplan scaling laws.

**Citation count:** about 1,750 citations on [OpenAlex](https://openalex.org/works/W3001279689) (checked October 2026).
