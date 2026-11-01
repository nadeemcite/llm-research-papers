---
title: "Superposition Scaling: Talk & Q&A"
parent: "037 Superposition Scaling"
nav_order: 2
---
# TALK — Superposition Yields Robust Neural Scaling (Superposition Scaling)

## Talks, Slides and Press Coverage

1. **NeurIPS 2025 — Best Paper Runner-up (Oral):** The paper was selected as Best Paper Runner-up and presented as an Oral at NeurIPS 2025. The NeurIPS proceedings record lists it in *Advances in Neural Information Processing Systems 38* (2025), pages 159269–159305. ([arXiv abstract](https://arxiv.org/abs/2505.10465), [DOI](https://doi.org/10.52202/085713-5320))
2. **GitHub repository:** The authors released code at [github.com/liuyz0/SuperpositionScaling](https://github.com/liuyz0/SuperpositionScaling), including all experiment scripts and figure-generation code.

No additional press coverage, blog posts, or recorded talks were found as of November 2026.

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2505.10465).

**Q1: Why does strong superposition produce a universal 1/m scaling?**
A1 (from the paper, Section 3.2): In the strong superposition regime, all features are represented with overlapping vectors. The key geometric fact is that when many more vectors are squeezed into a lower-dimensional space, their mean-square overlaps scale inversely proportional to the square root of dimension m. Since loss is approximately proportional to mean-square overlaps (shown in Appendix A.2), the loss scales as 1/m regardless of the data distribution.

**Q2: What is the toy model and why is it relevant to LLMs?**
A2 (from the paper, Section 2): "We adopt the toy model of superposition from Anthropic [27] (an autoencoder) with minor modifications." The toy model captures representation constraints that LLMs face: "To represent more than fifty thousand tokens — or even more abstract concepts — within a hidden space of at most a few thousand dimensions, the quality of representations is inevitably constrained by the model dimension or width." The paper confirms LLMs operate in the same regime (Section 3.3).

**Q3: Do real LLMs actually operate in the strong superposition regime?**
A3 (from the paper, Section 3.3): "We confirmed that open-sourced LLMs operate in the strong superposition regime and have loss scaling inversely with model dimension." The authors analyzed OPT, Pythia, and Qwen2.5 models, finding that squared overlaps of token representations scale as 1/m and the model exponent α_m = 0.91 ± 0.04, close to 1.

**Q4: How does weight decay control superposition?**
A4 (from the paper, Section 2): Positive weight decay (γ ≥ 0) shrinks weights toward zero, so only the most frequent features are represented (weak superposition, φ ≈ m/n). Negative weight decay (γ < 0) pushes row norms toward 1, encouraging all features to be represented (strong superposition, φ ≈ 1). "We found that weight decay can tune superposition for all models we trained."

**Q5: When will neural scaling laws break down?**
A5 (from the paper, Section 5): "Based on our naive connection between features and tokens, the answer is that when the model dimension reaches the vocabulary size, the loss limited by width will deviate from a power law and vanish." However, the authors note the vocabulary size may underestimate the true number of independent concepts, so the power law could continue longer.

**Q6: Can the scaling exponent be improved beyond 1?**
A6 (from the paper, Section 5): "The answer is no for natural languages but may be yes for domain tasks with super skewed feature frequencies." The 1/m scaling from geometric overlaps is a robust lower bound for natural language where feature frequencies are relatively flat (Zipf's law, α ≈ 1).

## Common Misconceptions

1. **"This paper explains all neural scaling laws."** — It explains the *width* scaling (loss vs model dimension m), specifically the representation-limited component. The paper explicitly notes (Section 5) that LLM loss also includes "losses due to parsing or processing in the transformer layers" and that dataset-size and training-step scaling were not studied.

2. **"The toy model is a language model."** — The toy model is a simple autoencoder with MSE loss on synthetic sparse data, not a transformer trained on text. The paper shows (Appendix A.2) that the scaling behavior is the same for MSE and cross-entropy loss, making the toy model a valid abstraction for studying representation-limited scaling.

3. **"Superposition is always beneficial."** — The paper notes (Section 5) that "encouraging superposition may cause difficulties for the mechanistic interpretation of models and AI safety." Also, in the weak superposition regime, superposition is not preferred when features are sparse and error correction is difficult.

## Real Citations

1. Yizhou Liu, Ziming Liu, Jeff Gore (2025). "Superposition Yields Robust Neural Scaling." *NeurIPS 2025.* [arXiv:2505.10465](https://arxiv.org/abs/2505.10465) — the original paper.
2. Nelson Elhage, Tristan Hume, Catherine Olsson, et al. (2022). "Toy Models of Superposition." *Transformer Circuits Thread.* [link](https://transformer-circuits.pub/2022/toy_model/index.html) — the Anthropic toy model this paper builds on.
3. Jared Kaplan, Sam McCandlish, et al. (2020). "Scaling Laws for Neural Language Models." [arXiv:2001.08361](https://arxiv.org/abs/2001.08361) — the original neural scaling laws paper.
4. Jordan Hoffmann, Sebastian Borgeaud, et al. (2022). "Training Compute-Optimal Large Language Models." [arXiv:2203.15556](https://arxiv.org/abs/2203.15556) — the Chinchilla scaling laws, confirmed consistent with 1/m behavior.

**Citation count:** about 1 citation on [OpenAlex](https://openalex.org/works/W4416635575) (checked November 2026).
