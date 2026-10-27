---
title: "Speculative Decoding: Talk & Q&A"
parent: "025 Speculative Decoding"
nav_order: 2
---
# TALK — Fast Inference from Transformers via Speculative Decoding (Speculative Decoding)

## Talks, Slides and Press Coverage

1. **ICML 2023 Oral — "Fast Inference from Transformers via Speculative Decoding" (Yaniv Leviathan, 25 Jul 2023):** The paper was presented as an Oral at the 40th International Conference on Machine Learning in Honolulu, Hawaii. The talk covered the speculative sampling method, the draft-then-verify algorithm, and the 2X-3X speedup results on T5-XXL. ([conference schedule](https://icml.cc/Conferences/2023/Schedule))

2. **ICML 2023 Virtual — Paper page and video:** The paper is available on the ICML 2023 virtual site with the recorded oral presentation. ([paper page](https://icml.cc/virtual/2023/oral/24986))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2211.17192).

**Q1: Why does speculative decoding produce identical outputs?**
A1 (from the paper, Section 2.3): The speculative sampling method guarantees this. For any draft token x sampled from q(x), it is accepted with probability min(1, p(x)/q(x)). On rejection, a token is resampled from norm(max(0, p(x) - q(x))). The paper proves in Section A.1 that "for any distributions p(x) and q(x), and x sampled in this way, indeed x ~ p(x)" — the output distribution is provably identical to sampling from the target model alone.

**Q2: Does the method require retraining or architecture changes?**
A2 (from the paper, Abstract): "Our method can accelerate existing off-the-shelf models without retraining or architecture changes." The approximation model M_q can be any model, and the target model M_p is used as-is.

**Q3: What is the acceptance rate alpha and how is it related to the draft model size?**
A3 (from the paper, Section 3.2 and Section 4.2): alpha = E(min(p, q)) (Corollary 3.6), measuring the overlap between target and draft distributions. The paper finds that "approximation models that are a couple of orders of magnitude smaller than the target model tend to produce alpha values between 0.5 and 0.9" (Section 4.2). Larger draft models give higher alpha but also higher cost c.

**Q4: Can trivial approximation models also provide speedup?**
A4 (from the paper, Section 3.6): Yes. "For the English-German translation task, with M_p being T5-XXL 11B and M_q being a trivial bigram model, we get alpha approximately 0.2 which leads to an inference speed improvement factor of 1.25X with gamma = 3." N-gram models have c approximately 0, so even small alpha values yield improvement.

**Q5: What is the optimal number of draft tokens gamma?**
A5 (from the paper, Section 3.5): The optimal gamma maximizes the walltime improvement (Theorem 3.8) and "can be easily found numerically." It depends on both alpha and c. The paper uses gamma = 7 for translation and gamma = 5 for summarization (Table 2).

**Q6: When does speculative decoding NOT help?**
A6 (from the paper, Section 6): "Our method is not helpful for configurations where additional computation resources are not available." The method trades increased arithmetic operations (up to gamma+1x more) for reduced wall-clock time, so it only helps when memory bandwidth is the bottleneck and compute resources are available.

## Common Misconceptions

1. **"Speculative decoding changes the output distribution slightly."** — No. The paper proves (Section A.1) that the output distribution is mathematically identical to sampling from the target model alone, for any choice of draft model. The speculative sampling correction (resampling from norm(max(0, p - q)) on rejection) exactly compensates for the draft model's biases.

2. **"The draft model needs to be trained separately for speculative decoding."** — No. The paper explicitly states the method works with "off-the-shelf" models (Abstract). Any existing smaller model can serve as M_q, including n-gram models that require no training at all (Section 3.6).

3. **"Speculative decoding speeds up all inference equally."** — No. The speedup depends on the acceptance rate alpha and cost ratio c. If alpha is low (draft model is a poor approximation) or c is high (draft model is not much cheaper), the method may not help. The paper notes it is "not helpful for configurations where additional computation resources are not available" (Section 6).

4. **"The speedup comes from running the draft model instead of the target."** — The draft model runs serially (gamma sequential calls), which is additional work. The speedup comes from the target model processing gamma+1 positions in **one** parallel forward pass instead of gamma+1 sequential passes. The target model is always called; it just processes more positions per call.

## Real Citations

1. Yaniv Leviathan, Matan Kalman, Yossi Matias (2022). "Fast Inference from Transformers via Speculative Decoding." *ICML 2023 (Oral).* [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) — the original paper.

2. Tianle Cai et al. (2024). "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads." [arXiv:2401.10774](https://arxiv.org/abs/2401.10774) — follow-up that replaces the autoregressive draft model with parallel multi-head prediction, eliminating the serial draft phase.

3. Jiashuo Li et al. (2024). "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty." [arXiv:2401.15077](https://arxiv.org/abs/2401.15077) — improves draft quality by predicting at the feature level rather than the token level, achieving higher acceptance rates.

**Citation count:** about 71 citations on [OpenAlex](https://openalex.org/W4310561894) (checked October 2026).
