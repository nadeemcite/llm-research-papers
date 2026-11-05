---
title: "Weak-to-Strong: Talk & Q&A"
parent: "051 Weak-to-Strong"
nav_order: 2
---
# TALK — Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision (W2S)

## Talks, Slides and Press Coverage

1. **OpenAI — "Introducing Superalignment" (Jul 2023):** OpenAI announced the Superalignment team and its plan to build a human-level automated alignment researcher, with weak-to-strong generalization as a core research direction. ([link](https://openai.com/blog/introducing-superalignment))
2. **OpenAI — "Weak-to-strong generalization" (Dec 2023):** The official OpenAI blog post accompanying the paper, summarising the key findings and their significance for superalignment. ([link](https://openai.com/research/weak-to-strong-generalization))
3. **GitHub — openai/weak-to-strong:** The official open-source code repository for the paper's experiments, including NLP, chess, and reward modeling setups. ([link](https://github.com/openai/weak-to-strong))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2312.09390).

**Q1: Why should weak-to-strong generalization be possible at all?**
A1 (from the paper, Section 1): "Strong pretrained models should already have good representations of the alignment-relevant tasks we care about. For example, if a model can generate complicated code, then it should intuitively also know whether that code faithfully adheres to the user's instructions. As a result, for the purposes of alignment we do not need the weak supervisor to teach the strong model new capabilities; instead, we simply need the weak supervisor to elicit what the strong model already knows."

**Q2: What is the Performance Gap Recovered (PGR)?**
A2 (from the paper, Section 3): PGR measures the fraction of the performance gap between the weak supervisor and the strong ceiling that is recovered with weak supervision. If weak-to-strong performance equals the strong ceiling, PGR = 1. If the weak-to-strong model does no better than the weak supervisor, PGR = 0.

**Q3: How much does the auxiliary confidence loss help?**
A3 (from the paper, Section 4.3.2): "With the smallest weak supervisor and largest strong student, the confidence loss increases median PGR from about 25% to nearly 80%." The loss is closely related to conditional entropy minimisation (Grandvalet & Bengio, 2004).

**Q4: Does the naive baseline fully recover the strong model's capabilities?**
A4 (from the paper, Section 1): "We are still far from recovering the full capabilities of strong models with naive finetuning alone, suggesting that techniques like RLHF may scale poorly to superhuman models without further work."

**Q5: What are the key disanalogies between this setup and the real superalignment problem?**
A5 (from the paper, Section 3): Two main disanalogies: (1) Imitation saliency — future superhuman models will have salient representations of human behaviors, so imitating the weak supervisor may be easier to avoid in this setting than it will be in the future. (2) Pretraining leakage — current pretraining data contains human-level supervision, making it artificially easy to elicit strong model capabilities.

## Common Misconceptions

1. **"Weak-to-strong generalization means the weak model trains the strong model to be better than itself"** — Not quite. The strong model is already pretrained and has the capabilities; the weak labels merely elicit what the strong model already knows. The weak supervisor does not teach new capabilities — it provides a (noisy) signal that activates existing knowledge.

2. **"The confidence loss is a new loss function invented for this paper"** — The confidence loss is conditional entropy minimisation, a well-known technique from semi-supervised learning (Grandvalet & Bengio, 2004). The paper's contribution is applying it to the weak-to-strong setting and showing it dramatically improves PGR.

3. **"If PGR > 0, alignment is solved"** — The paper explicitly says PGR is far from 1 in most settings, especially reward modeling (PGR ~10%). The authors frame their methods as "proofs-of-concept that weak-to-strong generalization is tractable, rather than practical solutions we recommend deploying today" (Section 4.3).

## Real Citations

1. Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, Ilya Sutskever, Jeff Wu (2023). "Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision." *Preprint.* [arXiv:2312.09390](https://arxiv.org/abs/2312.09390) — the original paper.
2. Paul F. Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei (2017). "Deep Reinforcement Learning from Human Preferences." [arXiv:1706.03741](https://arxiv.org/abs/1706.03741) — RLHF, the alignment technique whose scalability this paper questions.
3. Yves Grandvalet, Yoshua Bengio (2004). "Semi-supervised Learning by Entropy Minimization." *CAP 2004.* — the conditional entropy minimisation technique adapted as the auxiliary confidence loss.
4. Leopold Aschenbrenner, Jared Kaplan, Swyx (2023). "OpenAI Superalignment Blog Post." [link](https://openai.com/blog/introducing-superalignment) — the announcement of the Superalignment team this paper was part of.

**Citation count:** about 250 citations on [OpenAlex](https://openalex.org/W4390698830) (checked November 2026).
