---
title: "Grokking: Talk & Q&A"
parent: "038 Grokking"
nav_order: 2
---
# TALK — Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets (Grokking)

## Talks, Slides and Press Coverage

1. **Google PAIR — "Do Machine Learning Models Memorize or Generalize?" (Adam Pearce, Asma Ghandeharioun, Nada Hussein, Nithum Thain, Martin Wattenberg, Lucas Dixon, August 2023):** An interactive explorable article that visualizes grokking on modular addition (mod 67) with a one-layer MLP. Shows how the model transitions from memorization to generalization, with live weight heatmaps and a reverse-engineering of the generalizing solution. Calls grokking "the fruit fly of mechanistic interpretability." ([link](https://pair.withgoogle.com/explorables/grokking/))

2. **Neel Nanda — "Progress Measures for Grokking via Mechanistic Interpretability" (ICLR 2023):** Not a talk by the original authors, but the most influential follow-up. Nanda reverse-engineered the circuits a grokking network learns on modular addition, showing that grokking corresponds to a structured circuit gradually replacing a memorization circuit. ([arXiv:2301.05217](https://arxiv.org/abs/2301.05217))

3. **Ziming Liu, Eric Michaud, Max Tegmark — "Towards Understanding Grokking: An Effective Theory of Representation Learning" (ICLR 2023):** Provides a theoretical framework for grokking, attributing it to the competition between "sluggish" and "grokking" modules in the network. ([arXiv:2205.10343](https://arxiv.org/abs/2205.10343))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their
section in the [arXiv paper](https://arxiv.org/abs/2201.02177).

**Q1: Why does grokking happen so late — long after the model has already memorized the training data?**
A1 (from the paper, Section 3.1): The authors observe that validation accuracy "starts increasing beyond chance level only after 1000 times more optimization steps than are required for training accuracy to get close to optimal." They do not claim to fully explain *why* — they document the phenomenon and note that it "can happen well past the point of overfitting." In the Discussion (Section 4), they speculate that noise from SGD "might induce the optimization to find flatter minima that generalize better."

**Q2: What is the role of weight decay?**
A2 (from the paper, Section 3.3): "Adding weight decay has a very large effect on data efficiency, more than halving the amount of samples needed compared to most other interventions." They also found that weight decay towards the initialization is effective but not as effective as weight decay towards the origin, suggesting "the prior, that approximately zero weights are suitable for small algorithmic tasks, explains part, but not all of the superior performance of weight decay."

**Q3: Does grokking happen for all operations?**
A3 (from the paper, Section 3.2): No. "Some operations (for example x³ + xy² + y (mod 97)) didn't lead to generalization within the allowed optimization budget at any percentage of data up to 95%. The converged models effectively just memorized the training dataset without finding any real patterns in the data." Grokking is typical for operations near the minimum dataset size, but not guaranteed.

**Q4: How does dataset size affect grokking?**
A4 (from the paper, Section 3.1.1): "In the vicinity of 25-30% of data, a decrease of 1% of training data leads to an increase of 40-50% in median time to generalization." The number of steps until validation accuracy >99% "grows quickly as dataset size decreases," while steps until training accuracy >99% "generally trends down as dataset size decreases and stays in the range of 10³–10⁴ optimization steps."

## Common Misconceptions

1. **"Grokking means the model suddenly understands the concept."** — Grokking is defined operationally as a sharp increase in validation accuracy after prolonged overfitting. The paper does not claim the model "understands" anything — it documents a measurable phenomenon (validation accuracy jumping from chance to perfect). The mechanistic explanation (structured circuits replacing memorization circuits) came later, in Nanda (2023).

2. **"Grokking only happens with weight decay."** — The paper shows grokking happens even without weight decay (their headline Figure 1 used no weight decay to emphasize how late generalization can begin), but weight decay makes it happen much faster and with less data. The paper also found that gradient noise (from minibatch SGD or explicit Gaussian noise) helps.

3. **"Grokking is the same as double descent."** — The paper explicitly distinguishes them (Appendix A.3): "we believe that the phenomenon we describe might be distinct from the double descent phenomena described in Nakkiran et al. (2019); Belkin et al. (2018) because we observe the second descent in loss far past the first time the training loss becomes very small (tens of thousands of epochs in some of our experiments), and we don't observe a non-monotonic behavior of accuracy."

## Real Citations

1. Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra (2022). "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets." *arXiv preprint.* [arXiv:2201.02177](https://arxiv.org/abs/2201.02177) — the original paper.

2. Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt (2023). "Progress Measures for Grokking via Mechanistic Interpretability." *ICLR 2023.* [arXiv:2301.05217](https://arxiv.org/abs/2301.05217) — reverse-engineers the circuits behind grokking on modular addition.

3. Ziming Liu, Eric Michaud, Max Tegmark (2023). "Towards Understanding Grokking: An Effective Theory of Representation Learning." *ICLR 2023.* [arXiv:2205.10343](https://arxiv.org/abs/2205.10343) — theoretical framework attributing grokking to competition between sluggish and grokking modules.

**Citation count:** about 83 citations on [OpenAlex](https://openalex.org/works/W4226434736) (checked November 2026).
