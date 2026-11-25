---
title: "Toy Superposition: Talk & Q&A"
parent: "089 Toy Superposition"
nav_order: 2
---
# TALK — Toy Models of Superposition (Toy Superposition)

## Talks, Slides and Press Coverage

1. **3Blue1Brown — "But what is a neural network?" / interpretability coverage:** The Toy Models of Superposition paper's geometric visualizations have been widely discussed in the ML interpretability community. The paper's interactive visualizations on transformer-circuits.pub are frequently cited in blog posts and tutorials about mechanistic interpretability. ([paper page](https://transformer-circuits.pub/2022/toy_model/index.html))

2. **Anthropic Blog — "Toy Models of Superposition" (Transformer Circuits Thread, 14 Sep 2022):** The original publication, published as part of Anthropic's Transformer Circuits Thread, a series of papers on interpretability. ([link](https://transformer-circuits.pub/2022/toy_model/index.html))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [paper](https://arxiv.org/abs/2209.10652).

**Q1: Why do neural networks have polysemantic neurons?**
A1 (from the paper, Background & Motivation): The authors hypothesize that polysemanticity arises from superposition — "models storing additional sparse features in superposition." When features are sparse (rarely active), the model can represent more features than it has neurons by packing them into nearly-orthogonal directions, at the cost of interference that requires nonlinear filtering.

**Q2: What causes the phase change between representing and not representing features?**
A2 (from the paper, Superposition as a Phase Change): The authors find that "whether features are stored in superposition is governed by a phase change." The transition is controlled by the sparsity of features: as sparsity increases past a threshold, the benefit of representing additional features (feature benefit) outweighs the cost of interference, causing a sudden jump from representing $m$ features to representing all $n$ features.

**Q3: Why do features arrange into geometric structures like pentagons?**
A3 (from the paper, The Geometry of Superposition): The authors connect this to the Thomson problem in physics. When features have uniform importance and unit norm, the interference term becomes "a generalized Thomson problem — we're just packing points on the surface of the sphere with a slightly unusual energy function." This is why features form regular polytopes.

**Q4: What is the connection between superposition and adversarial examples?**
A4 (from the paper, Relationship to Adversarial Robustness): The authors show that in superposition, the effective weight matrix $W^TW$ has small off-diagonal interference terms that "create an obvious way for an adversary to attack." They find that "vulnerability to adversarial examples sharply increases as superposition forms (increasing by >3×)."

## Common Misconceptions

1. **"Superposition means features are perfectly orthogonal"** — No, superposition means features are *nearly* but not perfectly orthogonal. The whole point is that features interfere with each other; the ReLU nonlinearity filters out small interferences. If features were perfectly orthogonal, there would be no superposition — just a standard orthogonal basis.

2. **"This paper proves superposition happens in real neural networks"** — The paper is careful to say it demonstrates superposition in *toy models*, and notes that "it's very unclear what to generalize to real networks." The results show superposition can occur in simple ReLU networks under natural conditions, suggesting it *may* occur in practice, but this is not proven for large models.

3. **"Superposition and polysemanticity are the same thing"** — Superposition is the *cause*; polysemanticity is the *symptom*. Superposition is the strategy of packing multiple features into fewer dimensions. Polysemanticity (one neuron responding to multiple features) is what we observe when features in superposition happen to align with neurons in a privileged basis.

## Real Citations

1. Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., Hatfield-Dodds, Z., Lasenby, R., Drain, D., Chen, C., Grosse, R., McCandlish, S., Kaplan, J., Amodei, D., Wattenberg, M., & Olah, C. (2022). "Toy Models of Superposition." *Transformer Circuits Thread.* [arXiv:2209.10652](https://arxiv.org/abs/2209.10652) — the original paper.

2. Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., et al. (2023). "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning." *Transformer Circuits Thread.* [link](https://transformer-circuits.pub/2023/monosemantic-features) — direct follow-up using sparse autoencoders to decompose superposed features, motivated by this paper's findings.

**Citation count:** about 1,700 citations ([OpenAlex](https://openalex.org/works/W4296932880), checked November 2026).
