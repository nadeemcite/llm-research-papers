---
title: "Monosemanticity: Talk & Q&A"
parent: "090 Monosemanticity"
nav_order: 2
---
# TALK — Towards Monosemanticity: Decomposing Language Models With Dictionary Learning (Monosemanticity)

## Talks, Slides and Press Coverage

1. **Anthropic Blog — "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning" (Transformer Circuits Thread, 4 Oct 2023):** The original publication, released as part of Anthropic's Transformer Circuits Thread series on interpretability. Includes interactive feature visualizations. ([link](https://transformer-circuits.pub/2023/monosemantic-features))

2. **Neel Nanda — Replication & Tutorial:** Independent researcher Neel Nanda replicated the paper's core results using an open-source 1-layer GELU language model and published a tutorial with trained autoencoders. He found that "a significant fraction of the latent space features were interpretable," corroborating the paper's claims. His work is available on [GitHub](https://github.com/neelnanda). The replication was included as a comment in the original paper.

3. **Anthropic Research Post — "Scaling Monosemanticity" (21 May 2024):** The direct follow-up by Templeton et al., applying sparse autoencoders to Claude 3 Sonnet and extracting millions of interpretable features from a production-scale model. ([link](https://transformer-circuits.pub/2024/scaling-monosemanticity))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [paper](https://transformer-circuits.pub/2023/monosemantic-features).

**Q1: Why use a sparse autoencoder instead of other dictionary learning methods?**

A1 (from the paper, Using Sparse Autoencoders To Find Good Decompositions): The authors explain that they tried several classic dictionary learning methods but "ultimately decided to focus on a simpler sparse autoencoder approximation." Two reasons: (1) a sparse autoencoder "can readily scale to very large datasets," which they believe is necessary; and (2) they have "a concern that iterative dictionary learning methods might be 'too strong'" — able to recover features the model itself cannot access, since exact compressed sensing is NP-hard and "the neural network certainly isn't doing" it.

**Q2: Why not just build models without superposition (architectural approaches)?**

A2 (from the paper, Why Not Architectural Approaches): The authors found that even when they eliminated superposition by pushing activation sparsity to the extreme (1-hot activations), "individual neurons can be polysemantic even in the absence of superposition." They provide a toy example showing that a single neuron representing two features achieves lower cross-entropy loss (ln 2 ≈ 0.7) than representing one feature and ignoring the rest (¾ ln 3 ≈ 0.8). They conclude that "models trained on cross-entropy loss will generally prefer to represent more features polysemantically."

**Q3: How much of the model do the features explain?**

A3 (from the paper, How Much of the Model Does Our Interpretation Explain?): For their primary run A/1 (4,096 features), "79% of the log-likelihood loss reduction provided by the MLP layer is recovered by our features." With A/5 (131,072 features, lower L1 coefficient), this rises to 94.5%. They caution that "framing this question in terms of fraction of loss may be misleading" because of a long tail of features.

**Q4: Are the features real or just artifacts of the dictionary learning process?**

A4 (from the paper, Universality): The authors argue that universality provides evidence features are real: "it suggests that the features we're finding are not just artifacts of the dictionary learning process — or at least that if they come from the dictionary learning process, it's in some consistent way." They also note that features have "interpretable causal effects on model outputs" which "must be a property of the model" since downstream effects "are not an input to the dictionary learning process."

**Q5: What is feature splitting?**

A5 (from the paper, Feature Splitting): "As we go from 512 features in A/0 to 4,096 features in A/1 and to 16,384 features in A/2, the number of features specific to base64 contexts goes from 1 to 3 to many more." The authors conjecture that "there is some idealized set of features that dictionary learning would return if we provided it with an unlimited dictionary size," and that restricted dictionary sizes return features that "cover approximately the same territory... at the cost of being somewhat less specific."

## Common Misconceptions

1. **"Sparse autoencoders prove that features are the 'true' units of neural networks."** — The paper takes an "agnostic position" on whether features are fundamental or "just a convenient post-hoc description." While their universality results are suggestive, they do not claim to have proven that features have ontological status beyond individual models.

2. **"This paper solved superposition."** — The paper demonstrates that SAEs can extract interpretable features from a *one-layer* transformer. The authors note that "it's very unclear what to generalize to real networks" and that scaling to frontier models remains an open engineering challenge (potentially requiring autoencoders with billions of parameters).

3. **"All SAE features are perfectly monosemantic."** — The paper does not claim this. They acknowledge that "we do not claim that our interpretations catch all aspects of features' behaviors" and that "we don't believe our features are completely monosemantic (some polysemanticity may be hiding in low activations)."

4. **"The features are just neurons that happen to be interpretable."** — A key finding is the opposite: features like Arabic script and Hebrew are "effectively invisible if we only analyzed the model in terms of neurons." The most correlated neuron to the Arabic feature responds to "a mixture of different non-English languages," not Arabic specifically.

5. **"This is the first paper to use dictionary learning on neural networks."** — The paper explicitly acknowledges "a significant amount of prior work, especially in using dictionary learning and related methods on neural network activations," and cites Cunningham et al. and Sharkey et al. as having independently investigated the SAE approach in response to Toy Models of Superposition.

## Real Citations

1. Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., et al. (2023). "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning." *Transformer Circuits Thread.* [link](https://transformer-circuits.pub/2023/monosemantic-features) — the original paper.

2. Templeton, A., Conerly, T., Marcus, J., Lindsey, J., Bricken, T., et al. (2024). "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet." *Transformer Circuits Thread.* [link](https://transformer-circuits.pub/2024/scaling-monosemanticity) — direct follow-up scaling SAEs to Claude 3 Sonnet.

3. Elhage, N., Hume, T., Olsson, C., et al. (2022). "Toy Models of Superposition." *Transformer Circuits Thread.* [arXiv:2209.10652](https://arxiv.org/abs/2209.10652) — the predecessor paper that motivated this work.

4. Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). "Sparse Autoencoders Find Highly Interpretable Features in Language Models." [arXiv:2309.08600](https://arxiv.org/abs/2309.08600) — independent, parallel investigation of SAEs for feature extraction.

5. Bills, S., Cammarata, N., Mossing, D., et al. (2023). "Language Models Can Explain Neurons in Language Models." [link](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html) — the automated interpretability methodology the paper builds on.

**Citation count:** The paper is not indexed on OpenAlex or Semantic Scholar as of November 2026 (published on transformer-circuits.pub rather than arXiv). Its direct follow-up, *Scaling Monosemanticity* (Templeton et al., 2024), has about 587 citations on [OpenAlex](https://openalex.org/works/W7162776923) (checked November 2026), indicating the substantial impact of this line of work.
