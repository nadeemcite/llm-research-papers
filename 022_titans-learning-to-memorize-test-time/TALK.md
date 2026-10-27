---
title: "Titans: Talk & Q&A"
parent: "022 Titans"
nav_order: 2
---
# TALK — Titans: Learning to Memorize at Test Time (Titans)

## Talks, Slides and Press Coverage

1. **AI-in-PM — "Titans: Learning to Memorize at Test Time" (2025):** An interactive multi-agent demonstration platform for the Titans architecture, featuring 7 AI agents that reason about the memory problem from different perspectives. The repo includes visualisations of the surprise-based update mechanism. ([link](https://github.com/ai-in-pm/Titans---Learning-to-Memorize-at-Test-Time))

2. **Aedelon — "Titans PyTorch/MLX Implementation" (2025):** A complete PyTorch and MLX implementation of the Titans architecture, including the neural long-term memory module, all three architectural variants (MAC, MAG, MAL), and pretraining scripts. 33 stars on GitHub. ([link](https://github.com/Aedelon/titans-pytorch-mlx))

3. **Google Research — arXiv preprint page:** The official preprint with abstract, PDF, and version history. The paper has not been published at a peer-reviewed venue as of October 2026. ([link](https://arxiv.org/abs/2501.00663))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2501.00663).

**Q1: Why use gradient magnitude as the "surprise" metric?**
A1 (from the paper, Section 3.1): The authors define surprise as "the gradient of the neural network with respect to the input in associative memory loss." The larger the gradient, the more different the input is from the past data. They cite neuropsychology research (Mandler, 2014) showing that "an event that violates the expectations (being surprising) is more memorable" in human cognition.

**Q2: What is the relationship between Titans' weight decay and the gating in modern RNNs?**
A2 (from the paper, Section 3.1): The authors show that their adaptive forgetting mechanism (weight decay with gate α_t) "is in fact the generalization of forgetting mechanism in modern recurrent models" including Mamba2, GLA, and LRU. The key difference is that Titans combine this with the surprise-based update and momentum, while modern RNNs use gating without the gradient-based surprise metric.

**Q3: Why does Titans use a deep (multi-layer) memory module instead of a single matrix?**
A3 (from the paper, Section 3.1 and 5.5): The authors argue that "deep memory modules (i.e., L_M >= 2)" are more expressive, aligning with theoretical results that "MLPs with at least two layers are strictly more expressive than linear models" (Hornik et al., 1989). In experiments (Figure 7), deeper memory achieves better perplexity across all sequence lengths, with the advantage most pronounced for smaller models on longer sequences.

**Q4: How do the three Titans variants (MAC, MAG, MAL) differ in practice?**
A4 (from the paper, Section 5.2 and 5.9): MAC and MAG have close performance on language modeling and reasoning, but "MAC achieves significantly better performance in long-context NIAH." Both outperform MAL. The authors attribute this to the architectural design: MAC lets attention decide whether long-term memory information is needed for the current context, while MAL is limited by each layer independently.

**Q5: Did Titans release official code?**
A5: As of October 2026, Google Research has not released an official code repository for Titans. Multiple unofficial implementations exist on GitHub, but the paper itself does not link to one. The paper states (Section 5.1) that experimental details follow the training procedure of Yang et al. (2024a) for Gated DeltaNet.

## Common Misconceptions

1. **"Titans is just another linear attention variant"** — This is wrong. Linear attention compresses the past into a fixed matrix via additive updates (Eq. 4-5). Titans use a neural network (MLP) as the memory, updated by gradient descent at test time. The memory is deep and non-linear, not a single matrix. The authors explicitly contrast their approach with linear Transformers in Section 2.1.

2. **"The surprise metric is just a learning rate"** — The surprise metric is the gradient of the associative memory loss, which determines the direction and magnitude of the memory update. It is combined with momentum (past surprise) and weight decay (forgetting gate), making it a full meta-learning update rule, not a simple learning rate schedule. The ablation (Table 5) shows removing momentum drops long-context accuracy by 5.56 points.

3. **"Titans replace attention entirely"** — Titans augment attention, not replace it. All three variants (MAC, MAG, MAL) include an attention module as "short-term memory" for the current context window. The neural long-term memory handles long-range dependencies that exceed the attention window. The LMM-only variant (Titans LMM) is evaluated separately and is competitive but not the best performer.

## Real Citations

1. Ali Behrouz, Peilin Zhong, Vahab Mirrokni (2024). "Titans: Learning to Memorize at Test Time." *arXiv preprint.* [arXiv:2501.00663](https://arxiv.org/abs/2501.00663) — the original paper.

2. Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Dhiman, Kuan-Chih Huang, et al. (2024). "Test-Time Training of Linear Attention by Gradient Descent." [arXiv:2407.04620](https://arxiv.org/abs/2407.04620) — TTT, the gradient-based recurrent model that Titans extend with momentum and weight decay.

3. Songlin Yang, Jan Kautz, Ali Hatamizadeh (2024). "Gated DeltaNet: A Simple, Powerful Recurrent Network." [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) — Gated DeltaNet, the strongest non-Titans baseline that combines gating with the delta rule.

**Citation count:** about 470 citations on [Google Scholar](https://scholar.google.com/scholar?q=%22Titans+Learning+to+Memorize+at+Test+Time%22) and 23 on [OpenAlex](https://openalex.org/W4406032263) (checked October 2026).
