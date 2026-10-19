---
title: "RMSNorm: Talk & Q&A"
parent: "004 RMSNorm"
nav_order: 2
---
# TALK — Root Mean Square Layer Normalization (RMSNorm)

## Talks, Slides and Press Coverage

1. **NeurIPS 2019 poster presentation** — The paper was presented as a poster at the 33rd Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada. The NeurIPS proceedings page is at [NeurIPS 2019](https://proceedings.neurips.cc/paper/2019).

2. **Official code repository** — The authors released the source code at [github.com/bzhangGo/rmsnorm](https://github.com/bzhangGo/rmsnorm), with implementations in TensorFlow, PyTorch, and Theano.

3. **Hugging Face documentation** — RMSNorm is documented as the normalization layer in modern architectures like LLaMA and Mistral on the [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/model_doc/llama), where it is described as the default normalization.

4. **Adoption in LLaMA** — Meta's LLaMA paper (Touvron et al., 2023) explicitly cites RMSNorm and uses it in place of LayerNorm, as noted in the [LLaMA paper](https://arxiv.org/abs/2302.13971).

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/1910.07467).

**Q1: Why did the authors hypothesize that re-centering invariance is dispensable?**

A1 (from the paper, Section 1): "We argue that this mean normalization does not reduce the variance of hidden states or model gradients, and hypothesize that it has little impact on the success of LayerNorm."

**Q2: What is the key difference between RMSNorm and LayerNorm?**

A2 (from the paper, Section 4): "Intuitively, RMSNorm simplifies LayerNorm by totally removing the mean statistic in Eq. (3) at the cost of sacrificing the invariance that mean normalization affords. When the mean of summed inputs is zero, RMSNorm is exactly equal to LayerNorm."

**Q3: Why does RMSNorm use RMS instead of L2 norm (Euclidean norm)?**

A3 (from the paper, Section 4): "Although Euclidean norm which only differs from RMS by a factor of √n has been successfully explored [22], we empirically find that it does not work for layer normalization. We hypothesize that scaling the sphere with the size of the input vector is important because it makes the normalization more robust across vectors of different size."

**Q4: What is the implicit learning rate adaptation property?**

A4 (from the paper, Section 4.2, Eq. 9–10): The gradient of the weight matrix W is "negatively correlated with both input and weight matrix scaling." The authors state this "acts as an implicit learning rate adaptor and dynamically controls the norm of gradients which avoids large-norm weight matrix and improves model convergence."

**Q5: How much speedup does RMSNorm provide?**

A5 (from the paper, Abstract): "RMSNorm achieves comparable performance against LayerNorm but reduces the running time by 7%∼64% on different models." The speedup depends on "the framework, hardware, neural network architecture and relative computational cost of other components" (Section 7).

**Q6: Can RMSNorm be used with partial inputs?**

A6 (from the paper, Section 5): "Considering that neurons in one layer often have independent identically distributed structure, we argue that the RMS can be estimated on a subset of these neurons rather than all of them." The authors use p = 6.25% and find "competitive performance compared to RMSNorm."

**Q7: What happened when the Transformer was trained without any normalization?**

A7 (from the paper, Section 6.1, Table 4): The baseline (no normalization) failed to train — "we fail to train this model and BLEU score is 0." This shows normalization is essential for the Transformer.

## Common Misconceptions

1. **"RMSNorm is just LayerNorm without the learnable bias."** — It is not just about the bias. RMSNorm removes the entire mean computation (the $\mu$ subtraction and the variance computation). The normalization statistic changes from $\sigma = \sqrt{\frac{1}{n}\sum(a_i - \mu)^2}$ to $\text{RMS} = \sqrt{\frac{1}{n}\sum a_i^2}$. This is a different mathematical operation, not just removing a parameter.

2. **"RMSNorm always trains faster than LayerNorm."** — The speedup depends on the model architecture, framework, and hardware. The paper reports 7–64% speedup across different setups. On the Transformer, the speedup is only 7–9% because there are "significantly fewer sequential normalization operations" (Section 6.1). On RNNs with many time steps, the speedup is larger.

3. **"RMSNorm is more accurate than LayerNorm."** — The paper claims "comparable performance," not superiority. On most tasks the BLEU or accuracy numbers are within noise of each other. On CIFAR-10, RMSNorm (8.83% error) actually does better than LayerNorm (10.49%), but this is the exception, not the rule.

4. **"pRMSNorm is always faster than RMSNorm."** — The paper notes that "though in theory the amount of computation in pRMSNorm is less than that in RMSNorm, pRMSNorm (p = 6.25%) sometimes tends to be slower" due to "non-optimal implementation of tensor slicing operation" (Section 6.1).

## Real Citations

1. Zhang, B. & Sennrich, R. (2019). "Root Mean Square Layer Normalization." *NeurIPS 2019.* [arXiv:1910.07467](https://arxiv.org/abs/1910.07467) — the original paper.

2. Ba, J. L., Kiros, J. R. & Hinton, G. E. (2016). "Layer Normalization." [arXiv:1607.06450](https://arxiv.org/abs/1607.06450) — the LayerNorm paper that RMSNorm simplifies.

3. Touvron, H. et al. (2023). "LLaMA: Open and Efficient Foundation Language Models." [arXiv:2302.13971](https://arxiv.org/abs/2302.13971) — adopted RMSNorm as the default normalization, popularising it in the open-weight LLM ecosystem.

4. Vaswani, A. et al. (2017). "Attention Is All You Need." *NIPS 2017, now NeurIPS.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) — the Transformer architecture where RMSNorm is tested as a drop-in replacement.

**Citation count:** about 203 citations on [OpenAlex](https://openalex.org/W2970903692) (checked October 2026).
