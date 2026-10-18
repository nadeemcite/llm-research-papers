---
title: "Transformer: Talk & Q&A"
parent: "020 Transformer"
nav_order: 2
---
# TALK.md — Attention Is All You Need (Transformer)

## Press / Blog Coverage

1. **Google Research Blog — "Transformer: A Novel Neural Network Architecture for Language Understanding" (Jakob Uszkoreit, August 31, 2017):** Google's announcement of the paper. It reports that the Transformer outperformed both recurrent and convolutional models on the WMT English-to-German and English-to-French translation benchmarks, while needing less computation to train and fitting modern ML hardware much better. (URL: https://research.google/blog/transformer-a-novel-neural-network-architecture-for-language-understanding/)

2. **Harvard NLP — "The Annotated Transformer" (Alexander Rush, April 2018):** A line-by-line PyTorch implementation of the paper, with the paper's text placed next to the code that implements each part. (URL: https://nlp.seas.harvard.edu/2018/04/03/attention.html)

3. **Jay Alammar's "The Illustrated Transformer" (2018):** A widely read blog post that walks through the Transformer architecture step by step with diagrams. (URL: https://jalammar.github.io/illustrated-transformer/)

4. **Lilian Weng's Blog — "Attention? Attention!" (2018):** A survey of attention mechanisms leading up to and including the Transformer. (URL: https://lilianweng.github.io/posts/2018-06-24-attention/)

## Q&A (answers from the paper)

**Q1: Why did you decide to remove recurrence and convolutions entirely?**
A1 (from the paper, Sections 1 and 4): Recurrent models compute hidden states one position at a time, and the paper notes that this inherently sequential nature precludes parallelization within training examples. Self-attention connects every pair of positions with a constant number of sequential operations, versus O(n) for a recurrent layer, and shortens the path between long-range dependencies to O(1).

**Q2: What was the role of the scaling factor 1/√d_k in scaled dot-product attention?**
A2 (from the paper, Section 3.2.1): The authors note that additive attention beats unscaled dot-product attention when d_k is large. They *suspect* the reason is that large d_k makes the dot products large in magnitude, pushing the softmax into regions with extremely small gradients, so they scale by 1/√d_k to counteract it. Their footnote explains why: if the query and key entries are independent with mean 0 and variance 1, their dot product has variance d_k, so dividing by √d_k brings the variance back to 1.

**Q3: Why use sinusoidal positional encodings instead of learned ones?**
A3 (from the paper, Sections 3.5 and 6.2): The authors *hypothesized* that sinusoids would make it easy for the model to learn to attend by relative position, since for any fixed offset k, PE(pos+k) is a linear function of PE(pos). They also tried learned positional embeddings and got nearly identical results (Table 3, row E). They kept the sinusoidal version because it *may* allow the model to extrapolate to sequences longer than those seen in training. The paper does not test that claim.

**Q4: How important was the multi-head attention mechanism?**
A4 (from the paper, Section 3.2.2): Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this. In the ablation (Table 3, row A, keeping total computation fixed), single-head attention was 0.9 BLEU worse than the best setting, but quality also dropped off with too many heads.

**Q5: Did the authors expect the Transformer to apply beyond machine translation?**
A5 (from the paper, Section 7): The conclusion says the authors planned to apply attention-based models to other tasks, extend the Transformer to input and output modalities other than text, and investigate local, restricted attention for large inputs such as images, audio and video. The paper itself already showed one step in that direction: English constituency parsing (Section 6.3).

## Common Misconceptions

1. **"The Transformer has no notion of word order."** False. Self-attention itself is permutation-invariant, but positional encodings explicitly inject order information. The model absolutely uses position — it just does so additively rather than structurally.

2. **"Multi-head attention means multiple separate Transformer models."** No. It means the Q/K/V are projected into multiple lower-dimensional subspaces within a single attention layer, each attending independently, then concatenated. It's one layer with parallel sub-attentions.

3. **"The Transformer eliminated attention — it's all about something else."** The opposite: the Transformer made attention the *only* mechanism, removing recurrence and convolutions. The title means attention replaces everything else, not that attention itself is removed.

4. **"Scaled dot-product attention is the only type of attention."** The paper discusses additive (Bahdanau) attention too, noting that dot-product attention is faster and more space-efficient in practice. The 1/√d_k scaling is what the authors added to keep dot-product attention competitive when d_k is large.

5. **"The original Transformer was used for general language modeling."** No — it was designed for sequence-to-sequence machine translation and evaluated on English→German and English→French (WMT 2014). The paper also tried English constituency parsing (Section 6.3), but it was not used as a general language model. Its use for general pretraining (GPT, BERT) came later, though the architecture proved general enough.

## Real Citations

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). "Attention Is All You Need." *Advances in Neural Information Processing Systems 30 (NIPS 2017, now called NeurIPS).* arXiv:1706.03762. — The original paper, with about 197,000 citations on Semantic Scholar (October 2026).

2. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." arXiv:1810.04805. — Applied the Transformer encoder for bidirectional pretraining.

3. Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). "Improving Language Understanding by Generative Pre-Training." (GPT) — Applied the Transformer decoder for autoregressive language modeling.

4. Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." arXiv:2010.11929. — Adapted the Transformer to vision (ViT).

5. Bahdanau, D., Cho, K., & Bengio, Y. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate." arXiv:1409.0473. — Introduced additive attention to seq2seq, the direct predecessor that the Transformer built upon.

6. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). "Sequence to Sequence Learning with Neural Networks." arXiv:1409.3215. — The RNN encoder-decoder that the Transformer replaced.
