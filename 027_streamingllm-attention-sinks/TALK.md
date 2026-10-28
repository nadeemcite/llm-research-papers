---
title: "StreamingLLM: Talk & Q&A"
parent: "027 StreamingLLM"
nav_order: 2
---
# TALK — Efficient Streaming Language Models with Attention Sinks (StreamingLLM)

## Talks, Slides and Press Coverage

1. **MIT HAN Lab — "Streaming LLM" project page:** The official project page hosts the code, datasets, and interactive visualisations of attention sinks across Llama-2, MPT, Falcon, and Pythia models. ([link](https://github.com/mit-han-lab/streaming-llm))

2. **ICLR 2024 — "Efficient Streaming Language Models with Attention Sinks":** The paper was published at the 12th International Conference on Learning Representations (Vienna, Austria, May 2024). The paper is available at [arXiv:2309.17453](https://arxiv.org/abs/2309.17453).

3. **ICLR 2024 OpenReview — "Efficient Streaming Language Models with Attention Sinks":** The paper's review page at ICLR 2024, with reviewer discussions and the final decision. ([link](https://openreview.net/forum?id=NJfbDtEKqC))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2309.17453).

**Q1: Why do initial tokens receive so much attention even when they are not semantically important?**
A1 (from the paper, Section 3.1): "The nature of the SoftMax function (Equation 1) prevents all attended tokens from having zero values. This requires aggregating some information from other tokens across all heads in all layers, even if the current embedding has sufficient self-contained information for its prediction. Consequently, the model tends to dump unnecessary attention values to specific tokens." Initial tokens are the natural sink because "initial tokens are visible to almost all subsequent tokens because of the autoregressive language modeling nature, making them more readily trained to serve as attention sinks."

**Q2: How was it confirmed that the semantic content of the initial tokens does not matter?**
A2 (from the paper, Table 1, Section 3.1): The authors replaced the first four tokens with linebreak tokens ("\n"). The perplexity with original initial tokens (4+1020) was 5.40; with linebreak substitutes (4"\n"+1020) it was 5.60 — nearly identical. "This suggests that the absolute position of the starting tokens, rather than their semantic value, holds greater significance."

**Q3: How many sink tokens are needed?**
A3 (from the paper, Table 2, Section 3.1): "Introducing four initial tokens generally suffices; further additions have diminishing returns." For Llama-2-7B, 0+4096 gives PPL 3359.95, 1+4095 gives 11.88, 2+4094 gives 10.51, and 4+4092 gives 9.59. Going to 8+4088 gives 9.54 — barely better.

**Q4: What is the sink token pre-training idea and does it hurt normal performance?**
A4 (from the paper, Section 3.3 and Table 4): A single learnable placeholder token is prepended to every training sample. "Including a sink token during pre-training has no negative impact on model convergence and subsequent performance on a range of NLP benchmarks." Across 7 benchmarks (ARC-c, ARC-e, HellaSwag, LAMBADA, OpenbookQA, PIQA, Winogrande), the sink-token model scores equal or slightly higher than vanilla.

**Q5: Does StreamingLLM extend the model's context window?**
A5 (from the paper, Section 1): "StreamingLLM efficiently generates coherent text from tokens within the KV cache without extending the LLMs' context length. It suits continuous operation needs with minimal memory use and past data reliance. Additionally, StreamingLLM can complement context extension methods to increase the attendable recent context." The model attends only to the sink tokens plus the recent window — it cannot retrieve information from tokens that have been evicted.

**Q6: How does StreamingLLM handle position encoding when tokens are evicted?**
A6 (from the paper, Section 3.2): "When determining the relative distance and adding positional information to tokens, StreamingLLM focuses on positions within the cache rather than those in the original text." If the cache has tokens [0, 1, 2, 3, 6, 7, 8] and is decoding token 9, positions are [0, 1, 2, 3, 4, 5, 6, 7] — contiguous. "For encoding like RoPE, we cache the Keys of tokens prior to introducing the rotary transformation."

## Common Misconceptions

1. **"StreamingLLM lets the model remember everything from the conversation"** — No. StreamingLLM only attends to the sink tokens and a sliding window of recent tokens. Once a token is evicted from the rolling cache, it is gone. The model cannot retrieve information from the middle of a long conversation. The paper explicitly states (Section 1) that it does not enhance the model's "memory and usage on long texts."

2. **"Attention sinks mean the first tokens are semantically important"** — The opposite. Table 1 shows that replacing the first four tokens with meaningless linebreak tokens restores perplexity equally well. The sinks are a side effect of softmax requiring attention to sum to one, not a sign that the model found the beginning of the text meaningful.

3. **"You need to fine-tune the model to use StreamingLLM"** — No fine-tuning is required for existing models. The paper shows that simply keeping four initial KV pairs alongside the rolling window works on Llama-2, MPT, Falcon, and Pythia without any training. The sink token pre-training is an optional recommendation for *future* models.

## Real Citations

1. Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis (2024). "Efficient Streaming Language Models with Attention Sinks." *ICLR 2024.* [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) — the original paper.

2. Zhen Zhang, Yuxiao Dong, Sheng Wang, Jue Wang, Ming-Xie, Amir Houmansadr (2023). "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models." *NeurIPS 2023.* [arXiv:2306.14048](https://arxiv.org/abs/2306.14048) — builds on the attention sink observation to design a KV cache eviction policy based on heavy-hitter tokens.

3. Press Ofir, Smith Noah A., Lewis Mike (2022). "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." *ICLR 2022.* [arXiv:2108.12409](https://arxiv.org/abs/2108.12409) — ALiBi, one of the position encodings that StreamingLLM is compatible with; the paper tests MPT which uses ALiBi.

4. Su Jianlin, Lu Yu, Pan Shengfeng, Murtadha Ahmed, Wen Bo, Liu Yunfeng (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) — RoPE, the other position encoding tested; StreamingLLM caches pre-rotation keys and re-rotates at each step.

**Citation count:** about 55 citations on [OpenAlex](https://openalex.org/works/W4387294617) (checked October 2026).
