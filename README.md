<p align="center"><img src="assets/banner.svg" alt="100 Papers That Changed AI — Season 2: The LLM Era: reimplemented from scratch, validated on Kaggle" width="100%"></p>

<p align="center">
  <img alt="papers" src="https://img.shields.io/badge/papers-100%20%2F%20100-4ea8de">
  <a href="https://www.kaggle.com/nadymsazad"><img alt="Kaggle" src="https://img.shields.io/badge/Kaggle-run%20every%20notebook-20BEFF?logo=kaggle&logoColor=white"></a>
  <a href="https://www.youtube.com/@BackpropDaily"><img alt="YouTube" src="https://img.shields.io/badge/YouTube-Backprop%20Daily-FF0000?logo=youtube&logoColor=white"></a>
  <a href="https://github.com/nadeemcite/llm-research-papers/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-57cc99"></a>
</p>

**100 landmark papers from the LLM era**, from RoPE (2019) to Artificial Hivemind (2026), each turned into a small, runnable, from-scratch PyTorch reimplementation with a plain-English explainer. Every notebook in this repo was **executed top to bottom on Kaggle** and is published with its real outputs, so you can open it, see it work, and fork it on a free GPU.

## How it's made: agent-generated, GPU-validated, human-reviewed

This repo is also an experiment in what AI agents can build when the bar is *"it actually runs."*

1. **Curated list:** I picked 100 landmark LLM-era papers and wrote, for each one, a spec of what its notebook should demonstrate.
2. **Agent-generated:** an AI agent reads the paper, writes the from-scratch implementation, and drafts the explainer docs.
3. **GPU-validated:** every notebook is pushed to Kaggle and run end-to-end, on a T4 GPU or on CPU for the small ones. A notebook ships only if every cell runs with zero errors, and the executed version, with its outputs, is what gets committed.
4. **Human-reviewed:** I review the code, results and claims before publishing, and re-run or rewrite the notebooks whose results don't hold up.

> **Scaled down on purpose.** These are teaching-sized reproductions: small models, synthetic or small datasets, minutes of training instead of days. They show *how* each idea works and reproduce the qualitative effect, not the paper's headline numbers. Each `CODE_ARCHITECTURE.md` lists the simplifications made.

## The family tree

<p align="center"><a href="assets/family-tree.svg"><img src="assets/family-tree.svg" alt="Family tree of the papers in this repo, by year and domain" width="100%"></a></p>

## What's in each folder

| File | What it gives you |
|---|---|
| `README.md` | A 5-minute read: summary, core idea, key method details, influence, and the problem explained like you're 10 |
| `solution.ipynb` | The from-scratch implementation, executed on Kaggle with all outputs |
| `CODE_ARCHITECTURE.md` | A guided walkthrough of the notebook: data flow, tensor shapes, and simplifications vs. the paper |
| `TALK.md` | Coverage, Q&A answered from the paper, common misconceptions, and citations |

## Papers by domain

[Architecture & Attention](#architecture--attention) · [State Space & Linear Models](#state-space--linear-models) · [Inference Efficiency](#inference-efficiency) · [Quantization](#quantization) · [Scaling Laws & Training](#scaling-laws--training) · [Training Methods](#training-methods) · [Data & Curation](#data--curation) · [Alignment & Preference](#alignment--preference) · [Reasoning](#reasoning) · [Agents & Tools](#agents--tools) · [RAG & Retrieval](#rag--retrieval) · [Mixture of Experts](#mixture-of-experts) · [Multimodal](#multimodal) · [Diffusion LMs](#diffusion-lms) · [Interpretability](#interpretability) · [Safety & Alignment](#safety--alignment)

### Architecture & Attention

| # | Year | Paper | Run it |
|---|---|---|---|
| 004 | 2019 | [Root Mean Square Layer Normalization (RMSNorm)](004_rmsnorm-root-mean-square-layer-norm/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/rmsnorm-root-mean-square-layer-normalization) |
| 003 | 2020 | [GLU Variants Improve Transformer (SwiGLU)](003_swiglu-glu-variants-improve-transformer/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/swiglu-glu-variants-improve-transformer) |
| 001 | 2021 | [RoFormer: Enhanced Transformer with Rotary Position Embedding](001_roformer-rotary-position-embedding/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/roformer-rotary-position-embedding) |
| 002 | 2021 | [Train Short Test Long: Attention with Linear Biases (ALiBi)](002_alibi-attention-with-linear-biases/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/alibi-attention-with-linear-biases) |
| 005 | 2023 | [GQA: Training Generalized Multi-Query Transformer Models](005_gqa-generalized-multi-query-attention/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/gqa-generalized-multi-query-attention) |
| 006 | 2023 | [LLaMA: Open and Efficient Foundation Language Models](006_llama-open-and-efficient-foundation-language-models/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/llama-open-and-efficient-foundation-language-models) |
| 007 | 2023 | [Mistral 7B](007_mistral-7b-sliding-window-attention/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/mistral-7b-sliding-window-attention) |
| 008 | 2023 | [YaRN: Efficient Context Window Extension of Large Language Models](008_yarn-efficient-context-window-extension/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/yarn-efficient-context-window-extension) |
| 009 | 2024 | [DeepSeek-V2: Multi-head Latent Attention (MLA)](009_deepseek-v2-multi-head-latent-attention/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/deepseek-v2-multi-head-latent-attention) |
| 010 | 2024 | [Better and Faster LLMs via Multi-token Prediction](010_multi-token-prediction-llms/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/multi-token-prediction-llms) |
| 011 | 2024 | [Differential Transformer](011_differential-transformer/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/differential-transformer) |
| 015 | 2024 | [Byte Latent Transformer: Patches Scale Better Than Tokens](015_byte-latent-transformer/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/byte-latent-transformer) |
| 012 | 2025 | [Gated Attention for Large Language Models](012_gated-attention-llm/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/gated-attention-large-language-models) |
| 013 | 2025 | [mHC: Manifold-Constrained Hyper-Connections](013_mhc-manifold-constrained-hyper-connections/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/mhc-manifold-constrained-hyper-connections) |
| 014 | 2026 | [Conditional Memory via Scalable Lookup (Engram)](014_engram-conditional-memory-scalable-lookup/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/engram-conditional-memory-scalable-lookup) |

### State Space & Linear Models

| # | Year | Paper | Run it |
|---|---|---|---|
| 016 | 2021 | [Efficiently Modeling Long Sequences with Structured State Spaces (S4)](016_s4-structured-state-spaces/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/s4-structured-state-spaces) |
| 019 | 2023 | [RWKV: Reinventing RNNs for the Transformer Era](019_rwkv-reinventing-rnns-transformer-era/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/rwkv-reinventing-rnns-transformer-era) |
| 017 | 2024 | [xLSTM: Extended Long Short-Term Memory](017_xlstm-extended-lstm/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/xlstm-extended-long-short-term-memory) |
| 018 | 2024 | [Transformers are SSMs: Generalized Models via Structured State Space Duality](018_mamba-2-structured-state-space-duality/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/mamba-2-structured-state-space-duality) |
| 020 | 2024 | [Gated Delta Networks: Improving Mamba2 with Delta Rule](020_gated-deltanet-improving-mamba2/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/gated-deltanet-improving-mamba2) |
| 021 | 2025 | [Kimi Linear: An Expressive Efficient Attention Architecture](021_kimi-linear-attention-architecture/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/kimi-linear-attention-architecture) |
| 022 | 2025 | [Titans: Learning to Memorize at Test Time](022_titans-learning-to-memorize-test-time/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/titans-learning-to-memorize-test-time) |

### Inference Efficiency

| # | Year | Paper | Run it |
|---|---|---|---|
| 025 | 2022 | [Fast Inference from Transformers via Speculative Decoding](025_speculative-decoding-fast-inference/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/speculative-decoding-fast-inference) |
| 023 | 2023 | [FlashAttention-2: Faster Attention with Better Parallelism](023_flashattention-2-faster-attention/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/flashattention-2-faster-attention) |
| 024 | 2023 | [Efficient Memory Management for LLM Serving with PagedAttention (vLLM)](024_pagedattention-vllm-memory-management/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/pagedattention-vllm-memory-management) |
| 027 | 2023 | [Efficient Streaming Language Models with Attention Sinks](027_streamingllm-attention-sinks/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/streamingllm-attention-sinks) |
| 026 | 2024 | [Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads](026_medusa-multiple-decoding-heads/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/medusa-multiple-decoding-heads) |
| 032 | 2025 | [Native Sparse Attention: Hardware-Aligned and Natively Trainable](032_native-sparse-attention/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/native-sparse-attention) |
| 033 | 2025 | [DeepSeek-V3.2: Pushing the Frontier of Open LLMs (DeepSeek Sparse Attention)](033_deepseek-sparse-attention/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/deepseek-sparse-attention) |

### Quantization

| # | Year | Paper | Run it |
|---|---|---|---|
| 028 | 2022 | [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](028_llm-int8-8-bit-matmul/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/llm-int8-8-bit-matrix-multiplication) |
| 029 | 2022 | [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](029_gptq-post-training-quantization/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/gptq-post-training-quantization) |
| 030 | 2023 | [AWQ: Activation-aware Weight Quantization for LLM Compression](030_awq-activation-aware-weight-quantization/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/awq-activation-aware-weight-quantization) |
| 031 | 2024 | [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](031_bitnet-b1.58-1-bit-llms/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/bitnet-b1.58-1-bit-llms) |

### Scaling Laws & Training

| # | Year | Paper | Run it |
|---|---|---|---|
| 034 | 2020 | [Scaling Laws for Neural Language Models (Kaplan)](034_scaling-laws-neural-language-models/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/scaling-laws-neural-language-models) |
| 035 | 2024 | [DeepSeek-V3 Technical Report](035_deepseek-v3-technical-report/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/deepseek-v3-technical-report) |
| 036 | 2025 | [Muon is Scalable for LLM Training](036_muon-scalable-llm-training/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/muon-scalable-llm-training) |
| 037 | 2025 | [Superposition Yields Robust Neural Scaling](037_superposition-robust-neural-scaling/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/superposition-robust-neural-scaling) |

### Training Methods

| # | Year | Paper | Run it |
|---|---|---|---|
| 038 | 2022 | [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](038_grokking-generalization-beyond-overfitting/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/grokking-generalization-beyond-overfitting) |
| 039 | 2023 | [QLoRA: Efficient Finetuning of Quantized LLMs](039_qlora-efficient-finetuning-quantized-llms/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/qlora-efficient-finetuning-quantized-llms) |

### Data & Curation

| # | Year | Paper | Run it |
|---|---|---|---|
| 040 | 2022 | [Self-Instruct: Aligning Language Models with Self-Generated Instructions](040_self-instruct-aligning-language-models/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/self-instruct-aligning-language-models) |
| 041 | 2023 | [LIMA: Less Is More for Alignment](041_lima-less-is-more-for-alignment/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/lima-less-is-more-for-alignment) |
| 042 | 2023 | [Textbooks Are All You Need (phi-1)](042_phi-1-textbooks-are-all-you-need/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/phi-1-textbooks-are-all-you-need) |
| 043 | 2023 | [TinyStories: How Small Can Language Models Be and Still Speak Coherent English](043_tinystories-small-language-models/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/tinystories-small-language-models) |
| 044 | 2023 | [The Curse of Recursion: Training on Generated Data Makes Models Forget](044_model-collapse-generated-data/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/model-collapse-generated-data) |

### Alignment & Preference

| # | Year | Paper | Run it |
|---|---|---|---|
| 045 | 2022 | [Constitutional AI: Harmlessness from AI Feedback](045_constitutional-ai-harmlessness-ai-feedback/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/constitutional-ai-harmlessness-ai-feedback) |
| 046 | 2023 | [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](046_dpo-direct-preference-optimization/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/dpo-direct-preference-optimization) |
| 051 | 2023 | [Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision](051_weak-to-strong-generalization/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/weak-to-strong-generalization) |
| 047 | 2024 | [KTO: Model Alignment as Prospect Theoretic Optimization](047_kto-prospect-theoretic-optimization/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/kto-prospect-theoretic-optimization) |
| 048 | 2024 | [ORPO: Monolithic Preference Optimization without Reference Model](048_orpo-monolithic-preference-optimization/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/orpo-monolithic-preference-optimization) |
| 049 | 2024 | [SimPO: Simple Preference Optimization with a Reference-Free Reward](049_simpo-reference-free-preference-optimization/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/simpo-reference-free-preference-optimization) |
| 050 | 2024 | [Self-Rewarding Language Models](050_self-rewarding-language-models/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/self-rewarding-language-models) |

### Reasoning

| # | Year | Paper | Run it |
|---|---|---|---|
| 052 | 2022 | [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](052_chain-of-thought-prompting-reasoning/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/chain-of-thought-prompting-reasoning) |
| 053 | 2022 | [Large Language Models are Zero-Shot Reasoners](053_zero-shot-cot-reasoning/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/zero-shot-cot-reasoning) |
| 056 | 2022 | [STaR: Bootstrapping Reasoning With Reasoning](056_star-bootstrapping-reasoning/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/star-bootstrapping-reasoning) |
| 054 | 2023 | [Self-Consistency Improves Chain of Thought Reasoning in Language Models](054_self-consistency-chain-of-thought/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/self-consistency-chain-of-thought-reasoning) |
| 055 | 2023 | [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](055_tree-of-thoughts-deliberate-problem-solving/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/tree-of-thoughts-deliberate-problem-solving) |
| 057 | 2023 | [Let's Verify Step by Step (Process Reward Models)](057_process-reward-models-verify-step-by-step/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/process-reward-models-verify-step-by-step) |
| 058 | 2024 | [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](058_scaling-test-time-compute-optimally/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/scaling-test-time-compute-optimally) |
| 059 | 2024 | [DeepSeekMath: Pushing the Limits of Mathematical Reasoning (GRPO)](059_grpo-deepseek-mathematical-reasoning/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/grpo-deepseek-mathematical-reasoning) |
| 065 | 2024 | [Training Large Language Models to Reason in a Continuous Latent Space (Coconut)](065_coconut-continuous-latent-space-reasoning/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/coconut-continuous-latent-space-reasoning) |
| 060 | 2025 | [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](060_deepseek-r1-reinforcement-learning-reasoning/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/deepseek-r1-reinforcement-learning-reasoning) |
| 061 | 2025 | [s1: Simple test-time scaling](061_s1-simple-test-time-scaling/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/s1-simple-test-time-scaling) |
| 062 | 2025 | [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](062_dapo-open-source-llm-rl-system/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/dapo-open-source-llm-rl-system) |
| 063 | 2025 | [Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO)](063_dr-grpo-critical-perspective-r1-zero/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/dr-grpo-critical-perspective-r1-zero) |
| 064 | 2025 | [Does Reinforcement Learning Really Incentivize Reasoning Capacity Beyond the Base Model](064_rl-reasoning-beyond-base-model/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/rl-reasoning-beyond-base-model) |
| 066 | 2025 | [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](066_absolute-zero-reinforced-self-play-reasoning/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/absolute-zero-reinforced-self-play-reasoning) |
| 067 | 2025 | [Less is More: Recursive Reasoning with Tiny Networks](067_tiny-recursive-model-reasoning/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/tiny-recursive-model-reasoning) |
| 068 | 2026 | [Learning to Discover at Test Time](068_learning-to-discover-test-time/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/learning-to-discover-test-time) |

### Agents & Tools

| # | Year | Paper | Run it |
|---|---|---|---|
| 069 | 2022 | [ReAct: Synergizing Reasoning and Acting in Language Models](069_react-synergizing-reasoning-acting/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/react-synergizing-reasoning-acting) |
| 070 | 2023 | [Toolformer: Language Models Can Teach Themselves to Use Tools](070_toolformer-teaching-lms-to-use-tools/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/toolformer-teaching-lms-to-use-tools) |
| 071 | 2023 | [Reflexion: Language Agents with Verbal Reinforcement Learning](071_reflexion-verbal-reinforcement-learning/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/reflexion-verbal-reinforcement-learning) |
| 072 | 2023 | [Generative Agents: Interactive Simulacra of Human Behavior](072_generative-agents-interactive-simulacra/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/generative-agents-interactive-simulacra) |
| 073 | 2023 | [Voyager: An Open-Ended Embodied Agent with Large Language Models](073_voyager-open-ended-embodied-agent/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/voyager-open-ended-embodied-agent) |
| 074 | 2023 | [SWE-bench: Can Language Models Resolve Real-World GitHub Issues](074_swe-bench-resolve-github-issues/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/swe-bench-resolve-github-issues) |
| 075 | 2024 | [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](075_swe-agent-agent-computer-interfaces/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/swe-agent-agent-computer-interfaces) |
| 076 | 2025 | [AlphaEvolve: A coding agent for scientific and algorithmic discovery](076_alphaevolve-coding-agent-discovery/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/alphaevolve-coding-agent-discovery) |

### RAG & Retrieval

| # | Year | Paper | Run it |
|---|---|---|---|
| 077 | 2022 | [Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)](077_hyde-hypothetical-document-embeddings/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/hyde-hypothetical-document-embeddings) |
| 081 | 2022 | [Matryoshka Representation Learning](081_matryoshka-representation-learning/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/matryoshka-representation-learning) |
| 078 | 2023 | [Lost in the Middle: How Language Models Use Long Contexts](078_lost-in-the-middle-long-contexts/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/lost-in-the-middle-long-contexts) |
| 079 | 2023 | [Self-RAG: Learning to Retrieve Generate and Critique through Self-Reflection](079_self-rag-retrieve-generate-critique/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/self-rag-retrieve-generate-critique) |
| 080 | 2024 | [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](080_graphrag-graph-rag-query-summarization/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/graphrag-graph-rag-query-summarization) |
| 082 | 2025 | [LLMs Get Lost In Multi-Turn Conversation](082_llms-lost-in-multi-turn-conversation/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/llms-lost-in-multi-turn-conversation) |

### Mixture of Experts

| # | Year | Paper | Run it |
|---|---|---|---|
| 083 | 2024 | [Mixtral of Experts](083_mixtral-of-experts/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/mixtral-of-experts) |
| 084 | 2024 | [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts](084_deepseek-moe-expert-specialization/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/deepseek-moe-expert-specialization) |

### Multimodal

| # | Year | Paper | Run it |
|---|---|---|---|
| 086 | 2022 | [Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)](086_whisper-robust-speech-recognition/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/whisper-robust-speech-recognition) |
| 085 | 2023 | [Visual Instruction Tuning (LLaVA)](085_llava-visual-instruction-tuning/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/llava-visual-instruction-tuning) |

### Diffusion LMs

| # | Year | Paper | Run it |
|---|---|---|---|
| 087 | 2025 | [Large Language Diffusion Models (LLaDA)](087_llada-large-language-diffusion-models/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/llada-large-language-diffusion-models) |

### Interpretability

| # | Year | Paper | Run it |
|---|---|---|---|
| 088 | 2022 | [In-context Learning and Induction Heads](088_induction-heads-in-context-learning/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/induction-heads-in-context-learning) |
| 089 | 2022 | [Toy Models of Superposition](089_toy-models-of-superposition/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/toy-models-of-superposition) |
| 091 | 2022 | [Locating and Editing Factual Associations in GPT (ROME)](091_rome-locating-editing-factual-associations/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/rome-locating-editing-factual-associations) |
| 090 | 2023 | [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](090_towards-monosemanticity-dictionary-learning/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/towards-monosemanticity-dictionary-learning) |
| 093 | 2023 | [Representation Engineering: A Top-Down Approach to AI Transparency](093_representation-engineering-ai-transparency/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/representation-engineering-ai-transparency) |
| 098 | 2025 | [Persona Vectors: Monitoring and Controlling Character Traits in Language Models](098_persona-vectors-monitoring-controlling-traits/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/persona-vectors-monitoring-controlling-traits) |
| 099 | 2025 | [Reasoning Models Don't Always Say What They Think](099_reasoning-models-dont-say-what-they-think/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/reasoning-models-dont-say-what-they-think) |
| 100 | 2025 | [Artificial Hivemind: The Open-Ended Homogeneity of Language Models](100_artificial-hivemind-llm-homogeneity/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/artificial-hivemind-llm-homogeneity) |

### Safety & Alignment

| # | Year | Paper | Run it |
|---|---|---|---|
| 092 | 2023 | [Universal and Transferable Adversarial Attacks on Aligned Language Models (GCG)](092_gcg-adversarial-attacks-aligned-llms/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/gcg-adversarial-attacks-aligned-llms) |
| 094 | 2024 | [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](094_sleeper-agents-deceptive-llms/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/sleeper-agents-deceptive-llms) |
| 095 | 2024 | [Alignment Faking in Large Language Models](095_alignment-faking-large-language-models/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/alignment-faking-large-language-models) |
| 096 | 2025 | [Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs](096_emergent-misalignment-narrow-finetuning/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/emergent-misalignment-narrow-finetuning) |
| 097 | 2025 | [Subliminal Learning: Language models transmit behavioral traits via hidden signals](097_subliminal-learning-hidden-signals/) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/subliminal-learning-hidden-signals) |

<sub>Year = first public release (usually arXiv). Numbers follow the order of the curated reading list.</sub>

## Watch: Backprop Daily

Short videos on the same 100 papers: *learn AI the way models learn, one step at a time.* They're made by an automated pipeline that writes the script, narrates it, generates the visuals and renders the video.

More at **[youtube.com/@BackpropDaily](https://www.youtube.com/@BackpropDaily?sub_confirmation=1)**.

## About me

I'm **Nadeem Sajjad**, a Senior AI Engineer and forward-deployed engineer with 12+ years across startups and scale-ups.

[![Website](https://img.shields.io/badge/devnadeem.com-0d1117?logo=googlechrome&logoColor=white)](https://devnadeem.com)
[![GitHub](https://img.shields.io/badge/GitHub-nadeemcite-181717?logo=github)](https://github.com/nadeemcite)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-nadeem--sajjad-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nadeem-sajjad/)
[![X](https://img.shields.io/badge/X-@nadym__sazad-000000?logo=x)](https://x.com/nadym_sazad)
[![YouTube](https://img.shields.io/badge/YouTube-@BackpropDaily-FF0000?logo=youtube&logoColor=white)](https://www.youtube.com/@BackpropDaily)
[![Kaggle](https://img.shields.io/badge/Kaggle-nadymsazad-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/nadymsazad)

If this helped you understand a paper, a ⭐ helps others find it.

---

<sub>Licensed under the [MIT License](https://github.com/nadeemcite/llm-research-papers/blob/main/LICENSE). Generated by `scripts/build_readme.py` from `scripts/papers.tsv`; edit those, not this file.</sub>
