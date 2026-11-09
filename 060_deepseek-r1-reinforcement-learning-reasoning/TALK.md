---
title: "DeepSeek-R1: Talk & Q&A"
parent: "060 DeepSeek-R1"
nav_order: 2
---
# TALK — DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (DeepSeek-R1)

## Talks, Slides and Press Coverage

1. **DeepSeek-AI — "DeepSeek-R1" (DeepSeek-AI, 22 Jan 2025):** The official release of DeepSeek-R1-Zero and DeepSeek-R1, open-sourcing the models and describing the RL training approach. ([GitHub](https://github.com/deepseek-ai/DeepSeek-R1))
2. **Hugging Face — "DeepSeek-R1" model card:** The official Hugging Face release with model weights, usage instructions, and benchmark results. ([Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1))
3. **OpenAlex — DeepSeek-R1 work page:** Bibliographic record with citation count and reference network for the paper. ([OpenAlex](https://openalex.org/works/W4391631327))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2501.12948).

**Q1: Why does R1-Zero skip supervised fine-tuning entirely?**
A1 (from the paper, Section 1.1): "We directly apply RL to the base model without relying on supervised fine-tuning (SFT) as a preliminary step. This approach allows the model to explore chain-of-thought (CoT) for solving complex problems, resulting in the development of DeepSeek-R1-Zero. DeepSeek-R1-Zero demonstrates capabilities such as self-verification, reflection, and generating long CoTs, marking a significant milestone for the research community."

**Q2: What rewards does R1-Zero use?**
A2 (from the paper, Section 2.2.2): "To train DeepSeek-R1-Zero, we adopt a rule-based reward system that mainly consists of two types of rewards: Accuracy rewards [which] evaluate whether the response is correct [...] and Format rewards [which] enforce the model to put its thinking process between '' and '' tags. We do not apply the outcome or process neural reward model in developing DeepSeek-R1-Zero, because we find that the neural reward model may suffer from reward hacking in the large-scale reinforcement learning process."

**Q3: What is the "aha moment"?**
A3 (from the paper, Section 2.2.4): "A particularly intriguing phenomenon observed during the training of DeepSeek-R1-Zero is the occurrence of an 'aha moment'. This moment occurs in an intermediate version of the model. During this phase, DeepSeek-R1-Zero learns to allocate more thinking time to a problem by reevaluating its initial approach. This behavior is not only a testament to the model's growing reasoning abilities but also a captivating example of how reinforcement learning can lead to unexpected and sophisticated outcomes."

**Q4: Does RL on smaller models work as well as distillation from larger models?**
A4 (from the paper, Section 4.1): "The 32B base model, after large-scale RL training, achieves performance on par with QwQ-32B-Preview. However, DeepSeek-R1-Distill-Qwen-32B, which is distilled from DeepSeek-R1, performs significantly better than DeepSeek-R1-Zero-Qwen-32B across all benchmarks." The paper concludes: "distilling more powerful models into smaller ones yields excellent results, whereas smaller models relying on the large-scale RL mentioned in this paper require enormous computational power and may not even achieve the performance of distillation."

**Q5: What approaches did the authors try that did not work?**
A5 (from the paper, Section 4.2): The authors tried Process Reward Models (PRM) and Monte Carlo Tree Search (MCTS). For PRM: "it is challenging to explicitly define a fine-grain step in general reasoning" and "once a model-based PRM is introduced, it inevitably leads to reward hacking." For MCTS: "unlike chess, where the search space is relatively well-defined, token generation presents an exponentially larger search space" and "training a fine-grained value model is inherently difficult."

## Common Misconceptions

1. **"DeepSeek-R1 and DeepSeek-R1-Zero are the same model"** — They are not. R1-Zero applies RL directly to the base model with no SFT, while R1 uses a multi-stage pipeline (cold-start data, reasoning RL, rejection sampling and SFT, final RL for all scenarios). R1-Zero achieves 71.0% pass@1 on AIME 2024; R1 achieves 79.8%. R1-Zero also has issues with "poor readability, and language mixing" that R1 addresses (Section 2.2.4, 2.3).

2. **"R1-Zero uses no pre-training at all"** — R1-Zero starts from DeepSeek-V3-Base, a 671B-parameter model that was pre-trained on trillions of tokens. The "no SFT" claim refers specifically to supervised fine-tuning on reasoning traces, not to the base model's pre-training. The base model already has strong language and arithmetic capabilities before RL begins.

3. **"GRPO was invented in this paper"** — GRPO was introduced in DeepSeekMath (Shao et al., 2024). This paper adopts GRPO as its RL framework. The contribution of this paper is applying GRPO at scale for reasoning without SFT, not the algorithm itself.

## Real Citations

1. DeepSeek-AI (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." *arXiv preprint.* [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) — the original paper.
2. Zhihong Shao, Peiyi Wang, Qihao Zhu, et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) — introduces GRPO, the RL algorithm used in this paper.
3. Zichen Liu et al. (2025). "Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO)." [arXiv:2503.20783](https://arxiv.org/abs/2503.20783) — analyzes and simplifies GRPO further, providing critical insights into R1-Zero training.

**Citation count:** about 2,100 citations on [OpenAlex](https://www.openalex.org/works/W4391631327) (checked November 2026).
