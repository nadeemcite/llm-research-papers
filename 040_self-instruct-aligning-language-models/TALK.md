---
title: "Self-Instruct: Talk & Q&A"
parent: "040 Self-Instruct"
nav_order: 2
---
# TALK — Self-Instruct: Aligning Language Models with Self-Generated Instructions (Self-Instruct)

## Talks, Slides and Press Coverage

1. **Official code and data release — yizhongw/self-instruct on GitHub (2022):** The authors released the full Self-Instruct pipeline code, the 52K generated instruction dataset, and the 175 seed tasks used in the paper. The repository has become a reference implementation for self-generated instruction tuning. ([GitHub](https://github.com/yizhongw/self-instruct))

2. **Hugging Face dataset — yizhongw/self_instruct (2023):** The 52K Self-Instruct dataset was published on Hugging Face, making it easily accessible for the open-source community. It has been widely used in follow-up instruction-tuning projects. ([Hugging Face](https://huggingface.co/datasets/yizhongw/self_instruct))

3. **ACL 2023 conference paper — Self-Instruct was accepted to ACL 2023:** The paper was initially released as an arXiv preprint on 20 December 2022 and accepted to the ACL 2023 conference. ([ACL 2023](https://aclanthology.org/2023.acl-long.754/))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their
section in the [arXiv paper](https://arxiv.org/abs/2212.10560).

**Q1: Why does Self-Instruct use only 175 seed tasks?**
A1 (from the paper, Section 2.2): The seed set is intentionally small — 175 tasks — because the goal is to demonstrate that a minimal amount of human effort can bootstrap a much larger dataset. The authors state: "the overall process is an iterative bootstrapping algorithm which starts off with a limited (e.g., 175 in our study) seed set of manually-written tasks that are used to guide the overall generation."

**Q2: How does Self-Instruct compare to InstructGPT?**
A2 (from the paper, Abstract): "we demonstrate a 33% absolute improvement over the original model on Super-NaturalInstructions, on par with the performance of InstructGPT-001, which was trained with private user data and human annotations." The paper notes only a 5% absolute gap remains on the novel user-oriented instruction set.

**Q3: What filtering rules does Self-Instruct use to ensure data quality?**
A3 (from the paper, Section 2.2, Filtering and Postprocessing): The pipeline applies several rules: removing instructions with ROUGE-L overlap greater than 0.7 with existing instructions, skipping instructions longer than 500 tokens, discarding instances with repetitive or empty text, and limiting to 6 instances per instruction.

**Q4: Does Self-Instruct work on smaller models, or does it require GPT3-scale?**
A4 (from the paper, Section 8, Limitations): The authors explicitly note a dependence on large models: "the quality of the generated data depends on the quality of the language model used for generation." They report that running Self-Instruct with smaller models (GPT2-large, OPT-6.7B) produced lower-quality instructions, suggesting the approach benefits from strong base models.

## Common Misconceptions

1. **"Self-Instruct is fully automatic with no human involvement"** — Not quite. The paper calls it "semi-automated" (Section 1). It requires 175 human-written seed tasks to start, and the evaluation on novel tasks uses expert-written instructions. The "self" refers to the model generating additional data from the seeds, not to zero human effort.

2. **"Self-Instruct replaces RLHF"** — Self-Instruct is a supervised fine-tuning method, not a reinforcement learning method. The paper compares against InstructGPT-001 (which uses supervised fine-tuning) and notes that InstructGPT-003 (which uses RLHF/PPO) is more powerful but harder to compare with directly (footnote 1).

3. **"The 52K instructions are all high quality"** — The paper acknowledges quality issues. Section 3.3 shows that a significant fraction of generated instances have errors (e.g., 1.8% of generation task instances had mismatched input-output formats). The filtering rules catch obvious problems but do not guarantee perfection.

## Real Citations

1. Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N.A., Khashabi, D., Hajishirzi, H. (2022). "Self-Instruct: Aligning Language Models with Self-Generated Instructions." *ACL 2023.* [arXiv:2212.10560](https://arxiv.org/abs/2212.10560) — the original paper.

2. Taori, R. et al. (2023). "Stanford Alpaca: An Instruction-following LLaMA Model." [GitHub](https://github.com/tatsu-lab/stanford_alpaca) — used the Self-Instruct method to generate 52K instructions from GPT3.5 (text-davinci-003) and fine-tune LLaMA, becoming one of the most popular open instruction-tuning recipes.

3. Yuan, W. et al. (2024). "Self-Rewarding Language Models." [arXiv:2401.10020](https://arxiv.org/abs/2401.10020) — extends the self-bootstrapping idea to reward modeling, where the model generates its own preference data for DPO training.

**Citation count:** about 726 citations on [OpenAlex](https://openalex.org/W4385572634) (checked November 2026).
