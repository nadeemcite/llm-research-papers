---
title: "STaR: Talk & Q&A"
parent: "056 STaR"
nav_order: 2
---
# TALK — STaR: Bootstrapping Reasoning With Reasoning (STaR)

## Talks, Slides and Press Coverage

1. **NeurIPS 2022 — Poster presentation:** STaR was accepted as a poster at NeurIPS 2022. The paper is listed in the conference proceedings. ([NeurIPS 2022 proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html))

No other talks, blog posts, or press coverage could be verified from primary sources at the time of writing.

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2203.14465).

**Q1: Why does STaR need rationalization in addition to rationale generation bootstrapping?**
A1 (from the paper, Section 3.2): "Since the model is only trained on the examples which it answers correctly, improvement ends when the model fails to solve new problems in the training set. This is fundamentally due to the fact that the algorithm cannot obtain any training signal from failed examples." Rationalization solves this by providing the correct answer as a hint, allowing the model to "reason backwards" and generate a useful rationale for problems it could not solve on its own.

**Q2: How does STaR compare to a 30× larger model?**
A2 (from the paper, Abstract): "performs comparably to fine-tuning a 30× larger state-of-the-art language model on CommonsenseQA." Specifically, STaR with rationalization on GPT-J (6B) achieved 72.5% on the CQA dev set, vs. 73.0% for a directly fine-tuned GPT-3 (175B).

**Q3: What base model did the authors use and why?**
A3 (from the paper, Section 4.1): "We used GPT-J as our base language model... We chose GPT-J, a 6B-parameter model, because the checkpoint and fine-tuning code are publicly available, and the model is large enough to generate rationales of non-trivial quality to be bootstrapped from."

**Q4: Does STaR work with any model size?**
A4 (from the paper, Section 6): "In order for the first iteration of STaR to succeed, few-shot performance must be above chance, implying that the initial model must be big enough to have some reasoning capabilities. For instance we found that GPT-2 was not able to bootstrap from few-shot reasoning in even the arithmetic domain." This is a notable limitation — the notebook's use of GPT-2 small (124M) is at the edge of what can bootstrap.

**Q5: Is STaR related to reinforcement learning?**
A5 (from the paper, Section 3.1): "STaR can be seen as an approximation to an RL-style policy gradient objective." The rationale is treated as a discrete latent variable, the indicator reward function (correct answer) filters rationales, and the log-derivative trick gives the policy gradient. STaR approximates this by greedily decoding and taking multiple gradient steps.

**Q6: What is the role of temperature in STaR?**
A6 (from the paper, Section 5): "in practice, we found that [higher temperature] is counterproductive. In general, it substantially increases the likelihood of a correct answer despite incorrect reasoning, and training on bad or irrelevant reasoning prevents generalization." The authors recommend low-temperature sampling over high-temperature alternatives to rationalization.

## Common Misconceptions

1. **"STaR is the same as chain-of-thought prompting."** — No. Chain-of-thought (CoT) prompting (Wei et al., 2022) uses a few examples with rationales in the prompt at inference time, without any fine-tuning. STaR uses CoT-style rationales as a starting point, but the key contribution is the iterative bootstrapping loop: the model generates rationales, filters by correctness, fine-tunes, and repeats. CoT is the input; STaR is the training method.

2. **"Rationalization means the model is just memorizing answers."** — The paper explicitly addresses this (Section 3.2): when adding rationalization-generated rationales to the training set, "we do not include the hint in its corresponding prompt, as if the model had come up with the rationale without the hint." The model must learn to produce the rationale independently. The hint is only used during data generation, not during training or inference.

3. **"STaR requires a massive dataset of human-annotated rationales."** — The opposite is true. The paper's central contribution is starting with only a small set of few-shot rationale examples (e.g., 10 for CommonsenseQA) and bootstrapping the rest from the model's own generations. This is explicitly contrasted with prior work requiring "massive rationale datasets" (Section 1).

4. **"STaR is just self-distillation."** — While STaR does involve a model learning from its own outputs, the key differences are (1) the correctness filter — only rationales leading to correct answers are kept, providing a training signal, and (2) rationalization, which introduces new information (the correct answer) that the model did not generate on its own. Self-distillation typically involves no such filtering or external signal.

## Real Citations

1. Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman (2022). "STaR: Bootstrapping Reasoning With Reasoning." *NeurIPS 2022.* [arXiv:2203.14465](https://arxiv.org/abs/2203.14465) — the original paper.
2. Jiaxin Huang, Difan Li, et al. (2023). "Large Language Models Cannot Self-Correct Reasoning Yet" (STaR\*). [arXiv:2310.01798](https://arxiv.org/abs/2310.01798) — extends STaR with self-consistency filtering, showing limitations of self-correction.
3. Caglar Gulcehre, et al. (2023). "Reinforced Self-Training (ReST) for Language Modeling." [arXiv:2312.06585](https://arxiv.org/abs/2312.06585) — separates the growing and improving phases of STaR-style bootstrapping into a structured framework.
4. Thomas Anthony, et al. (2017). "Expert Iteration." [arXiv:1703.08681](https://arxiv.org/abs/1703.08681) — the Expert Iteration framework that inspired STaR's iterative loop.
5. Jason Wei, et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) — the CoT prompting technique that STaR builds upon.

**Citation count:** about 249 citations on [OpenAlex](https://openalex.org/W7133192525) (checked November 2026).
