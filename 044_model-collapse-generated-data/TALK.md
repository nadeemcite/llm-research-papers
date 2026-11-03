---
title: "Model Collapse: Talk & Q&A"
parent: "044 Model Collapse"
nav_order: 2
---
# TALK — The Curse of Recursion: Training on Generated Data Makes Models Forget (Model Collapse)

## Talks, Slides and Press Coverage

1. **VentureBeat — "Generative inbreeding and its risk to human culture" (Sharon Goldman, 14 Aug 2023):** Discusses the model collapse paper's findings in the context of AI-generated content polluting future training data, describing the phenomenon as "AI inbreeding." ([link](https://venturebeat.com/business/generative-inbreeding-and-its-risk-to-human-culture))

2. **Axios — "AI content flood could cause 'model collapse'" (Ryan Heath, 28 Aug 2023):** Reports on the paper's warning that a flood of AI-generated content online could degrade future AI models through model collapse. ([link](https://www.axios.com/2023/08/28/ai-content-flood-model-collapse))

3. **Tom's Hardware — "Generative AI goes mad when trained on artificial data over five times" (Andy Edser, 2024):** Covers the Nature publication of the paper, noting that models trained recursively on their own output degrade rapidly. ([link](https://www.tomshardware.com/news/generative-ai-goes-mad-when-trained-on-artificial-data-over-five-times))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2305.17493).

**Q1: Why does model collapse happen even with perfect functional approximation?**
A1 (from the paper, Section 3.1): The primary cause is statistical approximation error, which "arises due to the number of samples being finite, and disappears as the number of samples tends to infinity." Even with a perfect model, finite sampling means "there is a non-zero probability that information can get lost at every step of re-sampling." Once rare events are missed, they cannot reappear.

**Q2: What is the difference between model collapse and catastrophic forgetting?**
A2 (from the paper, Section 3): "This process is different from the process of catastrophic forgetting in that we are considering multiple models over time, in which our models do not forget previously learned data, but rather start misinterpreting what they believe to be real, by reinforcing their own beliefs."

**Q3: Can mixing in original data prevent model collapse?**
A3 (from the paper, Section 5.2): The paper tested 10% original data preservation with 10 epochs. They "find that preservation of the original data allows for better model fine-tuning and leads to only minor degradation of performance." However, degradation still occurs — perplexity rises from ~17 to ~25 over 9 generations, showing that partial preservation slows but does not stop collapse.

**Q4: Why was the paper originally called "model dementia"?**
A4 (from the paper, footnote 1): "The original version of this paper referred to this effect as 'model dementia', but we decided to change this following feedback that it trivialised the medical notion of 'dementia' and could cause offence."

**Q5: Does using a repetition penalty fix the problem?**
A5 (from the paper, Section 5.2): No. "We find that this causes the models to produce lower score continuations to avoid using repeats, which as a result causes the consequent models to perform even worse." In fact, "enforcing this for the LLM experiments causes the perplexity to double, compared to the original. Models remain as susceptible to Model Collapse, if not more."

## Common Misconceptions

1. **"Model collapse means the model produces garbage immediately."** — In early model collapse, the model still produces plausible outputs; it is the tails (rare events) that disappear first. The paper distinguishes "early model collapse" (tails lost) from "late model collapse" (distribution converges to a point estimate). Degradation is gradual over generations.

2. **"Just use more data per generation to fix it."** — The paper proves that the Wasserstein distance grows as the sum of 1/M terms across generations. Even with large but constant M, the distance diverges. Only superlinearly increasing sample sizes (e.g., quadratic growth) can keep it bounded, and even then the expected distance remains non-zero.

3. **"Model collapse is the same as GAN mode collapse."** — While the name is "inspired by the Generative Adversarial Networks (GAN) literature on mode collapse," the mechanism is different. GAN mode collapse happens during training of a single model; model collapse happens across generations of separately trained models, driven by recursive training on generated data.

## Real Citations

1. Shumailov, I., Shumaylov, Z., Zhao, Y., Gal, Y., Papernot, N. & Anderson, R. (2024). "The Curse of Recursion: Training on Generated Data Makes Models Forget." *Nature*, 631, 755-759. [arXiv:2305.17493](https://arxiv.org/abs/2305.17493) — the original paper.

2. Shumailov, I., Shumaylov, Z., Zhao, Y., Papernot, N., Gal, Y. & Anderson, R. (2024). "AI models collapse when trained on recursively generated data." *Nature*, 631, 755-759. [doi:10.1038/s41586-024-07566-y](https://doi.org/10.1038/s41586-024-07566-y) — the Nature publication of the same work.

**Citation count:** about 155 citations on [OpenAlex](https://openalex.org/W4378765257) (checked November 2026).
