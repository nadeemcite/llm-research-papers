---
title: "Toolformer: Talk & Q&A"
parent: "070 Toolformer"
nav_order: 2
---
# TALK — Toolformer: Language Models Can Teach Themselves to Use Tools (Toolformer)

## Talks, Slides and Press Coverage

1. **Harvard CMSA — "Timo Schick \| Toolformer: Language Models Can Teach Themselves to Use Tools" (26 Apr 2023):** Timo Schick presents the Toolformer method and results in a talk at the Harvard Center for Mathematical Sciences and Applications. ([video](https://www.youtube.com/watch?v=UID_oXuN-0Y))
2. **Arize AI — "Deep Papers Episode 3 - Toolformer: Training LLMs To Use Tools" (21 Mar 2023):** A discussion of the Toolformer paper in the Deep Papers podcast series by Arize AI. ([video](https://www.youtube.com/watch?v=pSKHDduKt_g))
3. **arXiv — "Toolformer: Language Models Can Teach Themselves to Use Tools" (9 Feb 2023):** The official arXiv page hosting the paper abstract, PDF, and author affiliations. ([link](https://arxiv.org/abs/2302.04761))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2302.04761).

**Q1: Why can't the model just be trained on tool calls provided by humans?**
A1 (from the paper, Section 1): The authors note that creating a large dataset of examples where a language model uses tools "requires a large number of human annotations" and is "expensive and time-consuming." Toolformer's self-supervised approach avoids this by having the model generate and filter its own training data, requiring only "a handful of demonstrations for each API."

**Q2: What is the filtering criterion and why does it work?**
A2 (from the paper, Section 2, "Filtering API Calls"): An API call is kept only if L⁻ − L⁺ ≥ τ_f, where L⁺ is the weighted loss with the API call and its result, and L⁻ is the minimum of (i) the loss without any API call and (ii) the loss with the API call but without the result. The intuition: "an API call is helpful to M if it leads to a lower loss" — the criterion ensures the call's result genuinely helps the model predict subsequent tokens, not just that the model inserted a call.

**Q3: Can Toolformer chain multiple tool calls together?**
A3 (from the paper, Section 7, "Limitations"): No. "One such limitation is the inability of Toolformer to use tools in a chain (i.e., using the output of one tool as an input for another tool). This is due to the fact that API calls for each tool are generated independently; as a consequence, there are no examples of chained tool use in the finetuning dataset."

**Q4: Does learning to use tools hurt the model's language modeling ability?**
A4 (from the paper, Section 4.3, "Language Modeling"): No. The paper reports perplexity on WikiText and a CCNet validation set and finds that "training on C* (our dataset augmented with API calls) does not lead to any deterioration in language modeling performance" compared to fine-tuning on plain CCNet. This confirms tool use comes "without sacrificing its core language modeling abilities."

**Q5: Why does Toolformer with API calls disabled still outperform the baseline on math?**
A5 (from the paper, Section 4.2.2, "Math Datasets"): The authors observe that "Toolformer achieves stronger results even when API calls are disabled. We surmise that this is because the model is finetuned on many examples of API calls and their results, improving its own mathematical capabilities." Seeing many calculator results during fine-tuning appears to teach the model some arithmetic indirectly.

**Q6: How sample-efficient is the approach?**
A6 (from the paper, Section 7, "Limitations"): Not very, for some tools. "Depending on the tool, our method is also very sample-inefficient; for example, processing more than a million documents results in only a few thousand examples of useful calls to the calculator API." The authors suggest "iteratively applying our approach" as a potential solution.

## Common Misconceptions

1. **"Toolformer uses reinforcement learning to teach the model tool use."** — No. Toolformer uses a self-supervised filtering approach: the model samples API calls, executes them, and filters by loss reduction. There is no reward model, no policy gradient, and no RLHF. The filtering criterion is a simple loss comparison, not a learned reward signal.

2. **"Toolformer can use multiple tools in a single chain of calls."** — The paper explicitly lists this as a limitation (Section 7). API calls for each tool are generated independently, so there are no chained tool-use examples in the training data. A model that needs to search for a fact and then translate the result cannot do so in a single pass.

3. **"Toolformer is a tool-use agent like ReAct."** — ReAct (Yao et al., 2022) prompts a frozen model to interleave reasoning and tool calls at inference time. Toolformer instead fine-tunes the model on self-labeled tool-call data so it learns to insert calls autonomously. The two approaches are complementary: ReAct needs no training but requires careful prompting; Toolformer needs training but then works zero-shot.

## Real Citations

1. Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom (2023). "Toolformer: Language Models Can Teach Themselves to Use Tools." *NeurIPS 2023.* [arXiv:2302.04761](https://arxiv.org/abs/2302.04761) — the original paper.
2. Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2022). "ReAct: Synergizing Reasoning and Acting in Language Models." [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) — a prompting-based approach to tool use that Toolformer complements with a training-based approach.
3. Shishir Patil, Tianjun Zhang, Xinyang Wang, Joseph E. Gonzalez (2023). "Gorilla: Large Language Model Connected with Massive APIs." [arXiv:2306.07824](https://arxiv.org/abs/2306.07824) — a follow-up that fine-tunes LLMs on API documentation for tool use.

**Citation count:** about 3,245 citations on [OpenAlex](https://openalex.org/works/W7133214447) (checked November 2026).
