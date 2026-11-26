---
title: "GCG Attack: Talk & Q&A"
parent: "092 GCG Attack"
nav_order: 2
---
# TALK — Universal and Transferable Adversarial Attacks on Aligned Language Models (GCG)

## Talks, Slides and Press Coverage

1. **Project website and live demo — "Universal and Transferable Attacks on Aligned Language Models" (Zou et al., 2023):** The authors maintain a website at [llm-attacks.org](https://llm-attacks.org) with interactive demos showing adversarial suffixes jailbreaking ChatGPT, Claude, and Bard — the demos display actual model outputs from the GCG attack on production LLMs. ([link](https://llm-attacks.org))
2. **Official code repository — "LLM Attacks" (Zou et al., 2023):** The GitHub repository contains the full implementation, demo notebook, and `nanogcg` fast library. ([link](https://github.com/llm-attacks/llm-attacks))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2307.15043).

**Q1: Why does targeting an affirmative response ("Sure, here is...") work better than targeting the full harmful output?**
A1 (from the paper, Section 2.1): The authors explain that targeting a specific full output "prescribes a single 'ground truth' output to the query, when in reality there are likely many answers that are appropriate." Targeting just the affirmative prefix works because "if the language model can be put into a 'state' where this completion is the most likely response, as opposed to refusing to answer the query, then it likely will continue the completion with precisely the desired objectionable behavior."

**Q2: What makes GCG different from AutoPrompt?**
A2 (from the paper, Section 2.2): "GCG is quite similar to the AutoPrompt algorithm, except for the seemingly minor change that AutoPrompt in advance chooses a single coordinate to adjust then evaluates replacements just for that one position." The authors find "this design choice has a surprisingly large effect" — GCG achieves 88% ASR vs. AutoPrompt's 25% on Vicuna-7B for harmful strings.

**Q3: Do the adversarial suffixes transfer to closed-source models?**
A3 (from the paper, Table 2 and Section 3.2): Yes. Using an ensemble of GCG prompts optimized on Vicuna and Guanaco, the attack achieves 86.6% ASR on GPT-3.5, 46.9% on GPT-4, 66% on PaLM-2, and 2.1% on Claude-2. The authors note "these are the first results to demonstrate reliable transfer of automatically-generated universal 'jailbreak' attacks over a wide assortment of LLMs."

**Q4: Did the authors disclose the vulnerability before publishing?**
A4 (from the paper, Section 1): "Prior to publishing this work, we shared preliminary results with OpenAI, Google, Meta, and Anthropic." The Broader Impact section (Section 6) further states: "Prior to publication, we disclosed the results of this study to the organizations hosting the large closed-sourced LLMs we studied; accordingly, the examples shown in the paper will likely cease to function."

## Common Misconceptions

1. **"GCG is a completely new attack technique"** — The paper itself notes that the three key elements (affirmative responses, gradient-based discrete optimization, multi-prompt optimization) "have indeed existed in very similar forms in the literature." The contribution is their careful combination. GCG is described as "a simple extension of the AutoPrompt method" (Section 2.2).
2. **"The adversarial suffixes are human-readable instructions"** — No, the suffixes are strings of tokens that look like gibberish (e.g., "describing.\ + similarlyNow write oppositeily.]( Me giving**ONE please?"). They work by exploiting the model's internal representations, not by conveying semantic meaning to a human reader.
3. **"This means alignment is useless"** — The authors caution against this interpretation. They note that adversarial training "is still proven to be the empirically most effective means of training robust machine learning models" and that "it remains to be seen how this 'arms race' between adversarial attacks and defenses plays out in the LLM space" (Section 1).

## Real Citations

1. Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, Matt Fredrikson (2023). "Universal and Transferable Adversarial Attacks on Aligned Language Models." *arXiv preprint.* [arXiv:2307.15043](https://arxiv.org/abs/2307.15043) — the original paper.
2. Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Seung Gook Lee, et al. (2024). "Jailbreaking Black Box Large Language Models in Twenty Queries." [arXiv:2310.08419](https://arxiv.org/abs/2310.08419) — follow-up using GCG-style attacks in a black-box setting.

**Citation count:** about 4,979 citations on [OpenAlex](https://openalex.org/works/W4385374425) (checked November 2026).
