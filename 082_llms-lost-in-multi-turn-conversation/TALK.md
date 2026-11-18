---
title: "Multi-Turn Lost: Talk & Q&A"
parent: "082 Multi-Turn Lost"
nav_order: 2
---
# TALK — LLMs Get Lost In Multi-Turn Conversation (Multi-Turn Lost)

## Talks, Slides and Press Coverage

1. **Microsoft Research — "LLMs Get Lost In Multi-Turn Conversation" (6 Jun 2025):** Microsoft Research published a blog post about the paper on their official research blog, describing the sharding simulation framework and the finding that "all the top open- and closed-weight LLMs we test exhibit significantly lower performance in multi-turn conversations than single-turn, with an average drop of 39% across six generation tasks." ([link](https://www.microsoft.com/en-us/research/blog/llms-get-lost-in-multi-turn-conversation/))
2. **Hugging Face Papers — "LLMs Get Lost In Multi-Turn Conversation" (9 May 2025):** The paper was featured on Hugging Face Daily Papers, receiving significant community engagement. ([link](https://huggingface.co/papers/2505.06120))
3. **Datasets release:** The authors released the sharding framework and simulated conversation datasets at [Microsoft/lost_in_conversation](https://huggingface.co/datasets/Microsoft/lost_in_conversation) on Hugging Face, and code at [github.com/microsoft/lost_in_conversation](https://github.com/microsoft/lost_in_conversation).

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2505.06120).

**Q1: Why do LLMs perform worse in multi-turn conversations?**
A1 (from the paper, Section 6.2): The authors decompose the performance degradation into two components: "a minor loss in aptitude and a significant increase in unreliability." Model aptitude degrades by an average of 16% between full and sharded settings, while "unreliability skyrockets with an average increase of 112% (more than doubling)." The authors identify four root causes: premature answer attempts, answer bloat, over-adjustment based on last turn, and overly verbose responses (Appendix F).

**Q2: Do stronger models (like Gemini 2.5 Pro, GPT-4.1) handle multi-turn conversations better?**
A2 (from the paper, Section 6.1): "Surprisingly, more performant models (Claude 3.7 Sonnet, Gemini 2.5, GPT-4.1) get equally lost in conversation compared to smaller models (Llama3.1-8B-Instruct, Phi-4), with average degradations of 30-40%." This is partly because stronger models have higher absolute single-turn scores, giving more scope for degradation.

**Q3: Does lowering the temperature fix the unreliability problem?**
A3 (from the paper, Section 7.2, Table 3): No. "GPT-4o-mini does not see improvements in reliability as AT is decreased (in all user-temperature settings), and GPT-4o only sees minor improvements, on the order of 15-20%. Even when both the user and assistant temperatures are set to 0.0, there remains a large unreliability of around 30%." The authors explain: "one token difference in an early turn of a multi-turn conversation can lead to cascading deviations."

**Q4: Do reasoning models (o3, DeepSeek-R1) handle multi-turn better thanks to additional test-time compute?**
A4 (from the paper, Section 6.1): "Additional test-time compute (reasoning tokens) does not help models navigate multi-turn underspecification, as the two reasoning models included in the experiment (o3, Deepseek-R1) deteriorate in similar ways to non-reasoning models." The authors identify a root cause: "reasoning models tend to generate lengthier responses (on avg. 33% longer than non-reasoning LLMs)," which contain more assumptions that can derail the conversation.

**Q5: Can agent frameworks (like Recap or Snowball) fix the multi-turn problem?**
A5 (from the paper, Section 7.1, Table 2): Both Recap (adding a final recap turn) and Snowball (restating all previous shards at each turn) show "some level of success, with improvements over Sharded simulations, but the performance still lags behind Full or Concat." Snowball can "mitigate the Full-to-Sharded performance deterioration by 15-20%," but the authors argue "LLMs should natively support multi-turn interaction" rather than relying on agent frameworks.

**Q6: What can users do to get better results from LLMs in multi-turn conversations?**
A6 (from the paper, Section 7.4): The authors offer two recommendations: "If time allows, try again" — starting a new conversation may yield better outcomes than continuing one where the model is lost. And "Consolidate before retrying" — asking the LLM to consolidate everything told so far, then bringing that to a new conversation.

## Common Misconceptions

1. **"The performance drop is because the model forgets information from earlier turns"** — The paper shows the issue is not memory but *reliability*. The Concat simulation (all shards in one turn, bullet-pointed) achieves 95.1% of Full performance, proving the model has access to all information. The degradation in Sharded is due to the model making premature assumptions and over-relying on wrong previous answers, not information loss (Section 6.1).

2. **"Bigger or smarter models will naturally handle multi-turn better"** — The paper explicitly shows the opposite: "no matter how strong an LLM's single-turn performance is, we observe large performance degradations in the multi-turn setting" (Section 6.1). Even Gemini 2.5 Pro and GPT-4.1 show 30-40% degradation.

3. **"Setting temperature to 0 will make multi-turn reliable"** — The paper's temperature experiment (Table 3) shows that even at T=0 for both user and assistant, multi-turn unreliability remains around 30%. The cascading nature of conversation — where one token difference in an early turn leads to divergent trajectories — makes deterministic decoding insufficient (Section 7.2).

## Real Citations

1. Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, Jennifer Neville (2025). "LLMs Get Lost In Multi-Turn Conversation." *arXiv preprint.* [arXiv:2505.06120](https://arxiv.org/abs/2505.06120) — the original paper.
2. Herlihy, C., Neville, J., Schnabel, T., & Swaminathan, A. (2024). "On Overcoming Miscalibrated Conversational Priors in LLM-based Chatbots." [arXiv:2406.01633](https://arxiv.org/abs/2406.01633) — prior work on underspecification in LLM chat logs, cited extensively by this paper.
3. Liu, N. F., et al. (2024). "Lost in the Middle: How Language Models Use Long Contexts." *TACL.* [arXiv:2307.03172](https://arxiv.org/abs/2307.03172) — the earlier "Lost in the Middle" paper that this work extends from long-context to multi-turn settings.

**Citation count:** about 499 citations on [OpenAlex](https://openalex.org/W4417086184) (checked November 2026).
