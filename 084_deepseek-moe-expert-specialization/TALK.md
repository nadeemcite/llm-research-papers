---
title: "DeepSeekMoE: Talk & Q&A"
parent: "084 DeepSeekMoE"
nav_order: 2
---
# TALK — DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models (DeepSeekMoE)

## Talks, Slides and Press Coverage

1. **DeepSeek-AI Blog — "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models" (11 January 2024):** The official announcement post describing the DeepSeekMoE architecture, the 2B validation experiments, the 16B model release, and benchmark results. ([link](https://api-docs.deepseek.com/news/deepseek-moe))
2. **Hugging Face Papers — "DeepSeekMoE" (January 2024):** The paper's page on Hugging Face Daily Papers, showing community engagement. ([link](https://huggingface.co/papers/2401.06066))
3. **GitHub — DeepSeek-AI/DeepSeek-MoE:** The official repository releasing the 16B model checkpoint and inference code under a permissive license. ([link](https://github.com/deepseek-ai/DeepSeek-MoE))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2401.06066).

**Q1: What are the two main strategies of DeepSeekMoE?**
A1 (from the paper, Section 3): "(1) Fine-Grained Expert Segmentation: while maintaining the number of parameters constant, we segment the experts into a finer grain by splitting the FFN intermediate hidden dimension." And "(2) Shared Expert Isolation: we isolate certain experts to serve as shared experts that are always activated, aiming at capturing and consolidating common knowledge across varying contexts."

**Q2: How much better is DeepSeekMoE than GShard at the 2B scale?**
A2 (from the paper, Table 1): DeepSeekMoE 2B achieves Pile loss 1.808 vs GShard's 1.867, HellaSwag accuracy 54.8% vs 50.5%, and TriviaQA EM 16.6% vs 10.2% — all with the same total and activated parameters.

**Q3: Does DeepSeekMoE match the dense upper bound?**
A3 (from the paper, Section 4.3): "DeepSeekMoE nearly approaches the performance of a dense model with 16 times FFN parameters, which sets the strict upper bound of MoE models in terms of the model capacity." On Pile, both achieve loss 1.808.

**Q4: What happens when the shared expert is removed?**
A4 (from the paper, Section 4.5): "we disable it and activate one more routed expert. The evaluation on Pile shows a significant increase in the Pile loss, rising from 1.808 to 2.414, even though we maintain the same computational cost." This confirms the shared expert captures "fundamental and essential knowledge not shared with routed experts."

**Q5: How does the 16B model compare to LLaMA2 7B?**
A5 (from the paper, Table 4): DeepSeekMoE 16B achieves HellaSwag 77.1% (vs 75.6%), HumanEval Pass@1 26.8% (vs 14.6%), and GSM8K 18.8% (vs 15.5%), with only 39.6% of the compute.

**Q6: What is the combinatorial advantage of fine-grained segmentation?**
A6 (from the paper, Section 3.1): With N=16 experts and top-2, there are C(16,2)=120 possible combinations. Splitting each into 4 (m=4) gives 64 experts with top-8, yielding C(64,8)≈4,426,165,368 potential combinations — "The surge in combinatorial flexibility enhances the potential for achieving more accurate and targeted knowledge acquisition."

## Common Misconceptions

1. **"DeepSeekMoE has more parameters than GShard, so of course it's better"** — No. DeepSeekMoE and GShard have the same total expert parameters and the same activated expert parameters. The improvement comes from how those parameters are organized (more, smaller experts + shared experts), not from having more of them (Table 1).

2. **"The shared expert is just another routed expert that happens to always be selected"** — No. The shared expert is not routed at all — it is deterministically activated for every token, regardless of the router output. The paper shows it captures knowledge "not shared with routed experts, making it irreplaceable by routed ones" (Section 4.5). Disabling it causes loss to jump from 1.808 to 2.414.

3. **"Fine-grained segmentation just means more experts"** — It specifically means splitting each expert into m smaller ones by reducing the FFN intermediate dimension to 1/m, while proportionally increasing the number of activated experts to mK. The total parameters and compute stay constant. The key benefit is combinatorial flexibility, not raw capacity.

## Real Citations

1. Dai, D., Deng, C., Zhao, C., Xu, R.X., Gao, H., Chen, D., et al. (2024). "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models." [arXiv:2401.06066](https://arxiv.org/abs/2401.06066) — the original paper.
2. Lepikhin, D., et al. (2020). "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding." [arXiv:2006.16668](https://arxiv.org/abs/2006.16668) — the GShard architecture that DeepSeekMoE improves upon (conventional top-2 routing).
3. Fedus, W., Zoph, B., & Shazeer, N. (2021). "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." [arXiv:2101.03961](https://arxiv.org/abs/2101.03961) — top-1 routing MoE baseline compared in the paper.
4. Shazeer, N., et al. (2017). "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." *ICLR 2017.* [arXiv:1701.06538](https://arxiv.org/abs/1701.06538) — introduced top-K softmax gating and load balance loss.
5. Rajbhandari, S., et al. (2022). "DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale." [arXiv:2201.05596](https://arxiv.org/abs/2201.05596) — credited by the paper as the engineering prototype of shared expert isolation (Section 3.2).

**Citation count:** about 1,595 citations ([OpenAlex](https://openalex.org/W4402671950), checked November 2026).
