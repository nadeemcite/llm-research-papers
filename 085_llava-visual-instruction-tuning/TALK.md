---
title: "LLaVA: Talk & Q&A"
parent: "085 LLaVA"
nav_order: 2
---
# TALK — Visual Instruction Tuning (LLaVA)

## Talks, Slides and Press Coverage

1. **NeurIPS 2023 Oral:** LLaVA was accepted as an Oral at NeurIPS 2023. The project page with demo, data, and model links is at [LLaVA Project Page](https://llava-vl.github.io/).

2. **LLaVA GitHub repository (haotian-liu/LLaVA):** The official codebase, model checkpoints, and GPT-4-generated visual instruction tuning data are released at [github.com/haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA). The README describes the NeurIPS 2023 acceptance and links to the paper.

3. **Microsoft Research — Open Source Visual Chatbot:** LLaVA was developed partly at Microsoft Research; the project is listed on the [MSR Visual Chat](https://github.com/haotian-liu/LLaVA) GitHub page alongside the team's other vision-language work.

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2304.08485).

**Q1: Why use a simple linear projector instead of a more sophisticated cross-attention mechanism?**
A1 (from the paper, Section 4.1): "Note that our simple projection scheme is lightweight, which allows us to iterate data centric experiments quickly. More sophisticated schemes to connect the image and language representations can also be considered, such as gated cross-attention in Flamingo and Q-former in BLIP-2. We leave exploring possibly more effective and sophisticated architecture designs for LLaVA as future work."

**Q2: How did the authors generate multimodal instruction-following data without human annotation?**
A2 (from the paper, Section 3): They used language-only GPT-4/ChatGPT as a teacher, feeding it symbolic representations of images (captions and bounding boxes) rather than the images themselves. They generated three types: conversation (58K), detailed description (23K), and complex reasoning (77K), totaling 158K samples from COCO images.

**Q3: Why is feature alignment (Stage 1) necessary before end-to-end fine-tuning?**
A3 (from the paper, Section 5.2, Table 8): Skipping pre-training and training on ScienceQA from scratch drops accuracy from 90.92% to 85.81% — a 5.11% absolute degradation. The authors state this "indicates the importance of our pre-training stage, in aligning multimodal features while preserving the vast pre-trained knowledge."

**Q4: What visual features work best — before or after CLIP's last Transformer layer?**
A4 (from the paper, Section 5.2, Table 8): Features before the last layer yield 90.92% vs 89.96% for last-layer features. The authors hypothesize "this is because CLIP's last layer features may focus more on global and abstract image properties compared to the layer before it, which can focus more on localized properties."

**Q5: How does LLaVA compare to BLIP-2 and OpenFlamingo on instruction following?**
A5 (from the paper, Section 5.1, Table 5): On LLaVA-Bench (In-the-Wild), LLaVA achieves 67.3% relative score vs BLIP-2 at 38.1% and OpenFlamingo at 19.1%. The authors note BLIP-2 and OpenFlamingo "focus on describing the image, instead of following the user instruction to answer in an appropriate manner."

## Common Misconceptions

1. **"LLaVA was the first multimodal large language model"** — It was not. Flamingo (Alayrac et al., 2022), BLIP-2 (Li et al., 2023), and KOSMOS-1 (Huang et al., 2023) preceded it. LLaVA's contribution is being the first to use *visual instruction tuning* — training on GPT-4-generated instruction-following data — not the first vision-language model.

2. **"LLaVA uses GPT-4V to generate training data"** — It uses text-only GPT-4, not the multimodal version. The key insight is that image content is encoded as text (captions and bounding boxes) and fed to text-only GPT-4, which generates instruction-following data without ever "seeing" the image.

3. **"The linear projector is the optimal architecture"** — The paper explicitly says it is a simple, lightweight choice for rapid experimentation and leaves more sophisticated designs (cross-attention, Q-former) as future work. LLaVA-1.5 later replaced the linear projector with a 2-layer MLP.

## Real Citations

1. Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee (2023). "Visual Instruction Tuning." *NeurIPS 2023.* [arXiv:2304.08485](https://arxiv.org/abs/2304.08485) — the original paper.
2. Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, Yong Jae Lee (2023). "Improved Baselines with Visual Instruction Tuning" (LLaVA-1.5). [arXiv:2310.03744](https://arxiv.org/abs/2310.03744) — follow-up that upgraded the projector to a 2-layer MLP and improved performance.

**Citation count:** about 4,300 citations on [OpenAlex](https://openalex.org/W4366330503) (checked November 2026).
