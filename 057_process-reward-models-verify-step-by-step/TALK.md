---
title: "Process Reward Models: Talk & Q&A"
parent: "057 Process Reward Models"
nav_order: 2
---
# TALK — Let's Verify Step by Step (Process Reward Models)

## Talks, Slides and Press Coverage

1. **OpenAI PRM800K Repository (OpenAI, GitHub):** OpenAI released the PRM800K dataset — 800,000 step-level human feedback labels — alongside the paper on GitHub, with documentation on data collection, labeling guidelines, and usage. ([repository](https://github.com/openai/prm800k))

No conference presentation or other press coverage could be verified from primary sources at the time of writing. The paper was released on arXiv and was not formally published at a conference venue.

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2305.20050).

**Q1: What is the difference between outcome supervision and process supervision?**
A1 (from the paper, Abstract): Outcome supervision "provides feedback for a final result," while process supervision "provides feedback for each intermediate reasoning step." The paper finds that "process supervision significantly outperforms outcome supervision for training models to solve problems from the challenging MATH dataset."

**Q2: Why can outcome supervision be misleading?**
A2 (from the paper, Section 2.5): "We note the automatic grading used to determine ORM targets is not perfectly reliable: false positives — solutions that reach the correct answer with incorrect reasoning — will be misgraded." This means the ORM can accidentally reward solutions that get the right answer through wrong reasoning.

**Q3: How is the PRM score for a full solution computed?**
A3 (from the paper, Section 2.6 and Appendix F.2): The PRM predicts correctness at the last token of each step. The solution-level score is defined as "the probability that every step is correct," computed as the product of per-step correctness probabilities. Appendix F.2 notes that "the best performing strategy is to take the product of step-level scores and to consider the neutrals as positives, but the difference in performance between all strategies is minor."

**Q4: How does the PRM compare to the ORM and majority voting?**
A4 (from the paper, Figure 3): At best-of-1860, the ORM solves 72.4% of test problems, the PRM solves 78.2%, and majority voting solves 69.6%. The PRM outperforms both.

**Q5: What is PRM800K and why was it released?**
A5 (from the paper, Abstract and Section 8): PRM800K is "the complete dataset of 800,000 step-level human feedback labels used to train our best reward model." The authors release it with "the hope that removing this significant barrier to entry will catalyze related research on the alignment of large language models."

**Q6: What role does active learning play?**
A6 (from the paper, Section 4.2): A selector model (PRM_selector) trained on 1 sample per problem scores 1000 samples. "80% are the most convincing wrong-answer samples, and 20% are the most convincing samples that remain." This focuses human labeling effort on solutions where the PRM is likely mistaken, improving data efficiency.

**Q7: Does the PRM generalize to out-of-distribution problems?**
A7 (from the paper, Section 5, Table 1): On 224 held-out STEM questions (AP Physics, AP Calculus, AP Chemistry, AMC10/12), the PRM achieves 72.9% (best-of-100), compared to 63.8% for the ORM and 61.3% for majority voting, demonstrating out-of-distribution generalization.

**Q8: How does this paper differ from Uesato et al. (2022)?**
A8 (from the paper, Section 7.1): Uesato et al. compared outcome and process supervision on grade-school math and "found that both methods led to similar final-answer error rates." The authors note "three main differences: we use a more capable base model, we use significantly more human feedback, and we train and test on the more challenging MATH dataset."

**Q9: What are the alignment benefits of process supervision?**
A9 (from the paper, Section 6.2): "Process supervision is more likely to produce interpretable reasoning, since it encourages models to follow a process endorsed by humans. Process supervision is also inherently safer: it directly rewards alignment with human intent."

## Common Misconceptions

1. **"The PRM is used to train the generator via RL."** — No. The paper explicitly scopes this out (Section 2.1): "We do not attempt to improve the generator with reinforcement learning (RL)." The PRM is trained and evaluated only by its ability to perform best-of-N reranking over generator samples. Using the PRM for RL training of the generator is mentioned as "a natural next step" but "intentionally not the focus of this work."

2. **"Process supervision means humans check the final answer step by step."** — Not exactly. Human labelers assign each step a label of positive, negative, or neutral (Section 2.4). A neutral label "indicates ambiguity" — for example, a step that is "subtly misleading" or "a poor suggestion that is technically still valid." The neutral label lets annotators defer ambiguity decisions to test time, where neutrals can be treated as either positive or negative.

3. **"The 78% solve rate is the generator's accuracy."** — No. The 78.2% figure is the best-of-1860 reranking accuracy — the PRM selects the best solution from 1,860 candidates per problem. The generator itself has a much lower single-digit solve rate on many MATH problems, as noted in Section 6.3: "the PRM regularly surfaces correct solutions to MATH problems that have a low single-digit percentage solve-rate under the generator."

4. **"PRM800K contains 800K problems."** — No. PRM800K contains 800,000 step-level human feedback labels across solutions to MATH problems. The number of unique problems is much smaller; the 800K refers to individual step-level annotations.

## Real Citations

1. Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, Karl Cobbe (2023). "Let's Verify Step by Step." [arXiv:2305.20050](https://arxiv.org/abs/2305.20050) — the original paper.
2. Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, Irina Higgins (2022). "Solving math word problems with process- and outcome-based feedback." [arXiv:2211.14275](https://arxiv.org/abs/2211.14275) — the prior comparison of outcome and process supervision on grade-school math that this paper builds on.
3. Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, John Schulman (2021). "Training Verifiers to Solve Math Word Problems." [arXiv:2110.14168](https://arxiv.org/abs/2110.14168) — the ORM training methodology (GSM8K + verifiers) that this paper extends.
4. Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt (2021). "Measuring Mathematical Problem Solving With the MATH Dataset." [arXiv:2103.03874](https://arxiv.org/abs/2103.03874) — the MATH dataset used for training and evaluation.

**Citation count:** about 2,154 citations on [OpenAlex](https://openalex.org/W4379089709) (checked November 2026).
