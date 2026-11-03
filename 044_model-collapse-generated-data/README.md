---
title: "044 Model Collapse"
nav_order: 45
permalink: "/044_model-collapse-generated-data/"
---
# The Curse of Recursion: Training on Generated Data Makes Models Forget (Model Collapse)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/model-collapse-generated-data)

- **Paper:** [The Curse of Recursion: Training on Generated Data Makes Models Forget](https://arxiv.org/abs/2305.17493)
- **Authors:** Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Yarin Gal, Nicolas Papernot, Ross Anderson
- **Lab, year:** University of Oxford, University of Cambridge, Imperial College London, University of Toronto and Vector Institute, 2023 (arXiv May 2023; published in Nature, July 2024)

## Summary

In May 2023, a team from Oxford, Cambridge, Imperial College, and Toronto showed that if generative models are trained recursively on data produced by previous generations of models, they suffer an irreversible degeneration they named **model collapse**. The tails (rare events) of the original data distribution disappear first, then the model entangles different modes and converges to a narrow distribution with very small variance — a point estimate that carries little resemblance to the original. The paper demonstrates this effect in Gaussian Mixture Models (a statistical clustering tool), Variational Autoencoders (a generative model), and fine-tuned language models (OPT-125m on wikitext2).

The core insight is that two sources of error compound over generations. **Statistical approximation error** — the primary cause — arises because finite samples inevitably miss rare events; once those events are absent from the training set, they can never reappear. **Functional approximation error** — the secondary cause — comes from the model family being imperfect (e.g. fitting a single Gaussian to a mixture of two). The paper proves that for a 1D Gaussian, the Wasserstein-2 distance (a measure of distributional distance) from the original grows as the sum of 1/M terms across generations, where M is the sample size — and only superlinearly increasing sample sizes can keep it bounded.

This notebook demonstrates model collapse at toy scale with two experiments: (1) a single Gaussian fit to a 3-mode mixture, where functional approximation error causes rapid collapse (std drops 4.24 to 3.00, Wasserstein grows 0.43 to 1.27 over 30 generations), and (2) a 3-component GMM with the correct number of components, where only statistical error remains (Wasserstein grows 0.28 to 0.88).

## What Problem Does It Solve

Imagine a school where every year, students learn not from original textbooks but from the notes written by last year's students. The first class reads the real textbooks and writes good notes. The next class reads only those notes and writes their own. After a few years, the notes are copies of copies of copies — details get dropped, rare topics vanish, and everyone ends up writing the same shallow summary. The original richness is gone forever.

That is what happens when AI models train on AI-generated content. Each generation of model produces data that is a slightly blurred version of its training data. The next model trains on that blurred data, blurs it further, and so on. Rare patterns — the "tails" of the distribution — are the first to vanish, because finite samples are unlikely to include them. After enough generations, the model produces a narrow, repetitive output with no diversity. The paper asks: what will happen to GPT-{n} as n grows, if LLMs contribute most of the text online?

## Core Idea

- **Model collapse is degenerative and irreversible:** Once information about the tails is lost, it cannot be recovered — the model's perception of reality narrows with each generation, even if each individual model is trained perfectly.
- **Statistical approximation error is the primary driver:** Finite sampling means rare events have probability 1-q of being missed at each generation (where q is their probability). Over generations, this compounds and the distribution converges to a delta function (a single point).
- **Functional approximation error accelerates collapse:** When the model family cannot represent the true distribution (e.g. a single Gaussian for a mixture), collapse is faster and more severe, as the paper shows with VAEs and LLMs.
- **Mixing in original data helps but does not fully prevent collapse:** The paper shows that even with 10% of original data preserved, perplexity still degrades over generations in the OPT-125m fine-tuning experiments.

## Key Method Details

- **Gaussian Mixture Model experiment:** A GMM fits two Gaussians; over 2000 generations of resampling, the distribution converges to a point estimate with very little variance (Section 5.1).
- **VAE experiment:** A VAE trained on MNIST generates data for the next generation; by generation 20, the generated digits are blurred mixtures with little resemblance to the original classes (Section 5.1).
- **Language model experiment:** OPT-125m fine-tuned on wikitext2 (a Wikipedia text dataset), then each generation fine-tuned on data generated by the previous. Perplexity (a measure of how well a model predicts text; lower is better) rises from about 20 to 28 over 9 generations with no original data preserved; from about 17 to 25 with 10% original data preserved.
- **Theoretical result (1D Gaussian):** The Wasserstein-2 distance from the original grows as (3/2)\u03c3\u00b2 \u03a3(1/M\u1d62), requiring superlinearly increasing sample sizes to remain bounded (Section 4.2).

### Simplifications in This Notebook

- 1D Gaussian mixture with 3 modes (means at -5, 0, 5) and 200 samples per generation, instead of the paper's 2D GMM, VAE on MNIST, or OPT-125m on wikitext2
- sklearn GaussianMixture and scipy.stats for fitting, instead of neural network-based models
- 30 generations with 200 samples each, instead of up to 2000 generations
- Two experiments: single Gaussian (functional + statistical error) and 3-component GMM (statistical error only)
- Diversity measured by standard deviation and Wasserstein-1 distance, instead of perplexity

### What the Notebook Result Shows

The single Gaussian experiment shows standard deviation shrinking from 4.24 (generation 0) to 3.00 (generation 30) and Wasserstein-1 distance growing from 0.43 to 1.27, as the three original modes merge into a single narrower peak. The 3-component GMM experiment (correct model family) shows a slower but still growing Wasserstein distance, from 0.28 to 0.88 over 30 generations. The histograms visually show the multi-modal distribution losing its structure. This demonstrates the paper's core finding: both functional and statistical approximation errors compound over generations, causing irreversible distributional collapse. These numbers are not comparable to the paper's OPT-125m perplexity results — they demonstrate the principle with 1D synthetic data and classical models.

## Influence

The paper was published in Nature in July 2024 (vol. 631, pp. 755-759), giving it significant visibility. It directly influenced discussions about AI-generated content polluting training data, sometimes called "AI inbreeding" or "model autophagy disorder." The concept has been cited in policy discussions about data provenance and the long-term sustainability of web-scraped training data. About 155 citations ([OpenAlex](https://openalex.org/W4378765257), checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/044_model-collapse-generated-data/solution.ipynb) | NumPy/sklearn implementation showing model collapse via single Gaussian and GMM fitting on a 1D Gaussian mixture across 30 generations (std drops 4.24 to 3.00, Wasserstein grows 0.43 to 1.27 for single Gaussian) |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, array shapes, and key design decisions |
| [TALK.md](TALK.md) | Press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
