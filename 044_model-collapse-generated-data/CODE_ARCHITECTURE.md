---
title: "Model Collapse: Code architecture"
parent: "044 Model Collapse"
nav_order: 1
---
# Code Architecture — Model Collapse Notebook

## Overview

The notebook demonstrates model collapse on a 1D 3-component Gaussian mixture with two experiments. Experiment 1 fits a single Gaussian (functional + statistical approximation error) recursively for 30 generations; std drops from 4.24 to 3.00 and Wasserstein-1 distance grows from 0.43 to 1.27. Experiment 2 fits a 3-component GMM (statistical error only) for 30 generations; Wasserstein grows from 0.28 to 0.88. Each generation uses 200 samples.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants:
- `SEED = 42` — reproducibility
- `N_ORIGINAL = 10_000` — large sample from the true distribution (ground truth for Wasserstein)
- `N_GENERATIONS = 30` — recursive fitting-sampling cycles
- `N_SAMPLES_PER_GEN = 200` — small sample to amplify statistical approximation error
- `ORIGINAL_MEANS = [-5, 0, 5]`, `ORIGINAL_STDS = [1, 1, 1]`, `ORIGINAL_WEIGHTS = [1/3, 1/3, 1/3]`

### 2. Original Distribution

**Function: `sample_original(n)`** — draws n samples from the true 3-component Gaussian mixture by first choosing a component by weights, then sampling from that Gaussian. Returns a 1D array of shape `(n,)`. Plots a histogram of 10,000 original samples.

### 3. Experiment 1: Single Gaussian (Functional + Statistical Error)

**Function: `simulate_single_gaussian_collapse(data, n_generations, n_samples_per_gen)`** — at each generation, estimates mean and std from the current data, then samples `n_samples_per_gen` new points from the fitted Gaussian. Returns `all_data` (list of arrays, each shape `(200,)`) and `metrics` dict.

### 4. Experiment 2: GMM with Correct Components (Statistical Error Only)

**Function: `simulate_gmm_collapse(data, n_generations, n_samples_per_gen)`** — at each generation, fits a `GaussianMixture(n_components=3)` to the current data, then samples `n_samples_per_gen` new points. Same return structure.

### 5. Distribution Evolution Histograms

2-row grid: single Gaussian (top) and GMM (bottom), at generations 0, 5, 10, 20, 25, 30. Shows the three modes merging in the single Gaussian case.

### 6. Diversity Loss Metrics

Two line plots: standard deviation and Wasserstein-1 distance over generations, both experiments overlaid with the original std as a reference line.

### 7. Final Evaluation

Prints std, Wasserstein, and mean for generation 0 and generation 30 for both experiments, plus min/max values.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `sample_original(n)` | Draw n samples from the true 3-component Gaussian mixture |
| `simulate_single_gaussian_collapse(data, n_gen, n_per_gen)` | Recursive single-Gaussian fit-sample loop with metrics |
| `simulate_gmm_collapse(data, n_gen, n_per_gen)` | Recursive 3-component GMM fit-sample loop with metrics |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model type | GMM, VAE, OPT-125m LLM | Single Gaussian + 3-component GMM |
| Data | 2D GMM, MNIST, wikitext2 | 1D 3-component Gaussian mixture |
| Generations | Up to 2000 (GMM) | 30 |
| Sample size per gen | 1000 (GMM) | 200 |
| Functional approx. error | Present (VAE, LLM) | Present (single Gaussian) + absent (GMM) |
| Metrics | L2 distance, perplexity | Std, Wasserstein-1 distance |
