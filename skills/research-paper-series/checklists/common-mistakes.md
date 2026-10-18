# Common mistakes (bad vs good)

Every class of mistake below was found in a real audit of Season 1. Paper details are only examples;
the rule is what matters. `[auto]` = `check_paper.py` catches it, `[manual]` = only a human/agent check does.

## A. Invented or misattributed material (TALK.md)

| # | Mistake | Bad | Good |
|---|---|---|---|
| A1 | Fabricated named-person quote [manual; checker WARNs on unsourced quotes] | Q1 answer in first person, "Jakob Uszkoreit, in various public talks" | A real quote from a linked article (e.g. WIRED 2024 quoting Uszkoreit), or an answer "from the paper, Section 1" |
| A2 | Paper text presented as an interview [auto] | `## Interview Q&A`, "Joseph Redmon: ..." (the abstract, verbatim) | `## Q&A (answers from the paper)`, "From the paper (Abstract): "..."" |
| A3 | Quote not in the paper or misquoted [manual] | "each grid cell only predicts one box" | "each grid cell only predicts two boxes and can only have one class" (Section 2.4, word for word) |
| A4 | Invented press item / title [auto: no link = FAIL; manual: title] | "MIT Technology Review (2017) called it ..." with no URL | Real article, live title, date, link to the exact page |
| A5 | Homepage-only link [auto WARN] | `[MIT TR](https://www.technologyreview.com/)` | `https://www.technologyreview.com/2018/02/21/145289/the-ganfather-...` |
| A6 | Dead link with an invented URL [auto with --links] | `wired.com/2015/02/google-ai-beats-atari-games/` (404) | `wired.com/2015/02/google-ai-plays-atari-like-pros/` (200, real title) |
| A7 | Link that doesn't support the claim [manual] | "Spinning Up provides DQN reference implementations" | "Spinning Up places DQN in its map of RL algorithms; it has no DQN implementation" |
| A8 | Later work credited to the paper [manual; auto WARN on later years in "from the paper" answers] | "(from the paper, Section 2.2): the target network ..." for the 2013 DQN | "(from the 2015 Nature follow-up, not this paper): ..." |
| A9 | Section numbers that don't exist [manual] | "(Section 2.1)" when the paper has no 2.1 | Open the PDF and cite the real section |
| A10 | Lost hedging [manual] | "sinusoids allow the model to extrapolate" | "they *may* allow it to extrapolate; the paper doesn't test it, and learned embeddings gave nearly identical results" |
| A11 | Analogy presented as origin story [manual] | "inspired by the world of counterfeit money" | "the paper explains it with its own analogy: counterfeiters vs the police" |
| A12 | True facts labelled as misconceptions [manual] | Misconception: "YOLOv1 has no anchors" | Misconception: "YOLOv1 uses anchor boxes." Then: it does not; anchors came in YOLOv2 |
| A13 | Unsourced "context" about people [manual] | "Goodfellow has expressed surprise ..." | Neutral, unattributed context, or a linked source |

## B. Facts (README and TALK)

| # | Mistake | Bad | Good |
|---|---|---|---|
| B1 | Venue renamed retroactively [auto] | "NeurIPS 2017" | "NIPS 2017 (now NeurIPS)" |
| B2 | Affiliation flattened [manual] | "(all at Google Brain)" | "Google Brain / Google Research (Gomez: University of Toronto, work done at Google Brain)" |
| B3 | Numbers from the wrong source [manual] | "Vocabulary: 3 million words" (the later released file) | "1 million most frequent words" (the paper) |
| B4 | Dataset list padded [manual] | "MNIST, TFD, CIFAR-10 and SVHN" | "MNIST, TFD and CIFAR-10" (as in the paper) |
| B5 | Ensemble number presented as single model [manual] | "reached 4.94% top-5 error" | "an ensemble reached 4.94%; the best single model 5.71%" |
| B6 | Overclaim the paper doesn't make [auto WARN on "the first ..."] | "the first real-time deep-learning detector" | "one of the first deep-learning detectors fast enough for live video" |
| B7 | Invented framing / lineage [manual] | "AlphaFold uses AlphaZero's planning" | Drop it, or "DeepMind's later science work" |
| B8 | Citation count without source/date, or contradictory [auto WARN] | "6,800+ (Semantic Scholar)" and "150,000+ (Google Scholar)" in the same file | "about 197,000 on Semantic Scholar (October 2026)" |
| B9 | Wrong algorithm detail [manual] | "selects moves with high visit counts" | "low visit count, high move probability and high value" |
| B10 | Self-contradiction across files [manual] | TALK Q5 mentions constituency parsing, Misconception 5 says "translation only" | One consistent statement in every file |

## C. Notebook honesty

| # | Mistake | Bad | Good |
|---|---|---|---|
| C1 | Success claimed for a failed run [manual; auto FAIL if no summary with numbers] | GAN collapsed (D_loss 0.0000, identical blobs) but "generates diverse digits" | Fix the run, or "the GAN collapsed: every z gives the same blob" |
| C2 | Metric is 0 / chance but text reads as success [manual] | "Mean IoU 0.0000" + "Detection visualization complete" | Train until it works, report "mAP@0.5 about 0.90", or say it failed |
| C3 | Hyperparameters differ across files [auto WARN] | CODE_ARCHITECTURE `EPOCHS = 25`, notebook `EPOCHS = 3`, README "20-30 epochs" | One value, copied from the code cell |
| C4 | Unsaved experiment claimed [auto WARN] | "collapses on every seed we tested", "trains in about a minute" | Print it in the notebook, or "in earlier runs (not shown)" |
| C5 | Plot description doesn't match picture [manual] | "a clean diagonal" | "a near-diagonal pattern; the final EOS step attends elsewhere" |
| C6 | Wrong physics/interpretation in printed text [manual] | "prefer push left when the pole tilts right" | "push right when it leans right: the cart moves back under the pole" |
| C7 | Paper's features credited to the notebook or vice versa [manual] | Summary table: "Paper: target network ... Identical" | "Paper vs this notebook" table that separates 2013, 2015 and our additions |
| C8 | Wrong baseline [manual] | "(vs random baseline ~10)" while cell 8 measured 55 | Measure the baseline and quote that number |
| C9 | GitHub vs Kaggle numbers differ [auto WARN with --kaggle] | README "99.6%" (CPU run) while Kaggle shows 100% | Commit the Kaggle run's outputs, or "99.6–100%" |
| C10 | Boilerplate that is false for this notebook [auto WARN, partly] | "from-scratch PyTorch" for a NumPy notebook or a pretrained GPT-2 | Describe what the notebook really is |

## D. Notebook format

| # | Mistake | Bad | Good |
|---|---|---|---|
| D1 | Markdown line breaks lost [auto] | `## 3. HyperparametersKey parameters ...` (one line) | Heading, blank line, text; lists one item per line |
| D2 | Leftover validator cell [auto] | `# --- Auto-injected by kaggle_validator ...` | Delete it on GitHub AND Kaggle |
| D3 | Plots suppressed [auto] | `matplotlib.use('Agg')` + "Loss plot saved." | Inline backend + `plt.show()` |
| D4 | Progress spam [auto] | 5,206 tqdm stderr chunks from a CIFAR download | `contextlib.redirect_stderr(io.StringIO())` around the download |
| D5 | Giant inline animation [auto] | 3.4 MB HTML animation, 1.18 MB GIF | Fewer frames, dpi 50, small palette; notebook well under 1 MB |
| D6 | Stray empty figure [auto WARN] | `<Figure size 640x480 with 0 Axes>` | Create the figure once, then `plt.show()` |
| D7 | Wrong comment [manual] | `# Show a few training images` drawing `test_ds` | `# Show a few test images` |

## E. Template, site and Kaggle

| # | Mistake | Bad | Good |
|---|---|---|---|
| E1 | Bare pipe [auto] | Pr(Class_i, a bare pipe, Object) written with a plain pipe in a paragraph or inline code: kramdown builds a table | Write the pipe as `\|`, even inside inline code |
| E2 | Video line edited by hand / stale [auto] | "coming on 18 Oct 2026" after the Short went public | media.tsv `status=public` + `shorts_url`, then `python scripts/readme_media.py NNN` |
| E3 | Kaggle slug mismatch [auto] | README badge `...-learning`, papers.tsv `...reinforcement` (404 on the root README) | Same real slug in both, then `scripts/build_readme.py` |
| E4 | Kaggle source differs from GitHub [auto with --kaggle] | Fix committed to GitHub only | `kaggle kernels push` the same file, wait for COMPLETE, `pull -m` and diff |
| E5 | Kaggle-only spam [auto WARN with --kaggle] | MNIST download bars in the Kaggle log only | Silence the download in the notebook |
| E6 | Legacy credentials [manual; auto FAIL on token patterns] | `~/.kaggle/kaggle.json`, `KAGGLE_KEY` in a cell | `KAGGLE_API_TOKEN` env var + latest CLI; never print it |
| E7 | Private files committed [auto] | `.ignore/` pipeline folder pushed (had to be scrubbed from history) | Keep it in `.gitignore`; review `git diff --cached --stat` |
| E8 | Long jargon paragraph [auto WARN] | Summary: 245 words, "explicit likelihood", "MCMC" | 2–3 paragraphs under 120 words, each term glossed |
