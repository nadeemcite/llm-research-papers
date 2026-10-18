# Pre-commit checklist (paper NNN)

Do not commit until **`python scripts/check_paper.py <repo_root> NNN --links` prints `RESULT: PASS`** (and
`--kaggle` too when `KAGGLE_API_TOKEN` is set) **and every box below is ticked**. Every WARN must be
fixed or consciously accepted (write why in the commit message). The script cannot judge facts or honesty;
these boxes are where you do.

## 1. README
- [ ] `python scripts/build_readme.py` was run after adding/changing the `scripts/papers.tsv` row (front matter, root README, family tree).
- [ ] Order: front matter, `# Title`, banner (only if media.tsv row is private/public), exactly one Kaggle badge, bullets Paper / Authors / Lab, year / Video, then Summary, What Problem Does It Solve, Core Idea, Key Method Details, Influence, `---`, Resources.
- [ ] Banner and Video line were written by `python scripts/readme_media.py NNN`, not by hand; they match `scripts/media.tsv`.
- [ ] Kaggle badge slug = the real kernel slug = the `kaggle` column in papers.tsv (the root README uses papers.tsv).
- [ ] Easy English: each piece of jargon is glossed the first time (BLEU, mAP, regression, policy, ...); no paragraph over ~120 words.
- [ ] Facts checked against the arXiv PDF, not memory: authors (all, in order), affiliations as printed (with exceptions), year = first public release, venue named as it was then (NIPS before 2018), every number and benchmark name.
- [ ] No claim stronger than the paper ("one of the first", not "the first", unless the paper or a source says so). Lineage claims are hedged.
- [ ] Citation counts carry a source and a check date ("about 53,000, OpenAlex, checked October 2026"), or are removed.
- [ ] Resources table describes the notebook accurately (not "from-scratch" if pretrained, not "PyTorch" if NumPy) and TALK as "Q&A (answers from the paper)".

## 2. TALK.md
- [ ] No fabricated or unsourced named-person quotes. Every quote is word-for-word in the paper (with section) or on a linked real page.
- [ ] Q&A heading says the answers come from the paper; nothing is called an "interview"; questions say "the authors", not "you".
- [ ] Retellings are labelled ("as reported by [MIT Technology Review](link), a retelling, not a direct quote").
- [ ] Hedging matches the paper ("may allow", "we suspect", "hypothesized").
- [ ] Nothing from later work is credited to this paper (e.g. the 2015 Nature DQN target network is not in the 2013 paper). Section numbers exist.
- [ ] Press items: real outlet, real live title, date, link to the exact page that supports the claim. No item without a link; no homepage links.
- [ ] Misconceptions are real wrong beliefs (not true facts labelled as misconceptions) and their heading matches their body.

## 3. Links
- [ ] `--links` run: every external link returns 200 with GET + a browser User-Agent. 403/429 WARNs opened by hand in a browser.
- [ ] nbviewer links: the GitHub blob was checked (nbviewer blocks bots). Kaggle checked with GET (HEAD gives 404).
- [ ] Each link supports the sentence it sits in (title and claim match the live page). Moved pages updated to their redirect target.

## 4. Notebook honesty
- [ ] Every claim in README, TALK, CODE_ARCHITECTURE and notebook markdown is visible in a saved output or saved image. A failed run is described as failed.
- [ ] README has `### What the Notebook Result Shows` and the notebook has `### What the run above shows`, both with the real numbers.
- [ ] Hyperparameters, epochs, layer counts, parameter count, function names agree across code, notebook text, README and CODE_ARCHITECTURE.
- [ ] "On every seed", "across 3 seeds", "trains in a minute" etc. are backed by printed output, or reworded "in earlier runs (not shown)", or removed.
- [ ] Plot descriptions match the pictures (look at every PNG), including visible mistakes.

## 5. Notebook format
- [ ] Markdown cells keep their line breaks (headings, lists and tables render on GitHub AND Kaggle).
- [ ] No download / tqdm / progress-bar spam (on GitHub or in the Kaggle log); no stray `<Figure ... with 0 Axes>`.
- [ ] No `kaggle_validator` / `__notebook__.ipynb` copy cell; no `/kaggle/working` paths.
- [ ] No non-interactive matplotlib backend; plots display inline.
- [ ] File well under 1 MB (no giant inline GIF/HTML animation; shrink frames, dpi or palette).

## 6. Kaggle
- [ ] Kernel `nadymsazad/<slug>` pushed with the latest `kaggle` CLI using the `KAGGLE_API_TOKEN` env var (never kaggle.json, never echo the token).
- [ ] `kaggle kernels status` = COMPLETE; kernel public; internet on; GPU only if the notebook needs it.
- [ ] `kaggle kernels pull -m` source is identical to GitHub `solution.ipynb` cell for cell.
- [ ] GitHub outputs are the Kaggle run's outputs, or at least every quoted number holds for both runs (give a range otherwise).

## 7. Git & Pages
- [ ] Committing as `nadeemcite`; real timestamps (no `--date` / `GIT_*_DATE` tricks); no force-push without Nadeem's OK.
- [ ] `git status` / `git diff --cached --stat` reviewed: no `.ignore/`, `.kaggle_output/`, `kernel-metadata.json`, `.env`, tokens, scratch files.
- [ ] After push (~1 min): `check_paper.py ... --pages` passes: paper page, TALK page, CODE_ARCHITECTURE page and banner return 200, no stray tables, formulas readable, page shows the new text.
