---
name: research-paper-series
description: Use when adding, fixing or reviewing a paper folder (NNN_slug/) in nadeemcite/llm-research-papers ("100 LLM Era Papers", Season 2), or before ANY commit/push to that repo. Covers the README/TALK/CODE_ARCHITECTURE/notebook templates, the Kaggle run, the Backprop Daily Video line, release-day updates, and the mandatory pre-commit gate (scripts/check_paper.py + manual checklist).
---

# Research-paper series (Season 2)

**Purpose:** every paper in https://github.com/nadeemcite/llm-research-papers must look the same, be
factually right against the arXiv paper, and be honest about what its notebook actually shows.
This skill gives you the templates, the workflow, and a gate you must pass before every commit.

Site: GitHub Pages + Jekyll + kramdown + Just the Docs (`remote_theme: just-the-docs/just-the-docs@v0.12.0`),
served at https://nadeemcite.github.io/llm-research-papers/. Kaggle user: `nadymsazad`. GitHub user: `nadeemcite`.
YouTube channel: [Backprop Daily](https://www.youtube.com/@BackpropDaily).

## Files in this skill

| Path | Use it for |
|---|---|
| `templates/README.md`, `templates/TALK.md`, `templates/CODE_ARCHITECTURE.md` | Start every new paper from these (placeholders in `{{...}}`) |
| `templates/solution_template.ipynb` | Notebook skeleton: sections, quiet downloads, inline plots, required summary cell |
| `templates/front-matter-snippets.md` | Exact nav front matter (written by `scripts/build_readme.py`) |
| `checklists/pre-commit-checklist.md` | The manual half of the gate: the 7 checks |
| `checklists/release-day-checklist.md` | D-1 / D / D+1 steps for media.tsv, readme_media.py and the Sheet |
| `checklists/common-mistakes.md` | Bad vs good examples from the Season 1 audits (read once before writing) |
| `scripts/check_paper.py` | The automated half of the gate (stdlib only) |
| `scripts/README.md` | How to run the checker, flags, exit codes |
| `examples/` | Golden README.md and TALK.md (032 YOLO, after its fixes) |

## Repo map (what you touch, what is generated)

- `NNN_slug/` holds exactly: `README.md`, `TALK.md`, `CODE_ARCHITECTURE.md`, `solution.ipynb`. Nothing else is public by intent.
- `scripts/papers.tsv` (num, short, year, domain, title, folder, kaggle): source of truth for the root README,
  family tree, banner SVG and **all nav front matter**. `python scripts/build_readme.py` (needs matplotlib) regenerates
  them. Never edit the root `README.md` by hand.
- `scripts/media.tsv` (num, short, banner, status, shorts_url, release_date): source of truth for the banner
  line and the `- **Video:**` line. `python scripts/readme_media.py NNN` rewrites only those two lines.
- `assets/banners/NNN.jpg`: banner image (only papers with a media.tsv row show it).
- `_config.yml`, `_includes/`: site config. Don't add per-file `defaults:` (slows Pages builds to minutes).
- `.gitignore` keeps `/.ignore/`, `.kaggle_output/`, `kernel-metadata.json`, `.env` out. Keep it that way.

## Workflow: adding a new paper

1. **Sync.** `git pull` (history may be force-pushed by the owner; if pull fails, re-clone, never force anything yourself).
2. **Read the paper.** arXiv abs + PDF (`pdftotext`). Write down: title, all authors in order, affiliations as
   printed (incl. "work done while at"), first-version year, venue *as named then* (NIPS before 2018), headline
   numbers with benchmark names, and section numbers you will cite.
3. **Register it.** Add the row to `scripts/papers.tsv` (`kaggle` = the real kernel slug you will use), add
   lineage edges to `EDGES` in `build_readme.py` if relevant, run `python scripts/build_readme.py`.
4. **Write the notebook** from `templates/solution_template.ipynb`: one hyperparameter cell, quiet downloads,
   inline plots (`plt.show()`), every number the docs will quote printed as `Name: value`, a final
   `## Summary` cell with `### What the run above shows` built from the real outputs.
5. **Run it on Kaggle** (latest `kaggle` CLI, auth by `KAGGLE_API_TOKEN` env var only): `kaggle kernels push`,
   public, internet on, GPU only if needed. Wait for `kaggle kernels status` = COMPLETE. Save the executed
   notebook with outputs as `solution.ipynb` (same source as the kernel; ideally the Kaggle run's outputs).
6. **Look at every output and image.** If the result failed, fix the model or say so. Never write success first.
7. **Write the docs** from the templates: README (section order is fixed), CODE_ARCHITECTURE (values copied
   from the code cell), TALK (Q&A answered from the paper; press only with real, exact-page links).
8. **Media state.** New paper = state 1 (no media.tsv row: no banner, no Video line). When the Short exists,
   add the media.tsv row (`private`, release_date) + banner, run `python scripts/readme_media.py NNN`.
9. **Gate** (below). Fix every FAIL; fix or justify every WARN; tick the manual checklist.
10. **Commit and push** as `nadeemcite`, only the paper folder + the generated/registry files you meant to change.
11. **After push:** `check_paper.py ... --pages` (paper page, TALK page, banner = 200; no stray tables).
12. **Update the Sheet** "Video Release Order" (Build Status) when the paper is done.

## The pre-commit gate (mandatory)

**Do not commit until `scripts/check_paper.py NNN` passes and the manual checklist is ticked.**

```bash
python /path/to/research-paper-series/scripts/check_paper.py <repo_root> NNN --links          # always
python /path/to/research-paper-series/scripts/check_paper.py <repo_root> NNN --links --kaggle # when KAGGLE_API_TOKEN is set
python /path/to/research-paper-series/scripts/check_paper.py <repo_root> NNN --pages          # after push
```

- `RESULT: PASS` (exit 0) is required. Exit 1 = at least one FAIL. `--strict` makes WARN fatal.
- The script can't judge truth. `checklists/pre-commit-checklist.md` (7 sections: README, TALK, Links,
  Notebook honesty, Notebook format, Kaggle, Git & Pages) must be fully ticked too.
- If a fix touches the notebook, it goes to GitHub AND Kaggle, then the gate runs again.

## Hard rules

**Template (README.md), in this order**
1. Front matter (`title: "NNN Short"`, `nav_order: NNN+1`, `permalink: "/folder/"`), then `# Title`.
2. Banner `![Short banner](../assets/banners/NNN.jpg)` directly under the title, only when media.tsv has a private/public row.
3. Exactly one Kaggle badge: `[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/<slug>)`.
4. Bullets `- **Paper:**` (arXiv abs link), `- **Authors:**`, `- **Lab, year:**`, `- **Video:**` (states 2/3 only).
5. `## Summary`, `## What Problem Does It Solve`, `## Core Idea`, `## Key Method Details` (may hold
   `### Simplifications in This Notebook`; must hold `### What the Notebook Result Shows`), `## Influence`, `---`, `## Resources` table.

**Video line (3 states, only via media.tsv + `python scripts/readme_media.py NNN`)**
1. Paper created: no Video line, no banner (no media.tsv row, or `status=none`).
2. Private: `- **Video:** YouTube Short coming on 19 Oct 2026 on [Backprop Daily](https://www.youtube.com/@BackpropDaily)` (date = `release_date` as `D Mon YYYY`).
3. Public: `- **Video:** [Watch the YouTube Short](https://youtube.com/shorts/<id>) on [Backprop Daily](https://www.youtube.com/@BackpropDaily)`.
Never put a private video URL anywhere in the repo.

**Writing**
- Easy English: gloss jargon on first use; paragraphs under ~120 words; one analogy in "What Problem".
- Kramdown: write every `|` outside a table as `\|`, even inside inline code. Every paper .md has nav front matter.
- Facts match the arXiv paper: authors, affiliations, year (first release), venue as named then, every number.
- Citation counts only with source + check date. Awards and press only with a working link to the exact page.

**TALK.md**
- No fabricated or unsourced named-person quotes; every quote traceable to the paper (with section) or a linked real page.
- Q&A heading "Q&A (answers from the paper)"; never "interview"; ask about "the authors", not "you".
- Match the paper's hedging; don't credit later work to this paper (e.g. the Nature 2015 DQN target network).

**Notebook**
- Saved outputs support every claim in README, TALK, CODE_ARCHITECTURE and notebook text; failures are reported as failures.
- Same hyperparameters/epochs everywhere; seed/timing claims only with a saved output.
- Markdown cells keep line breaks; no progress/download spam; no `kaggle_validator` cell; no non-interactive
  matplotlib backend; at least one displayed plot; file well under 1 MB.

**Kaggle**
- `nadymsazad/<slug>`; slug identical in the README badge and papers.tsv. Source identical to `solution.ipynb`.
- COMPLETE, public, internet on, GPU only when needed. GitHub outputs = Kaggle run, or never contradict it.
- Auth only via `KAGGLE_API_TOKEN` with the latest CLI. Never kaggle.json, never print or commit a token.

**Git**
- Commit and push only as `nadeemcite`. No force-push, history rewrite or `--no-verify` without Nadeem's explicit OK.
- Never commit secrets, `.ignore/`, `.kaggle_output/`, `kernel-metadata.json`, `.env`, scratch files. Real timestamps only.

**Pages & release**
- After push: paper page, TALK page and banner return 200; formulas render without stray tables.
- One Short goes public per day around 6:30 PM IST (Sheet "Video Release Order"); the paper must pass the gate
  before its release day. On release day follow `checklists/release-day-checklist.md`.

## When something can't be verified

Bot-blocked sites (403/429, nbviewer, Kaggle badge image, Google Scholar) are *unverified*, not *working*:
open them in a real browser or replace the source. If you can't find a real source for a claim, delete the claim.
