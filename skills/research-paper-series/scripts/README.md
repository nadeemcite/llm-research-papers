# check_paper.py

Automated half of the pre-commit gate for one paper of nadeemcite/llm-research-papers. Python 3.8+,
standard library only (the `kaggle` CLI is used only for `--kaggle`).

```bash
python check_paper.py <repo_root> NNN [--links] [--pages] [--kaggle] [--strict] [--today YYYY-MM-DD] [--now-hhmm HH:MM] [--json]

# typical
python check_paper.py ~/llm-research-papers 032                    # offline, ~1 s
python check_paper.py ~/llm-research-papers 032 --links            # + GET every link, arXiv API facts (~20 s)
KAGGLE_API_TOKEN=... python check_paper.py ~/llm-research-papers 032 --links --kaggle
python check_paper.py ~/llm-research-papers 032 --pages            # after push: live GitHub Pages
```

Exit code: `0` no FAIL (PASS), `1` at least one FAIL (or a WARN with `--strict`), `2` bad repo root.
Output: one block per check group, each line `PASS` / `FAIL` / `WARN` / `INFO` / `SKIP`, then
`RESULT: PASS|FAIL  PASS n  FAIL n  WARN n  SKIP n`.

## What it checks

| Group | Offline (always) | With a flag |
|---|---|---|
| 0 Files & hygiene | papers.tsv row + folder; the 4 required files; extra files; token patterns (KGAT_, ghp_, sk-, hf_, AKIA, kaggle.json key, the live `KAGGLE_API_TOKEN` value); tracked `.ignore/`, `.kaggle_output/`, `kernel-metadata.json`, `kaggle.json`, `.env`; git user.name | |
| 1 README | front matter of all 3 .md files vs papers.tsv (as build_readme.py writes it); `# Title`; banner + Video line vs media.tsv (same logic as readme_media.py, plus `readme_media.py --check`); release-day-passed warning; exactly one Kaggle badge, user `nadymsazad`, slug = papers.tsv; bullets Paper/Authors/Lab, year/Video in order; year vs papers.tsv; H2 order; `---` before Resources; Resources rows + nbviewer link; bare `\|` outside tables (and unescaped `\|` in inline code in tables) in all 3 .md files; NIPS/NeurIPS naming; citation counts without source+date; paragraphs > 120 words; unglossed jargon | `--links`: arXiv API authors / title / first-version year |
| 2 TALK | "interview" in headings; Q&A heading must mention the paper; questions addressed to "you"; answers without a section/link; "from the paper" answers that mention later years; press/talk items without a link; quotes and "X said" statements without a traceable source; paper's arXiv id cited | |
| 3 Links | relative links resolve; homepage-only links | `--links`: GET with a Chrome User-Agent, redirects followed; 404/410/5xx/timeout = FAIL, 401/403/429 = WARN (blocked, not verified), captcha pages = WARN, redirect-to-homepage = FAIL; nbviewer checked via the GitHub blob; also the root-README Kaggle URL built from papers.tsv |
| 4 Notebook honesty | `### What the Notebook Result Shows` in README Key Method Details, with numbers; notebook summary cell "What the run above shows" with numbers; >= 1 image output; no error outputs; executed; epochs / lr / `NAME = value` mentioned in CODE_ARCHITECTURE, README notebook sections and notebook markdown vs the code (WARN); seed / timing / strong success claims (WARN) | |
| 5 Notebook format | > 1 MB FAIL (> 700 KB WARN); markdown cells that lost line breaks; kaggle_validator / `__notebook__.ipynb` cell; non-interactive matplotlib backend; tqdm/download progress in outputs; outputs > 300 KB; empty `<Figure ... 0 Axes>`; `/kaggle/...` paths | |
| 6 Kaggle | | `--kaggle` (needs `KAGGLE_API_TOKEN` + `kaggle` on PATH, else SKIP): `kernels status` = COMPLETE; `kernels pull -m` → cell-by-cell source diff vs solution.ipynb; is_private false; internet on; GPU reported; run log (`kernels output --file-pattern .log`) progress spam; `Name: value` lines compared with GitHub outputs; CPU-vs-GPU run mismatch. Temp files deleted, token redacted from all output. |
| 7 Pages | | `--pages`: paper page, TALK.html, CODE_ARCHITECTURE.html, banner return 200; more `<table>` in HTML than tables in the markdown (stray table) = FAIL; live page shows the current first paragraph |

WARNs are heuristics: read each one and fix it or accept it consciously. PASS does not replace
`../checklists/pre-commit-checklist.md`: facts, quotes, hedging and "does the picture show it" are manual.

## Kaggle CLI setup (once)

```bash
python3 -m venv ~/.venvs/kaggle && ~/.venvs/kaggle/bin/pip install --upgrade kaggle
export PATH=~/.venvs/kaggle/bin:$PATH
export KAGGLE_API_TOKEN=...   # new-style token; never kaggle.json, never echo it, never commit it
```

## Self-test

Run it on a good paper (032 passed at 1ad774b) and on a broken copy (add a bare pipe, a second
badge, an "Interview Q&A" heading, `matplotlib.use('Agg')`, a validator cell); each fault must show up as FAIL.
