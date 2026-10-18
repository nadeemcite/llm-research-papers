#!/usr/bin/env python3
"""check_paper.py - pre-commit gate for one paper of nadeemcite/llm-research-papers.

Usage:
    python check_paper.py <repo_root> NNN [--links] [--pages] [--kaggle] [--strict]
                                          [--today YYYY-MM-DD] [--now-hhmm HH:MM] [--json]

    (no flag)  offline checks: files, front matter, README template, Video line vs scripts/media.tsv,
               bare pipes, TALK integrity heuristics, notebook format/honesty heuristics,
               hyperparameter consistency, paragraph length, secrets/private files.
    --links    GET every external link with a browser User-Agent (non-200 -> FAIL/WARN) and
               compare the README against the arXiv API (authors, year, title).
    --pages    (after push) GitHub Pages: paper page, TALK page, CODE_ARCHITECTURE page and banner
               return 200, no stray tables, page shows the current text.
    --kaggle   needs KAGGLE_API_TOKEN in the env and the `kaggle` CLI on PATH: kernel status,
               pull source and diff every cell against solution.ipynb, check metadata, compare the
               run log's final numbers with the GitHub outputs. The token is never printed.
    --strict   treat WARN as FAIL.

Exit code: 0 = no FAIL, 1 = at least one FAIL, 2 = usage error. Standard library only.
"""
import argparse
import csv
import datetime as dt
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
CHANNEL = '[Backprop Daily](https://www.youtube.com/@BackpropDaily)'
KAGGLE_USER = 'nadymsazad'
GH_USER = 'nadeemcite'
SITE = 'https://nadeemcite.github.io/llm-research-papers'
UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
      'Chrome/126.0.0.0 Safari/537.36')
REQUIRED_FILES = ['README.md', 'TALK.md', 'CODE_ARCHITECTURE.md', 'solution.ipynb']
README_H2 = ['Summary', 'What Problem Does It Solve', 'Core Idea', 'Key Method Details', 'Influence', 'Resources']
META_KEYS = ['Paper', 'Authors', 'Lab, year', 'Video']
PARA_LIMIT = 120
NB_MAX = 1_000_000
NB_WARN = 700_000

GROUPS = {
    0: 'Files & hygiene', 1: 'README', 2: 'TALK', 3: 'Links', 4: 'Notebook honesty',
    5: 'Notebook format', 6: 'Kaggle', 7: 'Pages',
}


class Report:
    def __init__(self):
        self.items = []

    def add(self, level, group, msg):
        self.items.append((level, group, msg))

    def ok(self, g, m): self.add('PASS', g, m)
    def fail(self, g, m): self.add('FAIL', g, m)
    def warn(self, g, m): self.add('WARN', g, m)
    def skip(self, g, m): self.add('SKIP', g, m)
    def info(self, g, m): self.add('INFO', g, m)

    def count(self, level):
        return sum(1 for l, _, _ in self.items if l == level)


SECRET = None  # the KAGGLE_API_TOKEN value, used only for redaction / leak detection


def redact(s):
    s = str(s)
    if SECRET:
        s = s.replace(SECRET, '[REDACTED]')
    return re.sub(r'KGAT_[A-Za-z0-9]+', 'KGAT_[REDACTED]', s)


# ----------------------------------------------------------------------------- helpers

def read(p):
    return Path(p).read_text(encoding='utf-8')


def load_tsv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def split_front_matter(text):
    """Return (dict or None, body, body_start_line_index)."""
    lines = text.split('\n')
    if not lines or lines[0].strip() != '---':
        return None, text, 0
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == '---')
    except StopIteration:
        return None, text, 0
    fm = {}
    for ln in lines[1:end]:
        if ':' not in ln:
            continue
        k, v = ln.split(':', 1)
        v = v.strip()
        try:
            v = json.loads(v)
        except Exception:
            v = v.strip('"\'')
        fm[k.strip()] = v
    return fm, '\n'.join(lines[end + 1:]), end + 1


def fmt_date(iso):
    d = dt.date.fromisoformat(iso)
    return f'{d.day} {d.strftime("%b")} {d.year}'


def md_blocks(text, start_line=0):
    """Yield (kind, lineno(1-based), lines) blocks: code, table, heading, list item, para, blank-separated."""
    lines = text.split('\n')
    i, n = 0, len(lines)
    blocks = []
    while i < n:
        ln = lines[i]
        s = ln.strip()
        if s.startswith('```') or s.startswith('~~~'):
            fence = s[:3]
            j = i + 1
            while j < n and not lines[j].strip().startswith(fence):
                j += 1
            blocks.append(('code', start_line + i + 1, lines[i:j + 1]))
            i = j + 1
            continue
        if not s:
            i += 1
            continue
        if s.startswith('#'):
            blocks.append(('heading', start_line + i + 1, [ln]))
            i += 1
            continue
        if s.startswith('|'):
            j = i
            while j < n and lines[j].strip().startswith('|'):
                j += 1
            blk = lines[i:j]
            is_table = any(re.match(r'^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$', b.strip()) for b in blk)
            blocks.append(('table' if is_table else 'pseudotable', start_line + i + 1, blk))
            i = j
            continue
        if re.match(r'^\s*([-*+]|\d+[.)])\s+', ln):
            j = i + 1
            while j < n and lines[j].strip() and not re.match(r'^\s*([-*+]|\d+[.)])\s+', lines[j]) \
                    and not lines[j].strip().startswith(('#', '|', '```')):
                j += 1
            blocks.append(('item', start_line + i + 1, lines[i:j]))
            i = j
            continue
        j = i + 1
        while j < n and lines[j].strip() and not lines[j].strip().startswith(('#', '|', '```', '~~~')) \
                and not re.match(r'^\s*([-*+]|\d+[.)])\s+', lines[j]):
            j += 1
        blocks.append(('para', start_line + i + 1, lines[i:j]))
        i = j
    return blocks


def words(s):
    s = re.sub(r'\]\([^)]*\)', ']', s)          # drop link targets
    s = re.sub(r'https?://\S+', '', s)
    return len(re.findall(r"[A-Za-z0-9][\w'’.-]*", s))


LINK_RE = re.compile(r'\]\(\s*<?([^)\s>]+)>?(?:\s+"[^"]*")?\s*\)')  # every [..](target), incl. nested badges
BARE_RE = re.compile(r'(?<![(<"])\bhttps?://[^\s)<>\]"\'`]+')


def extract_links(text):
    out = []
    for i, ln in enumerate(text.split('\n'), 1):
        found = set()
        for m in LINK_RE.finditer(ln):
            found.add(m.group(1))
        for m in BARE_RE.finditer(ln):
            u = m.group(0).rstrip('.,;:')
            if not any(u in f for f in found):
                found.add(u)
        for u in found:
            out.append((i, u))
    return out


def sections(body):
    """Return list of (level, title, start_idx, end_idx) over body lines (code fences ignored)."""
    lines = body.split('\n')
    heads, in_code = [], False
    for i, ln in enumerate(lines):
        if ln.strip().startswith('```'):
            in_code = not in_code
        if in_code:
            continue
        m = re.match(r'^(#{1,6})\s+(.*?)\s*#*\s*$', ln)
        if m:
            heads.append((len(m.group(1)), m.group(2), i))
    out = []
    for k, (lvl, t, i) in enumerate(heads):
        end = len(lines)
        for lvl2, _, j in heads[k + 1:]:
            if lvl2 <= lvl:
                end = j
                break
        out.append((lvl, t, i, end))
    return out, lines


def cell_src(c):
    s = c.get('source', '')
    return ''.join(s) if isinstance(s, list) else s


# ----------------------------------------------------------------------------- checks

def check_files(R, root, folder, num):
    d = root / folder
    for f in REQUIRED_FILES:
        if (d / f).is_file():
            R.ok(0, f'{f} present')
        else:
            R.fail(0, f'{f} missing in {folder}/')
    extra = sorted(p.name for p in d.iterdir() if p.name not in REQUIRED_FILES)
    if extra:
        R.warn(0, f'extra files in paper folder (make sure they are meant to be public): {", ".join(extra)}')


def check_hygiene(R, root, folder):
    # secrets in the paper folder
    pats = [
        (r'KGAT_[A-Za-z0-9]{16,}', 'Kaggle API token'),
        (r'ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}', 'GitHub token'),
        (r'\bsk-[A-Za-z0-9]{20,}', 'API secret key'),
        (r'\bhf_[A-Za-z0-9]{30,}', 'Hugging Face token'),
        (r'AKIA[0-9A-Z]{16}', 'AWS key'),
        (r'"key"\s*:\s*"[0-9a-f]{32}"', 'legacy kaggle.json key'),
        (r'KAGGLE_(?:KEY|API_TOKEN)\s*=\s*[\'"]?[A-Za-z0-9_]{12,}', 'hard-coded Kaggle credential'),
    ]
    leaks = []
    for p in (root / folder).rglob('*'):
        if not p.is_file() or p.stat().st_size > 20_000_000:
            continue
        try:
            t = p.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        for rx, what in pats:
            if re.search(rx, t):
                leaks.append(f'{p.name}: {what}')
        if SECRET and len(SECRET) >= 8 and SECRET in t:
            leaks.append(f'{p.name}: the live KAGGLE_API_TOKEN value')
        if 'kaggle.json' in t and re.search(r'open\([^)]*kaggle\.json', t):
            leaks.append(f'{p.name}: reads legacy kaggle.json')
    if leaks:
        for l in leaks:
            R.fail(0, f'possible secret: {l} (never commit tokens)')
    else:
        R.ok(0, 'no secrets / token patterns in the paper folder')
    # private pipeline files tracked or about to be added
    if shutil.which('git') and (root / '.git').exists():
        try:
            tracked = subprocess.run(['git', '-C', str(root), 'ls-files'], capture_output=True, text=True,
                                     timeout=60).stdout.split('\n')
            bad = [t for t in tracked if re.search(r'(^|/)(\.ignore/|\.kaggle_output/|kernel-metadata\.json$|'
                                                   r'kaggle\.json$|\.env$|\.claude/|\.serena/)', t)]
            if bad:
                R.fail(0, f'private/pipeline files are tracked by git: {", ".join(bad[:8])}')
            else:
                R.ok(0, 'no private pipeline files (.ignore/, kernel-metadata.json, kaggle.json, .env) tracked')
            st = subprocess.run(['git', '-C', str(root), 'status', '--porcelain', '--', folder], capture_output=True,
                                text=True, timeout=60).stdout.strip()
            if st:
                R.info(0, 'uncommitted changes in folder: ' + '; '.join(st.split('\n')[:8]))
            name = subprocess.run(['git', '-C', str(root), 'config', 'user.name'], capture_output=True,
                                  text=True).stdout.strip()
            if name and name != GH_USER:
                R.warn(0, f'git user.name is "{name}"; commit and push only as {GH_USER}')
            gi = (root / '.gitignore')
            if gi.exists() and '.ignore' not in gi.read_text():
                R.warn(0, '.gitignore no longer ignores /.ignore/')
        except Exception as e:
            R.skip(0, f'git checks skipped: {e}')
    else:
        R.skip(0, 'not a git checkout; git hygiene checks skipped')


def check_front_matter(R, root, folder, num, short):
    exp = {
        'README.md': {'title': f'{num} {short}', 'nav_order': int(num) + 1, 'permalink': f'/{folder}/'},
        'CODE_ARCHITECTURE.md': {'title': f'{short}: Code architecture', 'parent': f'{num} {short}', 'nav_order': 1},
        'TALK.md': {'title': f'{short}: Talk & Q&A', 'parent': f'{num} {short}', 'nav_order': 2},
    }
    for fname, want in exp.items():
        p = root / folder / fname
        if not p.exists():
            continue
        fm, _, _ = split_front_matter(read(p))
        if fm is None:
            R.fail(1, f'{fname}: no YAML front matter (run `python scripts/build_readme.py`)')
            continue
        bad = [f'{k}={fm.get(k)!r} (want {v!r})' for k, v in want.items() if fm.get(k) != v]
        if fname == 'README.md' and 'parent' in fm:
            bad.append('README must not have parent')
        if bad:
            R.fail(1, f'{fname}: front matter mismatch: ' + '; '.join(bad) + ' (run scripts/build_readme.py)')
        else:
            R.ok(1, f'{fname}: front matter ok ({", ".join(want)})')


def media_state(root, num, today, now_hhmm):
    p = root / 'scripts' / 'media.tsv'
    if not p.exists():
        return None
    rows = [r for r in load_tsv(p) if r.get('num') == num]
    return rows[0] if rows else {}


def check_readme(R, root, folder, paper, media, today, now_hhmm):
    num, short = paper['num'], paper['short']
    p = root / folder / 'README.md'
    if not p.exists():
        return {}
    text = read(p)
    fm, body, off = split_front_matter(text)
    lines = body.split('\n')
    nz = [(i, l) for i, l in enumerate(lines) if l.strip()]
    if not nz or not nz[0][1].startswith('# '):
        R.fail(1, 'first line after front matter must be "# Title"')
        return {}
    ti = nz[0][0]
    R.ok(1, f'title: {lines[ti][2:].strip()}')

    # ---- media state (banner + Video line), mirrors scripts/readme_media.py
    status = (media or {}).get('status', '').strip().lower()
    want_banner = want_video = None
    if status == 'public':
        want_banner = f"![{media['short']} banner](../{media['banner'].strip()})"
        url = (media.get('shorts_url') or '').strip()
        want_video = f'- **Video:** [Watch the YouTube Short]({url}) on {CHANNEL}' if url else None
        if not url:
            R.fail(1, 'media.tsv: status=public but shorts_url is empty')
    elif status == 'private':
        want_banner = f"![{media['short']} banner](../{media['banner'].strip()})"
        try:
            want_video = f"- **Video:** YouTube Short coming on {fmt_date(media['release_date'].strip())} on {CHANNEL}"
        except Exception:
            R.fail(1, f'media.tsv: bad release_date {media.get("release_date")!r} (want YYYY-MM-DD)')
        rd = media.get('release_date', '').strip()
        if rd:
            rdd = dt.date.fromisoformat(rd)
            if rdd < today or (rdd == today and now_hhmm >= '18:30'):
                R.warn(1, f'media.tsv says private but release day {rd} 6:30 PM IST has passed: if the Short is public, '
                          'set status=public + shorts_url, run `python scripts/readme_media.py ' + num + '`, update the sheet')
            elif (rdd - today).days <= 1:
                R.info(1, f'release day {rd} is imminent: this paper must fully pass all checks now')
    elif status in ('', 'none'):
        pass
    else:
        R.fail(1, f'media.tsv: unknown status {status!r}')
    if want_banner:
        bpath = root / media['banner'].strip()
        if not bpath.is_file():
            R.fail(1, f'banner file missing: {media["banner"]}')
        elif media['banner'].strip() != f'assets/banners/{num}.jpg':
            R.warn(1, f'banner path {media["banner"]} is not assets/banners/{num}.jpg')

    # banner line
    nxt = ti + 1
    banner_re = re.compile(r'^!\[[^\]]*\]\((?:\.\./)?assets/banners/[^)]+\)\s*$')
    has_banner = nxt < len(lines) and banner_re.match(lines[nxt])
    if want_banner:
        if not has_banner:
            R.fail(1, f'banner line missing right under the title (want: {want_banner}); run scripts/readme_media.py {num}')
        elif lines[nxt].strip() != want_banner:
            R.fail(1, f'banner line is "{lines[nxt].strip()}", want "{want_banner}"')
        else:
            R.ok(1, 'banner line matches media.tsv')
        badge_i = nxt + 1
    else:
        if has_banner:
            R.fail(1, 'banner present but media.tsv has no private/public row for this paper')
            badge_i = nxt + 1
        else:
            R.info(1, 'no media.tsv row (state 1 "paper created"): no banner and no Video line expected')
            badge_i = nxt

    # Kaggle badge
    badges = [i for i, l in enumerate(lines) if 'open-in-kaggle.svg' in l]
    slug = paper.get('kaggle', '').strip()
    want_badge = (f'[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)]'
                  f'(https://www.kaggle.com/code/{KAGGLE_USER}/{slug})')
    if len(badges) != 1:
        R.fail(1, f'want exactly one Kaggle badge, found {len(badges)}')
    else:
        b = lines[badges[0]].strip()
        if badges[0] != badge_i:
            R.fail(1, 'Kaggle badge must be the line right after the title/banner')
        m = re.search(r'kaggle\.com/code/([^/\s)]+)/([^/\s)]+)', b)
        if m:
            paper['badge_slug'] = m.group(2)
        if not m:
            R.fail(1, 'Kaggle badge does not link to kaggle.com/code/<user>/<slug>')
        elif m.group(1) != KAGGLE_USER:
            R.fail(1, f'Kaggle badge user is {m.group(1)}, want {KAGGLE_USER}')
        elif slug and m.group(2) != slug:
            R.fail(1, f'Kaggle badge slug {m.group(2)} != papers.tsv kaggle column {slug} (the root README links the '
                      'papers.tsv slug; fix whichever is wrong, then run scripts/build_readme.py)')
        elif b != want_badge:
            R.warn(1, f'Kaggle badge markup differs from the standard: {b}')
        else:
            R.ok(1, f'one Kaggle badge -> {KAGGLE_USER}/{slug}')

    # metadata bullet list
    first_h2 = next((i for i in range(ti, len(lines)) if lines[i].startswith('## ')), len(lines))
    meta = [(i, l) for i, l in enumerate(lines[:first_h2]) if re.match(r'^- \*\*[^*]+:\*\* ', l)]
    keys = [re.match(r'^- \*\*([^*]+):\*\*', l).group(1) for _, l in meta]
    want_keys = META_KEYS if want_video else META_KEYS[:3]
    if keys != want_keys:
        R.fail(1, f'metadata bullets are {keys}, want {want_keys} in that order')
    else:
        R.ok(1, 'metadata bullets: ' + ', '.join(keys))
    if meta and badges and meta[0][0] <= badges[0]:
        R.fail(1, 'metadata bullets must come after the Kaggle badge')
    md = {k: l for k, (_, l) in zip(keys, meta)}
    arxiv = None
    if 'Paper' in md:
        m = re.search(r'\]\((https?://arxiv\.org/abs/([\w.\-/]+?))(v\d+)?\)', md['Paper'])
        if m:
            arxiv = m.group(2)
        elif 'http' not in md['Paper']:
            R.fail(1, '"- **Paper:**" has no link')
        else:
            R.warn(1, '"- **Paper:**" does not link to an arXiv abs page')
    if 'Lab, year' in md:
        yrs = re.findall(r'\b(19\d\d|20\d\d)\b', md['Lab, year'])
        ty = str(paper.get('year', ''))
        if not yrs:
            R.fail(1, '"Lab, year" has no year')
        elif ty and yrs[0] != ty:
            R.warn(1, f'"Lab, year" first year {yrs[0]} != papers.tsv year {ty} (year = first public release, venue in brackets)')
        else:
            R.ok(1, f'year {yrs[0]} matches papers.tsv')
    # Video line
    vids = [l for l in lines if l.startswith('- **Video:**')]
    fmt_ok = re.compile(r'^- \*\*Video:\*\* (\[Watch the YouTube Short\]\(https://(www\.)?youtube\.com/shorts/[\w-]+\)'
                        r'|YouTube Short coming on \d{1,2} [A-Z][a-z]{2} \d{4}) on '
                        + re.escape(CHANNEL) + r'$')
    if len(vids) > 1:
        R.fail(1, f'{len(vids)} Video lines, want at most 1')
    if want_video:
        if not vids:
            R.fail(1, f'Video line missing (want: {want_video}); run scripts/readme_media.py {num}')
        elif vids[0] != want_video:
            R.fail(1, f'Video line does not match media.tsv: has "{vids[0]}", want "{want_video}"')
        else:
            R.ok(1, f'Video line matches media.tsv ({status})')
    elif vids:
        R.fail(1, 'Video line present but media.tsv has no private/public row (state 1 has no Video line)')
    for v in vids:
        if not fmt_ok.match(v):
            R.fail(1, f'Video line is not one of the valid formats: {v}')

    # sections
    secs, blines = sections(body)
    h2 = [t for lvl, t, _, _ in secs if lvl == 2]
    if [h.lower() for h in h2] != [h.lower() for h in README_H2]:
        R.fail(1, f'H2 sections are {h2}, want {README_H2}')
    elif h2 != README_H2:
        R.warn(1, f'H2 section capitalisation differs: {h2}')
    else:
        R.ok(1, 'sections in order: ' + ' > '.join(h2))
    kmd = next(((s, e) for lvl, t, s, e in secs if lvl == 2 and t.lower() == 'key method details'), None)
    h3_in_kmd = [t for lvl, t, s, e in secs if lvl == 3 and kmd and kmd[0] < s < kmd[1]]
    wns = [t for t in h3_in_kmd if re.match(r'what the notebook result shows', t, re.I)]
    if not wns:
        anywhere = [t for lvl, t, _, _ in secs if re.match(r'what the notebook result shows', t, re.I)]
        if anywhere:
            R.fail(4, '"### What the Notebook Result Shows" must sit inside "## Key Method Details"')
        else:
            R.fail(4, 'README has no "### What the Notebook Result Shows" section (real numbers from the saved run)')
    else:
        s0 = next(s for lvl, t, s, e in secs if t == wns[0])
        e0 = next(e for lvl, t, s, e in secs if t == wns[0])
        chunk = '\n'.join(blines[s0 + 1:e0])
        if not re.search(r'\d', chunk):
            R.fail(4, '"What the Notebook Result Shows" has no numbers')
        else:
            R.ok(4, 'README has "What the Notebook Result Shows" with numbers')
    other = [t for t in h3_in_kmd if not re.match(r'(simplifications|what the notebook result shows)', t, re.I)]
    if other:
        R.info(1, f'extra ### headings in Key Method Details: {other}')
    # --- before Resources
    res_i = next((s for lvl, t, s, e in secs if lvl == 2 and t == 'Resources'), None)
    if res_i is not None:
        prev = next((blines[j].strip() for j in range(res_i - 1, -1, -1) if blines[j].strip()), '')
        if prev != '---':
            R.fail(1, 'a "---" line must come right before "## Resources"')
        else:
            R.ok(1, '"---" before Resources')
        tbl = '\n'.join(blines[res_i:])
        for f in ['README.md', 'solution.ipynb', 'CODE_ARCHITECTURE.md', 'TALK.md']:
            if f'[{f}]' not in tbl:
                R.fail(1, f'Resources table has no [{f}] row')
        nbv = f'https://nbviewer.org/github/{GH_USER}/llm-research-papers/blob/main/{folder}/solution.ipynb'
        if nbv not in tbl:
            R.fail(1, f'Resources solution.ipynb link must be {nbv}')
        if re.search(r'interview', tbl, re.I):
            R.warn(2, 'Resources row describes TALK.md as "interview Q&A"; say "Q&A (answers from the paper)"')
        if re.search(r'from-scratch', tbl, re.I) and re.search(r'from_pretrained|pretrained=True', nb_code_text(root, folder)):
            R.warn(4, 'Resources says "from-scratch" but the notebook loads pretrained weights')
    return {'arxiv': arxiv, 'authors': md.get('Authors', ''), 'body': body, 'tsv_slug': slug,
            'year': str(paper.get('year', ''))}


def nb_code_text(root, folder):
    p = root / folder / 'solution.ipynb'
    try:
        nb = json.loads(read(p))
        return '\n'.join(cell_src(c) for c in nb['cells'] if c['cell_type'] == 'code')
    except Exception:
        return ''


def check_placeholders(R, root, folder):
    hits = []
    for fname in ['README.md', 'TALK.md', 'CODE_ARCHITECTURE.md', 'solution.ipynb']:
        p = root / folder / fname
        if p.exists():
            for i, l in enumerate(read(p).split('\n'), 1):
                if re.search(r'\{\{[^}]{1,200}\}\}', l):
                    hits.append(f'{fname}:{i}')
    if hits:
        R.fail(0, f'unfilled template placeholders {{{{...}}}}: {", ".join(hits[:12])}' + (' ...' if len(hits) > 12 else ''))
    else:
        R.ok(0, 'no unfilled template placeholders')


def check_pipes(R, root, folder):
    for fname in ['README.md', 'TALK.md', 'CODE_ARCHITECTURE.md']:
        p = root / folder / fname
        if not p.exists():
            continue
        _, body, off = split_front_matter(read(p))
        bad = []
        for kind, ln, blk in md_blocks(body, off):
            if kind == 'code':
                continue
            for k, l in enumerate(blk):
                if kind == 'table':
                    for code in re.findall(r'`([^`]*)`', l):
                        if re.search(r'(?<!\\)\|', code):
                            bad.append((ln + k, l.strip(), 'unescaped | inside inline code in a table cell'))
                    continue
                if re.search(r'(?<!\\)\|', l):
                    bad.append((ln + k, l.strip(), 'bare | outside a table'))
        if bad:
            for ln, l, why in bad[:10]:
                R.fail(1, f'{fname}:{ln}: {why} (kramdown makes it a table; write \\|): {l[:90]}')
        else:
            R.ok(1, f'{fname}: no bare | outside tables')


JARGON = ['BLEU', 'mAP', 'SOTA', 'MCMC', 'Markov chain', 'regression problem', 'backbone', 'logit', 'softmax',
          'autoregressive', 'ablation', 'perplexity', 'IoU', 'FLOPs', 'RNN', 'LSTM', 'Q-value', 'minimax',
          'maxout', 'function approximator', 'non-stationary', 'bootstrap', 'transduction', 'permutation-invariant']


def check_prose(R, root, folder, paper):
    year = int(paper.get('year') or 0)
    for fname in ['README.md', 'TALK.md', 'CODE_ARCHITECTURE.md']:
        p = root / folder / fname
        if not p.exists():
            continue
        text = read(p)
        _, body, off = split_front_matter(text)
        g = 2 if fname == 'TALK.md' else 1
        long = []
        for kind, ln, blk in md_blocks(body, off):
            if kind in ('para', 'item'):
                w = words(' '.join(blk))
                if w > PARA_LIMIT:
                    long.append((ln, w, ' '.join(blk).strip()[:60]))
        for ln, w, s in long:
            R.warn(g, f'{fname}:{ln}: paragraph is {w} words (> {PARA_LIMIT}); split it and explain jargon: "{s}..."')
        if not long:
            R.ok(g, f'{fname}: no paragraph over {PARA_LIMIT} words')
        for i, l in enumerate(body.split('\n'), off + 1):
            for m in re.finditer(r'NeurIPS\W{0,3}(20[01]\d)', l):
                if int(m.group(1)) < 2018:
                    R.fail(g, f'{fname}:{i}: "NeurIPS {m.group(1)}": the venue was called NIPS until 2018 '
                              f'(write "NIPS {m.group(1)}, now NeurIPS")')
            for m in re.finditer(r'\bNIPS\W{0,3}(20[12]\d)', l):
                if int(m.group(1)) >= 2018:
                    R.warn(g, f'{fname}:{i}: "NIPS {m.group(1)}": from 2018 on the venue is NeurIPS')
            if re.search(r'\b(cited|citations?)\b', l, re.I) and re.search(r'\d[\d,.]*\s*(\+|k\b|,000|000)', l):
                src = re.search(r'Semantic Scholar|OpenAlex|Google Scholar|Scholar|Crossref', l)
                when = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|'
                                 r'December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+20\d\d|checked|as of', l)
                if not (src and when):
                    R.warn(g, f'{fname}:{i}: citation count without a named source AND check date: {l.strip()[:90]}')
            for rx, why in [(r'\bon every seed\b|\bevery seed we tested\b|\bacross \d+ (random )?seeds\b',
                             'seed claim needs a saved multi-seed output or "earlier runs (not shown)"'),
                            (r'\btrains? in (about |under |less than )?(a|one|\d+) (minute|second)s?\b',
                             'timing claim needs printed timing in the saved output'),
                            (r'\bclearly (learned|learns|works)\b|\bperfectly (learn|balanc|reconstruct|separat)|'
                             r'\b(is|was|gets) solved\b|\bsolves (the task|cartpole|it)\b',
                             'strong success claim: check the saved outputs really show it'),
                            (r'\bthe first real-time\b|\bfor the first time\b|\bfirst-ever\b|\bfirst ever\b',
                             '"first" claim: make sure the paper or a source says so')]:
                if re.search(rx, l, re.I):
                    R.warn(4 if 'output' in why else g, f'{fname}:{i}: {why}: {l.strip()[:90]}')
        if fname == 'README.md':
            secs, blines = sections(body)
            for lvl, t, s, e in secs:
                if lvl == 2 and t in ('Summary', 'What Problem Does It Solve'):
                    chunk = '\n'.join(blines[s + 1:e])
                    unglossed = []
                    for term in JARGON:
                        m = re.search(r'\b' + re.escape(term), chunk)
                        if m and '(' not in chunk[m.end():m.end() + 40] and '—' not in chunk[m.end():m.end() + 6]:
                            unglossed.append(term)
                    if unglossed:
                        R.warn(1, f'README "{t}": jargon that may need a plain-English gloss: {", ".join(unglossed)}')


def check_talk(R, root, folder, paper, arxiv):
    p = root / folder / 'TALK.md'
    if not p.exists():
        return
    text = read(p)
    _, body, off = split_front_matter(text)
    secs, blines = sections(body)
    year = int(paper.get('year') or 0)
    heads = [(lvl, t, s) for lvl, t, s, e in secs]
    bad = [t for lvl, t, s in heads if re.search(r'interview', t, re.I)]
    for t in bad:
        R.fail(2, f'TALK heading "{t}": never call it an interview; use "Q&A (answers from the paper)"')
    qa = [(t, s, e) for lvl, t, s, e in secs if lvl == 2 and re.search(r'Q\s*&\s*A|Q&amp;A', t)]
    if not qa:
        R.fail(2, 'TALK.md has no "## Q&A (answers from the paper)" section')
    else:
        for t, s, e in qa:
            if not re.search(r'paper', t, re.I):
                R.fail(2, f'TALK Q&A heading "{t}" must say the answers come from the paper, e.g. '
                          '"## Q&A (answers from the paper)"')
            elif not bad:
                R.ok(2, f'Q&A heading ok: "{t}"')
            chunk = blines[s + 1:e]
            # questions / answers
            qidx = [i for i, l in enumerate(chunk) if re.match(r'^\*\*Q\d*[:.]', l.strip())]
            for k, qi in enumerate(qidx):
                q = chunk[qi]
                ans = '\n'.join(chunk[qi + 1:(qidx[k + 1] if k + 1 < len(qidx) else len(chunk))])
                ln = off + s + 2 + qi
                if re.search(r'\byou(r)?\b', q, re.I):
                    R.warn(2, f'TALK:{ln}: question addresses "you" (reads like an interview): {q.strip()[:80]}')
                if not re.search(r'from the paper|Section|§|Abstract|Table|Figure|Appendix|\]\(https?://|https?://', ans):
                    R.warn(2, f'TALK:{ln}: answer has no source (paper section or linked source): {q.strip()[:70]}')
                if re.search(r'from the paper', ans, re.I):
                    later = [y for y in re.findall(r'\b(19\d\d|20\d\d)\b', ans) if year and int(y) > year + 1]
                    if later and not re.search(r'later|follow-up|subsequent|after', ans, re.I):
                        R.warn(2, f'TALK:{ln}: answer "from the paper" mentions later year(s) {later}: '
                                  'is later work being credited to this paper?')
            if not qidx:
                R.warn(2, f'Q&A section "{t}" has no **Q...** questions')
    # press / talks items need real links
    press = [(t, s, e) for lvl, t, s, e in secs if lvl == 2 and re.search(r'press|coverage|blog|talk|media|slides', t, re.I)]
    for t, s, e in press:
        items = [(s + 1 + i, l) for i, l in enumerate(blines[s + 1:e]) if re.match(r'^\s*(\d+[.)]|[-*])\s+', l)]
        nolink = [(i, l) for i, l in items if not re.search(r'https?://', l)]
        for i, l in nolink:
            R.fail(2, f'TALK:{off + i + 1}: press/talk item without a link (only real, working links allowed): {l.strip()[:80]}')
        if items and not nolink:
            R.ok(2, f'every item in "{t}" has a link ({len(items)} items)')
        if not items:
            R.warn(2, f'section "{t}" has no list items')
    if not press:
        R.warn(2, 'TALK has no press / talks section')
    # quotes
    qrx = re.compile(r'["“]([^"“”]{25,})["”]')
    skip_rng = [(s, e) for lvl, t, s, e in secs if lvl == 2 and
                re.search(r'misconception|citation|reference|press|coverage|blog|talk|slides|media', t, re.I)]
    for kind, ln, blk in md_blocks(body, off):
        if kind not in ('para', 'item'):
            continue
        rel = ln - off - 1
        if any(s <= rel < e for s, e in skip_rng):
            continue
        para = ' '.join(blk)
        for m in qrx.finditer(para):
            qt = m.group(1)
            if len(qt.split()) < 6:
                continue
            sourced = re.search(r'Section|§|Abstract|Table|Fig\.?|Figure|Appendix|from the paper|\]\(https?://|https?://|'
                                r'arXiv', para)
            if not sourced:
                R.warn(2, f'TALK:{ln}: quote without an attributed, traceable source: "{qt[:70]}..."')
        if re.search(r'^\s*(\d+[.)]\s*)?\*\*[A-Z][a-z]+ [A-Z][a-z]+( \([^)]*\))?:\*\*', para) and not re.search(r'https?://', para):
            R.warn(2, f'TALK:{ln}: named-person speaker label with no linked source: {para.strip()[:70]}')
        if (re.search(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)? (said|says|recalled|told|explained)\b', para)
                or re.search(r'in (an|various|several|multiple) (public )?(interview|talk)s?', para)) \
                and not re.search(r'\b(The|the|This|this) (paper|conclusion|abstract|article|section)\b', para) \
                and not re.search(r'https?://', para):
            R.warn(2, f'TALK:{ln}: attributed statement ("said/told") without a link: {para.strip()[:70]}')
    if arxiv and arxiv not in text:
        R.warn(2, f'TALK.md never cites the paper arXiv id {arxiv}')


def check_relative_links(R, root, folder):
    d = root / folder
    bad = 0
    for fname in ['README.md', 'TALK.md', 'CODE_ARCHITECTURE.md']:
        p = d / fname
        if not p.exists():
            continue
        for ln, u in extract_links(read(p)):
            if re.match(r'^(https?:|mailto:|#)', u):
                continue
            target = (d / urllib.parse.unquote(u.split('#')[0])).resolve()
            if not target.exists():
                bad += 1
                R.fail(3, f'{fname}:{ln}: relative link does not resolve: {u}')
    if not bad:
        R.ok(3, 'all relative links resolve on disk')
    # homepage-only links
    for fname in ['README.md', 'TALK.md', 'CODE_ARCHITECTURE.md']:
        p = d / fname
        if not p.exists():
            continue
        for ln, u in extract_links(read(p)):
            if u.startswith('http'):
                pu = urllib.parse.urlparse(u)
                if pu.path in ('', '/') and not pu.query:
                    R.warn(3, f'{fname}:{ln}: homepage-only link supports nothing; link the exact page: {u}')


def nb_load(R, root, folder):
    p = root / folder / 'solution.ipynb'
    if not p.exists():
        return None, 0
    try:
        return json.loads(read(p)), p.stat().st_size
    except Exception as e:
        R.fail(5, f'solution.ipynb is not valid JSON: {e}')
        return None, 0


PROGRESS = re.compile(r'\d+%\|[█▉▊▋▌▍▎▏ #]*\||\bit/s\]|\d+/\d+ \[\d\d:\d\d<|Downloading\b|Extracting\b|'
                      r'\r[^\n]|━━━|\[=+>?\.*\]')


def check_notebook(R, root, folder, nb, size):
    if nb is None:
        return
    cells = nb.get('cells', [])
    if nb.get('nbformat') != 4:
        R.warn(5, f'nbformat is {nb.get("nbformat")}, expected 4')
    if size > NB_MAX:
        R.fail(5, f'notebook is {size/1e6:.2f} MB (> 1 MB): shrink GIF/HTML animations or images')
    elif size > NB_WARN:
        R.warn(5, f'notebook is {size/1e6:.2f} MB; aim for well under 1 MB')
    else:
        R.ok(5, f'notebook size {size/1e3:.0f} KB')
    broken, validator, agg, spam, imgs, errors, unexec, big = [], [], [], [], 0, [], [], []
    empty_fig = []
    for i, c in enumerate(cells):
        src = cell_src(c)
        if c['cell_type'] == 'markdown':
            raw = c.get('source', '')
            glued = re.search(r'[^\s#|]#{2,6} \S|[a-z.:)*]- \*\*|\S\|\|\S', src)
            listbug = isinstance(raw, list) and len(raw) > 1 and any(not s.endswith('\n') for s in raw[:-1])
            one = '\n' not in src.strip()
            if (one and len(src) > 200) or (one and src.lstrip().startswith('#') and len(src.strip()) > 80) \
                    or glued or listbug:
                broken.append(i)
        if 'kaggle_validator' in src or '__notebook__.ipynb' in src:
            validator.append(i)
        if c['cell_type'] == 'code':
            if re.search(r"matplotlib\.use\(\s*['\"]agg['\"]|%matplotlib\s+agg|MPLBACKEND", src, re.I):
                agg.append(i)
            if src.strip() and c.get('execution_count') is None:
                unexec.append(i)
            for o in c.get('outputs', []):
                ot = o.get('output_type')
                if ot == 'error':
                    errors.append(f'{i}: {o.get("ename")}: {o.get("evalue", "")[:60]}')
                if ot == 'stream':
                    t = ''.join(o.get('text', '')) if isinstance(o.get('text'), list) else o.get('text', '')
                    if PROGRESS.search(t) or (o.get('name') == 'stderr' and len(t) > 5000):
                        spam.append(i)
                    if re.search(r'<Figure size [^>]* with 0 Axes>', t):
                        empty_fig.append(i)
                data = o.get('data', {})
                if any(k.startswith('image/') for k in data):
                    imgs += 1
                tp = data.get('text/plain', '')
                tp = ''.join(tp) if isinstance(tp, list) else tp
                if re.search(r'<Figure size [^>]* with 0 Axes>', tp):
                    empty_fig.append(i)
                osz = len(json.dumps(o))
                if osz > 300_000:
                    big.append(f'cell {i} ({osz/1e3:.0f} KB {",".join(data) or ot})')
                html = data.get('text/html', '')
                html = ''.join(html) if isinstance(html, list) else html
                if re.search(r'tqdm|progress', html, re.I) and re.search(r'<progress|%\|', html):
                    spam.append(i)
    (R.fail(5, f'markdown cells lost their line breaks (one run-together line): {broken}') if broken
     else R.ok(5, 'markdown cells keep their line breaks'))
    (R.fail(5, f'kaggle_validator / __notebook__ copy cell present: cells {validator}') if validator
     else R.ok(5, 'no kaggle_validator cell'))
    (R.fail(5, f"matplotlib.use('Agg') (or agg backend) hides plots: cells {agg}") if agg
     else R.ok(5, "no matplotlib.use('Agg')"))
    (R.fail(5, f'download / progress-bar spam in outputs: cells {sorted(set(spam))}') if spam
     else R.ok(5, 'no progress-bar / download spam in outputs'))
    (R.fail(4, 'no image outputs: plots must display in the saved notebook') if imgs == 0
     else R.ok(4, f'{imgs} image output(s) displayed'))
    if errors:
        R.fail(4, 'error outputs saved: ' + '; '.join(errors[:5]))
    else:
        R.ok(4, 'no error outputs')
    code_cells = [i for i, c in enumerate(cells) if c['cell_type'] == 'code' and cell_src(c).strip()]
    if code_cells and len(unexec) == len(code_cells):
        R.fail(4, 'no code cell has been executed: commit the executed notebook (outputs from the Kaggle run)')
    elif unexec:
        R.warn(4, f'unexecuted code cells: {unexec}')
    if big:
        R.warn(5, 'very large outputs (inline GIF/HTML?): ' + ', '.join(big))
    if empty_fig:
        R.warn(5, f'stray empty "<Figure ... with 0 Axes>" output: cells {sorted(set(empty_fig))}')
    allsrc = '\n'.join(cell_src(c) for c in cells)
    if '/kaggle/working' in allsrc or '/kaggle/input' in allsrc:
        R.warn(5, 'hard-coded /kaggle/... paths: the notebook must also run outside Kaggle')
    if cells and not cell_src(cells[0]).lstrip().startswith('# '):
        R.warn(5, 'first cell should be a markdown "# Title" cell with the paper link')
    md_cells = [cell_src(c) for c in cells if c['cell_type'] == 'markdown']
    summ = [s for s in md_cells if re.search(r'what the (run above|notebook result) shows', s, re.I)]
    if not summ:
        R.fail(4, 'notebook has no summary cell with "### What the run above shows" (honest result, real numbers)')
    elif not re.search(r'\d', summ[-1]):
        R.fail(4, 'notebook "What the run above shows" has no numbers')
    else:
        R.ok(4, 'notebook summary has "What the run above shows" with numbers')
    fixed = []
    for i, c in enumerate(cells):
        if c['cell_type'] == 'code':
            fixed += [(i, t) for t in fixed_conclusions(cell_src(c))]
    if fixed:
        R.fail(4, 'code prints a fixed conclusion that is not computed from values (compute it, e.g. '
                  'print(f"decays: {late < early}")): ' + '; '.join(f'cell {i}: "{t[:70]}"' for i, t in fixed[:5]))
    else:
        R.ok(4, 'no hardcoded conclusion sentences in print() calls')


CONCLUSION = re.compile(r'\b(decays?|outperforms?|improves?|better than|collapses?|converges?)\b', re.I)


def fixed_conclusions(src):
    """Texts of print() calls whose arguments are all string literals (or f-strings with no placeholder)
    and that state a conclusion (decays, outperforms, improves, better than, collapses, converges)."""
    import ast
    code = '\n'.join('' if ln.lstrip().startswith(('%', '!')) else ln for ln in src.splitlines())
    try:
        tree = ast.parse(code)
    except SyntaxError:  # fall back to a line regex for sources ast cannot parse
        return [m.group(2) for m in re.finditer(r'''print\(\s*f?(['"])([^'"{}]*)\1\s*\)''', code)
                if CONCLUSION.search(m.group(2))]
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'print' and node.args \
                and not node.keywords:
            texts = []
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    texts.append(a.value)
                elif isinstance(a, ast.JoinedStr) and all(isinstance(v, ast.Constant) for v in a.values):
                    texts.append(''.join(v.value for v in a.values))
                else:
                    texts = None
                    break
            if texts and CONCLUSION.search(' '.join(texts)):
                out.append(' '.join(texts))
    return out


NUMW = r'(\d[\d,]*)'


def _f(x):
    try:
        return float(str(x).replace(',', '').replace('_', ''))
    except Exception:
        return None


def check_hparams(R, root, folder, nb):
    if nb is None:
        return
    code = '\n'.join(cell_src(c) for c in nb['cells'] if c['cell_type'] == 'code')
    nbmd = '\n'.join(cell_src(c) for c in nb['cells'] if c['cell_type'] == 'markdown')
    assigns = {}
    for m in re.finditer(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([-+]?[\d_.]+(?:e[-+]?\d+)?)\s*(?:#.*)?$', code, re.M):
        assigns[m.group(1)] = _f(m.group(2))
    for m in re.finditer(r'\blr\s*=\s*([\d.]+(?:e[-+]?\d+)?)', code):
        assigns.setdefault('lr', _f(m.group(1)))
    ep_vals = {v for k, v in assigns.items() if re.search(r'epoch|episode', k, re.I) and not re.search(r'per_epoch|every', k, re.I) and v}
    all_vals = {v for v in assigns.values() if v is not None}
    lr_vals = {v for k, v in assigns.items() if re.search(r'(^|_)lr$|^lr|learning_rate', k, re.I) and v}
    if not ep_vals and not lr_vals:
        R.info(4, 'no EPOCHS/lr assignments found in the notebook; hyperparameter cross-check skipped')
        return
    R.info(4, f'notebook hyperparameters: epochs/episodes={sorted(ep_vals) or "-"}, lr={sorted(lr_vals) or "-"}')
    docs = []
    p = root / folder / 'CODE_ARCHITECTURE.md'
    if p.exists():
        docs += [('CODE_ARCHITECTURE.md', i, l) for i, l in enumerate(read(p).split('\n'), 1)]
    p = root / folder / 'README.md'
    if p.exists():
        rl = read(p).split('\n')
        cur = ''
        for i, l in enumerate(rl, 1):
            if l.startswith('#'):
                cur = l
            nbsec = re.search(r'simplif|notebook|this repo', cur, re.I)
            if nbsec or re.search(r'notebook|this repo|our run|we train', l, re.I):
                docs.append(('README.md', i, l))
    docs += [('solution.ipynb[md]', i, l) for i, l in enumerate(nbmd.split('\n'), 1)]
    issues = 0
    paperish = re.compile(r'paper|original|instead of|vs\.?|versus|Nature|authors|~|about \d|up to', re.I)
    for f, i, l in docs:
        flagged = False
        # direct NAME = value mentions
        for m in re.finditer(r'`?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([\d_.]+(?:e[-+]?\d+)?)`?', l):
            name, val = m.group(1), _f(m.group(2))
            if name in assigns and val is not None and assigns[name] is not None and val != assigns[name] \
                    and not paperish.search(l):
                issues += 1
                flagged = True
                R.warn(4, f'{f}:{i}: says {name} = {m.group(2)} but the notebook sets {name} = {assigns[name]:g}')
        if flagged:
            continue
        if ep_vals:
            for m in re.finditer(NUMW + r'\s+(epochs?|episodes?)\b', l, re.I):
                v = _f(m.group(1))
                pre = l[max(0, m.start() - 12):m.start()].lower()
                if re.search(r'(every|after|first|last|by|within|at|until|per|[–-])\s*(about\s*)?$', pre) \
                        or re.match(r'\s*[–-]', l[m.end(1):m.end(1) + 2]):
                    continue
                if v and v not in ep_vals and v not in all_vals and not paperish.search(l) and v > 1:
                    issues += 1
                    R.warn(4, f'{f}:{i}: mentions {m.group(0)} but the notebook uses {sorted(ep_vals)}: {l.strip()[:80]}')
        if lr_vals:
            for m in re.finditer(r'\b(?:lr|learning rate)\s*(?:of|=|:)?\s*`?([\d.]+(?:e-?\d+)?)', l, re.I):
                v = _f(m.group(1))
                if v and v not in lr_vals and v not in all_vals and not paperish.search(l):
                    issues += 1
                    R.warn(4, f'{f}:{i}: mentions lr {m.group(1)} but the notebook uses {sorted(lr_vals)}: {l.strip()[:80]}')
    if not issues:
        R.ok(4, 'epochs / lr mentioned in docs agree with the notebook (best effort)')


# ----------------------------------------------------------------------------- network

def http_get(url, timeout=25, maxbytes=400_000):
    req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': 'text/html,application/xhtml+xml,'
                                               'application/xml;q=0.9,image/*,*/*;q=0.8',
                                               'Accept-Language': 'en-US,en;q=0.9'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.geturl(), r.read(maxbytes).decode('utf-8', errors='ignore')
    except urllib.error.HTTPError as e:
        try:
            body = e.read(maxbytes).decode('utf-8', errors='ignore')
        except Exception:
            body = ''
        return e.code, url, body
    except Exception as e:
        return None, url, f'{type(e).__name__}: {e}'


def check_links(R, root, folder, readme_info, nb):
    urls = {}
    for fname in ['README.md', 'TALK.md', 'CODE_ARCHITECTURE.md']:
        p = root / folder / fname
        if p.exists():
            for ln, u in extract_links(read(p)):
                if u.startswith('http'):
                    urls.setdefault(u, []).append(f'{fname}:{ln}')
    if nb:
        for i, c in enumerate(nb['cells']):
            if c['cell_type'] == 'markdown':
                for ln, u in extract_links(cell_src(c)):
                    if u.startswith('http'):
                        urls.setdefault(u, []).append(f'solution.ipynb cell {i}')
    good = 0
    slug = (readme_info or {}).get('tsv_slug')
    if slug:
        urls.setdefault(f'https://www.kaggle.com/code/{KAGGLE_USER}/{slug}', []).append('papers.tsv (root README link)')
    n_checked = 0
    for u, where in sorted(urls.items()):
        w = ', '.join(where[:3])
        if 'kaggle.com/static/images/open-in-kaggle.svg' in u:
            R.info(3, f'Kaggle badge image not fetched (Kaggle serves bots a reCAPTCHA; it renders via GitHub camo)')
            continue
        n_checked += 1
        target, note = u, ''
        m = re.match(r'https://nbviewer\.org/github/([^/]+)/([^/]+)/blob/(.+)$', u)
        if m:
            target = f'https://github.com/{m.group(1)}/{m.group(2)}/blob/{m.group(3)}'
            note = ' (nbviewer blocks bots; checked the GitHub blob instead)'
        st, final, body = http_get(target)
        if st == 200 and re.search(r'Blocking suspicious activity|g-recaptcha|captcha-delivery', body[:5000]):
            R.warn(3, f'{u} -> 200 but a bot/captcha page: verify by hand [{w}]')
        elif st == 200:
            pf, po = urllib.parse.urlparse(final), urllib.parse.urlparse(target)
            if pf.path in ('', '/') and po.path not in ('', '/'):
                R.fail(3, f'{u} redirects to a homepage ({final}): the page is gone [{w}]')
            else:
                good += 1
                if note:
                    R.info(3, f'{u}: 200{note}')
        elif st in (401, 403, 429, 999):
            R.warn(3, f'{u} -> {st} (blocked for bots, NOT proof it works): open it in a browser [{w}]')
        elif st is None:
            R.fail(3, f'{u} -> no response ({body[:60]}) [{w}]')
        else:
            R.fail(3, f'{u} -> HTTP {st} [{w}]')
    (R.ok if good == n_checked else R.info)(3, f'{good}/{n_checked} external links returned 200 with GET + browser User-Agent')
    # arXiv facts
    ax = (readme_info or {}).get('arxiv')
    if ax:
        st, _, body = http_get(f'https://export.arxiv.org/api/query?id_list={ax}', timeout=30)
        if st != 200:
            R.warn(1, f'arXiv API not reachable ({st}); check authors/year by hand')
            return
        try:
            ns = {'a': 'http://www.w3.org/2005/Atom'}
            e = ET.fromstring(body).find('a:entry', ns)
            title = ' '.join(e.find('a:title', ns).text.split())
            pub = e.find('a:published', ns).text[:4]
            authors = [a.find('a:name', ns).text for a in e.findall('a:author', ns)]
        except Exception as ex:
            R.warn(1, f'could not parse arXiv API answer: {ex}')
            return
        line = readme_info.get('authors', '')
        norm = lambda s: re.sub(r'[^a-z]', '', s.lower().translate(str.maketrans('áàäéèëíïóöúüçłńšž', 'aaaeeeiioouuclnsz')))
        missing = [a for a in authors if norm(a.split()[-1]) not in norm(line)]
        if missing:
            R.warn(1, f'README Authors may be missing/misspelt vs arXiv: {", ".join(missing)}')
        else:
            R.ok(1, f'all {len(authors)} arXiv authors appear in the Authors line')
        body_txt = readme_info.get('body', '')
        if norm(title)[:30] not in norm(body_txt):
            R.warn(1, f'arXiv title "{title}" not found in README; check the Paper line')
        ty = (readme_info or {}).get('year', '')
        if ty and ty != pub:
            R.warn(1, f'arXiv {ax} first version is {pub} but papers.tsv year is {ty} (year = first public release)')
        else:
            R.ok(1, f'arXiv {ax}: first version {pub} matches papers.tsv year')


def check_pages(R, folder, num, media_row, root):
    base = f'{SITE}/{folder}/'
    pages = [('paper page', base), ('TALK page', base + 'TALK.html'), ('CODE_ARCHITECTURE page', base + 'CODE_ARCHITECTURE.html')]
    if media_row and media_row.get('status', '').strip().lower() in ('private', 'public'):
        pages.append(('banner', f'{SITE}/{media_row["banner"].strip()}'))
    for label, url in pages:
        st, final, body = http_get(url, maxbytes=2_000_000)
        if st != 200:
            R.fail(7, f'{label} {url} -> {st} (push first; Pages needs ~1 min to rebuild)')
            continue
        R.ok(7, f'{label} 200: {url}')
        if label.endswith('page'):
            fname = {'paper page': 'README.md', 'TALK page': 'TALK.md', 'CODE_ARCHITECTURE page': 'CODE_ARCHITECTURE.md'}[label]
            _, mdbody, _ = split_front_matter(read(root / folder / fname))
            n_md = sum(1 for k, _, _ in md_blocks(mdbody) if k == 'table')
            n_html = len(re.findall(r'<table\b', body))
            if n_html > n_md:
                R.fail(7, f'{label}: {n_html} <table> in HTML but {n_md} tables in {fname}: stray table from a bare |?')
            if '$$' in mdbody and 'katex' not in body.lower() and 'mathjax' not in body.lower():
                R.warn(7, f'{label}: $$ math in source; confirm formulas render (no math engine on the site)')
            sample = next((b for k, _, b in md_blocks(mdbody) if k == 'para' and words(' '.join(b)) > 20), None)
            if sample:
                plain = re.sub(r'!?\[([^\]]*)\]\([^)]*\)', r'\1', ' '.join(sample))
                plain = re.sub(r'[*_`]', '', plain)
                words_ = re.findall(r"[A-Za-z]{5,}", plain)[:8]
                import html as _html
                txt = _html.unescape(re.sub(r'<[^>]+>', ' ', body))
                if words_ and not all(w in txt for w in words_):
                    R.warn(7, f'{label}: first paragraph not found in live HTML; the site may be stale')


def check_kaggle(R, root, folder, paper, nb):
    if not os.environ.get('KAGGLE_API_TOKEN'):
        R.skip(6, 'KAGGLE_API_TOKEN not set: Kaggle checks skipped (never use the legacy kaggle.json key)')
        return
    kag = shutil.which('kaggle')
    if not kag:
        R.warn(6, 'kaggle CLI not on PATH (pip install --upgrade kaggle in a venv); Kaggle checks skipped')
        return
    slug = paper.get('badge_slug') or paper.get('kaggle', '').strip()
    ref = f'{KAGGLE_USER}/{slug}'
    env = dict(os.environ)
    env.pop('KAGGLE_USERNAME', None)
    env.pop('KAGGLE_KEY', None)

    def run(args, timeout=180):
        r = subprocess.run([kag] + args, capture_output=True, text=True, timeout=timeout, env=env)
        return r.returncode, redact(r.stdout + r.stderr)

    try:
        ver = run(['--version'], 60)[1].strip()
        R.info(6, f'{ver}; auth via KAGGLE_API_TOKEN env var (value not shown)')
        rc, out = run(['kernels', 'status', ref])
        if 'COMPLETE' in out:
            R.ok(6, f'{ref}: status COMPLETE')
        else:
            R.fail(6, f'{ref}: status is not COMPLETE: {out.strip()[:120]}')
        tmp = Path(tempfile.mkdtemp(prefix='kag_'))
        try:
            rc, out = run(['kernels', 'pull', ref, '-p', str(tmp), '-m'])
            ipynbs = list(tmp.glob('*.ipynb'))
            if rc != 0 or not ipynbs:
                R.fail(6, f'kernels pull failed: {out.strip()[:150]}')
                return
            meta = json.loads((tmp / 'kernel-metadata.json').read_text()) if (tmp / 'kernel-metadata.json').exists() else {}
            if meta.get('is_private') is False:
                R.ok(6, 'kernel is public')
            else:
                R.fail(6, f'kernel is_private={meta.get("is_private")}: must be public')
            (R.ok(6, 'internet enabled') if meta.get('enable_internet') else R.fail(6, 'enable_internet is off'))
            R.info(6, f'enable_gpu={meta.get("enable_gpu")} machine_shape={meta.get("machine_shape")}: '
                      'GPU only when the notebook needs it')
            kn = json.loads(ipynbs[0].read_text())
            gc = [(c['cell_type'], cell_src(c)) for c in (nb or {}).get('cells', [])]
            kc = [(c['cell_type'], cell_src(c)) for c in kn.get('cells', [])]
            if gc == kc:
                R.ok(6, f'Kaggle source identical to GitHub solution.ipynb ({len(kc)} cells)')
            else:
                diff = next((i for i in range(min(len(gc), len(kc))) if gc[i] != kc[i]), min(len(gc), len(kc)))
                R.fail(6, f'Kaggle source differs from GitHub: {len(kc)} vs {len(gc)} cells, first difference at cell {diff}'
                          ' (push the same notebook: kaggle kernels push)')
            # run log vs GitHub outputs
            odir = tmp / 'out'
            odir.mkdir()
            rc, out = run(['kernels', 'output', ref, '-p', str(odir), '--file-pattern', r'.*\.log$', '-q'], 300)
            logs = list(odir.glob('*.log'))
            if not logs:
                R.warn(6, 'could not download the Kaggle run log; compare outputs by hand')
                return
            try:
                ev = json.loads(logs[0].read_text())
            except Exception:
                ev = []
            kout = ''.join(e.get('data', '') for e in ev if e.get('stream_name') == 'stdout')
            kerr = ''.join(e.get('data', '') for e in ev if e.get('stream_name') == 'stderr')
            if PROGRESS.search(kerr) or PROGRESS.search(kout):
                R.warn(6, 'Kaggle run log shows download/progress-bar output (silence it so the Kaggle page is clean too)')
            gout = ''
            for c in (nb or {}).get('cells', []):
                for o in c.get('outputs', []):
                    if o.get('output_type') == 'stream' and o.get('name') == 'stdout':
                        t = o.get('text', '')
                        gout += ''.join(t) if isinstance(t, list) else t
            rx = re.compile(r'^\s*([A-Za-z][^\n:=]{2,60}?)\s*[:=]\s*([-+]?\d+(?:\.\d+)?)\s*(%|\b)', re.M)
            kd = {m.group(1).strip(): m.group(2) for m in rx.finditer(kout)}
            gd = {m.group(1).strip(): m.group(2) for m in rx.finditer(gout)}
            common = [k for k in gd if k in kd]
            diffs = [f'{k}: GitHub {gd[k]} vs Kaggle {kd[k]}' for k in common if gd[k] != kd[k]]
            dev_g = re.search(r'Device:\s*(\w+)', gout)
            dev_k = re.search(r'Device:\s*(\w+)', kout)
            if dev_g and dev_k and dev_g.group(1) != dev_k.group(1):
                R.warn(6, f'GitHub outputs come from a {dev_g.group(1)} run, Kaggle ran on {dev_k.group(1)}: '
                          'not the same run; numbers must not contradict')
            if diffs:
                R.warn(6, f'{len(diffs)} of {len(common)} logged values differ between GitHub outputs and the Kaggle run '
                          '(make sure docs quote numbers both runs support): ' + '; '.join(diffs[-6:]))
            elif common:
                R.ok(6, f'{len(common)} logged values identical between GitHub outputs and the Kaggle run')
            else:
                R.info(6, 'no comparable "name: value" lines found between the run log and GitHub outputs')
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    except subprocess.TimeoutExpired:
        R.warn(6, 'kaggle CLI timed out')


# ----------------------------------------------------------------------------- main

def main():
    global SECRET
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('repo_root')
    ap.add_argument('num', help='paper number, e.g. 032')
    ap.add_argument('--links', action='store_true')
    ap.add_argument('--pages', action='store_true')
    ap.add_argument('--kaggle', action='store_true')
    ap.add_argument('--strict', action='store_true')
    ap.add_argument('--today', help='override today (YYYY-MM-DD, IST) for release-date logic')
    ap.add_argument('--now-hhmm', help='override current IST time HH:MM')
    ap.add_argument('--json', action='store_true', help='also print the results as JSON')
    a = ap.parse_args()
    SECRET = os.environ.get('KAGGLE_API_TOKEN') or None
    root = Path(a.repo_root).resolve()
    num = a.num.zfill(3)
    now = dt.datetime.now(IST)
    today = dt.date.fromisoformat(a.today) if a.today else now.date()
    hhmm = a.now_hhmm or now.strftime('%H:%M')
    R = Report()

    ptsv = root / 'scripts' / 'papers.tsv'
    if not ptsv.exists():
        print(f'FAIL: {ptsv} not found: is {root} the repo root?')
        return 2
    rows = [r for r in load_tsv(ptsv) if r['num'] == num]
    folders = sorted(p.name for p in root.glob(f'{num}_*') if p.is_dir())
    if not rows:
        R.fail(0, f'{num} is not in scripts/papers.tsv (add a row, then run scripts/build_readme.py)')
        paper = {'num': num, 'short': '?', 'year': '', 'kaggle': '', 'folder': folders[0] if folders else ''}
    else:
        paper = rows[0]
        R.ok(0, f'papers.tsv row: {paper["short"]} ({paper["year"]}), Kaggle slug {paper["kaggle"]}')
    if len(folders) != 1:
        R.fail(0, f'expected exactly one {num}_* folder, found {folders}')
        return finish(R, a, num)
    folder = folders[0]
    if rows and paper.get('folder') != folder:
        R.fail(0, f'papers.tsv folder {paper.get("folder")} != {folder}')

    check_files(R, root, folder, num)
    check_hygiene(R, root, folder)
    check_front_matter(R, root, folder, num, paper['short'])
    media = media_state(root, num, today, hhmm)
    info = check_readme(R, root, folder, paper, media, today, hhmm)
    rm = root / 'scripts' / 'readme_media.py'
    if rm.exists() and media:
        r = subprocess.run([sys.executable, str(rm), '--check', num], capture_output=True, text=True, cwd=root)
        if 'would update' in r.stdout:
            R.fail(1, f'scripts/readme_media.py --check {num} would change README: run it')
        elif r.returncode != 0:
            R.warn(1, f'scripts/readme_media.py --check failed: {(r.stdout + r.stderr).strip()[:120]}')
        else:
            R.ok(1, 'scripts/readme_media.py --check: README already up to date')
    check_placeholders(R, root, folder)
    check_pipes(R, root, folder)
    check_prose(R, root, folder, paper)
    check_talk(R, root, folder, paper, (info or {}).get('arxiv'))
    check_relative_links(R, root, folder)
    nb, size = nb_load(R, root, folder)
    check_notebook(R, root, folder, nb, size)
    check_hparams(R, root, folder, nb)
    if a.links:
        check_links(R, root, folder, info, nb)
    else:
        R.skip(3, 'external links not fetched (run with --links)')
    if a.pages:
        check_pages(R, folder, num, media, root)
    else:
        R.skip(7, 'GitHub Pages not checked (run with --pages after pushing)')
    if a.kaggle:
        check_kaggle(R, root, folder, paper, nb)
    else:
        R.skip(6, 'Kaggle not checked (run with --kaggle; needs KAGGLE_API_TOKEN)')
    return finish(R, a, f'{num} {paper.get("short", "")}')


def finish(R, a, label):
    order = {'FAIL': 0, 'WARN': 1, 'PASS': 2, 'INFO': 3, 'SKIP': 4}
    print(f'check_paper.py  {label}')
    print('=' * 78)
    for g in sorted(GROUPS):
        its = [x for x in R.items if x[1] == g]
        if not its:
            continue
        print(f'\n[{g}] {GROUPS[g]}')
        for lvl, _, msg in sorted(its, key=lambda x: order[x[0]]):
            print(f'  {lvl:4}  {redact(msg)}')
    f, w, p = R.count('FAIL'), R.count('WARN'), R.count('PASS')
    verdict = 'FAIL' if f or (a.strict and w) else 'PASS'
    print('\n' + '=' * 78)
    print(f'RESULT: {verdict}   PASS {p}   FAIL {f}   WARN {w}   SKIP {R.count("SKIP")}')
    print('Automated checks only. Also tick checklists/pre-commit-checklist.md before committing.')
    if a.json:
        print(json.dumps([{'level': l, 'group': GROUPS[g], 'msg': redact(m)} for l, g, m in R.items], indent=1))
    return 1 if verdict == 'FAIL' else 0


if __name__ == '__main__':
    sys.exit(main())
