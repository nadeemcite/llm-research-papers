"""Add / update the banner image and the "- **Video:**" line in paper READMEs, from scripts/media.tsv.

Usage (from the repo root):
    python scripts/readme_media.py                 # apply every row of scripts/media.tsv
    python scripts/readme_media.py 032             # only these papers
    python scripts/readme_media.py --check         # report what would change, write nothing
    python scripts/readme_media.py --tsv other.tsv # alternate data file (.tsv or .json list of rows)

Columns: num, short, banner, status, shorts_url, release_date (YYYY-MM-DD)
    status = public   -> banner + "- **Video:** [Watch the YouTube Short](<shorts_url>) on [Backprop Daily](...)"
    status = private  -> banner + "- **Video:** YouTube Short coming on <18 Oct 2026> on [Backprop Daily](...)"
             (the private video URL is never written, even if present in the file)
    status = none     -> banner and Video line removed (plain paper template)

Only two lines of each README are touched: the banner line directly under the "# title" heading and the
"- **Video:**" line at the end of the metadata list. Everything else (front matter, prose) is left as is.
Re-running is idempotent. To publish a video: set status=public and shorts_url in media.tsv, then re-run.
"""
import argparse
import csv
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANNEL = '[Backprop Daily](https://www.youtube.com/@BackpropDaily)'
BANNER_RE = re.compile(r'^!\[[^\]]*\]\((?:\.\./)?assets/banners/[^)]+\)\s*$')
VIDEO_RE = re.compile(r'^- \*\*Video:\*\* ')
META_RE = re.compile(r'^- \*\*[^*]+:\*\* ')


def load_rows(path):
    path = Path(path)
    if path.suffix == '.json':
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else [dict(num=k, **v) for k, v in data.items()]
    with path.open(newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def fmt_date(iso):
    d = date.fromisoformat(iso)
    return f'{d.day} {d.strftime("%b")} {d.year}'


def wanted_lines(row):
    status = (row.get('status') or '').strip().lower()
    if status in ('', 'none'):
        return None, None
    banner = (row.get('banner') or '').strip()
    if not (ROOT / banner).is_file():
        raise SystemExit(f"{row['num']}: banner file not found: {banner}")
    banner_line = f"![{row['short']} banner](../{banner})"
    if status == 'public':
        url = (row.get('shorts_url') or '').strip()
        if not url:
            raise SystemExit(f"{row['num']}: status=public needs shorts_url")
        video = f'- **Video:** [Watch the YouTube Short]({url}) on {CHANNEL}'
    elif status == 'private':
        video = f"- **Video:** YouTube Short coming on {fmt_date(row['release_date'].strip())} on {CHANNEL}"
    else:
        raise SystemExit(f"{row['num']}: unknown status {status!r}")
    return banner_line, video


def rewrite(text, banner_line, video_line):
    lines = text.split('\n')
    # skip YAML front matter
    start = 0
    if lines and lines[0] == '---':
        start = lines.index('---', 1) + 1
    title = next(i for i in range(start, len(lines)) if lines[i].startswith('# '))
    # banner: the line right after the title
    if title + 1 < len(lines) and BANNER_RE.match(lines[title + 1]):
        if banner_line is None:
            del lines[title + 1]
        else:
            lines[title + 1] = banner_line
    elif banner_line is not None:
        lines.insert(title + 1, banner_line)
    # metadata list: first run of "- **X:** " lines before the first "## " heading
    first_h2 = next((i for i in range(title, len(lines)) if lines[i].startswith('## ')), len(lines))
    meta = [i for i in range(title, first_h2) if META_RE.match(lines[i])]
    if not meta:
        raise ValueError('no metadata list found')
    vid = [i for i in meta if VIDEO_RE.match(lines[i])]
    if vid:
        if video_line is None:
            del lines[vid[0]]
        else:
            lines[vid[0]] = video_line
    elif video_line is not None:
        lines.insert(meta[-1] + 1, video_line)
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('nums', nargs='*', help='paper numbers to process (default: all rows)')
    ap.add_argument('--tsv', default=ROOT / 'scripts' / 'media.tsv')
    ap.add_argument('--check', action='store_true', help='dry run')
    args = ap.parse_args()
    rows = [r for r in load_rows(args.tsv) if not args.nums or r['num'] in args.nums]
    changed = 0
    for row in rows:
        folders = sorted(ROOT.glob(f"{row['num']}_*/README.md"))
        if len(folders) != 1:
            raise SystemExit(f"{row['num']}: expected one paper folder, found {len(folders)}")
        path = folders[0]
        old = path.read_text()
        new = rewrite(old, *wanted_lines(row))
        if new != old:
            changed += 1
            print(f"{'would update' if args.check else 'updated'}: {path.relative_to(ROOT)}")
            if not args.check:
                path.write_text(new)
    print(f'{changed} file(s) {"would change" if args.check else "changed"}, {len(rows) - changed} already up to date')


if __name__ == '__main__':
    sys.exit(main())
