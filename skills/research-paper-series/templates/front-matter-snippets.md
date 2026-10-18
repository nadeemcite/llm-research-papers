# Front matter snippets (Just the Docs nav)

These blocks are WRITTEN BY `python scripts/build_readme.py` from `scripts/papers.tsv`
(columns: num, short, year, domain, title, folder, kaggle). Values are JSON-quoted strings; nav_order is a bare integer.
Prefer adding the papers.tsv row and running the script over typing these by hand. The checker compares them exactly.

README.md (top-level nav entry, ordered by number; nav_order = NNN + 1 because Home is 1):
```yaml
---
title: "032 YOLO"
nav_order: 33
permalink: "/032_you-only-look-once-yolo/"
---
```

CODE_ARCHITECTURE.md (child page):
```yaml
---
title: "YOLO: Code architecture"
parent: "032 YOLO"
nav_order: 1
---
```

TALK.md (child page):
```yaml
---
title: "YOLO: Talk & Q&A"
parent: "032 YOLO"
nav_order: 2
---
```

Notes
- `permalink` is required on README.md: jekyll-readme-index only turns front-matter-less READMEs into the folder index.
- `parent` must equal the README title exactly ("NNN Short").
- Never add per-file `defaults:` in `_config.yml` (≈300 scopes slow the Pages build from ~15 s to ~6 min).
- `.notebook_spec.json` files are excluded from the site by `_config.yml`; nothing else in a paper folder is.
