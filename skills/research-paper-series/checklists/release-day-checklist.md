# Release-day checklist (one Short per day, ~6:30 PM IST)

Schedule: Google Sheet "Video Release Order"
(https://docs.google.com/spreadsheets/d/1EzTm-kSGP1Y1GIW7nqKD6-5moKDmAPHPCAJZitdSFVg). Columns used here:
Release Day (day number, 1 = first day), Sr (paper number), Paper Title, Build Status, YouTube Link,
YouTube Status (e.g. "Uploaded (Private)", "Published (Public) 2026-10-18"). The calendar date lives in
`scripts/media.tsv` (`release_date`), not in the sheet. A paper must fully pass BEFORE its release day.

## The day before (D-1)
- [ ] Find tomorrow's paper: media.tsv row with `release_date` = tomorrow, and the matching Sheet row (Sr).
- [ ] `python scripts/check_paper.py . NNN --links --kaggle` gives PASS, and the whole pre-commit checklist is ticked.
- [ ] media.tsv row is `status=private` with the right `release_date`, banner `assets/banners/NNN.jpg` exists.
- [ ] `python scripts/readme_media.py NNN --check` reports "already up to date" (README says "YouTube Short coming on <D Mon YYYY>").
- [ ] Never write the private video URL into the README or media.tsv `shorts_url` while it is private (readme_media.py ignores it anyway).
- [ ] Sheet: Build Status = Done, YouTube Link filled, YouTube Status = "Uploaded (Private)".
- [ ] Pushed as nadeemcite; `--pages` passes (live page shows the banner and the "coming on" line).

## Release day (D, after the Short is public ~6:30 PM IST)
- [ ] Confirm the Short is public and plays in a logged-out browser: https://youtube.com/shorts/<id>.
- [ ] Edit `scripts/media.tsv`: `status=public`, `shorts_url=https://youtube.com/shorts/<id>` (keep release_date).
- [ ] `python scripts/readme_media.py NNN` (only that paper). `git diff` shows exactly 1 changed line in README.md.
- [ ] `python scripts/check_paper.py . NNN --links` gives PASS (Video line in state 3, URL returns 200).
- [ ] Commit ("NNN: Short is public") and push as nadeemcite. No force-push.
- [ ] Sheet: YouTube Status = "Published (Public) YYYY-MM-DD".
- [ ] After ~1 min: `check_paper.py . NNN --pages` gives PASS (live page shows "Watch the YouTube Short").
- [ ] If the root README "Watch: Backprop Daily" block should feature the new Short, change it in `scripts/build_readme.py` (the root README is generated: never edit README.md at the root by hand), then rerun it.

## After going public (D+1)
- [ ] Check tomorrow's paper (the next D-1 list).
- [ ] Re-run `check_paper.py NNN --links` for yesterday's paper: comments and new traffic surface dead links fast.
- [ ] Any fix after release goes through the full pre-commit gate again, and goes to BOTH GitHub and Kaggle (same notebook).
- [ ] The checker WARNs "release day has passed" while a row is still private: clear that WARN for every paper whose release time has passed.
