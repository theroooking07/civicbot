"""
change_detect.py - what changed on the portal since the previous crawl?

Compares the content_hash of every page in data/raw/pages.jsonl with the
baseline saved by make_snapshot.py in data/snapshot/page_hashes.csv:
  new      - URL not in the baseline
  changed  - same URL, different content hash (with a short text diff)
  removed  - in the baseline but not found in this crawl
             (may just mean the crawl cap was hit before reaching it)

Output: data/outputs/updates.json  (the app's "What's new" tab)

Usage:
    python -m crawler.change_detect
"""
import difflib
import json
import sys
from datetime import datetime, timezone

import pandas as pd

import config

PAGES_IN = config.RAW / "pages.jsonl"
BASELINE = config.SNAPSHOT / "page_hashes.csv"
BASELINE_TEXT = config.SNAPSHOT / "pages_clean.jsonl"     # for diffs (optional)
CURRENT_TEXT = config.CLEAN / "pages_clean.jsonl"
UPDATES_OUT = config.OUTPUTS / "updates.json"
MAX_DIFF_LINES = 12


def load_texts(path) -> dict:
    if not path.exists():
        return {}
    return {(r := json.loads(l))["url"]: r["text_full"] for l in open(path, encoding="utf-8")}


def short_diff(old: str, new: str) -> dict:
    """Added/removed lines between two versions of a page (first few only)."""
    diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0))
    added = [l[1:] for l in diff if l.startswith("+") and not l.startswith("+++")]
    removed = [l[1:] for l in diff if l.startswith("-") and not l.startswith("---")]
    return {"lines_added": len(added), "lines_removed": len(removed),
            "added_sample": added[:MAX_DIFF_LINES], "removed_sample": removed[:MAX_DIFF_LINES]}


def main():
    if not PAGES_IN.exists():
        sys.exit(f"Missing {PAGES_IN} - run  python -m crawler.run_crawl  first.")
    current = pd.DataFrame([json.loads(l) for l in open(PAGES_IN, encoding="utf-8")])
    current = current[["url", "title", "content_hash", "fetched_at"]]

    result = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "current_crawl": current.fetched_at.min(), "baseline_crawl": None,
              "first_run": False, "new": [], "changed": [], "removed": [], "unchanged": 0}

    if not BASELINE.exists():
        result["first_run"] = True
        result["message"] = ("No baseline yet - this is the first crawl. Run  python make_snapshot.py  "
                             "to save it; the next crawl will be compared against it.")
        UPDATES_OUT.write_text(json.dumps(result, indent=2))
        print(f"[changes] {result['message']}")
        print(f"[changes] saved {UPDATES_OUT}")
        return

    base = pd.read_csv(BASELINE)
    result["baseline_crawl"] = base.fetched_at.min()
    old_text, new_text = load_texts(BASELINE_TEXT), load_texts(CURRENT_TEXT)
    merged = current.merge(base, on="url", how="outer", suffixes=("", "_old"), indicator=True)

    for _, r in merged.iterrows():
        if r["_merge"] == "left_only":
            result["new"].append({"url": r.url, "title": r.title, "fetched_at": r.fetched_at})
        elif r["_merge"] == "right_only":
            result["removed"].append({"url": r.url, "title": r.title_old})
        elif r.content_hash != r.content_hash_old:
            item = {"url": r.url, "title": r.title, "fetched_at": r.fetched_at,
                    "previous_fetch": r.fetched_at_old}
            if r.url in old_text and r.url in new_text:
                item.update(short_diff(old_text[r.url], new_text[r.url]))
            result["changed"].append(item)
        else:
            result["unchanged"] += 1

    UPDATES_OUT.write_text(json.dumps(result, indent=2))
    print(f"[changes] baseline {result['baseline_crawl']} -> current {result['current_crawl']}")
    print(f"[changes] new {len(result['new'])} | changed {len(result['changed'])} | "
          f"removed/not reached {len(result['removed'])} | unchanged {result['unchanged']}")
    for c in result["changed"][:10]:
        print(f"   changed: {c['url']}  (+{c.get('lines_added', '?')} / -{c.get('lines_removed', '?')} lines)")
    for n in result["new"][:10]:
        print(f"   new:     {n['url']}")
    print(f"[changes] saved {UPDATES_OUT}")


if __name__ == "__main__":
    main()
