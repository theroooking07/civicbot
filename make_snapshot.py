"""
make_snapshot.py - save the current outputs to data/snapshot/ for demo mode.

The snapshot holds everything the Streamlit app needs, so the demo runs with
no network and no API key. It is also the baseline that change_detect.py
compares the next crawl against.

Kept small on purpose (it is committed to git):
  * pages.jsonl without the page text (url, title, depth, status, hash...)
  * raw visible text capped at RAW_WORDS_CAP words per page (raw vs clean view)
  * no raw HTML and no links.csv

Usage:
    python make_snapshot.py
"""
import json
import shutil
import sys

import pandas as pd

import config

RAW_WORDS_CAP = 1500

# (source file, name inside data/snapshot/)
COPY = [
    (config.OUTPUTS / "robots_report.json", None),
    (config.RAW / "crawl_report.json", None),
    (config.RAW / "documents.csv", None),
    (config.CLEAN / "pages_clean.jsonl", None),
    (config.CLEAN / "entities.csv", None),
    (config.CLEAN / "page_entities.csv", None),
    (config.OUTPUTS / "clusters.json", None),
    (config.OUTPUTS / "page_clusters.csv", None),
    (config.OUTPUTS / "link_report.json", None),
    (config.OUTPUTS / "link_metrics.csv", None),
    (config.OUTPUTS / "upcoming_deadlines.csv", None),
    (config.OUTPUTS / "eval_report.json", None),
    (config.OUTPUTS / "eval_results.csv", None),
    (config.OUTPUTS / "updates.json", None),
    (config.OUTPUTS / "index" / "faiss.index", "index/faiss.index"),
    (config.OUTPUTS / "index" / "chunks.jsonl", "index/chunks.jsonl"),
]
CHARTS = ["link_graph", "cluster_scatter", "silhouette", "cluster_sizes",
          "cluster_topic_heatmap", "deadline_timeline", "eval_recall"]


def main():
    pages_path = config.RAW / "pages.jsonl"
    needed = [pages_path] + [src for src, _ in COPY]
    absent = [str(p.relative_to(config.ROOT)) for p in needed if not p.exists()]
    if absent:
        sys.exit("Missing outputs - run the pipeline first:\n  " + "\n  ".join(absent))

    snap = config.SNAPSHOT
    (snap / "index").mkdir(parents=True, exist_ok=True)

    for src, name in COPY:
        shutil.copy2(src, snap / (name or src.name))
    for c in CHARTS:
        if (config.OUTPUTS / f"{c}.png").exists():
            shutil.copy2(config.OUTPUTS / f"{c}.png", snap / f"{c}.png")

    # Page records without text, plus the hash baseline for change detection.
    pages = [json.loads(l) for l in open(pages_path, encoding="utf-8")]
    with open(snap / "pages.jsonl", "w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps({k: v for k, v in p.items() if k not in ("text", "html_file")},
                               ensure_ascii=False) + "\n")
    pd.DataFrame(pages)[["url", "title", "content_hash", "fetched_at"]].to_csv(
        snap / "page_hashes.csv", index=False)

    # Raw visible text (menus included), capped, for the raw-vs-clean view.
    from preprocess.clean import raw_visible_text
    with open(snap / "raw_text.jsonl", "w", encoding="utf-8") as f:
        for p in pages:
            html_path = config.ROOT / p.get("html_file", "")
            raw = raw_visible_text(html_path.read_text(encoding="utf-8")) if html_path.is_file() else p["text"]
            words = raw.split(" ")
            if len(words) > RAW_WORDS_CAP:
                raw = " ".join(words[:RAW_WORDS_CAP]) + " [...]"
            f.write(json.dumps({"url": p["url"], "raw_text": raw}, ensure_ascii=False) + "\n")

    files = [p for p in snap.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files) / 1e6
    print(f"[snapshot] saved {len(files)} files ({size:.1f} MB) to {snap}")
    print("[snapshot] the app's 'Demo snapshot mode' now works offline, and the next crawl's "
          "change detection will compare against this baseline.")


if __name__ == "__main__":
    main()
