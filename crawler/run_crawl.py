"""
run_crawl.py - start a crawl, but only if the robots.txt check allows it.

Usage:
    python -m crawler.run_crawl                  # uses MAX_PAGES / MAX_DEPTH from config
    python -m crawler.run_crawl --max-pages 20   # small test crawl

Outputs (data/raw/): pages.jsonl, links.csv, documents.csv, crawl_report.json
"""
import argparse
import json
import sys
from datetime import datetime, timedelta, timezone

from scrapy.crawler import CrawlerProcess

import config
from crawler import robots_check
from crawler.civic_spider import CivicSpider
from crawler.settings import get_settings

REPORT = config.OUTPUTS / "robots_report.json"


def robots_ok() -> bool:
    """Re-run the robots check if the saved report is missing or older than a day."""
    report = None
    if REPORT.exists():
        report = json.loads(REPORT.read_text())
        age = datetime.now(timezone.utc) - datetime.fromisoformat(report["checked_at"])
        checked = {d["domain"] for d in report.get("domains", [])}
        if age > timedelta(days=1) or not set(config.ALLOWED_DOMAINS) <= checked:
            report = None  # stale, old format, or a domain was added since
    if report is None:
        print("[crawl] No fresh robots report - running robots check first.")
        report = robots_check.check()
    return bool(report.get("crawl_allowed"))


def main():
    ap = argparse.ArgumentParser(description="Polite crawl of the configured municipal portal.")
    ap.add_argument("--max-pages", type=int, default=config.MAX_PAGES)
    ap.add_argument("--max-depth", type=int, default=config.MAX_DEPTH)
    args = ap.parse_args()

    if args.max_pages > config.MAX_PAGES:
        sys.exit(f"[crawl] --max-pages {args.max_pages} exceeds the configured cap {config.MAX_PAGES}.")
    if not robots_ok():
        sys.exit("[crawl] robots.txt does not allow crawling the seed paths. Refusing to start.")

    print(f"[crawl] Target {config.BASE_URL}  (domains {', '.join(config.ALLOWED_DOMAINS)})")
    print(f"[crawl] max_pages={args.max_pages}  max_depth={args.max_depth}  "
          f"delay={config.DOWNLOAD_DELAY}s  concurrency=1  UA='{config.USER_AGENT}'")
    if "example.com" in config.CONTACT_EMAIL:
        print("[crawl] NOTE: set CIVICBOT_CONTACT_EMAIL in .env so the site owner can reach you.")

    process = CrawlerProcess(get_settings(args.max_pages, args.max_depth))
    process.crawl(CivicSpider)
    process.start()

    report_path = config.RAW / "crawl_report.json"
    if not report_path.exists():
        sys.exit("[crawl] Crawl ended without a report - check the errors above.")
    rep = json.loads(report_path.read_text())
    print("\n========== crawl report ==========")
    for key in ("finished_reason", "pages_fetched", "pages_saved", "skipped_by_robots", "skipped",
                "errors", "pdf_links", "document_links", "average_depth", "duration_s"):
        print(f"  {key:<18} {rep[key]}")
    print("  top linked pages:")
    for row in rep["top10_most_linked"]:
        print(f"    {row['in_links']:>4}  {row['url']}")


if __name__ == "__main__":
    main()
