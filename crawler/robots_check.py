"""
robots_check.py - Step 0 of the pipeline: check we are allowed to crawl.

For every allowed domain (config.SEEDS_BY_BASE) it fetches robots.txt with
urllib.robotparser, prints its rules, tests every seed path against them, and
saves a compliance report to data/outputs/robots_report.json. The crawler
reads that report and refuses to start unless "crawl_allowed" is true.

Usage:
    python -m crawler.robots_check
"""
import json
import sys
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests

import config


def fetch_robots(base_url: str) -> tuple[str, int]:
    """Download robots.txt and return (text, http_status)."""
    url = urljoin(base_url, "/robots.txt")
    resp = requests.get(url, headers={"User-Agent": config.USER_AGENT}, timeout=20)
    return resp.text, resp.status_code


def check_domain(base_url: str, seed_paths: list[str]) -> dict:
    """Parse one domain's robots.txt, test its seed paths and return a report dict."""
    robots_url = urljoin(base_url, "/robots.txt")
    print(f"\n[robots] Fetching {robots_url}")
    text, status = fetch_robots(base_url)

    parser = RobotFileParser()
    if status == 404:
        # No robots.txt means no restrictions (standard behaviour).
        parser.parse([])
    elif status >= 400:
        # Any other error: be conservative and treat the site as off-limits.
        print(f"[robots] HTTP {status} - treating site as NOT allowed.")
        parser.parse(["User-agent: *", "Disallow: /"])
    else:
        parser.parse(text.splitlines())

    print("----- robots.txt -----")
    print(text.strip() if status < 400 else f"(HTTP {status})")
    print("----------------------")

    results = {}
    for path in seed_paths:
        url = urljoin(base_url, path)
        ok = parser.can_fetch(config.USER_AGENT, url)
        results[path] = ok
        print(f"  {'ALLOWED   ' if ok else 'DISALLOWED'}  {path}")

    delay = parser.crawl_delay(config.USER_AGENT)
    sitemaps = parser.site_maps() or []
    print(f"Sitemaps: {sitemaps or 'none listed'}")
    print(f"Crawl-delay in robots.txt: {delay}")
    return {
        "domain": urlparse(base_url).netloc,
        "robots_url": robots_url,
        "http_status": status,
        "seed_results": results,
        "crawl_delay_from_robots": delay,
        "sitemaps": sitemaps,
        "crawl_allowed": all(results.values()),
        "robots_txt": text if status < 400 else "",
    }


def check() -> dict:
    """Check every allowed domain and save the combined report."""
    domains = [check_domain(base, paths) for base, paths in config.SEEDS_BY_BASE.items()]
    # Use the strictest crawl-delay any robots.txt asks for (never below our own).
    max_delay = max([config.DOWNLOAD_DELAY] + [d["crawl_delay_from_robots"] or 0 for d in domains])
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "user_agent": config.USER_AGENT,
        "domains": domains,
        "effective_delay": max_delay,
        "crawl_allowed": all(d["crawl_allowed"] for d in domains),
    }

    out = config.OUTPUTS / "robots_report.json"
    out.write_text(json.dumps(report, indent=2))
    print("\n========== summary ==========")
    for d in domains:
        print(f"  {d['domain']:<32} allowed={d['crawl_allowed']}")
    print(f"Effective delay: {max_delay}s")
    print(f"Crawl allowed for all seed paths: {report['crawl_allowed']}")
    print(f"Report saved to {out}")
    return report


if __name__ == "__main__":
    try:
        rep = check()
    except requests.RequestException as exc:
        print(f"[robots] Could not reach the portal: {exc}")
        sys.exit(2)
    sys.exit(0 if rep["crawl_allowed"] else 1)
