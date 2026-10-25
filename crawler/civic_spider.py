"""
civic_spider.py - the Scrapy spider that crawls the municipal portal.

What it does for every HTML page it fetches:
  * saves url, status, depth, title, headings, main text, fetched_at and a
    content hash to data/raw/pages.jsonl (plus the raw HTML in data/raw/html/
    so preprocessing can show "raw vs cleaned" side by side)
  * saves every hyperlink to data/raw/links.csv
  * records PDF / Office document links in data/raw/documents.csv
  * follows internal links only, service-related links first

It never follows login, account, payment, form or search-result links
(see config.EXCLUDE_SEGMENTS etc.), stays on config.ALLOWED_DOMAINS, and
stops at exactly MAX_PAGES fetched pages.
"""
import csv
import hashlib
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import PurePosixPath
from urllib.parse import urljoin, urlparse

import scrapy
import trafilatura
from bs4 import BeautifulSoup
from scrapy.exceptions import CloseSpider
from w3lib.url import canonicalize_url

import config

DOC_EXTENSIONS = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".odt", ".ods", ".csv", ".ppt", ".pptx", ".rtf"}

# "[PDF, 34Kb]" / "(PDF 1.2 MB)" - gov.uk sites put the file size in the link text.
SIZE_IN_TEXT = re.compile(r"(\d+(?:\.\d+)?)\s*(kb|mb|gb|bytes)\b", re.I)

# A page with fewer words than this AND almost no links is probably
# JavaScript-rendered (empty shell over plain HTTP) - we log and skip it.
MIN_WORDS = 20


def normalise_url(base: str, href: str) -> str | None:
    """Turn an href into an absolute, canonical URL (no #fragment). None if not http(s)."""
    href = (href or "").strip()
    if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
        return None  # mailto links are skipped on purpose: we collect no email addresses
    url = urljoin(base, href)
    if urlparse(url).scheme not in ("http", "https"):
        return None
    return canonicalize_url(url, keep_fragments=False)


def short_url(url: str) -> str:
    """Path plus query string, for compact log lines."""
    parsed = urlparse(url)
    return parsed.path + (f"?{parsed.query}" if parsed.query else "")


def is_excluded(url: str) -> bool:
    """True for login / account / payment / form / search pages we must not crawl.

    Checks each path segment against config.EXCLUDE_SEGMENTS / _PREFIXES /
    _SUFFIXES, and the query string against EXCLUDE_QUERY_KEYS.
    """
    parsed = urlparse(url.lower())
    for seg in filter(None, parsed.path.split("/")):
        if (seg in config.EXCLUDE_SEGMENTS
                or seg.startswith(config.EXCLUDE_PREFIXES)
                or seg.endswith(config.EXCLUDE_SUFFIXES)):
            return True
    query_keys = {kv.split("=")[0] for kv in parsed.query.split("&") if kv}
    return bool(query_keys & config.EXCLUDE_QUERY_KEYS)


def target_type(url: str) -> str:
    """Classify a link target as html, pdf, doc or external."""
    parsed = urlparse(url)
    ext = PurePosixPath(parsed.path).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in DOC_EXTENSIONS:
        return "doc"
    return "html" if parsed.netloc in config.ALLOWED_DOMAINS else "external"


def priority_for(url: str, anchor: str) -> int:
    """Count topic keywords in the path + anchor text. More hits = crawled sooner."""
    text = (urlparse(url).path + " " + anchor).lower()
    return sum(1 for kw in config.PRIORITY_KEYWORDS if kw in text)


def size_from_text(text: str) -> int | None:
    """Parse a size like '34Kb' from link text into bytes. None if absent."""
    m = SIZE_IN_TEXT.search(text or "")
    if not m:
        return None
    num, unit = float(m.group(1)), m.group(2).lower()
    factor = {"bytes": 1, "kb": 1024, "mb": 1024 ** 2, "gb": 1024 ** 3}[unit]
    return int(num * factor)


def main_text(html: str, soup: BeautifulSoup) -> str:
    """Main readable text: trafilatura first, BeautifulSoup <main> as a fallback."""
    text = trafilatura.extract(html, include_comments=False, include_tables=True, favor_recall=True)
    if text and len(text.split()) >= MIN_WORDS:
        return text
    main = soup.find("main") or soup.find("article") or soup.body
    if main is None:
        return text or ""
    for tag in main.find_all(["script", "style", "nav", "noscript"]):
        tag.decompose()
    fallback = " ".join(main.get_text(" ", strip=True).split())
    return fallback if len(fallback) > len(text or "") else (text or "")


class CivicSpider(scrapy.Spider):
    """Breadth-limited, polite crawler for one municipal portal."""

    name = "civic"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allowed_domains = list(config.ALLOWED_DOMAINS)
        self.start_time = time.time()
        self.pages_fetched = 0          # every HTML response counts towards the cap
        self.pages_saved = 0
        self.depths = []
        self.skipped = Counter()        # reason -> count
        self.errors = []                # (url, reason)
        self.seen_docs = set()
        self.link_in_degree = Counter()

        config.RAW.mkdir(parents=True, exist_ok=True)
        (config.RAW / "html").mkdir(exist_ok=True)
        # Open output files once; each run starts fresh (change_detect keeps history).
        self.pages_f = open(config.RAW / "pages.jsonl", "w", encoding="utf-8")
        self.links_f = open(config.RAW / "links.csv", "w", newline="", encoding="utf-8")
        self.docs_f = open(config.RAW / "documents.csv", "w", newline="", encoding="utf-8")
        self.links_w = csv.writer(self.links_f)
        self.links_w.writerow(["source_url", "target_url", "anchor_text", "is_internal", "target_type"])
        self.docs_w = csv.writer(self.docs_f)
        self.docs_w.writerow(["url", "title", "file_type", "size_bytes", "size_source", "source_url"])

    @property
    def max_pages(self) -> int:
        return self.settings.getint("CIVICBOT_MAX_PAGES", config.MAX_PAGES)

    # ------------------------------------------------------------------ start
    async def start(self):
        """Seed requests (Scrapy >= 2.13 uses async start())."""
        for base, paths in config.SEEDS_BY_BASE.items():
            for path in paths:
                url = canonicalize_url(urljoin(base, path))
                yield scrapy.Request(url, callback=self.parse, errback=self.on_error,
                                     priority=100, meta={"anchor": "(seed)"})

    # ------------------------------------------------------------------ parse
    def parse(self, response):
        """Handle one fetched page: save it, record its links, queue new links."""
        if self.pages_fetched >= self.max_pages:
            raise CloseSpider("page cap reached")

        ctype = response.headers.get("Content-Type", b"").decode().lower()
        if "text/html" not in ctype:
            self.skipped["non_html"] += 1
            print(f"  skip  non-HTML ({ctype or 'unknown'}): {response.url}", flush=True)
            return
        if urlparse(response.url).netloc not in config.ALLOWED_DOMAINS or is_excluded(response.url):
            # e.g. a redirect that landed on a login page or another domain
            self.skipped["redirected_out_of_scope"] += 1
            print(f"  skip  redirected out of scope: {response.url}", flush=True)
            return

        self.pages_fetched += 1
        depth = response.meta.get("depth", 0)
        html = response.text
        soup = BeautifulSoup(html, "lxml")

        # ---- links (recorded even if the page text turns out to be empty) ----
        out_links = []
        for a in soup.find_all("a", href=True):
            target = normalise_url(response.url, a["href"])
            if not target or target == response.url:
                continue
            anchor = " ".join(a.get_text(" ", strip=True).split())[:200]
            ttype = target_type(target)
            internal = urlparse(target).netloc in config.ALLOWED_DOMAINS
            self.links_w.writerow([response.url, target, anchor, internal, ttype])
            if internal:
                self.link_in_degree[target] += 1
            if ttype in ("pdf", "doc"):
                self.record_document(target, anchor, ttype, response.url)
            elif ttype == "html":
                out_links.append((target, anchor))

        # ---- page content ----
        headings = [" ".join(h.get_text(" ", strip=True).split())
                    for h in soup.find_all(["h1", "h2", "h3"])]
        # Title: <title>, else og:title (the waste site has no <title>), else first <h1>.
        title = (soup.title.get_text(strip=True) if soup.title else "").strip()
        if not title:
            og = soup.find("meta", property="og:title")
            h1 = soup.find("h1")
            title = (og.get("content", "").strip() if og else "") or \
                    (" ".join(h1.get_text(" ", strip=True).split()) if h1 else "")
        text = main_text(html, soup)
        n_words = len(text.split())

        if n_words < MIN_WORDS and len(out_links) < 5:
            self.skipped["empty_or_js_rendered"] += 1
            print(f"  skip  [{self.pages_fetched:>3}/{self.max_pages}] no useful content "
                  f"(likely JS-rendered, {n_words} words): {response.url}", flush=True)
        else:
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            html_file = config.RAW / "html" / f"{hashlib.md5(response.url.encode()).hexdigest()}.html"
            html_file.write_text(html, encoding="utf-8")
            record = {
                "url": response.url,
                "status": response.status,
                "depth": depth,
                "title": title,
                "headings": headings,
                "text": text,
                "word_count": n_words,
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "content_hash": content_hash,
                "html_file": str(html_file.relative_to(config.ROOT)),
            }
            self.pages_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.pages_f.flush()
            self.pages_saved += 1
            self.depths.append(depth)
            print(f"  ok    [{self.pages_fetched:>3}/{self.max_pages}] d={depth} {response.status} "
                  f"{n_words:>5}w  {short_url(response.url)}  | {title[:50]}", flush=True)

        if self.pages_fetched >= self.max_pages:
            raise CloseSpider("page cap reached")

        # ---- queue internal HTML links (Scrapy dedupes + enforces DEPTH_LIMIT) ----
        for target, anchor in out_links:
            if is_excluded(target):
                self.skipped["excluded_pattern"] += 1
                continue
            yield scrapy.Request(target, callback=self.parse, errback=self.on_error,
                                 priority=priority_for(target, anchor) * 10 - depth,
                                 meta={"anchor": anchor})

    # ---------------------------------------------------------------- helpers
    def record_document(self, url, anchor, ttype, source_url):
        """Write a PDF/doc link once. Size comes from the link text (no download)."""
        if url in self.seen_docs:
            return
        self.seen_docs.add(url)
        size = size_from_text(anchor)
        title = SIZE_IN_TEXT.sub("", anchor).strip(" []()-,") or PurePosixPath(urlparse(url).path).name
        title = re.sub(r"\s*\[?\b(pdf|docx?|xlsx?|odt|csv)\b,?\s*\]?$", "", title, flags=re.I).strip()
        self.docs_w.writerow([url, title, PurePosixPath(urlparse(url).path).suffix.lstrip(".").lower() or ttype,
                              size if size is not None else "", "link_text" if size else "", source_url])

    def on_error(self, failure):
        """Log and skip failed requests (robots blocks, timeouts, 404s...)."""
        url = failure.request.url
        reason = failure.getErrorMessage() or failure.type.__name__
        if failure.type.__name__ == "IgnoreRequest":
            # Raised by RobotsTxtMiddleware (forbidden by robots.txt) or OffsiteMiddleware.
            self.skipped["ignored_by_middleware"] += 1
            return
        if hasattr(failure.value, "response") and failure.value.response is not None:
            reason = f"HTTP {failure.value.response.status}"
        self.errors.append({"url": url, "reason": reason})
        print(f"  ERROR {reason}: {url}", flush=True)

    # ------------------------------------------------------------------ close
    def closed(self, reason):
        """Close files and write crawl_report.json."""
        self.pages_f.close()
        self.links_f.close()
        self.docs_f.close()
        stats = self.crawler.stats.get_stats()
        report = {
            "finished_reason": reason,
            "base_url": config.BASE_URL,
            "max_pages": self.max_pages,
            "max_depth": self.settings.getint("DEPTH_LIMIT"),
            "download_delay_s": self.settings.getfloat("DOWNLOAD_DELAY"),
            "pages_fetched": self.pages_fetched,
            "pages_saved": self.pages_saved,
            "skipped_by_robots": stats.get("robotstxt/forbidden", 0),
            "offsite_redirects_dropped": stats.get("offsite/filtered", 0),
            "skipped": dict(self.skipped),
            "errors": len(self.errors),
            "error_list": self.errors[:50],
            "pdf_links": sum(1 for d in self.seen_docs if d.lower().endswith(".pdf")),
            "document_links": len(self.seen_docs),
            "average_depth": round(sum(self.depths) / len(self.depths), 2) if self.depths else 0,
            "duration_s": round(time.time() - self.start_time, 1),
            "top10_most_linked": [{"url": u, "in_links": n} for u, n in self.link_in_degree.most_common(10)],
            "http_status_counts": {k.split("/")[-1]: v for k, v in stats.items()
                                   if k.startswith("downloader/response_status_count/")},
        }
        (config.RAW / "crawl_report.json").write_text(json.dumps(report, indent=2))
        (config.OUTPUTS / "crawl_report.json").write_text(json.dumps(report, indent=2))
