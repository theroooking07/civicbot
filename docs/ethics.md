# CivicBot: ethics and compliance

CivicBot only collects **public** pages, follows each site's robots.txt and terms,
crawls slowly, avoids login, payment and form pages, and removes personal data
before anything is analysed or shown. Every answer links back to the official page.

Checked on 25 October 2026. The full machine-readable results are in
`data/snapshot/robots_report.json`.

## 1. robots.txt

The crawler will not start until `crawler/robots_check.py` has parsed each
domain's robots.txt with `urllib.robotparser` and found every seed path allowed.
Scrapy also enforces robots.txt on every request (`ROBOTSTXT_OBEY = True`).

| Domain | robots.txt | Result |
|---|---|---|
| `www.cambridge.gov.uk` | `User-agent: *` / `Allow: /` | all 7 seed paths allowed |
| `www.greatercambridgewaste.org` | none published (HTTP 404) | no restrictions (standard behaviour); 2 seed paths allowed |

Neither file sets a `Crawl-delay` or lists a sitemap. In the full crawl, **0
requests were blocked** by robots.txt.

## 2. Terms of use

### Cambridge City Council (`/terms-of-use`)
- **Copyright is reserved.** Extracts may be downloaded "for your personal reference";
  **no commercial use** without a licence.
  → CivicBot is non-commercial coursework. Pages are analysed locally; the app shows
  short passages and always links to the original.
- **Authorship must be acknowledged.** → Every answer cites its source URLs, and the
  sidebar credits Cambridge City Council and Greater Cambridge Shared Waste.
- **Materials must not be modified or used out of context.** → Answers come only from
  retrieved text. A guard rejects any LLM answer containing a fee, phone number or date
  that is not in the sources.
- **The site must not be framed.** → The app never embeds council pages in iframes; it
  only links to them.
- **No unauthorised access or denial of service** (Computer Misuse Act 1990).
  → No login areas, no authentication bypass, one request at a time.
- **Content may be out of date and is not advice.** → The app shows the crawl date and a
  disclaimer asking users to check the linked official page.

### Greater Cambridge Shared Waste (`/terms-and-conditions`)
These are contract terms for commercial waste customers (containers, invoices,
liability). They say nothing about website use or automated access. Its privacy
notice covers data that people submit to it; CivicBot submits none.

## 3. Politeness limits

| Setting | Value |
|---|---|
| `DOWNLOAD_DELAY` | 2.0 s per domain, random jitter off (never below 2 s) |
| `CONCURRENT_REQUESTS` / per domain | 1 / 1 |
| AutoThrottle | on (start 2 s, max 15 s) |
| `DEPTH_LIMIT` | 3 |
| Page cap | 300 (enforced exactly inside the spider) |
| User-Agent | `CivicBot-StudentProject/0.1 (+university web-mining coursework; contact: <email>)` |
| Domains | the two above only; off-domain redirects are dropped |
| Cookies | off (no sessions) |

The full crawl took 9.5 minutes for 300 pages.

## 4. Pages we never crawl

Path segments are matched against exclusion rules in `config.py`:
- **accounts / login:** `my-account`, `online-accounts`, `login`, `sign-in`, `register`, `…-account`
- **payments:** `pay`, `pay-…`, `payment(s)`, `basket`, `checkout`
- **application forms:** `form(s)`, `apply-online`, `apply-for-…`, `report-a-…`, `request-a-…`, `book-a-…`
- **search results:** `/search`, or query keys `q`, `query`, `search`, `keywords`

Information pages *about* these topics (e.g. "Paying your Council Tax") are crawled;
the transactional pages themselves (e.g. "Pay your Council Tax") are not. In the full
crawl, 891 links were skipped by these rules, and 6 redirects that landed on form
pages were refused. The postcode-based bin-day lookup is never submitted, so no
addresses are collected.

JavaScript-rendered pages: a page with almost no text and few links is logged and
skipped. No headless browser is used. In practice, both sites serve plain HTML.

## 5. Personal data

Nothing personal is collected on purpose, and incidental personal data is
scrubbed in `preprocess/clean.py` before analysis:

| What | Rule | Removed in full crawl |
|---|---|---|
| Names of individual people (officers, councillors, people quoted in news) | spaCy PERSON entities, kept only if titled ("Cllr", "Mr"…) or made of capitalised words that never appear in lowercase on the site | 48 |
| Personal email addresses | `firstname.lastname@` style, or single-word addresses on non-government domains that name no role | 10 |
| Mobile numbers | UK `07…` numbers | 3 |
| `mailto:` links | never recorded by the crawler | - |

**Kept:** team and department emails (`licensing@…`, `refuse@scambs.gov.uk`),
landline and helpline numbers, and office hours.

The rules were audited by hand on the full crawl. An earlier version wrongly removed
things like "Blue Bin" and village names, which led to the current rules. Some brand
names (for example "John Lewis") are still removed as if they were people; we accept
over-removal as the safer error.

Raw HTML (`data/raw/`, unscrubbed) stays on the local machine and is git-ignored.

## 6. Documents (PDFs)

254 document links were found (229 PDFs). For each, CivicBot records the link, its
title and its size (parsed from the link text, e.g. "[PDF, 34Kb]"). **No documents
were downloaded.** The optional sample download for text extraction (max 15) was not
implemented. Nothing is redistributed; the app links to the originals.

## 7. Sharing the repository

This repository is **public**. `data/snapshot/` contains the cleaned text of the 300
crawled pages (`pages_clean.jsonl`, `raw_text.jsonl`, `index/chunks.jsonl`) so that
the demo runs offline. That text is Cambridge City Council and Greater Cambridge
Shared Waste copyright. It is included only for this non-commercial university
project, with attribution and links to every original page. Raw HTML is not included.

The council's terms allow copies for personal reference only, so publishing this text
goes beyond what they explicitly permit. **If either rights holder asks, the text files
will be removed** (including from the git history); the code, charts and reports do
not depend on them, and the pipeline can rebuild them locally with
`python -m crawler.run_crawl`. Anyone reusing this repository should not
redistribute the snapshot text.

## 8. Accuracy and limits of the chatbot

- If retrieval confidence is below the threshold (0.50), CivicBot replies "I couldn't
  find that on the portal. Please contact the relevant department." and does not call
  the LLM.
- Dates without a year are never completed, and no value is ever guessed.
- The crawl is a snapshot. The app shows the crawl date, and users should confirm
  details on the linked official page.
