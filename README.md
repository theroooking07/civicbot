# CivicBot - a municipal services guide for Cambridge

CivicBot is a university web-mining mini project. It **crawls** two public
council websites, **cleans and extracts** facts from their service pages,
**analyses** the site structure and topics, and then **answers citizen
questions** (bins, council tax, licences, housing, fees, phone numbers) with a
small retrieval-augmented chatbot that only answers from the crawled pages and
always links to the official source.

| Target | Why |
|---|---|
| `https://www.cambridge.gov.uk/` | Cambridge City Council: council tax, licensing, housing, contact details |
| `https://www.greatercambridgewaste.org/` | Greater Cambridge Shared Waste Service: bins and recycling (`cambridge.gov.uk/bins-and-recycling` redirects here) |

The pipeline has four stages: **collection → preprocessing → mining/analysis → reporting**.

```
robots_check ─► run_crawl ─► clean ─► extract ─► cluster_pages ─► link_graph ─► deadline_timeline
   (Module 1)    (Module 1)  (Module 2)  (Module 2)  (Module 3)        (Module 3)    (Module 3)
                                                             │
                                          chat.index ─► chat.ask / chat.evaluate (Module 4)
                                                             │
                              change_detect ─► make_snapshot ─► streamlit app.py (Module 5)
```

## Setup

Requires Python 3.10+.

```bash
python -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env                  # then edit .env
```

In `.env`:

- `GEMINI_API_KEY=` - paste a free key from **Google AI Studio**
  (https://aistudio.google.com/apikey). Optional: without it CivicBot uses a local
  **Ollama** server if one is running (`ollama pull llama3.2`), and otherwise runs in
  **retrieval-only mode** (shows the best matching passages and links). It never crashes
  for a missing key. The sidebar shows which mode is active.
- `CIVICBOT_CONTACT_EMAIL=` - your email, sent in the crawler's User-Agent so the site
  owner can contact you.

Never commit `.env`; it is in `.gitignore`. The key is only read from the environment.

## Run order

```bash
python -m crawler.robots_check            # 1. check robots.txt for both domains (crawler refuses to start without it)
python -m crawler.run_crawl --max-pages 20   # 2. quick test crawl; omit --max-pages for the full 300
python -m preprocess.clean                # 3. clean text, scrub personal data
python -m preprocess.extract              # 4. deadlines, fees, phones, hours, schedules
python -m analysis.cluster_pages          # 5. TF-IDF + K-Means topics (run before link_graph)
python -m analysis.link_graph             # 6. PageRank, hubs, communities, link-cluster finding
python -m analysis.deadline_timeline      # 7. upcoming dates chart
python -m chat.index                      # 8. chunk + embed + FAISS/BM25 index
python -m chat.evaluate                   # 9. Recall@3 on eval_questions.json
python -m crawler.change_detect           # 10. compare with the last snapshot ("What's new")
python make_snapshot.py                   # 11. save outputs to data/snapshot/ for the demo
streamlit run app.py                      # 12. the app
```

Ask from the command line: `python -m chat.ask "What time do I put my bins out?"`

### Demo snapshot mode (no key, no network)

The app's sidebar toggle **Demo snapshot mode** loads everything from
`data/snapshot/` (committed, about 8 MB), so the demo works with **no API key and
no network**: the embedding model loads from the local cache, and answers run in
retrieval-only mode. Run the pipeline once on a machine with internet to download
the embedding model before going offline.

## What each module does

| Module | Files | Output |
|---|---|---|
| 1 Crawler | `crawler/robots_check.py`, `settings.py`, `civic_spider.py`, `run_crawl.py`, `change_detect.py` | `data/raw/pages.jsonl`, `links.csv`, `documents.csv`, `crawl_report.json`; `data/outputs/updates.json` |
| 2 Preprocessing | `preprocess/clean.py`, `extract.py` | `data/clean/pages_clean.jsonl`, `entities.csv`, `page_entities.csv` |
| 3 Analysis | `analysis/cluster_pages.py`, `link_graph.py`, `deadline_timeline.py`, `style.py` | charts and reports in `data/outputs/` |
| 4 Chatbot | `chat/index.py`, `ask.py`, `evaluate.py`, `eval_questions.json` | `data/outputs/index/`, `eval_report.json`, `eval_recall.png` |
| 5 App | `app.py`, `make_snapshot.py` | Streamlit app, `data/snapshot/` |

All settings (target URLs, seeds, exclusions, limits, LLM models) are in `config.py`
and can be overridden from `.env`.

## Results (crawl of 25 October 2026)

- **Crawl:** 300 pages (201 council, 99 waste service), 0 blocked by robots.txt,
  9 broken links logged, 254 document links recorded (229 PDFs), 9.5 minutes at 2 s/request per domain.
- **Cleaning:** 320,278 visible words → 159,130 clean words (menus, banners and duplicate
  lines removed); 48 names, 10 personal emails and 3 mobile numbers scrubbed.
- **Extraction:** 730 entities: 69 deadlines, 157 dates, 405 fees, 48 phone numbers,
  9 office-hour statements, 42 collection-schedule rules.
- **Topics:** k = 4 by mean silhouette: Waste & recycling, Taxi licensing,
  Council tax + Licences, Housing (87 thin hub pages not clustered). Silhouette is low
  (~0.07) for every k - the site's text does not separate into sharp topics.
- **Link-structure finding:** **81.9%** of linked content-page pairs share a topic cluster
  once site menus are removed, vs 61.4% with menus included.
- **Chatbot retrieval:** Recall@3 **90%** (27/30), Recall@1 77%, MRR 0.83;
  8/8 off-topic questions correctly refused at the 0.50 confidence threshold.

## Limitations

**Coverage**
1. **No actual bin days.** Collection days come from a postcode lookup form, which we
   deliberately don't automate. CivicBot can give the rules ("bins out by 6am") and link
   to the lookup, but not "your bin day is Tuesday".
2. **300-page cap.** Many service pages were never reached: planning, parking details,
   most news, and 87 pages that are mostly link lists. A question about an uncrawled page
   gets "couldn't find that" or a near-miss page.
3. **No birth/death certificate content.** In England, registration is a county council
   service, so it isn't on these sites.
4. **PDFs are only recorded.** 229 PDF links were recorded but none were downloaded
   or read, so fee tables or forms that exist only as PDFs are invisible to the chatbot.
5. **One-off snapshot.** Change detection has only one baseline so far; the "What's new"
   tab shows real changes only after the next crawl.

**Extraction and cleaning**
6. **Rule-based extraction.** Some "fees" are not citizen fees (budget figures, "for every
   £1 you pay" on a jobs page), and office hours were found on only 6 pages.
7. **Most deadlines have no exact date.** Only 16 of 69 have a normalised date; the rest
   are relative ("within 21 days") or have no year, and are left null on purpose.
8. **Imperfect name scrubbing.** Some brands are removed as if they were people ("John
   Lewis"), and an unusual name written in lowercase, or appearing as an ordinary word
   elsewhere on the site, could be missed.

**Analysis**
9. **Weak topic structure.** Silhouette is about 0.06-0.07 for every k from 3 to 8 (within
   seed noise), so the chosen k = 4 is not strongly supported, and one cluster mixes council
   tax with licences. Cluster names come from a hand-written keyword list.
10. **The link finding depends on a threshold.** "Menu links" are defined as targets linked
    from at least 25% of pages; another threshold would change the 81.9% / 61.4% figures
    somewhat.

**Chatbot**
11. **Small, self-written test set.** The 30 evaluation questions were written by us after
    seeing the crawled pages, so Recall@3 = 90% is probably optimistic for real users.
12. **Narrow confidence margin.** Off-topic questions scored up to 0.47 and in-scope ones as
    low as 0.54. Some real questions will be wrongly refused, and some off-topic ones
    answered.
13. **LLM modes are untested.** Gemini and Ollama were not tested on the development machine
    (no key, no Ollama); only retrieval-only mode was exercised end to end. The figure guard
    only checks fees, UK phone numbers and "day Month year" dates, not every kind of fact.
14. **English only**, and every answer is limited to the date of the crawl.

## Ethics and compliance

See [`docs/ethics.md`](docs/ethics.md): robots.txt and terms-of-use checks for both
domains, rate limits, excluded page types, personal-data scrubbing, and how the
app attributes and links to the original pages.

## Demo

See [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) for a 5-minute walkthrough.
