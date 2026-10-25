# CivicBot - 5-minute demo script

**Before you start:** run `streamlit run app.py` and make sure **Demo snapshot mode**
is ON in the sidebar. This works with no network and no API key. If you have a Gemini
key in `.env`, the sidebar shows "LLM: Gemini"; otherwise it shows retrieval-only mode.
Both are fine for the demo.

---

## 0:00 - 0:30 · The problem (no clicks)

> "Council websites hold the answers citizens need - bin days, council tax, licence
> fees - but the information is spread over hundreds of pages and two different
> websites. CivicBot crawls the public portal, extracts and analyses the content, and
> answers questions with links back to the official pages."

Point at the sidebar: data source and date, and the active LLM mode.

## 0:30 - 1:30 · Tab 1: Compliance & Crawl

1. **robots.txt check:** both domains show ✅. Open the "robots.txt contents" expander
   for cambridge.gov.uk (`Allow: /`). Note that the waste site publishes none.
2. **Politeness line:** 2 s per domain, one request at a time, depth 3, at most 300 pages,
   login/payment/form/search pages excluded.
3. **Metrics row:** 300 pages, **0 blocked by robots**, 9 errors logged, 229 PDF links,
   9.5 minutes.
4. Scroll the crawled-URL table, then show **Top 10 most-linked pages**: the menu hubs.

> "Bins are run by a separate shared waste service on its own domain, so we checked
> its robots.txt and terms separately and added it as a second domain."

*(Optional, only with network and snapshot mode OFF: start a 10-page live crawl to
show the progress bar.)*

## 1:30 - 2:15 · Tab 2: Cleaning & Extraction

1. The page selector defaults to **Council Tax bands and charges**.
   - Left, raw text: 532 words including menus.
   - Right, cleaned text: 244 words of real content.
2. Metric: **50% of words removed**, **61 personal-data items scrubbed**. Hover to see
   names, emails and mobiles.
3. **Found vs null** table: 405 fees, 69 deadlines, 48 phone numbers.
4. In the entity table, show one deadline, e.g. **"The vouchers expire after 31 March
   2026" → 2026-03-31**.

> "Dates without a year stay null. We never guess."

## 2:15 - 3:15 · Tab 3: Analysis

1. Read the green **finding** box: **82% of linked content pages share a topic cluster**
   once menus are removed, vs 61% with menus.
   > "Departments mostly link within their own topic; the menus are what connect topics."
2. **Link graph:** node size = PageRank, colour = topic. Point out the taxi cluster and
   the waste-site cluster.
3. **Silhouette plot:** be honest:
   > "Scores are low for every k. Council pages share a lot of vocabulary, so the topics
   > overlap."
4. **Heatmap:** waste, taxi and council-tax questions each match their own cluster most
   strongly.
5. **Deadline timeline:** River Cam fees consultation (30 Oct), rental property
   registration deadline (14 Apr 2027).

## 3:15 - 3:40 · Tab 4: What's New

Show the change-detection summary. The snapshot is the baseline, so the current crawl
shows 0 changes.

> "On the next crawl, changed pages appear here with the added and removed lines."

Then show the **upcoming dates** table.

## 3:40 - 4:45 · Tab 5: Ask CivicBot

1. Click **"How much is council tax for Band D?"**
   - The entity table shows the **Band D row first**: house value up to £88,000 (1991),
     £2,355.41 for 2025/26, £2,467.02 for 2026/27.
   - Sources link to the council tax bands page.
2. Type **"What time do I put my bins out?"** → "by 6am on the day of your collection",
   source: the waste site.
3. Type **"How do I renew my passport?"** → *"I couldn't find that on the portal…"*.
   > "Below the confidence threshold, the LLM is not called at all."
4. Open the **Retrieval quality** expander: **Recall@3 90%** on 30 test questions, and
   8/8 off-topic questions refused.

## 4:45 - 5:00 · Wrap-up

> "Four pipeline stages: polite crawling, cleaning with personal-data scrubbing, link
> and topic analysis, and a grounded chatbot that cites its sources. Main limitations:
> bin days need a postcode lookup we deliberately don't automate, the topic clusters
> overlap, and the confidence threshold has a narrow margin."
