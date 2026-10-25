"""
app.py - CivicBot Streamlit app (Module 5).

Five tabs:
  1. Compliance & Crawl     robots.txt result, live crawl, crawl report, crawled URLs
  2. Cleaning & Extraction  raw vs cleaned text side by side, extracted entities
  3. Analysis               link graph, clusters, silhouette, heatmap, deadline timeline
  4. What's New             change detection + upcoming dates
  5. Ask CivicBot           chatbot with entity table and source links

Sidebar: "Demo snapshot mode" reads everything from data/snapshot/ (no network,
no API key needed) and shows which LLM mode is active.

Run:
    streamlit run app.py
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

import config

st.set_page_config(page_title="CivicBot - Cambridge", page_icon="🏛️", layout="wide")

# Where each file lives in live mode; in snapshot mode the same file name is
# read from data/snapshot/ instead.
LIVE_FILES = {
    "robots_report": config.OUTPUTS / "robots_report.json",
    "crawl_report": config.RAW / "crawl_report.json",
    "pages": config.RAW / "pages.jsonl",
    "documents": config.RAW / "documents.csv",
    "pages_clean": config.CLEAN / "pages_clean.jsonl",
    "entities": config.CLEAN / "entities.csv",
    "page_entities": config.CLEAN / "page_entities.csv",
    "clusters": config.OUTPUTS / "clusters.json",
    "link_report": config.OUTPUTS / "link_report.json",
    "link_metrics": config.OUTPUTS / "link_metrics.csv",
    "upcoming": config.OUTPUTS / "upcoming_deadlines.csv",
    "eval_report": config.OUTPUTS / "eval_report.json",
    "updates": config.OUTPUTS / "updates.json",
    "index_dir": config.OUTPUTS / "index",
}
CHARTS = {
    "link_graph": "Link graph (node size = PageRank, colour = topic cluster)",
    "cluster_scatter": "Pages coloured by topic cluster (PCA)",
    "silhouette": "Choosing k by silhouette score",
    "cluster_sizes": "Cluster sizes",
    "cluster_topic_heatmap": "Clusters vs citizen-question topics",
    "deadline_timeline": "Upcoming dates found on the portal",
    "eval_recall": "Chatbot retrieval quality (Recall@3)",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def path(name: str, snapshot: bool) -> Path:
    live = LIVE_FILES.get(name, config.OUTPUTS / f"{name}.png")
    return config.SNAPSHOT / live.name if snapshot else live


@st.cache_data(show_spinner=False)
def read_json(p: str, mtime: float):
    return json.loads(Path(p).read_text())


@st.cache_data(show_spinner=False)
def read_jsonl(p: str, mtime: float) -> pd.DataFrame:
    return pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8")])


@st.cache_data(show_spinner=False)
def read_csv(p: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(p)


def load(name: str, snapshot: bool):
    """Load a file by logical name; returns None if it doesn't exist yet."""
    p = path(name, snapshot)
    if not p.exists():
        return None
    reader = {".json": read_json, ".jsonl": read_jsonl, ".csv": read_csv}[p.suffix]
    return reader(str(p), p.stat().st_mtime)     # mtime in the key = reload after a new run


def missing(what: str, cmd: str):
    st.info(f"No {what} yet. Run `{cmd}` (or switch on demo snapshot mode).")


def raw_text_for(url: str, snapshot: bool) -> str:
    """Raw visible text of a page: from the saved HTML (live) or the snapshot copy."""
    if snapshot:
        raw = load("raw_text", snapshot)
        if raw is not None:
            match = raw[raw.url == url]
            return match.iloc[0]["raw_text"] if len(match) else ""
        return ""
    from preprocess.clean import raw_vs_clean
    try:
        return raw_vs_clean(url)["raw"]
    except (KeyError, FileNotFoundError):
        return ""


LIVE_FILES["raw_text"] = config.SNAPSHOT / "raw_text.jsonl"   # only exists in the snapshot


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
snapshot_exists = (config.SNAPSHOT / "pages_clean.jsonl").exists()
st.sidebar.title("🏛️ CivicBot")
st.sidebar.caption("Municipal services guide built by crawling public council pages.")
snapshot = st.sidebar.toggle("Demo snapshot mode", value=snapshot_exists,
                             help="Use the saved crawl in data/snapshot/ - no network or API key needed.",
                             disabled=not snapshot_exists)
if not snapshot_exists:
    st.sidebar.caption("No snapshot yet - run `python make_snapshot.py` to create one.")

from chat import ask as civic_ask  # noqa: E402  (import after page config)

civic_ask.use_data(path("index_dir", snapshot), path("entities", snapshot))
mode = civic_ask.llm_mode()
MODE_TEXT = {
    "gemini": f"**LLM: Gemini** (`{config.GEMINI_MODEL}`) - answers are generated from retrieved passages.",
    "ollama": f"**LLM: Ollama** (`{config.OLLAMA_MODEL}`, local) - no GEMINI_API_KEY set.",
    "retrieval-only": "**Retrieval-only mode** - no GEMINI_API_KEY and no Ollama running. "
                      "CivicBot shows the best matching passages and links.",
}
(st.sidebar.success if mode != "retrieval-only" else st.sidebar.warning)(MODE_TEXT[mode])

crawl_report = load("crawl_report", snapshot)
if crawl_report is not None:
    pages_df = load("pages", snapshot)
    crawled_on = pages_df.fetched_at.min()[:10] if pages_df is not None else "?"
    st.sidebar.markdown(f"**Data:** {'snapshot' if snapshot else 'live'} crawl from {crawled_on}, "
                        f"{crawl_report['pages_saved']} pages")
st.sidebar.divider()
st.sidebar.caption("Content © Cambridge City Council and Greater Cambridge Shared Waste. "
                   "Information may be out of date - always check the linked official page. "
                   "Non-commercial student project.")

tabs = st.tabs(["1 · Compliance & Crawl", "2 · Cleaning & Extraction", "3 · Analysis",
                "4 · What's New", "5 · Ask CivicBot"])


# ---------------------------------------------------------------------------
# Tab 1: Compliance & Crawl
# ---------------------------------------------------------------------------
with tabs[0]:
    st.header("Compliance and crawl")
    robots = load("robots_report", snapshot)
    if robots is None:
        missing("robots.txt report", "python -m crawler.robots_check")
    else:
        verdict = "✅ Crawling allowed" if robots["crawl_allowed"] else "⛔ Crawling NOT allowed"
        st.subheader(f"robots.txt check: {verdict}")
        st.caption(f"Checked {robots['checked_at'][:16].replace('T', ' ')} UTC as `{robots['user_agent']}`")
        cols = st.columns(len(robots["domains"]))
        for col, d in zip(cols, robots["domains"]):
            with col:
                st.markdown(f"**{d['domain']}** - robots.txt HTTP {d['http_status']}")
                st.dataframe(pd.DataFrame({"seed path": list(d["seed_results"]),
                                           "allowed": list(d["seed_results"].values())}),
                             hide_index=True, width="stretch")
                with st.expander("robots.txt contents"):
                    st.code(d["robots_txt"] or "(no robots.txt published - no restrictions)")
        st.markdown(f"Politeness: **{robots['effective_delay']}s** delay per domain, 1 request at a time, "
                    f"AutoThrottle on, depth ≤ {config.MAX_DEPTH}, ≤ {config.MAX_PAGES} pages, "
                    "login/payment/form/search pages excluded. See `docs/ethics.md`.")

    st.subheader("Run a live crawl")
    if snapshot:
        st.caption("Switch off demo snapshot mode to run a live crawl.")
    else:
        c1, c2 = st.columns([1, 3])
        cap = c1.number_input("Page cap", min_value=5, max_value=config.MAX_PAGES, value=20, step=5)
        also_pipeline = c2.checkbox("Then rerun cleaning, extraction, analysis and the index", value=True)
        if st.button("▶ Start crawl", type="primary"):
            progress, log_box = st.progress(0.0, text="starting..."), st.empty()
            lines = []
            proc = subprocess.Popen([sys.executable, "-m", "crawler.run_crawl", "--max-pages", str(cap)],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    cwd=config.ROOT)
            for line in proc.stdout:
                lines.append(line.rstrip())
                m = re.search(r"\[\s*(\d+)/(\d+)\]", line)
                if m:
                    progress.progress(int(m.group(1)) / int(m.group(2)), text=f"{m.group(1)}/{m.group(2)} pages")
                log_box.code("\n".join(lines[-15:]))
            proc.wait()
            if proc.returncode != 0:
                st.error("Crawl failed - see the log above.")
            elif also_pipeline:
                steps = ["preprocess.clean", "preprocess.extract", "analysis.cluster_pages",
                         "analysis.link_graph", "analysis.deadline_timeline", "crawler.change_detect",
                         "chat.index"]
                for i, mod in enumerate(steps, 1):
                    progress.progress(i / len(steps), text=f"running {mod} ...")
                    r = subprocess.run([sys.executable, "-m", mod], capture_output=True, text=True, cwd=config.ROOT)
                    if r.returncode != 0:
                        st.error(f"{mod} failed:\n{r.stdout[-1500:]}{r.stderr[-1500:]}")
                        break
                else:
                    progress.progress(1.0, text="done")
                    civic_ask.load_index.cache_clear()
                    civic_ask.load_entities.cache_clear()
                    st.success("Crawl and pipeline finished.")
            else:
                st.success("Crawl finished.")

    if crawl_report is None:
        missing("crawl report", "python -m crawler.run_crawl")
    else:
        st.subheader("Crawl report")
        m = st.columns(6)
        m[0].metric("Pages saved", crawl_report["pages_saved"])
        m[1].metric("Skipped by robots", crawl_report["skipped_by_robots"])
        m[2].metric("Errors (logged)", crawl_report["errors"])
        m[3].metric("PDF links", crawl_report["pdf_links"])
        m[4].metric("Average depth", crawl_report["average_depth"])
        m[5].metric("Duration", f"{crawl_report['duration_s'] / 60:.1f} min")
        st.caption(f"Skipped: {crawl_report['skipped']} · finished because: {crawl_report['finished_reason']}")
        c1, c2 = st.columns([3, 2])
        with c1:
            pages_df = load("pages", snapshot)
            if pages_df is not None:
                st.markdown(f"**Crawled pages ({len(pages_df)})**")
                st.dataframe(pages_df[["url", "depth", "status", "word_count", "title"]],
                             hide_index=True, width="stretch", height=360,
                             column_config={"url": st.column_config.LinkColumn("url")})
        with c2:
            st.markdown("**Top 10 most-linked pages**")
            st.dataframe(pd.DataFrame(crawl_report["top10_most_linked"]), hide_index=True,
                         width="stretch", column_config={"url": st.column_config.LinkColumn("url")})
        docs = load("documents", snapshot)
        if docs is not None:
            with st.expander(f"Documents found ({len(docs)} links - recorded, not downloaded)"):
                st.dataframe(docs, hide_index=True, width="stretch",
                             column_config={"url": st.column_config.LinkColumn("url")})


# ---------------------------------------------------------------------------
# Tab 2: Cleaning & Extraction
# ---------------------------------------------------------------------------
with tabs[1]:
    st.header("Cleaning and extraction")
    clean = load("pages_clean", snapshot)
    if clean is None:
        missing("cleaned pages", "python -m preprocess.clean")
    else:
        totals = clean[["n_words_raw", "n_words_clean"]].sum()
        scrub = pd.DataFrame(list(clean.scrubbed)).sum()
        c = st.columns(4)
        c[0].metric("Raw visible words", f"{totals.n_words_raw:,}")
        c[1].metric("Clean words", f"{totals.n_words_clean:,}",
                    f"-{100 * (1 - totals.n_words_clean / totals.n_words_raw):.0f}%", delta_color="off")
        c[2].metric("Personal data removed", int(scrub.sum()) if len(scrub) else 0,
                    help=", ".join(f"{k}: {int(v)}" for k, v in scrub.items()))
        c[3].metric("Pages", len(clean))

        options = clean.sort_values("n_words_clean", ascending=False)
        label = {r.url: f"{r.title[:70]}  ({r.url.split('://')[1][:60]})" for r in options.itertuples()}
        default = next((i for i, u in enumerate(options.url) if u.endswith("council-tax-bands-and-charges")), 0)
        url = st.selectbox("Page", options.url, index=default, format_func=label.get)
        row = clean[clean.url == url].iloc[0]
        left, right = st.columns(2)
        raw = raw_text_for(url, snapshot)
        left.markdown(f"**Raw visible text** ({len(raw.split()):,} words - menus, footer, banners)")
        left.text_area("raw", raw, height=420, label_visibility="collapsed")
        right.markdown(f"**Cleaned `text_full`** ({row.n_words_clean:,} words)")
        right.text_area("clean", row.text_full, height=420, label_visibility="collapsed")
        with st.expander("`text_tokens` (lowercase lemmas, stop-words removed - used for TF-IDF)"):
            st.write(row.text_tokens[:3000])

    st.subheader("Extracted entities")
    ents = load("entities", snapshot)
    if ents is None:
        missing("entities", "python -m preprocess.extract")
    else:
        summary = load("page_entities", snapshot)
        if summary is not None:
            types = [c for c in summary.columns if c not in ("url", "title")]
            st.dataframe(pd.DataFrame({
                "field": types,
                "pages found": [int(summary[t].notna().sum()) for t in types],
                "pages null": [int(summary[t].isna().sum()) for t in types],
                "values": [int((ents.entity_type == t).sum()) for t in types]}),
                hide_index=True)
        pick = st.multiselect("Types", sorted(ents.entity_type.unique()), default=["deadline", "fee", "phone"])
        st.dataframe(ents[ents.entity_type.isin(pick)], hide_index=True, width="stretch", height=380,
                     column_config={"url": st.column_config.LinkColumn("url")})
        st.caption("`value_normalised` is empty when it can't be determined without guessing "
                   "(e.g. a date with no year).")


# ---------------------------------------------------------------------------
# Tab 3: Analysis
# ---------------------------------------------------------------------------
def chart(name: str):
    p = path(name, snapshot)
    if p.exists():
        st.image(str(p), caption=CHARTS[name], width="stretch")
    else:
        st.caption(f"{CHARTS[name]}: not generated yet.")


with tabs[2]:
    st.header("Analysis")
    link = load("link_report", snapshot)
    clusters = load("clusters", snapshot)
    if link is None or clusters is None:
        missing("analysis outputs", "python -m analysis.cluster_pages && python -m analysis.link_graph")
    else:
        f_full, f_content = link["finding_full_graph"], link["finding_content_graph"]
        st.success(f"**Finding:** {f_content['percent_same_cluster']}% of linked page pairs share a topic "
                   f"cluster once menu links are removed ({f_content['linked_pairs']} pairs), vs "
                   f"{f_full['percent_same_cluster']}% with menus included. Departments mostly link within "
                   "their own topic; the site menus are what tie topics together.")
        c = st.columns(4)
        c[0].metric("Pages (nodes)", link["nodes"])
        c[1].metric("Links (all)", f"{link['edges_full']:,}")
        c[2].metric("Content links", f"{link['edges_content']:,}")
        c[3].metric("Topic clusters (k)", clusters["k"])
        chart("link_graph")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top pages by PageRank (content graph)**")
            st.dataframe(pd.DataFrame(link["top_pagerank_content"]), hide_index=True, width="stretch")
        with c2:
            st.markdown("**Clusters and their top terms**")
            st.dataframe(pd.DataFrame([{"cluster": c["name"], "pages": c["size"],
                                        "top terms": ", ".join(c["top_terms"])} for c in clusters["clusters"]]),
                         hide_index=True, width="stretch")
            st.caption(f"{clusters['hub_pages']} thin hub/navigation pages were not clustered.")
        c1, c2 = st.columns(2)
        with c1:
            chart("cluster_scatter")
            chart("cluster_sizes")
        with c2:
            chart("silhouette")
            chart("cluster_topic_heatmap")
        chart("deadline_timeline")


# ---------------------------------------------------------------------------
# Tab 4: What's New
# ---------------------------------------------------------------------------
with tabs[3]:
    st.header("What's new")
    upd = load("updates", snapshot)
    if upd is None:
        missing("change report", "python -m crawler.change_detect")
    elif upd.get("first_run"):
        st.info(upd["message"])
    else:
        st.caption(f"Compared crawl of {str(upd['current_crawl'])[:10]} with baseline from "
                   f"{str(upd['baseline_crawl'])[:10]}.")
        c = st.columns(4)
        c[0].metric("New pages", len(upd["new"]))
        c[1].metric("Changed pages", len(upd["changed"]))
        c[2].metric("Removed / not reached", len(upd["removed"]))
        c[3].metric("Unchanged", upd["unchanged"])
        for item in upd["changed"]:
            with st.expander(f"✏️ {item['title']}"):
                st.markdown(f"[{item['url']}]({item['url']}) - "
                            f"+{item.get('lines_added', '?')} / -{item.get('lines_removed', '?')} lines")
                for line in item.get("added_sample", []):
                    st.markdown(f"🟢 {line}")
                for line in item.get("removed_sample", []):
                    st.markdown(f"🔴 ~~{line}~~")
        if upd["new"]:
            st.markdown("**New pages**")
            st.dataframe(pd.DataFrame(upd["new"]), hide_index=True, width="stretch",
                         column_config={"url": st.column_config.LinkColumn("url")})

    st.subheader("Upcoming dates found on the portal")
    up = load("upcoming", snapshot)
    if up is None or up.empty:
        st.caption("No upcoming dated deadlines found.")
    else:
        st.dataframe(up[["date", "entity_type", "context", "url"]], hide_index=True, width="stretch",
                     column_config={"url": st.column_config.LinkColumn("url"),
                                    "context": st.column_config.TextColumn("what", width="large")})


# ---------------------------------------------------------------------------
# Tab 5: Ask CivicBot
# ---------------------------------------------------------------------------
def show_answer(res: dict):
    if res.get("entities") is not None and len(res["entities"]):
        st.markdown("**Key facts found in the sources**")
        st.dataframe(res["entities"][["type", "value", "normalised", "url"]], hide_index=True,
                     width="stretch", column_config={"url": st.column_config.LinkColumn("source")})
    st.markdown(res["answer"])
    if res.get("note"):
        st.caption(f"⚠️ {res['note']}")
    if res["sources"]:
        st.markdown("**Sources:** " + " · ".join(f"[{s['n']}] [{s['title'].split(' - ')[0]}]({s['url']})"
                                                for s in res["sources"]))
    st.caption(f"mode: {res['mode']} · retrieval confidence {res['confidence']:.2f} "
               f"(threshold {civic_ask.MIN_CONFIDENCE})")


with tabs[4]:
    st.header("Ask CivicBot")
    st.caption("Answers come only from the crawled pages, with links to the official source. "
               "Dates, fees and phone numbers are never invented.")
    ev = load("eval_report", snapshot)
    if ev is not None:
        with st.expander(f"Retrieval quality on {ev['n_questions']} test questions: "
                         f"Recall@3 {ev['recall@3']:.0%}, MRR {ev['mrr']:.2f}"):
            chart("eval_recall")
            st.caption(f"Off-topic questions refused: {ev['off_topic_refused']}/{len(ev['off_topic'])} "
                       f"(threshold {ev['threshold']}).")
    if not path("index_dir", snapshot).exists():
        missing("search index", "python -m chat.index")
    else:
        if "history" not in st.session_state:
            st.session_state.history = []
        examples = ["What time do I put my bins out?", "How much is council tax for Band D?",
                    "How much does a burial cost?", "What is the council's phone number?"]
        cols = st.columns(len(examples))
        clicked = next((q for c, q in zip(cols, examples) if c.button(q, width="stretch")), None)
        for q, res in st.session_state.history:
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                show_answer(res)
        question = st.chat_input("Ask about bins, council tax, licences, housing...") or clicked
        if question:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Searching the portal..."):
                    res = civic_ask.ask(question)
                show_answer(res)
            st.session_state.history.append((question, res))
