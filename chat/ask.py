"""
ask.py - Module 4b: answer a citizen question from the crawled portal only.

ask(question) does:
  1. hybrid retrieval: FAISS (meaning) + BM25 (keywords), scores fused
  2. confidence check: if the best match is weak, reply with the fallback
     message and do NOT call the LLM
  3. entity table: deadlines / fees / phones found in the retrieved passages
  4. answer with the first available mode:
        gemini         - GEMINI_API_KEY is set
        ollama         - a local Ollama server is running
        retrieval-only - show the best passages and links
  5. safety check: if the LLM answer contains a fee, phone number or date that
     is not in the sources, the answer is discarded and the passages are shown

Usage:
    python -m chat.ask "When do I need to put my bins out?"
"""
import json
import re
import sys
from functools import lru_cache

import faiss
import numpy as np
import pandas as pd
import requests
from rank_bm25 import BM25Okapi

import config
from chat.index import EMBED_MODEL, INDEX_DIR, bm25_tokens, load_embedder

TOP_K = 4                    # chunks given to the LLM
CANDIDATES = 30              # dense candidates considered before fusion
DENSE_WEIGHT = 0.65          # fused = 0.65 * cosine + 0.35 * normalised BM25
MIN_CONFIDENCE = 0.50        # best cosine below this -> "couldn't find that". Set from chat.evaluate:
                             # in-scope questions scored >= 0.54, off-topic ones <= 0.47
NOT_FOUND = "I couldn't find that on the portal. Please contact the relevant department."
ENTITIES_IN = config.CLEAN / "entities.csv"

# Which entity types are relevant to which kinds of question.
INTENT_WORDS = {
    "fee": r"\b(fee|fees|cost|costs|price|charge|charges|how much|pay|£)\b",
    "deadline": r"\b(when|deadline|by when|due|date|expire|renew|last day|until|start)\b",
    "date": r"\b(when|date|start|from)\b",
    "phone": r"\b(phone|call|telephone|number|contact|helpline|ring)\b",
    "office_hours": r"\b(open|opening|hours|time|times|when can)\b",
    "schedule": r"\b(collect|collection|collected|bin day|bins out|weekly|fortnight)\b",
}

PROMPT = """You are CivicBot, a helpful assistant for Cambridge City Council and the
Greater Cambridge Shared Waste Service. Answer the citizen's question using ONLY
the numbered sources below.

Rules:
- If the sources do not contain the answer, reply exactly: "{not_found}"
- Never invent dates, fees, phone numbers or opening hours. Copy them exactly.
- Keep the answer short (3-6 sentences or a short list) and plain.
- Cite sources inline with their numbers, e.g. [1] or [2][3].

Sources:
{sources}

Question: {question}
Answer:"""


# ---------------------------------------------------------------------------
# Loading (cached so the Streamlit app only loads things once)
# ---------------------------------------------------------------------------
def use_data(index_dir, entities_csv):
    """Point the chatbot at another index/entities (the app's demo snapshot mode)."""
    global INDEX_DIR, ENTITIES_IN
    if (index_dir, entities_csv) != (INDEX_DIR, ENTITIES_IN):
        INDEX_DIR, ENTITIES_IN = index_dir, entities_csv
        load_index.cache_clear()
        load_entities.cache_clear()


@lru_cache(maxsize=1)
def load_index():
    """Return (faiss index, chunk list, BM25 model, embedder)."""
    if not (INDEX_DIR / "faiss.index").exists():
        raise FileNotFoundError(f"No index in {INDEX_DIR} - run  python -m chat.index  first.")
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    chunks = [json.loads(l) for l in open(INDEX_DIR / "chunks.jsonl", encoding="utf-8")]
    bm25 = BM25Okapi([c["bm25_tokens"] for c in chunks])
    return index, chunks, bm25, load_embedder()


@lru_cache(maxsize=1)
def load_entities() -> pd.DataFrame:
    return pd.read_csv(ENTITIES_IN) if ENTITIES_IN.exists() else pd.DataFrame()


@lru_cache(maxsize=1)
def ollama_available() -> bool:
    """True if a local Ollama server answers and has the configured model."""
    try:
        tags = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=1.5).json()
        names = [m.get("name", "") for m in tags.get("models", [])]
        return any(n.split(":")[0] == config.OLLAMA_MODEL.split(":")[0] for n in names)
    except (requests.RequestException, ValueError):
        return False


def llm_mode() -> str:
    """Which answer mode is active: 'gemini', 'ollama' or 'retrieval-only'."""
    if config.GEMINI_API_KEY:
        return "gemini"
    if ollama_available():
        return "ollama"
    return "retrieval-only"


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """Hybrid search. Returns the top-k chunks with dense, bm25 and fused scores."""
    index, chunks, bm25, model = load_index()
    qvec = model.encode([question], normalize_embeddings=True).astype("float32")
    dense_scores, dense_ids = index.search(qvec, CANDIDATES)
    dense = dict(zip(dense_ids[0].tolist(), dense_scores[0].tolist()))

    kw = bm25.get_scores(bm25_tokens(question))
    kw_top = np.argsort(kw)[::-1][:CANDIDATES]
    kw_max = kw.max() or 1.0

    # Candidates from either method; a chunk missing a dense score gets its real cosine.
    cand = set(dense) | set(kw_top.tolist())
    missing = [i for i in cand if i not in dense]
    if missing:
        vecs = np.vstack([index.reconstruct(int(i)) for i in missing])
        for i, s in zip(missing, (vecs @ qvec[0]).tolist()):
            dense[i] = s
    results = []
    for i in cand:
        fused = DENSE_WEIGHT * dense[i] + (1 - DENSE_WEIGHT) * (kw[i] / kw_max)
        results.append({**{k_: chunks[i][k_] for k_ in ("chunk_id", "url", "title", "cluster", "text")},
                        "dense": round(dense[i], 4), "bm25": round(float(kw[i]), 3),
                        "score": round(fused, 4)})
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]


def ranked_urls(question: str, n: int = 10) -> list[str]:
    """Unique page URLs in retrieval order (used by the evaluation)."""
    urls = []
    for r in retrieve(question, k=40):
        if r["url"] not in urls:
            urls.append(r["url"])
    return urls[:n]


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------
def relevant_entities(question: str, hits: list[dict], max_rows: int = 8) -> pd.DataFrame:
    """Entities whose sentence appears in a retrieved chunk and match the question's intent."""
    ents = load_entities()
    if ents.empty:
        return ents
    q = question.lower()
    wanted = [t for t, pat in INTENT_WORDS.items() if re.search(pat, q)]
    if not wanted:
        return ents.iloc[0:0]
    text_by_url = {}
    for h in hits:
        text_by_url.setdefault(h["url"], []).append(" ".join(h["text"].split()))
    rows = ents[ents.url.isin(text_by_url) & ents.entity_type.isin(wanted)].copy()
    in_chunk = [any(" ".join(str(c).split())[:80] in t for t in text_by_url[u])
                for u, c in zip(rows.url, rows.context)]
    rows = rows[in_chunk]
    # Rank by word overlap between the question and the entity's sentence, so
    # "band D" puts the Band D row of the council tax table first.
    q_tokens = set(bm25_tokens(question))
    rows["overlap"] = [len(q_tokens & set(bm25_tokens(str(c)))) for c in rows.context]
    rows = rows.sort_values("overlap", ascending=False, kind="stable")
    rows = rows.drop_duplicates(["entity_type", "value_raw"]).head(max_rows)
    return rows[["entity_type", "value_raw", "value_normalised", "context", "url"]].rename(
        columns={"entity_type": "type", "value_raw": "value", "value_normalised": "normalised"})


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------
def format_sources(hits: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {h['title']} ({h['url']})\n{h['text']}" for i, h in enumerate(hits, 1))


def call_gemini(prompt: str) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    resp = client.models.generate_content(
        model=config.GEMINI_MODEL, contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=600))
    return (resp.text or "").strip()


def call_ollama(prompt: str) -> str:
    resp = requests.post(f"{config.OLLAMA_URL}/api/generate", timeout=120, json={
        "model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.1}})
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def unsupported_figures(answer: str, hits: list[dict]) -> list[str]:
    """Fees, phone numbers and dates in the answer that do not appear in the sources."""
    source = " ".join(" ".join(h["text"].split()) for h in hits).lower()
    source_digits = re.sub(r"\D", "", source)
    missing = []
    for fee in re.findall(r"£\s?\d[\d,]*(?:\.\d+)?", answer):
        if fee.replace(" ", "").lower() not in source.replace(" ", ""):
            missing.append(fee)
    for phone in re.findall(r"\b0\d{2,4}[\s-]?\d{3,4}[\s-]?\d{3,4}\b", answer):
        if re.sub(r"\D", "", phone) not in source_digits:
            missing.append(phone)
    for d in re.findall(r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b", answer):
        if d.lower() not in source:
            missing.append(d)
    return missing


def best_sentences(question: str, text: str, n: int = 3) -> str:
    """Retrieval-only mode: the n sentences of a chunk sharing most words with the question."""
    q = set(bm25_tokens(question)) - {"the", "a", "i", "my", "do", "how", "what", "is", "to", "can", "of"}
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", text) if len(s.split()) >= 4]
    if not sents:
        return text[:400]
    ranked = sorted(range(len(sents)), key=lambda i: -len(q & set(bm25_tokens(sents[i]))))[:n]
    return " ".join(sents[i] for i in sorted(ranked))


def retrieval_only_answer(question: str, hits: list[dict]) -> str:
    lines = ["Here are the most relevant passages from the portal (no AI summary):"]
    for i, h in enumerate(hits, 1):
        lines.append(f"\n[{i}] {h['title']}\n> {best_sentences(question, h['text'])}")
    return "\n".join(lines)


def ask(question: str) -> dict:
    """Answer a question. Returns answer text, mode, confidence, sources and an entity table."""
    question = (question or "").strip()
    if not question:
        return {"answer": "Please type a question.", "mode": "none", "confident": False,
                "confidence": 0.0, "sources": [], "entities": pd.DataFrame(), "hits": []}
    hits = retrieve(question)
    confidence = max((h["dense"] for h in hits), default=0.0)
    sources = []
    for h in hits:
        if h["url"] not in [s["url"] for s in sources]:
            sources.append({"n": len(sources) + 1, "url": h["url"], "title": h["title"],
                            "score": h["score"]})
    result = {"confidence": round(confidence, 3), "hits": hits, "sources": sources}

    if confidence < MIN_CONFIDENCE:
        return {**result, "answer": NOT_FOUND, "mode": "below-threshold", "confident": False,
                "entities": pd.DataFrame(), "sources": []}

    entities = relevant_entities(question, hits)
    mode = llm_mode()
    note = None
    if mode == "retrieval-only":
        answer = retrieval_only_answer(question, hits)
    else:
        prompt = PROMPT.format(not_found=NOT_FOUND, sources=format_sources(hits), question=question)
        try:
            answer = call_gemini(prompt) if mode == "gemini" else call_ollama(prompt)
        except Exception as exc:          # network/quota/model errors: never crash the app
            note = f"{mode} failed ({type(exc).__name__}); showing passages instead."
            answer, mode = retrieval_only_answer(question, hits), "retrieval-only"
        else:
            bad = unsupported_figures(answer, hits)
            if bad:
                note = (f"The AI answer mentioned {', '.join(bad)}, which is not in the sources, "
                        "so the source passages are shown instead.")
                answer, mode = retrieval_only_answer(question, hits), "retrieval-only (guarded)"
    return {**result, "answer": answer, "mode": mode, "confident": True,
            "entities": entities, "note": note}


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python -m chat.ask "your question"')
    # Fresh clone: no live index yet (data/outputs is git-ignored), so use the demo snapshot.
    if not (INDEX_DIR / "faiss.index").exists() and (config.SNAPSHOT / "index" / "faiss.index").exists():
        print("[ask] no live index - using the demo snapshot in data/snapshot/\n")
        use_data(config.SNAPSHOT / "index", config.SNAPSHOT / "entities.csv")
    res = ask(" ".join(sys.argv[1:]))
    print(f"[mode: {res['mode']} | confidence {res['confidence']}]\n")
    if res.get("entities") is not None and not res["entities"].empty:
        print(res["entities"][["type", "value", "normalised"]].to_string(index=False), "\n")
    print(res["answer"])
    if res.get("note"):
        print(f"\nNote: {res['note']}")
    if res["sources"]:
        print("\nSources:")
        for s in res["sources"]:
            print(f"  [{s['n']}] {s['title']} - {s['url']}")


if __name__ == "__main__":
    main()
