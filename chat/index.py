"""
index.py - Module 4a: build the search index for the chatbot.

  1. split each cleaned page into ~300-word chunks with a 50-word overlap
  2. embed every chunk with sentence-transformers all-MiniLM-L6-v2
  3. store the vectors in a FAISS index (cosine similarity via inner product)
  4. keep tokenised chunks for BM25 keyword search (hybrid retrieval in ask.py)

Each chunk carries metadata: url, title, cluster.

Outputs (data/outputs/index/): faiss.index, chunks.jsonl

Usage:
    python -m chat.index
"""
import json
import re
import sys

import faiss
import numpy as np
import pandas as pd

import config

PAGES_IN = config.CLEAN / "pages_clean.jsonl"
CLUSTERS_IN = config.OUTPUTS / "page_clusters.csv"
INDEX_DIR = config.OUTPUTS / "index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORDS = 300
OVERLAP_WORDS = 50
MIN_PAGE_WORDS = 30          # skip near-empty hub pages


def chunk_words(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Split text into overlapping windows of `size` words."""
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words) - overlap, step)]


def bm25_tokens(text: str) -> list[str]:
    """Simple keyword tokens for BM25: lowercase words and numbers."""
    return re.findall(r"[a-z0-9£]+", text.lower())


def load_embedder():
    """Load the sentence-transformers model from the local cache; download only if missing.

    Loading from the cache first means the demo works with no network.
    """
    from sentence_transformers import SentenceTransformer
    try:
        return SentenceTransformer(EMBED_MODEL, local_files_only=True)
    except (OSError, ValueError):
        print(f"[index] downloading {EMBED_MODEL} (first run only) ...")
        return SentenceTransformer(EMBED_MODEL)


def main():
    if not PAGES_IN.exists():
        sys.exit(f"Missing {PAGES_IN} - run  python -m preprocess.clean  first.")
    pages = [json.loads(l) for l in open(PAGES_IN, encoding="utf-8")]
    cluster_of = {}
    if CLUSTERS_IN.exists():
        cl = pd.read_csv(CLUSTERS_IN)
        cluster_of = dict(zip(cl.url, cl.cluster_name))

    chunks = []
    for p in pages:
        if p["n_words_clean"] < MIN_PAGE_WORDS:
            continue
        for i, text in enumerate(chunk_words(p["text_full"])):
            chunks.append({"chunk_id": len(chunks), "url": p["url"], "title": p["title"],
                           "cluster": cluster_of.get(p["url"], "unclustered"),
                           "part": i, "text": text})
    print(f"[index] {len(chunks)} chunks from {len({c['url'] for c in chunks})} pages "
          f"(~{CHUNK_WORDS} words, {OVERLAP_WORDS} overlap)")

    print(f"[index] embedding with {EMBED_MODEL} ...")
    model = load_embedder()
    # The page title is prepended so short chunks still know what page they're from.
    texts = [f"{c['title']}\n{c['text']}" for c in chunks]
    vectors = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype="float32")

    index = faiss.IndexFlatIP(vectors.shape[1])   # inner product = cosine on normalised vectors
    index.add(vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c, t in zip(chunks, texts):
            f.write(json.dumps({**c, "bm25_tokens": bm25_tokens(t)}, ensure_ascii=False) + "\n")
    print(f"[index] saved FAISS index ({index.ntotal} vectors, dim {vectors.shape[1]}) "
          f"and chunks.jsonl to {INDEX_DIR}")


if __name__ == "__main__":
    main()
