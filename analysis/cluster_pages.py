"""
cluster_pages.py - Module 3b: group pages into topics with TF-IDF + K-Means.

Steps:
  1. TF-IDF on text_tokens (stop-words already removed in clean.py), plus a few
     extra site-wide words that say nothing about the topic ("council", "service")
  2. LSA: reduce TF-IDF to 100 dimensions with truncated SVD (standard for sparse text)
  3. K-Means for k = 3..8, keep the k with the best silhouette score
     Thin hub pages (< MIN_TOKENS words, mostly link lists) are not clustered;
     they get the fixed label "Navigation / hub pages".
  3. name each cluster from its top 8 terms (e.g. "Council tax", "Waste & recycling")
  4. charts: silhouette plot, PCA scatter, cluster sizes, and a heatmap of cosine
     similarity between cluster centroids and the topics in eval_questions.json

Outputs (data/outputs/): page_clusters.csv, clusters.json, silhouette.png,
cluster_scatter.png, cluster_sizes.png, cluster_topic_heatmap.png

Usage:
    python -m analysis.cluster_pages
"""
import json
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

import config
from analysis import style

PAGES_IN = config.CLEAN / "pages_clean.jsonl"
EVAL_QUESTIONS = config.ROOT / "eval_questions.json"
K_RANGE = range(3, 9)      # at most 8 clusters = 8 distinguishable colours
SEED = 42
MIN_TOKENS = 30            # pages shorter than this are hub/navigation pages
HUB_LABEL = "Navigation / hub pages"
SVD_DIMS = 100
N_SEEDS = 5                # K-Means runs averaged per k when choosing k

# Words on almost every council page, plus the "licensing portal temporarily
# offline" notice that appears on many licensing pages. They would otherwise
# pull unrelated pages together.
EXTRA_STOPWORDS = [
    "council", "cambridge", "city", "service", "provide", "information", "need", "make",
    "help", "find", "page", "use", "also", "may", "please", "contact", "include", "new",
    "portal", "restore", "offline", "temporarily", "unavailable", "supplier", "system",
]

# Human-readable names: a cluster gets the label whose keywords overlap most
# with its top terms. Falls back to its top two terms if nothing matches.
LABEL_KEYWORDS = {
    "Council tax": {"council", "tax", "band", "discount", "exemption", "reduction", "bill", "pay"},
    "Waste & recycling": {"bin", "waste", "recycling", "recycle", "collection", "food", "caddy",
                          "plastic", "rubbish", "litter", "green", "blue", "black"},
    "Taxi licensing": {"taxi", "driver", "vehicle", "hire", "private", "operator", "proprietor"},
    "Licences & permits": {"licence", "premises", "gambling", "alcohol", "permit", "trading",
                           "street", "gaming", "application", "animal"},
    "Waste reduction & reuse": {"product", "reusable", "nappy", "period", "reduce", "reuse",
                                "repair", "disposable", "scheme", "item", "refill"},
    "Housing": {"housing", "tenant", "leasehold", "property", "home", "rent", "landlord",
                "homeless", "lease", "repair"},
    "Privacy & data": {"data", "personal", "information", "privacy", "process", "right",
                       "law", "retain", "share"},
    "Environment & streets": {"tree", "street", "clean", "pollution", "noise", "dust",
                              "climate", "energy", "water"},
    "News & community": {"news", "community", "event", "festival", "project", "grant"},
    "Jobs & council": {"employee", "staff", "role", "election", "benefit", "career"},
}


def load_pages() -> pd.DataFrame:
    if not PAGES_IN.exists():
        sys.exit(f"Missing {PAGES_IN} - run  python -m preprocess.clean  first.")
    return pd.DataFrame([json.loads(l) for l in open(PAGES_IN, encoding="utf-8")])


def choose_k(X) -> tuple[int, dict]:
    """Return the k with the best mean silhouette (cosine, LSA space) over N_SEEDS runs.

    One K-Means run is noisy (silhouette varies by ~0.005 between seeds, more
    than the gap between neighbouring k), so we average several seeds.
    """
    scores = {}
    for k in K_RANGE:
        runs = [silhouette_score(X, KMeans(n_clusters=k, n_init=10, random_state=SEED + r).fit_predict(X),
                                 metric="cosine") for r in range(N_SEEDS)]
        scores[k] = float(np.mean(runs))
        print(f"[cluster]   k={k}  mean silhouette={scores[k]:.4f}  (std {np.std(runs):.4f}, {N_SEEDS} seeds)")
    return max(scores, key=scores.get), scores


def name_clusters(top_terms: dict[int, list[str]]) -> dict[int, str]:
    """Pick a readable name per cluster from LABEL_KEYWORDS; make names unique."""
    names, used = {}, defaultdict(int)
    for c, terms in top_terms.items():
        overlap = {label: len(set(terms) & kws) for label, kws in LABEL_KEYWORDS.items()}
        ranked = sorted(overlap, key=overlap.get, reverse=True)
        best, second = ranked[0], ranked[1]
        if overlap[best] < 2:
            name = " / ".join(t.title() for t in terms[:2])
        elif overlap[second] >= 2 and overlap[second] >= overlap[best] - 1:
            name = f"{best} + {second}"          # a mixed cluster: show both themes
        else:
            name = best
        used[name] += 1
        if used[name] > 1:                      # second cluster with the same theme
            name = f"{name} ({terms[0]})"
        names[c] = name
    return names


def plot_silhouette(scores: dict, best_k: int):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ks, vals = list(scores), list(scores.values())
    ax.plot(ks, vals, color=style.color_for(0), linewidth=2, marker="o", markersize=7)
    ax.plot([best_k], [scores[best_k]], marker="o", markersize=11, color=style.color_for(0),
            markeredgecolor=style.SURFACE, markeredgewidth=2)
    ax.annotate(f"best k = {best_k}", (best_k, scores[best_k]), textcoords="offset points",
                xytext=(8, 8), color=style.TEXT, fontsize=9)
    ax.set_xticks(ks)
    ax.set_xlabel("number of clusters (k)")
    ax.set_ylabel(f"mean silhouette (cosine, {N_SEEDS} seeds)")
    ax.grid(axis="y")
    ax.set_title("Choosing k for K-Means")
    fig.savefig(config.OUTPUTS / "silhouette.png")
    plt.close(fig)


def plot_scatter(X, labels, names):
    coords = PCA(n_components=2, random_state=SEED).fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in sorted(names):
        m = labels == c
        ax.scatter(coords[m, 0], coords[m, 1], s=34, color=style.color_for(c),
                   edgecolor=style.SURFACE, linewidth=0.8, label=f"{names[c]} ({m.sum()})")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_title(f"Content pages (PCA of LSA space), coloured by cluster")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.savefig(config.OUTPUTS / "cluster_scatter.png")
    plt.close(fig)


def plot_sizes(labels, names):
    sizes = pd.Series(labels).value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(sizes) + 1.2))
    bars = ax.barh([names[c] for c in sizes.index], sizes.values, height=0.6,
                   color=[style.color_for(c) for c in sizes.index])
    ax.bar_label(bars, padding=4, color=style.TEXT_MUTED, fontsize=9)
    ax.set_xlabel("pages")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Cluster sizes")
    fig.savefig(config.OUTPUTS / "cluster_sizes.png")
    plt.close(fig)


def plot_topic_heatmap(vectorizer, centroids, names):
    """Cosine similarity between each cluster centroid and each eval-question topic."""
    if not EVAL_QUESTIONS.exists():
        print(f"[cluster] {EVAL_QUESTIONS.name} missing - skipping heatmap")
        return None
    from preprocess.clean import load_nlp, load_stopwords, to_tokens
    questions = json.loads(EVAL_QUESTIONS.read_text())
    by_topic = defaultdict(list)
    for q in questions:
        by_topic[q["topic"]].append(q["question"])
    nlp, stop = load_nlp(), load_stopwords()
    topics = sorted(by_topic)
    topic_tokens = [to_tokens(nlp(" ".join(by_topic[t])), stop) for t in topics]
    sim = cosine_similarity(centroids, vectorizer.transform(topic_tokens))

    fig, ax = plt.subplots(figsize=(1.0 * len(topics) + 3, 0.55 * len(names) + 1.8))
    im = ax.imshow(sim, cmap=style.SEQUENTIAL, aspect="auto", vmin=0)
    ax.set_xticks(range(len(topics)), [t.replace("_", " ") for t in topics], rotation=35, ha="right")
    ax.set_yticks(range(len(names)), [names[c] for c in sorted(names)])
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if sim[i, j] > 0.6 * sim.max() else style.TEXT)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, shrink=0.8, label="cosine similarity")
    ax.set_title("Cluster centroids vs citizen-question topics")
    fig.savefig(config.OUTPUTS / "cluster_topic_heatmap.png")
    plt.close(fig)
    return pd.DataFrame(sim, index=[names[c] for c in sorted(names)], columns=topics)


def main():
    style.apply()
    df = load_pages()
    content = df["n_tokens"] >= MIN_TOKENS
    print(f"[cluster] {len(df)} pages: {content.sum()} content pages clustered, "
          f"{(~content).sum()} thin hub pages labelled '{HUB_LABEL}'")
    docs = df.loc[content, "text_tokens"]

    vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, sublinear_tf=True, stop_words=EXTRA_STOPWORDS)
    X = vectorizer.fit_transform(docs)
    Z = normalize(TruncatedSVD(n_components=SVD_DIMS, random_state=SEED).fit_transform(X))
    print(f"[cluster] TF-IDF {X.shape[0]} x {X.shape[1]} terms -> LSA {Z.shape[1]} dims")

    best_k, scores = choose_k(Z)
    labels = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit_predict(Z)

    # Centroids and top terms in the original TF-IDF space (readable words).
    centroids = np.vstack([np.asarray(X[labels == c].mean(axis=0)).ravel() for c in range(best_k)])
    terms = np.array(vectorizer.get_feature_names_out())
    top_terms = {c: list(terms[np.argsort(centroids[c])[::-1][:8]]) for c in range(best_k)}
    names = name_clusters(top_terms)

    print(f"[cluster] best k = {best_k}")
    for c in range(best_k):
        print(f"   {c}  {names[c]:<26} {int((labels == c).sum()):>3} pages | {', '.join(top_terms[c])}")

    plot_silhouette(scores, best_k)
    plot_scatter(Z, labels, names)
    plot_sizes(labels, names)
    heat = plot_topic_heatmap(vectorizer, centroids, names)

    out = df[["url", "title", "domain"]].copy()
    out["cluster"] = -1
    out.loc[content, "cluster"] = labels
    out["cluster_name"] = out["cluster"].map({**names, -1: HUB_LABEL})
    out.to_csv(config.OUTPUTS / "page_clusters.csv", index=False)
    summary = {
        "k": best_k,
        "content_pages": int(content.sum()),
        "hub_pages": int((~content).sum()),
        "silhouette_scores": {str(k): round(v, 4) for k, v in scores.items()},
        "clusters": [{"id": c, "name": names[c], "size": int((labels == c).sum()),
                      "top_terms": top_terms[c]} for c in range(best_k)],
        "topic_similarity": heat.round(3).to_dict() if heat is not None else None,
    }
    (config.OUTPUTS / "clusters.json").write_text(json.dumps(summary, indent=2))
    print(f"[cluster] saved page_clusters.csv, clusters.json and 4 charts to {config.OUTPUTS}")


if __name__ == "__main__":
    main()
