"""
link_graph.py - Module 3a: the site's hyperlink structure as a directed graph.

Nodes are crawled pages; an edge A -> B means page A links to page B.
We build two versions:
  * full graph     - every internal link, including site menus/footers
  * content graph  - menu/footer ("template") links removed, i.e. links to a
                     target that appears on >= TEMPLATE_SHARE of all pages.
                     This shows how departments actually link to each other.

Metrics: PageRank, in/out-degree, HITS hub score, communities (greedy
modularity). Finding: % of linked page pairs that share a topic cluster.

Outputs (data/outputs/): link_metrics.csv, link_report.json, link_graph.png

Usage:
    python -m analysis.link_graph      (run analysis.cluster_pages first for colours)
"""
import json
import sys

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.lines import Line2D
from networkx.algorithms.community import greedy_modularity_communities

import config
from analysis import style

LINKS_IN = config.RAW / "links.csv"
PAGES_IN = config.CLEAN / "pages_clean.jsonl"
CLUSTERS_IN = config.OUTPUTS / "page_clusters.csv"
TEMPLATE_SHARE = 0.25     # a target linked from >= 25% of pages is a menu/footer link
SEED = 42


def short(url: str) -> str:
    """'https://www.cambridge.gov.uk/council-tax' -> 'cambridge.gov.uk/council-tax'"""
    return url.split("://", 1)[-1].removeprefix("www.").rstrip("/")


def build_graphs(links: pd.DataFrame, pages: set[str]) -> tuple[nx.DiGraph, nx.DiGraph, set[str]]:
    """Return (full graph, content graph, template targets)."""
    internal = links[(links.is_internal) & (links.target_type == "html")
                     & links.source_url.isin(pages) & links.target_url.isin(pages)
                     & (links.source_url != links.target_url)]
    edges = internal.groupby(["source_url", "target_url"]).size().reset_index(name="weight")

    full = nx.DiGraph()
    full.add_nodes_from(pages)
    full.add_weighted_edges_from(edges.itertuples(index=False, name=None))

    # How many different pages link to each target?
    linking_pages = edges.groupby("target_url")["source_url"].nunique()
    template = set(linking_pages[linking_pages >= TEMPLATE_SHARE * len(pages)].index)
    content = full.copy()
    content.remove_edges_from([(u, v) for u, v in full.edges if v in template])
    return full, content, template


def metrics(g: nx.DiGraph, prefix: str) -> pd.DataFrame:
    """PageRank, degrees and HITS hub score for every node."""
    pr = nx.pagerank(g, alpha=0.85, weight="weight")
    try:
        hubs, _ = nx.hits(g, max_iter=500, normalized=True)
    except nx.PowerIterationFailedConvergence:
        hubs = {n: float("nan") for n in g}
    return pd.DataFrame({
        f"{prefix}_pagerank": pr,
        f"{prefix}_in_degree": dict(g.in_degree()),
        f"{prefix}_out_degree": dict(g.out_degree()),
        f"{prefix}_hub_score": hubs,
    })


def shared_cluster_share(g: nx.DiGraph, cluster_of: dict) -> dict:
    """Percentage of linked page pairs (both clustered content pages) in the same cluster."""
    pairs = {tuple(sorted((u, v))) for u, v in g.edges
             if cluster_of.get(u, -1) >= 0 and cluster_of.get(v, -1) >= 0}
    same = sum(cluster_of[u] == cluster_of[v] for u, v in pairs)
    return {"linked_pairs": len(pairs), "same_cluster": same,
            "percent_same_cluster": round(100 * same / len(pairs), 1) if pairs else None}


def draw(g: nx.DiGraph, pr: dict, cluster_of: dict, names: dict, path):
    """Spring layout: node size = PageRank, colour = topic cluster.

    Pages with no content links (isolates) are left out of the drawing.
    """
    isolates = [n for n in g if g.degree(n) == 0]
    g = g.subgraph([n for n in g if g.degree(n) > 0])
    und = g.to_undirected()
    pos = nx.spring_layout(und, k=0.22, iterations=200, seed=SEED)
    fig, ax = plt.subplots(figsize=(11, 9))
    nx.draw_networkx_edges(und, pos, ax=ax, width=0.4, edge_color=style.GRID, alpha=0.9)

    max_pr = max(pr.values())
    nodes = list(g.nodes)
    colors = [style.color_for(cluster_of[n]) if cluster_of.get(n, -1) >= 0 else style.NEUTRAL for n in nodes]
    sizes = [20 + 900 * pr[n] / max_pr for n in nodes]
    nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=colors, node_size=sizes, ax=ax,
                           edgecolors=style.SURFACE, linewidths=0.8)

    for n in sorted(pr, key=pr.get, reverse=True)[:10]:     # label the 10 most important pages
        x, y = pos[n]
        label = short(n).rsplit("/", 1)[-1] if "/" in short(n) else short(n)
        label = label if len(label) <= 28 else label[:26] + "..."
        ax.annotate(label, (x, y), xytext=(4, 4),
                    textcoords="offset points", fontsize=8, color=style.TEXT)

    handles = [Line2D([], [], marker="o", linestyle="", markersize=8, color=style.color_for(c),
                      label=names[c]) for c in sorted(names)]
    handles.append(Line2D([], [], marker="o", linestyle="", markersize=8, color=style.NEUTRAL,
                          label="Navigation / hub pages"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), title="Topic cluster")
    ax.set_title(f"Content link graph (menu links removed) - node size = PageRank\n"
                 f"{g.number_of_nodes()} linked pages shown, {len(isolates)} unlinked pages omitted",
                 loc="left")
    ax.axis("off")
    fig.savefig(path)
    plt.close(fig)


def main():
    style.apply()
    for f in (LINKS_IN, PAGES_IN):
        if not f.exists():
            sys.exit(f"Missing {f} - run the crawler and preprocessing first.")
    links = pd.read_csv(LINKS_IN)
    pages = {json.loads(l)["url"] for l in open(PAGES_IN, encoding="utf-8")}

    cluster_of, names = {}, {}
    if CLUSTERS_IN.exists():
        cl = pd.read_csv(CLUSTERS_IN)
        cluster_of = dict(zip(cl.url, cl.cluster))
        names = dict(cl[cl.cluster >= 0].drop_duplicates("cluster")[["cluster", "cluster_name"]].values)
    else:
        print("[graph] page_clusters.csv missing - run analysis.cluster_pages for colours")

    full, content, template = build_graphs(links, pages)
    print(f"[graph] full graph:    {full.number_of_nodes()} nodes, {full.number_of_edges()} edges")
    print(f"[graph] content graph: {content.number_of_edges()} edges "
          f"({len(template)} menu/footer targets removed, e.g. "
          f"{', '.join(sorted(short(t) for t in template)[:4])} ... - full list in link_report.json)")

    df = metrics(full, "full").join(metrics(content, "content"))
    communities = greedy_modularity_communities(content.to_undirected())
    community_of = {n: i for i, comm in enumerate(communities) for n in comm}
    df["community"] = pd.Series(community_of)
    df["cluster"] = pd.Series(cluster_of)
    df.index.name = "url"
    df.sort_values("content_pagerank", ascending=False).to_csv(config.OUTPUTS / "link_metrics.csv")

    def top(col, n=10):
        return [{"url": u, col: round(float(v), 4)} for u, v in df[col].nlargest(n).items()]

    finding_full = shared_cluster_share(full, cluster_of)
    finding_content = shared_cluster_share(content, cluster_of)
    big_communities = [c for c in communities if len(c) >= 5]
    report = {
        "nodes": full.number_of_nodes(),
        "edges_full": full.number_of_edges(),
        "edges_content": content.number_of_edges(),
        "template_targets": sorted(template),
        "top_pagerank_full": top("full_pagerank"),
        "top_pagerank_content": top("content_pagerank"),
        "top_in_degree_content": top("content_in_degree"),
        "top_hubs_content": top("content_hub_score"),
        "communities": [{"size": len(c), "example_pages": [short(u) for u in sorted(c)[:6]]}
                        for c in big_communities],
        "finding_full_graph": finding_full,
        "finding_content_graph": finding_content,
    }
    (config.OUTPUTS / "link_report.json").write_text(json.dumps(report, indent=2))
    draw(content, dict(df["content_pagerank"]), cluster_of, names, config.OUTPUTS / "link_graph.png")

    print("[graph] top pages by PageRank, full graph vs content graph:")
    for a, b in zip(report["top_pagerank_full"][:6], report["top_pagerank_content"][:6]):
        print(f"   {short(a['url']):<45} | {short(b['url'])}")
    print("[graph] top hubs (content graph):", ", ".join(short(h["url"]) for h in report["top_hubs_content"][:5]))
    print(f"[graph] {len(communities)} communities, {len(big_communities)} with 5+ pages")
    for k, f in (("full", finding_full), ("content", finding_content)):
        print(f"[graph] FINDING ({k} graph): {f['percent_same_cluster']}% of {f['linked_pairs']} linked "
              f"page pairs share a topic cluster")
    print(f"[graph] saved link_metrics.csv, link_report.json, link_graph.png to {config.OUTPUTS}")


if __name__ == "__main__":
    main()
