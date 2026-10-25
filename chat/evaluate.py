"""
evaluate.py - Module 4c: how well does retrieval find the right page?

For each of the 30 questions in eval_questions.json, a hit means one of the
question's expected URLs is among the top-k distinct pages retrieved.
Reports Recall@1, Recall@3 (the headline number), Recall@5 and MRR, per
topic as well, plus a check of the "couldn't find that" threshold using
off-topic questions that the portal cannot answer.

Outputs (data/outputs/): eval_results.csv, eval_report.json, eval_recall.png

Usage:
    python -m chat.evaluate
"""
import json
import sys

import matplotlib.pyplot as plt
import pandas as pd

import config
from analysis import style
from chat.ask import MIN_CONFIDENCE, ranked_urls, retrieve

EVAL_IN = config.ROOT / "eval_questions.json"

# Questions the portal cannot answer: these should fall below MIN_CONFIDENCE.
OFF_TOPIC = [
    "What is the capital of France?",
    "How do I renew my UK passport?",
    "What's the weather in Cambridge tomorrow?",
    "Which Premier League team won last season?",
    "How do I apply for a birth certificate?",       # county council service, not on these sites
    "What is the best pizza restaurant near me?",
    "How do I reset my iPhone?",
    "When is the next train to London?",
]


def reciprocal_rank(urls: list[str], expected: list[str]) -> float:
    for i, u in enumerate(urls, 1):
        if u in expected:
            return 1 / i
    return 0.0


def plot(per_topic: pd.DataFrame, overall: float):
    per_topic = per_topic.sort_values("recall@3")
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(per_topic) + 1.4))
    bars = ax.barh(per_topic.index.str.replace("_", " "), per_topic["recall@3"], height=0.6,
                   color=style.color_for(0))
    ax.bar_label(bars, labels=[f"{v:.0%}  (n={n})" for v, n in zip(per_topic["recall@3"], per_topic["n"])],
                 padding=4, fontsize=9, color=style.TEXT_MUTED)
    ax.axvline(overall, color=style.TEXT_MUTED, linestyle=":", linewidth=1)
    ax.annotate(f"overall {overall:.0%}", (overall, len(per_topic) - 0.4), xytext=(4, 0),
                textcoords="offset points", fontsize=9, color=style.TEXT_MUTED)
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Recall@3 (expected page in top 3 results)")
    ax.set_title("Retrieval quality by question topic")
    fig.savefig(config.OUTPUTS / "eval_recall.png")
    plt.close(fig)


def main():
    if not EVAL_IN.exists():
        sys.exit(f"Missing {EVAL_IN}")
    style.apply()
    questions = json.loads(EVAL_IN.read_text())
    rows = []
    for q in questions:
        urls = ranked_urls(q["question"], n=5)
        conf = max(h["dense"] for h in retrieve(q["question"]))
        hit_rank = next((i for i, u in enumerate(urls, 1) if u in q["expected_urls"]), None)
        rows.append({"id": q["id"], "topic": q["topic"], "question": q["question"],
                     "hit_rank": hit_rank, "recall@1": hit_rank == 1,
                     "recall@3": hit_rank is not None and hit_rank <= 3,
                     "recall@5": hit_rank is not None,
                     "rr": reciprocal_rank(urls, q["expected_urls"]),
                     "confidence": round(conf, 3), "answered": conf >= MIN_CONFIDENCE,
                     "top3": " | ".join(u.split("/", 3)[-1] for u in urls[:3])})
        mark = "OK  " if rows[-1]["recall@3"] else "MISS"
        print(f"  {mark} #{q['id']:<2} rank={str(hit_rank):<4} conf={conf:.2f}  {q['question']}")
    df = pd.DataFrame(rows)
    df.to_csv(config.OUTPUTS / "eval_results.csv", index=False)

    off = [{"question": q, "confidence": round(max(h["dense"] for h in retrieve(q)), 3)} for q in OFF_TOPIC]
    for o in off:
        o["refused"] = o["confidence"] < MIN_CONFIDENCE

    per_topic = df.groupby("topic").agg(n=("id", "count"), **{"recall@3": ("recall@3", "mean")})
    report = {
        "n_questions": len(df),
        "recall@1": round(df["recall@1"].mean(), 3),
        "recall@3": round(df["recall@3"].mean(), 3),
        "recall@5": round(df["recall@5"].mean(), 3),
        "mrr": round(df["rr"].mean(), 3),
        "per_topic_recall@3": per_topic["recall@3"].round(3).to_dict(),
        "threshold": MIN_CONFIDENCE,
        "in_scope_answered": int(df["answered"].sum()),
        "in_scope_min_confidence": float(df["confidence"].min()),
        "off_topic": off,
        "off_topic_refused": sum(o["refused"] for o in off),
        "misses": df[~df["recall@3"]][["id", "question", "top3"]].to_dict("records"),
    }
    (config.OUTPUTS / "eval_report.json").write_text(json.dumps(report, indent=2))
    plot(per_topic, report["recall@3"])

    print(f"\n[eval] Recall@1 {report['recall@1']:.0%} | Recall@3 {report['recall@3']:.0%} | "
          f"Recall@5 {report['recall@5']:.0%} | MRR {report['mrr']:.2f}")
    print(f"[eval] threshold {MIN_CONFIDENCE}: in-scope answered {report['in_scope_answered']}/{len(df)} "
          f"(lowest confidence {report['in_scope_min_confidence']:.2f}); "
          f"off-topic refused {report['off_topic_refused']}/{len(off)}")
    for o in off:
        print(f"   {'refused ' if o['refused'] else 'ANSWERED'} conf={o['confidence']:.2f}  {o['question']}")
    print(f"[eval] saved eval_results.csv, eval_report.json, eval_recall.png to {config.OUTPUTS}")


if __name__ == "__main__":
    main()
