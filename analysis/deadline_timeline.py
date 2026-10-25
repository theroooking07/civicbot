"""
deadline_timeline.py - Module 3c: timeline of upcoming dates found by extraction.

Takes deadline/date entities from data/clean/entities.csv that have an exact
(day-precision) normalised date on or after today, drops page metadata such as
"Next review due", and plots one row per event.

Outputs (data/outputs/): upcoming_deadlines.csv, deadline_timeline.png

Usage:
    python -m analysis.deadline_timeline [--today YYYY-MM-DD]
"""
import argparse
import sys
from datetime import date

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

import config
from analysis import style

ENTITIES_IN = config.CLEAN / "entities.csv"
METADATA_PHRASES = ("next review due", "last reviewed", "page last updated")
MAX_ROWS = 20


def upcoming(today: date) -> pd.DataFrame:
    """Day-precision dates on/after today, one row per (date, sentence)."""
    if not ENTITIES_IN.exists():
        sys.exit(f"Missing {ENTITIES_IN} - run  python -m preprocess.extract  first.")
    e = pd.read_csv(ENTITIES_IN)
    e = e[e.entity_type.isin(["deadline", "date"]) & (e.date_precision == "day")].copy()
    e["date"] = pd.to_datetime(e.value_normalised, errors="coerce")
    e = e[e.date >= pd.Timestamp(today)]
    e = e[~e.context.str.lower().str.contains("|".join(METADATA_PHRASES), na=False)]
    e = e.drop_duplicates(["date", "context"]).sort_values(["date", "entity_type"])
    return e[["date", "entity_type", "value_raw", "context", "url", "title"]]


def short_label(row) -> str:
    """A readable one-line label: page title + the start of the sentence."""
    title = str(row.title).split(" - ")[0].split(": ", 1)[-1]
    if "|" in str(row.context):                      # a row from a table (e.g. cleaning rounds)
        return f"{title}: schedule table entry"
    ctx = " ".join(str(row.context).split())
    text = f"{title}: {ctx}"
    return text if len(text) <= 70 else text[:68] + "..."


def plot(df: pd.DataFrame, today: date):
    df = df.head(MAX_ROWS).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 0.42 * max(len(df), 3) + 1.4))
    if df.empty:
        ax.text(0.5, 0.5, "No upcoming dated deadlines found on the portal",
                ha="center", va="center", color=style.TEXT_MUTED)
        ax.axis("off")
    else:
        y = range(len(df))[::-1]                       # earliest at the top
        is_deadline = df.entity_type == "deadline"
        for flag, color, label in ((True, style.color_for(0), "Deadline"),
                                   (False, style.NEUTRAL, "Other dated event")):
            m = is_deadline == flag
            ax.scatter(df.date[m], [yy for yy, mm in zip(y, m) if mm], s=70, color=color,
                       edgecolor=style.SURFACE, linewidth=1.5, zorder=3, label=label)
        for yy, (_, row) in zip(y, df.iterrows()):
            ax.hlines(yy, pd.Timestamp(today), row.date, color=style.GRID, linewidth=1, zorder=1)
            ax.annotate(f"{row.date:%d %b %Y}", (row.date, yy), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=8, color=style.TEXT_MUTED)
        ax.set_yticks(list(y), [short_label(r) for _, r in df.iterrows()], fontsize=8)
        ax.axvline(pd.Timestamp(today), color=style.TEXT_MUTED, linewidth=1, linestyle=":")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlim(pd.Timestamp(today) - pd.Timedelta(days=7), df.date.max() + pd.Timedelta(days=75))
        ax.legend(loc="upper right")
    ax.set_title(f"Upcoming dates found on the portal (from {today:%d %b %Y})")
    fig.savefig(config.OUTPUTS / "deadline_timeline.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--today", default=date.today().isoformat(), help="reference date (YYYY-MM-DD)")
    today = date.fromisoformat(ap.parse_args().today)
    style.apply()
    df = upcoming(today)
    df.to_csv(config.OUTPUTS / "upcoming_deadlines.csv", index=False)
    plot(df, today)
    print(f"[timeline] {len(df)} upcoming dated events ({(df.entity_type == 'deadline').sum()} deadlines)")
    for _, r in df.iterrows():
        print(f"   {r.date:%Y-%m-%d}  {r.entity_type:<8}  {short_label(r)}")
    print(f"[timeline] saved upcoming_deadlines.csv and deadline_timeline.png to {config.OUTPUTS}")


if __name__ == "__main__":
    main()
