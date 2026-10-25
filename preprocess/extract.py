"""
extract.py - Module 2b: pull structured facts out of the cleaned text.

Entity types (regex + spaCy NER, dates normalised with dateparser):
  deadline       a date or time limit in a sentence that sounds like a deadline
                 ("pay by 1 April 2026", "within 21 days")
  date           any other explicit date
  fee            money amounts (£45, £1,234.50)
  phone          landline / helpline numbers (mobiles were scrubbed in clean.py)
  office_hours   "Monday to Friday, 9am to 5pm"
  schedule       collection patterns ("every other week", "collected on Tuesday")

Rule: we never guess. A date without an explicit year keeps value_normalised
empty (null), and a page with no match for a type simply has no row for it.

Outputs:
  data/clean/entities.csv        one row per entity found
  data/clean/page_entities.csv   one row per page, first value per type or null

Usage:
    python -m preprocess.extract
"""
import json
import re
import sys
import warnings

import dateparser
import pandas as pd

import config
from preprocess.clean import load_nlp

PAGES_IN = config.CLEAN / "pages_clean.jsonl"
ENTITIES_OUT = config.CLEAN / "entities.csv"
SUMMARY_OUT = config.CLEAN / "page_entities.csv"
ENTITY_TYPES = ["deadline", "date", "fee", "phone", "office_hours", "schedule"]

# dateparser emits a noisy pytz deprecation warning on some setups
warnings.filterwarnings("ignore", category=DeprecationWarning)

MONTHS = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|" \
         r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
DAYS = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"

DATE_RE = re.compile(
    rf"\b(?:\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS}(?:\s+\d{{4}})?"      # 1 April 2026 / 1 April
    rf"|{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}"               # April 1, 2026
    rf"|{MONTHS}\s+\d{{4}}"                                              # April 2026
    rf"|\d{{1,2}}/\d{{1,2}}/\d{{4}})\b", re.I)                           # 01/04/2026
RELATIVE_DEADLINE_RE = re.compile(r"\bwithin\s+\d+\s+(?:working\s+)?(?:days?|weeks?|months?)\b", re.I)
DEADLINE_WORDS = re.compile(
    r"\b(deadline|due|by|before|no later than|closing date|closes|expires?|expiry|"
    r"must (?:be )?(?:paid|submitted|received|renewed)|renew(?:al)?|last date|until)\b", re.I)

FEE_RE = re.compile(r"£\s?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?(?:\s?(?:million|m|k)\b)?"
                    r"(?:\s?(?:per|a|each)\s(?:year|month|week|day|annum|item|bin|collection|hour))?", re.I)
PHONE_RE = re.compile(r"(?<![\d£])(?:\+44\s?\(?0?\)?\s?|\(?0)\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}(?!\d)")
HOURS_RE = re.compile(
    rf"\b{DAYS}(?:s)?(?:\s*(?:to|-|and)\s*{DAYS}(?:s)?)?[,:]?\s*(?:from\s+)?"
    r"\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm)\s*(?:to|-|until)\s*\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm)", re.I)
SCHEDULE_RE = re.compile(
    rf"\b(?:every\s+(?:other\s+)?(?:week|fortnight|two weeks|{DAYS})"
    rf"|(?:weekly|fortnightly|monthly)\s+(?:collection|service)s?"
    rf"|collected\s+(?:on\s+(?:a\s+)?\w+|every\s+(?:other\s+)?\w+)"
    rf"|(?:collection|bin)s?\s+day"
    r"|(?:by|from|before)\s+\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm)\s+on\s+(?:the\s+day\s+of\s+)?(?:your\s+)?collection(?:\s+day)?"
    r"|from\s+\d{4}\s+hours\s+on\s+the\s+day\s+of\s+collection)", re.I)


def sentences(text: str) -> list[str]:
    """Split text into short sentences/lines for context."""
    out = []
    for line in text.splitlines():
        out.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip())
    return out


def normalise_date(raw: str) -> tuple[str | None, str | None]:
    """Return (ISO date, precision) only when the year is written; never guess."""
    if not re.search(r"\b\d{4}\b", raw):
        return None, None
    settings = {"DATE_ORDER": "DMY", "REQUIRE_PARTS": ["day", "month", "year"]}
    dt = dateparser.parse(raw, languages=["en"], settings=settings)
    if dt:
        return dt.date().isoformat(), "day"
    dt = dateparser.parse(raw, languages=["en"],
                          settings={"REQUIRE_PARTS": ["month", "year"], "PREFER_DAY_OF_MONTH": "first"})
    if dt:
        return dt.strftime("%Y-%m"), "month"
    return None, None


def normalise_phone(raw: str) -> str | None:
    """Digits only, UK format check (10-11 digits starting 0). None if implausible."""
    digits = re.sub(r"\D", "", raw)
    if digits.startswith("44"):
        digits = "0" + digits[2:].lstrip("0")
    if not (digits.startswith("0") and len(digits) in (10, 11)) or digits.startswith("07"):
        return None
    return digits


def page_entities(page: dict, nlp_dates) -> list[dict]:
    """All entities for one page as a list of row dicts."""
    rows, seen = [], set()

    def add(etype, raw, norm, ctx, source, precision=None):
        key = (etype, raw.lower())
        if key in seen:
            return
        seen.add(key)
        rows.append({"url": page["url"], "title": page["title"], "entity_type": etype,
                     "value_raw": raw.strip(" ()"), "value_normalised": norm,
                     "date_precision": precision, "context": ctx[:250], "source": source})

    sents = sentences(page["text_full"])
    for s in sents:
        is_deadline = bool(DEADLINE_WORDS.search(s))
        for m in DATE_RE.finditer(s):
            iso, prec = normalise_date(m.group(0))
            add("deadline" if is_deadline else "date", m.group(0), iso, s, "regex", prec)
        for m in RELATIVE_DEADLINE_RE.finditer(s):
            add("deadline", m.group(0), None, s, "regex")
        for m in FEE_RE.finditer(s):
            amount = re.search(r"\d[\d,]*(?:\.\d+)?", m.group(0)).group(0).replace(",", "")
            add("fee", m.group(0), amount, s, "regex")
        for m in PHONE_RE.finditer(s):
            norm = normalise_phone(m.group(0))
            if norm:
                add("phone", m.group(0), norm, s, "regex")
        for m in HOURS_RE.finditer(s):
            add("office_hours", m.group(0), None, s, "regex")
        for m in SCHEDULE_RE.finditer(s):
            add("schedule", m.group(0), None, s, "regex")

    # spaCy DATE entities catch phrasing regex misses ("the end of March").
    # Only sentences that mention a month, weekday or year are sent to spaCy (speed).
    date_hint = re.compile(rf"{MONTHS}|{DAYS}|\b\d{{4}}\b", re.I)
    candidates = [s for s in sents if date_hint.search(s)]
    for s, doc in zip(candidates, nlp_dates.pipe(candidates, batch_size=64)):
        is_deadline = bool(DEADLINE_WORDS.search(s))
        for ent in doc.ents:
            if ent.label_ != "DATE" or not re.search(rf"{MONTHS}|\b(?:19|20)\d{{2}}\b", ent.text, re.I):
                continue  # skip vague ones like "today", "weekly"
            if re.search(r"\d{3}\s\d{3,4}", ent.text):
                continue  # spaCy sometimes tags part of a phone number as a DATE
            if re.fullmatch(r"(?:the\s+)?(?:19|20)\d{2}s?", ent.text.strip()):
                continue  # a bare year ("2018") is too vague to be a useful date
            if any(ent.text.lower() in r["value_raw"].lower() or r["value_raw"].lower() in ent.text.lower()
                   for r in rows if r["entity_type"] in ("date", "deadline")):
                continue  # regex already has it
            iso, prec = normalise_date(ent.text)
            add("deadline" if is_deadline else "date", ent.text, iso, s, "spacy", prec)
    return rows


def main():
    if not PAGES_IN.exists():
        sys.exit(f"Missing {PAGES_IN} - run  python -m preprocess.clean  first.")
    pages = [json.loads(l) for l in open(PAGES_IN, encoding="utf-8")]
    print(f"[extract] {len(pages)} pages")
    nlp_dates = load_nlp()
    nlp_dates.select_pipes(enable=["ner"])   # NER only - much faster (sm's NER has its own tok2vec)

    rows = []
    for i, page in enumerate(pages, 1):
        rows.extend(page_entities(page, nlp_dates))
        if i % 50 == 0:
            print(f"[extract]   {i}/{len(pages)} pages, {len(rows)} entities so far")

    cols = ["url", "title", "entity_type", "value_raw", "value_normalised",
            "date_precision", "context", "source"]
    ents = pd.DataFrame(rows, columns=cols)
    ents.to_csv(ENTITIES_OUT, index=False)

    # One row per page: first value of each type, or null when nothing was found.
    summary = pd.DataFrame({"url": [p["url"] for p in pages], "title": [p["title"] for p in pages]})
    for etype in ENTITY_TYPES:
        first = ents[ents.entity_type == etype].groupby("url")["value_raw"].first()
        summary[etype] = summary["url"].map(first)          # NaN (null) when not found
    summary.to_csv(SUMMARY_OUT, index=False)

    print(f"\n[extract] {len(ents)} entities saved to {ENTITIES_OUT}")
    print("[extract] pages with at least one value (found) vs null:")
    for etype in ENTITY_TYPES:
        found = summary[etype].notna().sum()
        print(f"   {etype:<13} found on {found:>3} pages | null on {len(summary) - found:>3} "
              f"| {int((ents.entity_type == etype).sum()):>4} values total")
    print(f"[extract] saved {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
