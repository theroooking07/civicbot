"""
clean.py - Module 2a: turn crawled pages into clean text.

For every page in data/raw/pages.jsonl:
  1. content extraction from the saved raw HTML (trafilatura, BeautifulSoup fallback)
  2. whitespace and special-character fixes, duplicate-line removal, and removal
     of boilerplate lines that repeat across many pages ("Is this page helpful?")
  3. personal-data scrubbing: personal email addresses, names of individual
     people (spaCy PERSON entities) and mobile numbers are replaced with
     placeholders. Department emails and landline/helpline numbers are kept.
  4. two versions of the text:
       text_full   - readable, original case (for embeddings and display)
       text_tokens - lowercase lemmas, NLTK stop-words removed (for TF-IDF / clustering)

Output: data/clean/pages_clean.jsonl

Usage:
    python -m preprocess.clean
"""
import json
import re
import sys
import unicodedata
from collections import Counter

import nltk
import spacy
import trafilatura
from bs4 import BeautifulSoup

import config

PAGES_IN = config.RAW / "pages.jsonl"
PAGES_OUT = config.CLEAN / "pages_clean.jsonl"

# A line that appears on at least this share of pages (and on 10+ pages) is
# boilerplate - e.g. promo banners ("Join us for fireworks night...") and the
# email-subscribe box. Measured on the full crawl: such blocks repeat on 30-47
# pages, while real content lines repeat on at most ~15.
BOILERPLATE_SHARE = 0.08
BOILERPLATE_MIN_PAGES = 10

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Email local parts containing one of these words belong to a team, not a person.
ROLE_EMAIL_WORDS = (
    "info", "enquir", "contact", "customer", "service", "help", "admin", "team",
    "office", "tax", "licens", "waste", "recycl", "bin", "planning", "housing",
    "building", "environment", "foi", "complaint", "benefit", "council", "rent",
    "parking", "democra", "election", "privacy", "data", "access", "feedback",
    "streets", "refuse", "commercial", "digital", "equalit", "noreply", "hello",
)
# "firstname.lastname@..." style local parts are personal.
PERSONAL_LOCAL_RE = re.compile(r"[a-z]+[._-][a-z]+\d*")
UK_MOBILE_RE = re.compile(r"(?<!\d)(?:\+44\s?7|07)\d{3}[\s-]?\d{3}[\s-]?\d{3}(?!\d)")
# Words that show a "PERSON" entity is really a place or an organisation.
NOT_A_PERSON = {"council", "street", "road", "lane", "centre", "house", "court", "park",
                "common", "college", "hall", "close", "avenue", "way", "cambridge",
                # Cambridge areas spaCy mistakes for people
                "hinton", "trumpington", "chesterton", "arbury", "romsey", "newnham",
                "coleridge", "petersfield", "abbey", "milton", "histon", "girton",
                # South Cambridgeshire villages / landmarks seen in the crawl
                "fen", "camps", "drayton", "ditton", "steeple", "morden", "abington",
                "pigotts", "weston", "colville", "queen", "harbour", "embassy", "mandir",
                # products / materials
                "pak", "chloride"}
# Titles that mark a real person's name ("Cllr Jane Doe").
PERSON_TITLES = {"cllr", "councillor", "mr", "mrs", "ms", "miss", "dr", "sir", "dame"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_stopwords() -> set[str]:
    """NLTK English stop-words (downloaded once if missing)."""
    try:
        return set(nltk.corpus.stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(nltk.corpus.stopwords.words("english"))


def load_nlp():
    """spaCy small English model (parser off - we only need tags, lemmas and NER)."""
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser"])
    except OSError:
        sys.exit("spaCy model missing: run  python -m spacy download en_core_web_sm")
    nlp.max_length = 2_000_000
    return nlp


def raw_visible_text(html: str) -> str:
    """Everything a browser would show (menus, footer...) - the 'before' view."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    return "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())


def strip_navigation(html: str) -> BeautifulSoup:
    """Remove menus, headers, footers and ARIA navigation landmarks.

    On short hub pages trafilatura can't find an article and returns the whole
    page, menu included - so we take the menus out before it runs.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["nav", "header", "footer", "script", "style", "noscript"]):
        tag.decompose()
    for tag in soup.find_all(attrs={"role": ["navigation", "banner", "contentinfo", "search"]}):
        tag.decompose()
    return soup


def extract_main(html: str) -> str:
    """Main content: trafilatura first, BeautifulSoup <main>/<article> as fallback."""
    soup = strip_navigation(html)
    text = trafilatura.extract(str(soup), include_comments=False, include_tables=True, favor_recall=True)
    if text and len(text.split()) >= 20:
        return text
    main = soup.find("main") or soup.find("article") or soup.body
    if main is None:
        return text or ""
    for tag in main(["script", "style", "nav", "noscript", "form"]):
        tag.decompose()
    fallback = "\n".join(l.strip() for l in main.get_text("\n").splitlines() if l.strip())
    return fallback if len(fallback) > len(text or "") else (text or "")


def normalise_chars(text: str) -> str:
    """Unicode normalisation, smart quotes/dashes to ASCII, tidy whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                                         "–": "-", "—": "-", " ": " ", "•": "-"}))
    text = re.sub(r"[^\S\n]+", " ", text)            # collapse spaces/tabs, keep newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def dedupe_lines(text: str, boilerplate: set[str]) -> str:
    """Drop repeated lines within the page and site-wide boilerplate lines."""
    seen, kept = set(), []
    for line in text.splitlines():
        key = line.strip().lower()
        if not key:
            continue
        if key in seen or key in boilerplate:
            continue
        seen.add(key)
        kept.append(line.strip())
    return "\n".join(kept)


def scrub_emails(text: str, counts: Counter) -> str:
    """Keep team/department addresses, remove personal-looking ones.

    Removed: firstname.lastname style addresses, and single-word addresses on
    non-government domains unless they name a role (e.g. nicole@charity.org).
    Kept: role addresses like refuse@scambs.gov.uk or licensing@cambridge.gov.uk.
    """
    def repl(m):
        local, domain = m.group(0).lower().split("@")
        personal = bool(PERSONAL_LOCAL_RE.fullmatch(local))
        role = any(word in local for word in ROLE_EMAIL_WORDS)
        gov = domain.endswith((".gov.uk", ".nhs.uk"))
        if not personal and (role or gov):
            return m.group(0)
        counts["emails_removed"] += 1
        return "[email removed]"
    return EMAIL_RE.sub(repl, text)


def scrub_mobiles(text: str, counts: Counter) -> str:
    """Mobile numbers are usually personal - remove them (landlines/helplines are kept)."""
    new, n = UK_MOBILE_RE.subn("[phone removed]", text)
    counts["mobiles_removed"] += n
    return new


def looks_like_person(ent, common_words: set[str]) -> bool:
    """Filter spaCy PERSON entities down to real names.

    spaCy's small model tags things like "Blue Bin" or "Repair Cafe" as people,
    so we only accept an entity when:
      * it is preceded by a title (Cllr, Mr, Dr...), or
      * it is 2-3 Capitalised words, none of which is a place word, and none of
        which also appears in lowercase elsewhere on the site ("bin", "cafe" do;
        "rosy", "moore" don't - so ordinary words are never mistaken for names).
    """
    if ent.label_ != "PERSON" or "\n" in ent.text:
        return False
    prev = ent.doc[ent.start - 1].text.lower().strip(".") if ent.start > 0 else ""
    words = [w.strip(".,'") for w in ent.text.split()]
    if words and words[0].lower().strip(".") in PERSON_TITLES:
        return len(words) >= 2
    if prev in PERSON_TITLES:
        return True
    if not 2 <= len(words) <= 3 or not all(w[:1].isupper() and w.isalpha() for w in words):
        return False
    return not any(w.lower() in NOT_A_PERSON or w.lower() in common_words for w in words)


def lowercase_vocabulary(texts: list[str]) -> set[str]:
    """Words that appear written in lowercase somewhere on the site (i.e. ordinary words)."""
    vocab = set()
    for t in texts:
        vocab.update(re.findall(r"\b[a-z]{2,}\b", t))
    return vocab


def scrub_names(text: str, doc, counts: Counter, common_words: set[str]) -> str:
    """Replace names of individual people (e.g. a named officer) with a placeholder."""
    spans = [(ent.start_char, ent.end_char) for ent in doc.ents if looks_like_person(ent, common_words)]
    for start, end in reversed(spans):       # replace from the end so offsets stay valid
        text = text[:start] + "[name removed]" + text[end:]
    counts["names_removed"] += len(spans)
    return text


def to_tokens(doc, stopwords: set[str]) -> str:
    """Lowercase lemmas of alphabetic words, stop-words and very short words removed."""
    toks = []
    for t in doc:
        if not t.is_alpha:
            continue
        lemma = t.lemma_.lower()
        if len(lemma) < 3 or lemma in stopwords or t.text.lower() in stopwords:
            continue
        toks.append(lemma)
    return " ".join(toks)


def find_boilerplate(texts: list[str]) -> set[str]:
    """Lines that appear on many different pages (headers, feedback widgets, footers)."""
    df = Counter()
    for text in texts:
        df.update({l.strip().lower() for l in text.splitlines() if l.strip()})
    min_pages = max(BOILERPLATE_MIN_PAGES, int(BOILERPLATE_SHARE * len(texts)))
    return {line for line, n in df.items() if n >= min_pages}


# ---------------------------------------------------------------------------
# Public helper for the Streamlit app
# ---------------------------------------------------------------------------
def raw_vs_clean(url: str) -> dict:
    """Return {'url', 'title', 'raw', 'clean', 'tokens'} for one URL (side-by-side view)."""
    page = next((json.loads(l) for l in open(PAGES_IN, encoding="utf-8")
                 if json.loads(l)["url"] == url), None)
    clean = next((json.loads(l) for l in open(PAGES_OUT, encoding="utf-8")
                  if json.loads(l)["url"] == url), None) if PAGES_OUT.exists() else None
    if page is None:
        raise KeyError(f"{url} is not in {PAGES_IN}")
    html_path = config.ROOT / page["html_file"]
    raw = raw_visible_text(html_path.read_text(encoding="utf-8")) if html_path.exists() else page["text"]
    return {
        "url": url,
        "title": page["title"],
        "raw": raw,
        "clean": clean["text_full"] if clean else "",
        "tokens": clean["text_tokens"] if clean else "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not PAGES_IN.exists():
        sys.exit(f"Missing {PAGES_IN} - run  python -m crawler.run_crawl  first.")
    pages = [json.loads(l) for l in open(PAGES_IN, encoding="utf-8")]
    print(f"[clean] {len(pages)} pages from {PAGES_IN.name}")

    stopwords = load_stopwords()
    nlp = load_nlp()

    # Pass 1: content extraction + character fixes.
    extracted = []
    for p in pages:
        html_path = config.ROOT / p.get("html_file", "")
        if html_path.is_file():
            html = html_path.read_text(encoding="utf-8")
            main_text, raw_words = extract_main(html), len(raw_visible_text(html).split())
        else:   # HTML not available (e.g. loaded from an old run) - use crawler text
            main_text, raw_words = p["text"], len(p["text"].split())
        extracted.append((normalise_chars(main_text), raw_words))

    boilerplate = find_boilerplate([t for t, _ in extracted])
    print(f"[clean] {len(boilerplate)} boilerplate lines found across pages")

    # Pass 2: dedupe, scrub, tokenise.
    totals = Counter()
    texts = [dedupe_lines(t, boilerplate) for t, _ in extracted]
    common_words = lowercase_vocabulary(texts)
    with open(PAGES_OUT, "w", encoding="utf-8") as out:
        for p, (_, raw_words), text, doc in zip(pages, extracted, texts, nlp.pipe(texts, batch_size=8)):
            counts = Counter()
            text = scrub_names(text, doc, counts, common_words)   # uses doc offsets, so do it first
            text = scrub_emails(text, counts)
            text = scrub_mobiles(text, counts)
            tokens = to_tokens(doc, stopwords)
            totals.update(counts)
            rec = {
                "url": p["url"],
                "domain": p["url"].split("/")[2],
                "title": p["title"],
                "depth": p["depth"],
                "fetched_at": p["fetched_at"],
                "content_hash": p["content_hash"],
                "text_full": text,
                "text_tokens": tokens,
                "n_words_raw": raw_words,
                "n_words_clean": len(text.split()),
                "n_tokens": len(tokens.split()),
                "scrubbed": dict(counts),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    raw_total = sum(r for _, r in extracted)
    clean_total = sum(len(t.split()) for t in texts)
    print(f"[clean] words: raw visible {raw_total:,} -> clean {clean_total:,} "
          f"({100 * (1 - clean_total / max(raw_total, 1)):.0f}% removed as menus/boilerplate/duplicates)")
    print(f"[clean] personal data scrubbed: {dict(totals) or 'none found'}")
    print(f"[clean] saved {PAGES_OUT}")


if __name__ == "__main__":
    main()
