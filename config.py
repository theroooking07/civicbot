"""
config.py - one place for every setting CivicBot uses.

Change the target portal here (or via .env) and every module picks it up.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads a local .env file if present (never committed)

# ---------------------------------------------------------------------------
# Target portal
# ---------------------------------------------------------------------------
BASE_URL = os.getenv("CIVICBOT_BASE_URL", "https://www.cambridge.gov.uk/")
ALLOWED_DOMAIN = os.getenv("CIVICBOT_ALLOWED_DOMAIN", "www.cambridge.gov.uk")

# Seed pages: service hubs we want to reach first.
SEED_PATHS = [
    "/",
    "/council-tax",
    "/services-a-to-z",          # A-Z hub links to every service page
    "/licences-and-permits",     # (/licensing is a 404)
    "/planning-and-building-control",
    "/housing",
    "/contact-us",
]

# Bins and recycling are run by the Greater Cambridge Shared Waste Service on
# its own domain (cambridge.gov.uk/bins-and-recycling redirects there), so we
# add it as a second allowed domain. robots_check.py checks it separately.
WASTE_BASE_URL = "https://www.greatercambridgewaste.org/"
WASTE_SEED_PATHS = ["/", "/residents/what-goes-in-which-bin"]

# Every domain the crawler may visit, and every seed as a full URL.
ALLOWED_DOMAINS = [ALLOWED_DOMAIN, "www.greatercambridgewaste.org"]
SEEDS_BY_BASE = {BASE_URL: SEED_PATHS, WASTE_BASE_URL: WASTE_SEED_PATHS}

# Paths we must never crawl (accounts, payments, forms, search results).
# Matched against whole path segments (the bits between "/"), not substrings,
# so information pages like /paying-your-council-tax are still crawled while
# transactional pages like /pay-your-council-tax or /my-account are not.
EXCLUDE_SEGMENTS = {               # segment is exactly one of these
    "login", "log-in", "logout", "sign-in", "signin", "register", "account",
    "my-account", "myaccount", "online-accounts", "payment", "payments", "pay",
    "basket", "checkout", "search", "form", "forms", "apply-online",
}
EXCLUDE_PREFIXES = (               # segment starts with one of these
    "pay-", "login", "sign-in", "my-account", "apply-online",
    "apply-for-", "report-a-", "request-a-", "book-a-",   # application forms
)
EXCLUDE_SUFFIXES = ("-account", "-login")   # e.g. ...-online-council-tax-account
EXCLUDE_QUERY_KEYS = {"q", "query", "search", "keywords"}   # search-result URLs

# Topic keywords used to prioritise links (higher priority = crawled sooner).
PRIORITY_KEYWORDS = [
    "service", "tax", "property", "water", "waste", "sanitation", "garbage",
    "bin", "recycling", "permit", "licence", "license", "certificate",
    "birth", "death", "trade", "building", "grievance", "complaint",
    "contact", "department", "schedule", "deadline", "fee", "charge",
    "form", "notice",
]

# ---------------------------------------------------------------------------
# Politeness limits (do not lower these)
# ---------------------------------------------------------------------------
DOWNLOAD_DELAY = float(os.getenv("CIVICBOT_DOWNLOAD_DELAY", "2.0"))  # seconds
MAX_DEPTH = int(os.getenv("CIVICBOT_MAX_DEPTH", "3"))
MAX_PAGES = int(os.getenv("CIVICBOT_MAX_PAGES", "300"))
MAX_PDF_DOWNLOADS = 15
CONTACT_EMAIL = os.getenv("CIVICBOT_CONTACT_EMAIL", "your-team-email@example.com")
USER_AGENT = f"CivicBot-StudentProject/0.1 (+university web-mining coursework; contact: {CONTACT_EMAIL})"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
CLEAN = DATA / "clean"
OUTPUTS = DATA / "outputs"
SNAPSHOT = DATA / "snapshot"
DOCS = ROOT / "docs"

for _d in (RAW, CLEAN, OUTPUTS, SNAPSHOT, DOCS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LLM (keys only ever come from the environment)
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
