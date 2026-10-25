"""
settings.py - Scrapy settings for CivicBot (the "polite crawling" rules).

Every limit here comes from config.py so there is one place to change it.
run_crawl.py passes this dict to Scrapy's CrawlerProcess.
"""
import config


def get_settings(max_pages: int = config.MAX_PAGES, max_depth: int = config.MAX_DEPTH) -> dict:
    """Return the Scrapy settings dict used for every crawl."""
    return {
        # --- Identify ourselves honestly ---
        "USER_AGENT": config.USER_AGENT,
        "BOT_NAME": "civicbot",

        # --- Compliance ---
        "ROBOTSTXT_OBEY": True,             # Scrapy re-checks robots.txt itself
        "COOKIES_ENABLED": False,           # we never log in or keep a session

        # --- Rate limits ---
        "DOWNLOAD_DELAY": config.DOWNLOAD_DELAY,     # >= 2 seconds between requests
        "DOWNLOAD_DELAY_JITTER": 0,                  # no random jitter: never below DOWNLOAD_DELAY
        "CONCURRENT_REQUESTS": 1,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
        "AUTOTHROTTLE_ENABLED": True,                # slows down further if the server is slow
        "AUTOTHROTTLE_START_DELAY": config.DOWNLOAD_DELAY,
        "AUTOTHROTTLE_MAX_DELAY": 15,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 1.0,
        "DOWNLOAD_TIMEOUT": 30,
        "RETRY_TIMES": 1,

        # --- Scope ---
        "DEPTH_LIMIT": max_depth,           # seeds are depth 0
        "DEPTH_STATS_VERBOSE": True,
        "CIVICBOT_MAX_PAGES": max_pages,    # enforced inside the spider (exact cap)

        # --- Output / logging: the spider prints its own progress lines ---
        "LOG_LEVEL": "WARNING",
        "TELNETCONSOLE_ENABLED": False,
    }
