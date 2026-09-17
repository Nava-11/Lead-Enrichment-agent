from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger("agent.crawler")

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 LeadEnrichmentBot/1.0"
)

PRIORITY_PATH_KEYWORDS = [
    "about", "team", "company", "leadership", "contact", "pricing",
    "careers", "who-we-are", "founders", "our-story",
]

MAX_PAGES_PER_DOMAIN = 5
REQUEST_TIMEOUT = 15.0


def _cache_path(url: str) -> Path:
    key = hashlib.sha256(url.encode()).hexdigest()
    return CACHE_DIR / f"{key}.html"


def _looks_js_shell(html: str) -> bool:
    text_len = len(re.sub(r"<[^>]+>", " ", html))
    script_len = sum(len(m) for m in re.findall(r"<script.*?</script>", html, re.S))
    return text_len < 500 or (script_len > 0 and script_len / max(text_len, 1) > 3)


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    reraise=True,
)
def _fetch_httpx(client: httpx.Client, url: str) -> Optional[str]:
    resp = client.get(url, timeout=REQUEST_TIMEOUT, follow_redirects=True)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.text


def _fetch_playwright(url: str) -> Optional[str]:

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.warning("Playwright not installed; skipping browser fallback for %s", url)
        return None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=USER_AGENT)
            page.goto(url, timeout=20_000, wait_until="networkidle")
            html = page.content()
            browser.close()
            return html
    except Exception as e:  
        logger.warning("Playwright fallback failed for %s: %s", url, e)
        return None


def fetch_page(client: httpx.Client, url: str, use_cache: bool = True) -> Optional[str]:
    """Fetch a single page, using cache -> httpx -> Playwright fallback, in that order."""
    cache_file = _cache_path(url)
    if use_cache and cache_file.exists():
        return cache_file.read_text(encoding="utf-8", errors="ignore")

    html: Optional[str] = None
    try:
        html = _fetch_httpx(client, url)
    except httpx.HTTPStatusError as e:
        logger.info("httpx blocked (%s) on %s, trying browser fallback", e.response.status_code, url)
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        logger.info("httpx failed (%s) on %s, trying browser fallback", e, url)

    if html is None or _looks_js_shell(html):
        browser_html = _fetch_playwright(url)
        if browser_html:
            html = browser_html

    if html:
        cache_file.write_text(html, encoding="utf-8")
    return html


def discover_links(html: str, base_url: str) -> List[str]:
   
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, re.I)
    domain = urlparse(base_url).netloc

    scored: Dict[str, int] = {}
    for href in hrefs:
        full = urljoin(base_url, href.split("#")[0])
        parsed = urlparse(full)
        if parsed.netloc != domain or not parsed.scheme.startswith("http"):
            continue
        path = parsed.path.lower()
        score = sum(2 for kw in PRIORITY_PATH_KEYWORDS if kw in path)
        if score > 0:
            scored[full] = max(scored.get(full, 0), score)

    ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
    return [url for url, _ in ranked]


def crawl_domain(domain: str, max_pages: int = MAX_PAGES_PER_DOMAIN) -> Dict[str, str]:
    """Crawl a single domain: homepage + top-scoring subpages.

    Returns {url: raw_html}. Never raises - failures are simply omitted from the dict,
    and the caller can check len(pages) to decide whether extraction is even worthwhile.
    """
    base_url = f"https://{domain}"
    pages: Dict[str, str] = {}

    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
    with httpx.Client(headers=headers) as client:
        home_html = fetch_page(client, base_url)
        if not home_html:
            logger.error("Could not fetch homepage for %s", domain)
            return pages
        pages[base_url] = home_html

        for link in discover_links(home_html, base_url)[: max_pages - 1]:
            try:
                html = fetch_page(client, link)
                if html:
                    pages[link] = html
            except Exception as e: 
                logger.warning("Skipping %s after repeated failures: %s", link, e)
                continue

    return pages
