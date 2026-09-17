from __future__ import annotations

import hashlib
from typing import Dict, List

import trafilatura
from bs4 import BeautifulSoup

STRIP_TAGS = ["script", "style", "svg", "nav", "footer", "header", "noscript", "form", "iframe"]
TOKEN_BUDGET_PER_DOMAIN = 7000  
CHARS_PER_TOKEN = 4


def _bs4_fallback(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(STRIP_TAGS):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def extract_clean_text(html: str) -> str:
    text = trafilatura.extract(html, include_links=True, include_tables=False) or ""
    if len(text.strip()) < 200:
        text = _bs4_fallback(html)
    return text.strip()


def _dedupe_paragraphs(pages_text: Dict[str, str]) -> Dict[str, str]:
    seen_hashes: set[str] = set()
    deduped: Dict[str, str] = {}

    for url, text in pages_text.items():
        kept_paragraphs: List[str] = []
        for para in text.split("\n"):
            para_stripped = para.strip()
            if len(para_stripped) < 20:  # too short to be meaningful boilerplate; keep as-is
                kept_paragraphs.append(para)
                continue
            h = hashlib.sha256(para_stripped.encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            kept_paragraphs.append(para)
        deduped[url] = "\n".join(kept_paragraphs)

    return deduped


def clean_pages(raw_pages: Dict[str, str]) -> Dict[str, str]:
   
    extracted = {url: extract_clean_text(html) for url, html in raw_pages.items()}
    deduped = _dedupe_paragraphs(extracted)

    def priority(url: str) -> int:
        low = url.lower()
        if low.rstrip("/").count("/") <= 2:  # homepage
            return 3
        if "contact" in low or "about" in low or "team" in low:
            return 2
        return 1

    ordered_urls = sorted(deduped.keys(), key=priority, reverse=True)

    budget_chars = TOKEN_BUDGET_PER_DOMAIN * CHARS_PER_TOKEN
    used = 0
    trimmed: Dict[str, str] = {}
    for url in ordered_urls:
        text = deduped[url]
        remaining = budget_chars - used
        if remaining <= 0:
            break
        chunk = text[:remaining]
        trimmed[url] = chunk
        used += len(chunk)

    return trimmed


def build_llm_context(clean_pages_dict: Dict[str, str], domain: str) -> str:
    parts = [f"# Source material for domain: {domain}\n"]
    for url, text in clean_pages_dict.items():
        if not text.strip():
            continue
        parts.append(f"\n## Page: {url}\n{text}")
    return "\n".join(parts)
