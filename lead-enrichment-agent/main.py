
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from agent.crawler import crawl_domain
from agent.cleaner import clean_pages, build_llm_context
from agent.extractor import extract_profile
from agent.schema import CompanyProfile

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agent.main")

DEFAULT_DOMAINS = ["postman.com", "supabase.com", "vapi.ai"]
DATA_DIR = Path(__file__).resolve().parent / "data"


def process_domain(domain: str) -> CompanyProfile:
    logger.info("=== %s: crawling ===", domain)
    try:
        raw_pages = crawl_domain(domain)
        if not raw_pages:
            return CompanyProfile.failed(domain, "No pages could be fetched (site unreachable "
                                                   "or fully blocked homepage).")

        logger.info("%s: fetched %d page(s), cleaning content", domain, len(raw_pages))
        cleaned = clean_pages(raw_pages)
        context = build_llm_context(cleaned, domain)

        logger.info("%s: sending ~%d chars of context to LLM", domain, len(context))
        profile = extract_profile(domain, context, pages_scraped=list(raw_pages.keys()))
        logger.info(
            "%s: done (confidence=%.2f, tokens=%s, cost=$%.5f)",
            domain, profile.confidence_score, profile.tokens_used, profile.estimated_cost_usd or 0.0,
        )
        return profile

    except Exception as e:  # noqa: BLE001 - top-level safety net per domain requirement
        logger.error("%s: FAILED - %s", domain, e)
        return CompanyProfile.failed(domain, str(e))


def run(domains: List[str]) -> List[CompanyProfile]:
    results: List[CompanyProfile] = []
    for domain in domains:
        start = time.time()
        results.append(process_domain(domain))
        logger.info("%s: took %.1fs\n", domain, time.time() - start)
    return results


def write_outputs(results: List[CompanyProfile]) -> None:
    DATA_DIR.mkdir(exist_ok=True)

    json_path = DATA_DIR / "output.json"
    json_path.write_text(
        json.dumps([r.model_dump() for r in results], indent=2), encoding="utf-8"
    )

    csv_path = DATA_DIR / "output.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "domain", "company_overview", "target_audience", "contact_points",
            "leadership", "confidence_score", "pages_scraped", "tokens_used",
            "estimated_cost_usd", "error",
        ])
        for r in results:
            writer.writerow([
                r.domain,
                r.company_overview,
                r.target_audience,
                "; ".join(r.contact_points),
                "; ".join(f"{m.name} ({m.role or 'n/a'})" for m in r.leadership),
                r.confidence_score,
                "; ".join(r.pages_scraped),
                r.tokens_used or "",
                r.estimated_cost_usd or "",
                r.error or "",
            ])

    logger.info("Wrote %s and %s", json_path, csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Lead Enrichment Agent")
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS,
                         help="Space-separated list of company domains to process.")
    args = parser.parse_args()

    results = run(args.domains)
    write_outputs(results)

    total_cost = sum(r.estimated_cost_usd or 0.0 for r in results)
    total_tokens = sum(r.tokens_used or 0 for r in results)
    logger.info(
        "Batch complete: %d/%d domains succeeded | %d total tokens | $%.5f estimated cost",
        sum(1 for r in results if r.error is None), len(results), total_tokens, total_cost,
    )


if __name__ == "__main__":
    sys.exit(main())
