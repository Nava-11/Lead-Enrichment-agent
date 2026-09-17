from __future__ import annotations

import os
from typing import Optional

from agent.schema import CompanyProfile


def find_linkedin_url(person_name: str, company_domain: str) -> Optional[str]:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return None

    try:
        from tavily import TavilyClient  
    except ImportError:
        return None

    client = TavilyClient(api_key=api_key)
    query = f'"{person_name}" {company_domain} LinkedIn'
    results = client.search(query, max_results=3)
    for r in results.get("results", []):
        if "linkedin.com/in/" in r.get("url", ""):
            return r["url"]
    return None


def enrich_leadership_with_search(profile: CompanyProfile) -> CompanyProfile:
    """Fill in missing linkedin_url fields via search, in place. No-op without TAVILY_API_KEY."""
    if not os.environ.get("TAVILY_API_KEY"):
        return profile

    for member in profile.leadership:
        if not member.linkedin_url:
            member.linkedin_url = find_linkedin_url(member.name, profile.domain)
    return profile
