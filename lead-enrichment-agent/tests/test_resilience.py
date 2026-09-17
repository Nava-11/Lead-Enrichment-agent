"""Proves Step 4 (Fallback & Resilience): process_domain() never raises, even when
crawling fails completely, and always returns a valid CompanyProfile."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest.mock import patch

from agent.schema import CompanyProfile
from main import process_domain


def test_unreachable_domain_returns_failed_profile_not_exception():
    with patch("main.crawl_domain", return_value={}):
        profile = process_domain("this-domain-does-not-exist-12345.com")
    assert isinstance(profile, CompanyProfile)
    assert profile.confidence_score == 0.0
    assert profile.error is not None


def test_extractor_exception_is_caught_and_recorded():
    with patch("main.crawl_domain", return_value={"https://example.com": "<html>hi</html>"}), \
         patch("main.extract_profile", side_effect=RuntimeError("LLM API down")):
        profile = process_domain("example.com")
    assert profile.error is not None
    assert "LLM API down" in profile.error


def test_schema_failed_factory_is_valid():
    p = CompanyProfile.failed("x.com", "boom")
    assert p.domain == "x.com"
    assert p.confidence_score == 0.0
    assert p.leadership == []
    assert p.contact_points == []


if __name__ == "__main__":
    test_unreachable_domain_returns_failed_profile_not_exception()
    test_extractor_exception_is_caught_and_recorded()
    test_schema_failed_factory_is_valid()
    print("All resilience tests passed.")
