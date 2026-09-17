from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, EmailStr, field_validator


class LeadershipMember(BaseModel):
    name: str = Field(..., description="Full name of the leader/team member.")
    role: Optional[str] = Field(None, description="Title or role, e.g. 'Co-founder & CEO'.")
    linkedin_url: Optional[str] = Field(
        None, description="LinkedIn profile URL if discoverable from the page content."
    )


class CompanyProfile(BaseModel):
    domain: str = Field(..., description="The input domain, e.g. 'postman.com'.")

    company_overview: str = Field(
        ..., description="A concise 2-sentence summary of what the company does."
    )
    target_audience: str = Field(
        ..., description="Who the product/service is built for (ICP), e.g. "
                          "'Developers building and testing backend APIs'."
    )
    contact_points: List[str] = Field(
        default_factory=list,
        description="Generic/public emails found on the site (contact@, sales@, support@, ...).",
    )
    leadership: List[LeadershipMember] = Field(
        default_factory=list, description="Key leadership / team members discoverable on the site."
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Estimated 0.0-1.0 score for the quality/completeness of the extracted data.",
    )

    # --- Pipeline metadata (not requested from the LLM, filled in by the pipeline itself) ---
    pages_scraped: List[str] = Field(default_factory=list, exclude=False)
    used_browser_fallback: bool = Field(default=False)
    error: Optional[str] = Field(
        default=None,
        description="Populated by the pipeline (not the LLM) if this domain failed; "
        "the record is still emitted instead of crashing the run.",
    )
    tokens_used: Optional[int] = Field(default=None)
    estimated_cost_usd: Optional[float] = Field(default=None)

    @field_validator("contact_points")
    @classmethod
    def dedupe_contacts(cls, v: List[str]) -> List[str]:
        return sorted(set(e.strip().lower() for e in v if e.strip()))

    @classmethod
    def failed(cls, domain: str, error: str) -> "CompanyProfile":
        """Build a placeholder record for a domain that could not be processed at all,
        so one bad domain never crashes the batch."""
        return cls(
            domain=domain,
            company_overview="N/A - extraction failed.",
            target_audience="N/A",
            contact_points=[],
            leadership=[],
            confidence_score=0.0,
            error=error,
        )


def extraction_tool_schema() -> dict:
    schema = CompanyProfile.model_json_schema()
    llm_fields = {
        "company_overview",
        "target_audience",
        "contact_points",
        "leadership",
        "confidence_score",
    }
    schema["properties"] = {k: v for k, v in schema["properties"].items() if k in llm_fields}
    schema["required"] = [f for f in schema.get("required", []) if f in llm_fields]
    schema.pop("$defs", None)
    schema["properties"]["leadership"] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "role": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "linkedin_url": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["name"],
        },
    }
    return schema
