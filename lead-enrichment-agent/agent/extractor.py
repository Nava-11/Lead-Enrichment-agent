from __future__ import annotations

import json
import logging
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_fixed,
    stop_after_attempt,
    before_sleep_log,
)

from .schema import CompanyProfile, extraction_tool_schema

load_dotenv()

logger = logging.getLogger("agent.extractor")

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.environ.get(
    "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
)
MODEL = os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
TOOL_NAME = "emit_company_profile"


PRICE_PER_INPUT_TOKEN = float(os.environ.get("PRICE_PER_INPUT_TOKEN", 0.0))
PRICE_PER_OUTPUT_TOKEN = float(os.environ.get("PRICE_PER_OUTPUT_TOKEN", 0.0))

SYSTEM_PROMPT = """You are a precise B2B lead-research analyst. You will be given cleaned \
text scraped from a company's public website. Extract ONLY facts that are explicitly present \
in the provided text. Do not invent names, titles, emails, or URLs that do not appear in the \
source material - if information is missing, leave the field empty rather than guessing. \
Call the emit_company_profile tool exactly once with your findings."""


def _build_client() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(api_key=api_key, base_url=NVIDIA_BASE_URL)


class _TransientAPIError(Exception):
    """Raised on transient OpenAI-compatible API failures (500/503/429) worth retrying."""


def _retry_delay_seconds(exc: BaseException) -> float:
    """Parse a server-provided retry delay (e.g. 'Please retry in 35.2s') out of the body."""
    msg = str(exc)
    m = re.search(r"retry in\s*([0-9]+(?:\.[0-9]+)?)\s*s", msg, re.I)
    if m:
        return float(m.group(1))
    return 0.0


def _is_transient(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in ("500", "503", "429", "internal server error",
                                   "resource_exhausted", "rate limit",
                                   "service unavailable", "5xx"))


def _make_wait():
    """Tenacity wait strategy honoring the server's retry delay when present."""
    class _ServerAwareWait:
        def __call__(self, retry_state):
            exc = retry_state.outcome.exception()
            delay = _retry_delay_seconds(exc) if exc else 0.0
            return max(delay, 5.0)
    return _ServerAwareWait()


@retry(
    stop=stop_after_attempt(4),
    wait=_make_wait(),
    retry=retry_if_exception_type(_TransientAPIError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_llm(client: OpenAI, model: str, **kwargs):
    try:
        return client.chat.completions.create(model=model, **kwargs)
    except Exception as e:
        if _is_transient(e):
            raise _TransientAPIError(str(e)) from e
        raise


def _parse_content_json(content: str, domain: str) -> dict:
    """Some NVIDIA models ignore tool_choice and emit JSON in the content field.
    Extract the first JSON object from the content so we still get structured output."""
    text = content.strip()
   
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Could not parse JSON from model content for {domain}: {e}"
            ) from e
    raise RuntimeError(f"No JSON object found in model content for {domain}")


def extract_profile(domain: str, context_text: str, pages_scraped: list[str]) -> CompanyProfile:
    """Call the LLM once per domain with the pre-cleaned context and return a validated
    CompanyProfile. Raises only on hard API failures - caller wraps this per-domain so one
    failure doesn't abort the whole batch."""

    client = _build_client()

    response = _call_llm(
        client,
        MODEL,
        max_tokens=4096,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context_text},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": TOOL_NAME,
                    "description": "Emit the structured company profile extracted from the source text.",
                    "parameters": extraction_tool_schema(),
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
    )

    message = response.choices[0].message
    tool_calls = message.tool_calls or []
    if tool_calls:
       
        target = next(
            (tc for tc in tool_calls if tc.function.name == TOOL_NAME), tool_calls[0]
        )
        data = json.loads(target.function.arguments)
    elif message.content:
       
        data = _parse_content_json(message.content, domain)
    else:
        raise RuntimeError(f"Model did not return a tool_call or JSON for {domain}")

    data["domain"] = domain
    data["pages_scraped"] = pages_scraped

    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    data["tokens_used"] = input_tokens + output_tokens
    data["estimated_cost_usd"] = round(
        input_tokens * PRICE_PER_INPUT_TOKEN + output_tokens * PRICE_PER_OUTPUT_TOKEN, 6
    )

    return CompanyProfile.model_validate(data)