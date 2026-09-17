# Autonomous Lead Enrichment Agent

A Python pipeline that takes a list of company domains, crawls their public web
presence, and uses an LLM with strict structured outputs to produce clean B2B lead
intelligence (company overview, ICP, contact points, leadership, confidence score).

## Architecture

```
main.py                    orchestrator: loops over domains, catches per-domain
                            failures, writes data/output.json + data/output.csv
agent/
  crawler.py                Step 1 - httpx first, headless Playwright fallback for
                             JS/SPA sites or bot blocks; discovers + ranks subpages
                             (about/team/company/contact/pricing); disk cache
  cleaner.py                Step 2 - trafilatura extraction with BeautifulSoup
                             fallback, cross-page boilerplate dedup, token-budget
                             enforcement (~7k tokens/domain)
  schema.py                 Pydantic CompanyProfile / LeadershipMember models;
                             doubles as the JSON-Schema sent to the LLM
extractor.py               Step 3 - NVIDIA NIM (OpenAI-compatible) function-calling
                              for forced structured output; tracks tokens + estimated
                              cost per domain
  search_enrichment.py      Bonus - optional Tavily lookup for LinkedIn URLs not
                             found on-site (no-op if TAVILY_API_KEY is unset)
tests/test_resilience.py    Proves Step 4 - a dead/blocked domain or LLM failure
                             never crashes the batch
data/output.json, output.csv  Sample output (see note below)
```

## Setup

```bash
git clone <your-repo-url>
cd lead-enrichment-agent
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium

cp .env.example .env
# edit .env and add your NVIDIA_API_KEY (get one free at https://build.nvidia.com/)
```

## Run

```bash
python main.py                                       # runs the 3 default test domains
python main.py --domains postman.com supabase.com vapi.ai
python main.py --domains yourcompany.com              # any arbitrary domain
```

Output is written to `data/output.json` and `data/output.csv`.

## Design notes

- **Never crash the batch.** Every domain runs inside its own try/except in
  `main.process_domain`. A 404, a bot block, a timeout, or an LLM error produces a
  `CompanyProfile.failed(...)` record with `confidence_score=0.0` and an `error`
  string instead of stopping the run. See `tests/test_resilience.py`.
- **Token discipline.** Raw HTML is never sent to the LLM. `cleaner.py` strips
  scripts/styles/nav/footer, dedupes repeated boilerplate paragraphs across pages by
  hashing, and enforces a hard character budget before the context is built.
- **Structured output, not regex.** `extractor.py` uses NVIDIA NIM's function-calling
  with `tool_choice` forced to a single tool, whose `input_schema` is generated directly
  from the Pydantic model — so the shape returned by the LLM is guaranteed to match
  `schema.py`, and Pydantic validates it again on the way back. Some NIM models ignore
  `tool_choice` and emit JSON in the content field; `_parse_content_json()` handles that
  fallback so structured output is still recovered.
- **Anti-hallucination.** The system prompt explicitly instructs the model to leave
  fields empty rather than invent names/emails/URLs not present in the source text.

## About the sample output in `data/`

`data/output.json` / `data/output.csv` were built from real content fetched from the
homepages of postman.com, supabase.com, and vapi.ai, to demonstrate the exact output
shape. Only homepages were captured for this sample (no leadership names or emails
appear on any of the three homepages, which is why those fields are empty — the
pipeline intentionally leaves fields blank rather than guessing).


