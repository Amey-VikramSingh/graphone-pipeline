# GraphOne pipeline

[![tests](https://github.com/Amey-VikramSingh/graphone-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/Amey-VikramSingh/graphone-pipeline/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
[![Data: Google Sheet](https://img.shields.io/badge/data-Google%20Sheet-34A853?logo=googlesheets&logoColor=white)](https://docs.google.com/spreadsheets/d/1kJ6kOPlylsO9UD-tOW9UeRlYpGYAS53Wj19uPaYS2kU/edit?usp=sharing)

A fault-tolerant ingestion pipeline for a knowledge graph of the AI ecosystem.
It collects startups, products, research papers (with GitHub star counts), job
postings and news, validates every record against a schema, links mentions of
the same company together, and stores the results in SQLite with an optional
Google Sheets export.

**Results from the latest run:** [view the data in Google Sheets](https://docs.google.com/spreadsheets/d/1kJ6kOPlylsO9UD-tOW9UeRlYpGYAS53Wj19uPaYS2kU/edit?usp=sharing) (six tabs:
startups, products, research papers, jobs, news and the entity mapping log).

The interesting part is what happens when sources misbehave: APIs rate-limit,
pages are too large for a model's context window, API keys expire, and some
sites block scrapers. Each of those has a defined path that degrades the run
instead of crashing it.

## Sources

| Data | Source | Notes |
|---|---|---|
| Research papers | arXiv API (cs.AI, cs.LG, cs.CL, cs.CV, stat.ML, cs.NE) | GitHub repo links found in the abstract and comments, then star counts from the GitHub API |
| Products | Product Hunt GraphQL API | Pricing model classified by an LLM |
| Startups | Wikidata SPARQL endpoint (CC0) | Companies whose industry is AI or machine learning; no key or LLM needed |
| Jobs | Greenhouse, Lever and Ashby boards (OpenAI, Anthropic, Scale AI, Cohere, ElevenLabs) | Only postings from the last 24 hours |
| News | RSS: TechCrunch AI, CSET, BAIR, NVIDIA, AWS ML | Article text extracted from HTML; only the last 24 hours |

## How it handles failure

- **Rate limits (429).** Requests to a model retry with exponential backoff and
  jitter, then move to the next provider in the chain: Gemini Flash, then Groq
  (gpt-oss-120b, then gpt-oss-20b), then DeepSeek.
- **Oversized input (413).** Text is chunked by token count at paragraph
  boundaries before it reaches a model. If a provider still rejects it, the
  chunk is split again and retried.
- **Dead accounts.** A provider that fails in a way retrying can't fix
  (rejected key, no balance, exhausted quota, retired model) is skipped for the
  rest of the run, so one bad account costs one request instead of one per item.
- **Never guessing.** A record that doesn't validate against its schema is
  dropped and logged rather than filled in. If every model fails, a product's
  pricing model is recorded as `UNKNOWN`, not a plausible default.
- **Freshness.** Dates are normalized ("2 hours ago", missing meta tags, ISO
  strings), and job and news items outside a 24-hour window are rejected. A
  seen-URL store prevents processing the same item twice.
- **Blocked pages.** News pages that refuse plain HTTP clients fall back to a
  headless Chromium browser (Playwright).
- **Missing credentials.** A source whose credentials are missing fails with an
  error naming what to set; the other sources still run.

## Entity resolution

Company names arrive in many forms ("OpenAI", "OpenAI, Inc.", "Open AI"). The
resolver normalizes them and matches them against a seed list of 51 AI
companies, first exactly, then by alias, then by fuzzy match (RapidFuzz, with a
conservative threshold to avoid false merges). It also finds company mentions in
free text such as news articles. Every decision is written to an entity mapping
log with the method and score used.

## Project structure

```
src/
├── main.py                  Entry point: runs the phases, stores, exports
├── schemas/models.py        Pydantic schemas for each record type
├── scrapers/                One module per source, plus the headless-browser fallback
├── llm/
│   ├── orchestrator.py      Provider fallback chain, 429 backoff, 413 rechunking
│   └── chunking.py          Token-aware chunking
├── entity_resolution/       Name normalization, seed list, fuzzy matching
├── storage/db.py            SQLite: deduplication, seen URLs, mapping log, run log
├── export/to_sheets.py      Google Sheets export, one tab per record type
└── utils/                   Date normalization, HTML-to-text
tests/                       Offline test suite (python tests/run_all.py)
architecture.pdf             Design notes: scaling, rate limits, freshness, storage
```

## Getting started

Requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium      # optional: enables the headless-browser fallback
cp .env.example .env             # then fill in the keys you have
```

```bash
python src/main.py --target-count 10 --skip-export   # small test run
python src/main.py                                   # full run: 1,000 per source, then export
python src/main.py --phases papers jobs              # only some sources
```

Results go to `data/pipeline.sqlite3` (change with `--db`). Each phase's outcome
is also written to the `run_log` table.

## Configuration

| Variable | Used for | Without it |
|---|---|---|
| `GEMINI_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY` | LLM fallback chain | Product pricing is recorded as `UNKNOWN` |
| `GITHUB_TOKEN` | Star counts for papers | Falls back to unauthenticated GitHub (60 requests/hour) |
| `PRODUCTHUNT_API_TOKEN` | Products | The products phase is skipped with an error |
| `GOOGLE_SERVICE_ACCOUNT_JSON`, `TARGET_SHEET_URL` | Sheets export | Export is skipped with a warning |

Startups, jobs and news need no credentials.

`truststore` (in `requirements.txt`) makes Python use the operating system's
certificate store, so machines whose antivirus or proxy re-signs HTTPS traffic
don't fail with `CERTIFICATE_VERIFY_FAILED`.

## Tests

```bash
python tests/run_all.py
```

| Module | Covers |
|---|---|
| `test_dates_logic.py` | Date parsing, the 24-hour window, the seen-URL store |
| `test_orchestrator_logic.py` | Fallback chain, 429 backoff, 413 rechunking (with fakes) |
| `test_jobs_logic.py` | Role classification, remote detection, Greenhouse and Lever parsing |
| `test_review_fixes.py` | Ashby parsing, Wikidata queries and parsing, entity resolution, HTML extraction, dead-provider skipping, CLI |

The tests run offline in CI. The scrapers' HTTP calls aren't mocked, so check
them with a small live run.

## Status

Latest full run (22 November 2026, default settings), exported to
[this Google Sheet](https://docs.google.com/spreadsheets/d/1kJ6kOPlylsO9UD-tOW9UeRlYpGYAS53Wj19uPaYS2kU/edit?usp=sharing):

| Data | Records | Notes |
|---|---|---|
| Research papers | 1,000 | 136 link a GitHub repo and carry its star count |
| Startups | 1,000 | From Wikidata, in about 40 seconds |
| Products | 982 | 1,000 fetched; 18 were duplicates |
| Jobs | 4 | Everything posted in the previous 24 hours |
| News | 5 | Everything published in the previous 24 hours |

Product pricing: 325 free, 91 freemium, 85 paid, 1 enterprise and 480 unknown.
Most unknowns are product descriptions that don't mention price, which the
classifier is told to leave as unknown rather than guess. The last 79 products
were left unknown because both Groq models reached their free-tier daily limits.

## License

[MIT](LICENSE)
