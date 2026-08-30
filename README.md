# GraphOne / FrontierAtlas — AI Engineer Assessment Pipeline

A fault-tolerant ingestion pipeline for the AI/venture Intelligence Graph:
startups, products, research papers (with live GitHub star tracking),
AI news, and AI job postings — built for the GraphOne AI Engineer take-home
assessment.

> **Status:** work in progress, built across multiple sessions. See
> [`HANDOFF.md`](./HANDOFF.md) for exactly what's done, what's left, and
> how to resume this project (including handing it to a different AI
> assistant with zero prior context).

---

## 1. What this repo contains right now

```
graphone-pipeline/
├── README.md              <- you are here
├── HANDOFF.md              <- objective + progress + what's left (read this next)
├── requirements.txt
├── src/
│   ├── schemas/models.py       Pydantic schemas (Startup/Product/Paper/Job/News)
│   ├── utils/dates.py          Date normalization + 24h freshness + seen-URL logic
│   ├── llm/
│   │   ├── chunking.py         Token-aware chunking (413 prevention)
│   │   └── orchestrator.py     Multi-provider LLM fallback chain (429/413 handling)
│   ├── entity_resolution/resolver.py   Canonicalization engine + 51-entity seed list
│   ├── storage/db.py           SQLite storage: dedup, seen-urls, entity mapping log
│   └── scrapers/
│       ├── research_papers.py  Arxiv API + GitHub stars       (COMPLETE)
│       ├── startups.py         YC Companies directory         (COMPLETE, needs live API creds)
│       ├── products.py         Product Hunt GraphQL + LLM pricing classification (COMPLETE, needs API token)
│       └── jobs.py             Greenhouse/Lever ATS boards + layered freshness   (COMPLETE, needs board token verification)
├── tests/
│   ├── _fakes.py                Local pydantic/litellm stand-ins used because this
│   │                            dev sandbox has no network/pip access
│   ├── run_all.py                Runs every test module below in one go
│   ├── test_dates_logic.py       Date parsing + freshness window + seen-URL store
│   ├── test_orchestrator_logic.py   LLM fallback chain, 429 backoff, 413 rechunk
│   └── test_jobs_logic.py        Role classification, remote detection, layered freshness
└── data/                        SQLite DB + exports land here (gitignored in practice)
```

**Not yet built:** News scraper, anti-bot/Playwright module,
Google Sheets exporter, top-level CLI/orchestrator script that runs all
phases end-to-end, `architecture.pdf`. Full detail in `HANDOFF.md`
(see the "Quick status" table at the top).

---

## 2. Why some things aren't tested against the real internet yet

This codebase was developed inside a sandboxed environment **with no
network access and no ability to `pip install`**. Every module was still
verified as thoroughly as possible without a network:

- Pure logic (chunking, date/freshness math, entity-name normalization,
  SQLite dedup) was run and asserted against directly — no mocking needed.
- The LLM orchestrator (fallback chain, 429 backoff, 413 rechunk-retry)
  was tested against small local fake stand-ins for `litellm` (see
  `tests/_fakes.py`) that simulate rate limits, context-overflow errors,
  and malformed responses. This caught and fixed one real bug (a lost
  trace entry on the 413 retry path).
- Anything requiring a live HTTP call (Arxiv API, GitHub API, YC's
  Algolia index, Product Hunt's GraphQL API) has its **parsing and
  validation logic** tested against realistic synthetic payloads shaped
  exactly like the real APIs' documented responses, but has **not been
  run against the live endpoints**.

**Before you trust this in production: run it against the real APIs
first**, on a small sample (e.g. `target_count=10`), and sanity-check the
output. Treat every scraper as "logic-verified, network-unverified."

---

## 3. Setup

### 3.1 Requirements
- Python 3.11+
- API keys / accounts (see §3.3)

### 3.2 Install

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium        # only needed once the anti-bot module is added
```

### 3.3 Environment variables

Create a `.env` file in the project root (loaded via `python-dotenv`):

```bash
# --- LLM providers (Phase III fallback chain) ---
GEMINI_API_KEY=...
GROQ_API_KEY=...
DEEPSEEK_API_KEY=...

# --- GitHub (research papers star-count enrichment) ---
GITHUB_TOKEN=...                   # personal access token, no special scopes needed
                                    # unauthenticated works too but caps at 60 req/hr

# --- YC startups source ---
# These are NOT hardcoded in the code on purpose (see src/scrapers/startups.py
# docstring) -- capture them yourself from a live browser session:
#   ycombinator.com/companies -> DevTools -> Network tab -> filter "algolia.net"
#   -> copy x-algolia-application-id / x-algolia-api-key from a live request
YC_ALGOLIA_URL=https://<APP_ID>-dsn.algolia.net/1/indexes/*/queries
YC_ALGOLIA_APP_ID=...
YC_ALGOLIA_API_KEY=...

# --- Product Hunt (products source) ---
PRODUCTHUNT_API_TOKEN=...          # register a free dev app at
                                    # https://api.producthunt.com/v2/oauth/applications

# --- Jobs source (Greenhouse/Lever) ---
# Board tokens in src/scrapers/jobs.py::DEFAULT_JOB_BOARDS are UNVERIFIED
# placeholders (see module docstring) -- verify each one before relying
# on it: check whether https://boards.greenhouse.io/<token> or
# https://jobs.lever.co/<token> actually loads for the intended company.
# No separate env var needed for this source (Greenhouse/Lever's public
# boards APIs are unauthenticated) -- just correct board tokens.

# --- Google Sheets export (not yet built, will need this) ---
GOOGLE_SERVICE_ACCOUNT_JSON=./secrets/service-account.json
```

litellm reads `GEMINI_API_KEY` / `GROQ_API_KEY` / `DEEPSEEK_API_KEY` from
the environment automatically for the model strings used in
`src/llm/orchestrator.py`'s `DEFAULT_CHAIN` (`gemini/gemini-1.5-flash`,
`groq/llama3-70b-8192`, `deepseek/deepseek-chat`).

---

## 4. Running what's built so far

There is no single top-level `main.py` yet (that's on the "what's left"
list). Until then, run each completed vertical directly:

```bash
python3 -c "
import asyncio
from src.scrapers.research_papers import run_research_papers_pipeline, ScraperConfig

async def main():
    records = await run_research_papers_pipeline(ScraperConfig(target_count=10))
    for r in records:
        print(r.content.title, '|', r.content.github_stars, '|', r.content.paper_url)

asyncio.run(main())
"
```

Similarly for startups (`src/scrapers/startups.py::run_startups_pipeline`),
products (`src/scrapers/products.py::run_products_pipeline`), and jobs
(`src/scrapers/jobs.py::run_jobs_pipeline`) once their respective
credentials/board tokens are set. Start with a small `target_count`
(10–20) on your first run of each to confirm the live API shapes still
match what the parsers expect before scaling up to 1,000+. For jobs,
also pass a `PipelineDB` instance so the layered freshness check
(Greenhouse's weak `updated_at` signal gated by "not seen before") has
somewhere to persist seen-URLs across runs:

```bash
python3 -c "
import asyncio
from src.storage.db import PipelineDB
from src.scrapers.jobs import run_jobs_pipeline

async def main():
    with PipelineDB('data/pipeline.sqlite3') as db:
        records = await run_jobs_pipeline(db=db)
        for r in records:
            print(r.content.company, '|', r.content.role_family, '|', r.source.url)

asyncio.run(main())
"
```

---

## 5. Running the test suite

```bash
# Pure-logic tests (no dependencies needed beyond stdlib):
python3 -m py_compile src/**/*.py     # syntax check everything

# Run everything at once:
python3 tests/run_all.py

# Or individually:
python3 tests/test_dates_logic.py          # date parsing + freshness window + seen-URL store
python3 tests/test_orchestrator_logic.py   # LLM fallback chain, 429 backoff, 413 rechunk
python3 tests/test_jobs_logic.py           # role classification, remote detection, layered freshness
```

Once `pip install -r requirements.txt` succeeds in a networked
environment, the fakes in `tests/_fakes.py` become unnecessary for
day-to-day development — they exist specifically to unblock testing in a
no-network sandbox and can be replaced with real `pydantic`/`litellm`
imports and, ideally, a proper `pytest` suite with HTTP mocking
(`respx` for `httpx`) for the network-touching scraper functions.

---

## 6. Design principles this codebase follows throughout

1. **Never fabricate a field.** Every `to_record()` function either
   produces a schema-valid record backed by a real source URL, or
   returns `None` (logged) and drops the record. Nothing is
   guessed/interpolated to fill a gap.
2. **LLM failure degrades to "unknown", never to a guess.** See
   `classify_pricing_model()` in `products.py` — total provider failure
   returns `PricingModel.UNKNOWN`, not a plausible-sounding default.
3. **Every external credential is either a real env var or an explicit
   `RuntimeError`** telling you what to go get and from where — never a
   fabricated placeholder that would silently fail or misrepresent
   itself as a working value.

See `HANDOFF.md` for the full project objective, evaluation criteria
this is being built against, and the detailed remaining task list.
