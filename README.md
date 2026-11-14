# GraphOne / FrontierAtlas — AI Engineer Assessment Pipeline

A fault-tolerant ingestion pipeline for the AI/venture Intelligence Graph:
startups, products, research papers (with live GitHub star tracking),
AI news, and AI job postings — built for the GraphOne AI Engineer take-home
assessment.

> **Status:** every phase is built and runs end to end from `src/main.py`.
> Live-verified on 2026-11-14 (small run, 10 records per source): research
> papers, products, jobs and news work against the real APIs. See §2 for
> what depends on your accounts. [`HANDOFF.md`](./HANDOFF.md) has the full
> objective and history.

---

## 1. What this repo contains

```
graphone-pipeline/
├── README.md / HANDOFF.md
├── architecture.pdf / .txt     Scale, 413/429 handling, freshness, storage design
├── requirements.txt            Pinned, verified together (Python 3.12)
├── src/
│   ├── main.py                 Entry point: runs all phases, stores, exports
│   ├── schemas/models.py       Pydantic schemas (Startup/Product/Paper/Job/News)
│   ├── utils/dates.py          Date normalization + 24h freshness + seen-URL logic
│   ├── utils/html_text.py      HTML -> readable article text (no extra dependency)
│   ├── llm/
│   │   ├── chunking.py         Token-aware chunking (413 prevention)
│   │   └── orchestrator.py     LLM fallback chain: 429 backoff, 413 rechunk,
│   │                           dead providers (bad key / no quota) skipped for the run
│   ├── entity_resolution/resolver.py   Canonicalization + 51-entity seed list +
│   │                                   entity mentions in free text
│   ├── storage/db.py           SQLite: dedup, seen-urls, entity mapping log, run log
│   ├── export/to_sheets.py     Google Sheets export (one tab per record type)
│   └── scrapers/
│       ├── research_papers.py  arXiv API + GitHub repo links (abstract + comments) + stars
│       ├── startups.py         YC companies directory (Algolia)
│       ├── products.py         Product Hunt GraphQL + LLM pricing classification
│       ├── jobs.py             Greenhouse / Lever / Ashby boards + layered freshness
│       ├── news.py             RSS feeds -> article text + entity mentions
│       └── antibot_browser.py  Headless Chromium fallback for pages that block bots
├── tests/                      python tests/run_all.py
└── data/                       SQLite DB lands here (gitignored)
```

---

## 2. What depends on your accounts

The code degrades instead of crashing when an account is unusable, and the
run log says which. As of the 2026-11-14 live run:

| Source | Needs | Behaviour without it |
|---|---|---|
| Research papers | `GITHUB_TOKEN` (optional) | Works; a rejected token falls back to unauthenticated GitHub (60 req/hr), so star counts thin out on big runs |
| Products | `PRODUCTHUNT_API_TOKEN` + at least one working LLM key | Without a working LLM, every `pricingModel` is `UNKNOWN` (never guessed) |
| Startups | `YC_ALGOLIA_APP_ID` + `YC_ALGOLIA_API_KEY` | Phase fails with a clear error; other phases continue |
| Jobs, News | nothing | — |
| Sheets export | `GOOGLE_SERVICE_ACCOUNT_JSON` + `TARGET_SHEET_URL` | Skipped with a warning |

The LLM chain stops calling a provider for the rest of a run once it fails
in a way retrying can't fix (rejected key, no balance, quota used up,
retired model), so one dead account costs one request, not one per item.

---

## 3. Setup

### 3.1 Requirements
- Python 3.12 (3.11+ should work)
- API keys / accounts (see §3.3)

### 3.2 Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium        # optional: enables the anti-bot fallback for news pages
```

`truststore` (in requirements) makes Python use the operating system's
certificate store. Without it, machines whose antivirus or corporate proxy
re-signs HTTPS traffic fail every API call with `CERTIFICATE_VERIFY_FAILED`.

### 3.3 Environment variables

Copy `.env.example` to `.env` (gitignored) and fill it in. **Never put real
keys in `.env.example`** -- it is committed.

```bash
# --- LLM providers (fallback chain, in order) ---
GEMINI_API_KEY=...      # model: gemini-3.8-flash
GROQ_API_KEY=...        # model: llama-3.3-70b-versatile
DEEPSEEK_API_KEY=...    # model: deepseek-flash (needs account balance)

# --- GitHub (paper star counts) ---
GITHUB_TOKEN=...        # personal access token, no scopes needed

# --- YC startups (capture from a live browser session) ---
#   ycombinator.com/companies -> DevTools -> Network -> filter "algolia.net"
#   -> copy x-algolia-application-id / x-algolia-api-key from a request
YC_ALGOLIA_APP_ID=...
YC_ALGOLIA_API_KEY=...
# YC_ALGOLIA_URL=...    # optional; derived from the app id by default

# --- Product Hunt ---
PRODUCTHUNT_API_TOKEN=...   # free dev app: https://api.producthunt.com/v2/oauth/applications

# --- Google Sheets export ---
GOOGLE_SERVICE_ACCOUNT_JSON=./secrets/service-account.json
TARGET_SHEET_URL=...
```

Model ids were checked against the providers' live model lists on
2026-11-14; the previous ones (`gemini-1.5-flash`, `llama3-70b-8192`,
`deepseek-chat`) have all been retired. Job board tokens in
`src/scrapers/jobs.py` were verified live the same day.

---

## 4. Running

From the repository root:

```bash
python src/main.py --target-count 10 --skip-export   # small smoke run first
python src/main.py                                   # full run (1,000 per source) + Sheets export
python src/main.py --phases papers jobs              # only some phases
```

Results go to `data/pipeline.sqlite3` (`--db` to change). Each phase's
outcome is also written to the `run_log` table.

---

## 5. Running the test suite

```bash
python tests/run_all.py
```

| Module | Covers |
|---|---|
| `test_dates_logic.py` | date parsing, 24h freshness window, seen-URL store |
| `test_orchestrator_logic.py` | LLM fallback chain, 429 backoff, 413 rechunk (uses the fakes in `_fakes.py`) |
| `test_jobs_logic.py` | role classification, remote detection, Greenhouse/Lever parsing, layered freshness |
| `test_review_fixes.py` | Ashby parsing, resolver seed list + mentions, HTML extraction, Algolia params, dead-provider skipping, CLI (uses the real libraries) |

The scrapers' HTTP calls aren't mocked; verify them with a small live run.

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
