# HANDOFF.md — Project Objective, Progress, and Resume Instructions

**Purpose of this file:** paste this entire file (or point to it) into a
new conversation with any AI assistant, in any tool, to resume this
project with zero prior context. It is written to be self-contained.

**Last updated:** 2026-11-14, after a code review + fix pass and a small
live run against the real APIs. §2-§3 below describe the original
sandbox-era state; this table is current.

## Quick status (update this block on every session)

| Phase | Item | Status |
|---|---|---|
| Schemas/infra | Schemas, dates/freshness, chunking, LLM orchestrator, entity resolver, SQLite storage | ✅ Built + tested |
| Phase I | Research papers (arXiv + GitHub stars) | ✅ Live-verified (repo links from abstract + arXiv comments; Papers with Code removed, it shut down) |
| Phase I | Startups (YC directory) | ✅ Built; ⚠️ configured Algolia app id doesn't exist (DNS fails) -- recapture from browser |
| Phase I | Products (Product Hunt + LLM pricing) | ✅ Live-verified; pricing is UNKNOWN until an LLM account works |
| Phase II | Jobs (Greenhouse/Lever/Ashby) | ✅ Live-verified, 5 boards (OpenAI, Anthropic, Scale AI, Cohere, ElevenLabs) |
| Phase II | News (RSS) | ✅ Live-verified: article text extraction + entity mentions |
| Phase III | LLM extraction engine | ✅ Built + tested; models updated; dead providers skipped per run |
| Phase IV | Entity resolution | ✅ Built + tested (51 seeds, fuzzy on aliases, mention detection) |
| Phase V | Anti-bot / Playwright | ✅ Built (fallback for blocked news pages); not live-tested -- needs `playwright install chromium` |
| Phase VI | architecture.pdf | ✅ Done |
| Deliverable | Google Sheets exporter | ✅ Built; not live-tested (no service account configured) |
| Deliverable | Top-level CLI/orchestrator | ✅ `python src/main.py` |

Account problems found on 2026-11-14 (not code issues): Gemini quota
exhausted, Groq key rejected (401), DeepSeek "Insufficient Balance",
GitHub token rejected (401), YC Algolia app id doesn't resolve.

Full detail on each row is in §3 below.

---

## 1. The Objective (original assignment, summarized)

This is a take-home technical assessment for an **AI Engineer** role at
**GraphOne / FrontierAtlas**, a company building an "Intelligence Graph"
for the AI/venture ecosystem (startups, founders, products, AI jobs,
research papers, news). The assignment is a 3-day trial task to build a
**scalable, fault-tolerant data ingestion pipeline**.

### Deliverables required at the end
1. **A public Google Sheet** with 6 tabs:
   - `Startups` (min. 1,000 rows)
   - `Products` (min. 1,000 rows)
   - `Research Papers` (min. 1,000 rows, must include GitHub star counts)
   - `Jobs` (all jobs found published within the last 24 hours)
   - `News` (all news found published within the last 24 hours)
   - `Entity Mapping Log` (raw name → canonical name, with method/score)
2. **A GitHub repository** containing:
   - `README.md` — setup instructions + architecture overview
   - `src/` — crawler, LLM orchestrator, entity resolver source code
   - `architecture.pdf` — max-3-page design doc (see §1.3 below)
3. Both submitted via this Google Form: `https://forms.gle/8bnrg78Ki4E25RAk8`

### 1.1 The six phases of work
- **Phase I — Massive one-time bulk extraction.** ≥1,000 unique startups,
  ≥1,000 unique products, ≥1,000 unique research papers (with GitHub
  stars). Must be architected to theoretically scale to 500,000+ records
  via infrastructure scaling alone, no code changes.
- **Phase II — High-fidelity signal ingestion.** Monitor 5 AI news
  sources + 5 AI job boards. Every ingested news/job item must be
  **guaranteed published within the last 24 hours** — includes
  normalizing relative dates ("2 hours ago"), handling missing date
  meta tags, and an "intelligent heuristic" fallback for sources with no
  reliable date signal at all.
- **Phase III — Multi-tier LLM extraction engine.** Raw HTML/text → the
  canonical JSON schema, via an LLM fallback chain (e.g. Gemini Flash →
  Groq Llama 3 → DeepSeek). Must handle 413 (payload too large, via
  intelligent chunking) and 429 (rate limit, via exponential backoff +
  jitter) gracefully.
- **Phase IV — Deterministic entity resolution.** Canonicalize messy
  startup/product name strings (e.g. "OpenAI", "OpenAI, Inc.", "Open AI"
  → "OpenAI") against a seed list of ~50 known AI startups (mocked is
  fine).
- **Phase V — Anti-bot & scale thinking.** Demonstrate or document a
  strategy for scraping Cloudflare/Datadome-protected or heavily
  JS-rendered sites without triggering captchas. Must run asynchronously
  (asyncio+aiohttp or Playwright async).
- **Phase VI — Architecture document** (max 3 pages) answering exactly
  four questions:
  1. Scale strategy — how to collect 500,000 startups/products/papers
     with no manual intervention.
  2. Exact strategy for handling 413s and 429s across thousands of
     concurrent extractions.
  3. Freshness tracking — how the architecture guarantees no
     news/job article is ever processed twice across **distributed**
     crawler nodes.
  4. Storage strategy — justify the primary DB choice and the
     vector/graph storage choice for mapping complex relationships.

### 1.2 The schemas (exact — see `src/schemas/models.py` for the
implementation, these are already built and tested)

**Startup:** `schemaVersion`, `recordType="STARTUP"`, `source.name`,
`source.url`, `content.entityName`, `content.data.employeeCount`,
`collectedAt`.

**Product:** `schemaVersion`, `recordType="PRODUCT"`, `source.name`,
`source.url`, `content.startupName`,
`content.pricingModel` (enum: `FREE`/`FREEMIUM`/`PAID`/`ENTERPRISE`),
`collectedAt`.

**Research Paper:** `schemaVersion`, `recordType="RESEARCH_PAPER"`,
`content.title`, `content.authors` (array), `content.paper_url`,
`content.github_url`, `content.github_stars`, `content.published_date`.

**Job:** `schemaVersion`, `recordType="JOB"`, `content.company`,
`content.date`, `content.is_remote`, `content.role_family`.

### 1.3 The single most important rule
> **"WARNING: Hallucinated data generated by the LLM will result in
> immediate disqualification. Every record must trace back to a
> legitimate, valid source URL."**

This rule governs every design decision in this codebase — see §4
below ("Design principles"). Do not weaken this in any future work on
this project.

### 1.4 Evaluation weighting (build effort should track this)
| Category | Weight | Focus |
|---|---|---|
| LLM Orchestration | 25% | Fallback chain robustness, payload chunking |
| Data Quality | 25% | Date parsing accuracy, 24h freshness, GitHub tracking |
| Scale Thinking | 20% | Ability to acquire "lakhs" of records, 500k+ architecture |
| Engineering Rigor | 20% | Async design, error handling, maintainability |
| Entity Resolution | 10% | Precision mapping messy strings → canonical forms |

---

## 2. Environment constraints this was built under (important context)

This codebase has been developed so far **inside a sandboxed
environment with no network access and no ability to `pip install`**.
That means:
- No module has been run against a real live API yet.
- All logic has instead been unit-tested using either (a) pure-Python
  code paths that need no external dependency, or (b) small local fake
  stand-ins for `pydantic` and `litellm` (see `tests/_fakes.py`) that
  simulate realistic behavior (rate limits, malformed JSON, context
  overflow) well enough to exercise real control flow.
- **Whoever resumes this in a networked environment should immediately
  run `pip install -r requirements.txt` and do small live test runs
  (`target_count=10`) of each scraper before trusting them at scale.**
  API response shapes (especially YC's Algolia index and Product Hunt's
  GraphQL schema) were written from documented/typical shapes and may
  have drifted from what's actually live.

---

## 3. Progress so far — exact inventory

### 3.1 Fully built AND tested (logic-verified without network)

| File | What it does | How it was tested |
|---|---|---|
| `src/schemas/models.py` | Pydantic v2 schemas for all 5 record types + `EntityMappingLogEntry`. Validators reject non-http URLs, wrong `recordType`, etc. | Constructed valid/invalid records against a local pydantic-behavior fake; confirmed required-field enforcement, nested model coercion, enum handling, custom validators all fire correctly. |
| `src/utils/dates.py` | `normalize_date()` (wraps `dateparser`, handles relative dates like "2 hours ago"), `is_within_freshness_window()` (strict 24h check, never treats `None` as fresh), `is_probably_new()` + `SeenUrlStore` (heuristic fallback for sources with no reliable date, based on seen-URL/content-hash tracking). | Ran directly (pure datetime math, no external dep needed for the tested paths) — freshness boundary cases (23h/25h/future-date/None) and seen-URL dedup all pass. |
| `src/llm/chunking.py` | `chunk_text()` — structural (paragraph-boundary) splitting with token budgets, falls back to sliding-window w/ overlap for oversized single paragraphs; `strip_boilerplate()` removes cookie-notice/newsletter/footer noise before chunking. | Ran directly against synthetic long texts (400 paragraphs, one giant 5000-word paragraph with no breaks). Verified every output chunk respects the token budget and that boilerplate is actually stripped. |
| `src/llm/orchestrator.py` | `extract_structured()` — walks a provider fallback chain (`DEFAULT_CHAIN = [gemini-1.5-flash, groq/llama3-70b, deepseek-chat]`), retries 429s with exponential backoff+jitter up to a cap then falls through providers, catches context-window-exceeded (413-equivalent) and re-chunks+retries in place, validates every response against the target Pydantic schema and treats a validation failure as a hard failure (never coerces). | **4 integration tests in `tests/test_orchestrator_logic.py`, all passing**, against a fake `litellm` that simulates: (1) 429-exhaustion on provider 1 → malformed JSON on provider 2 → success on provider 3; (2) a 413 that triggers rechunk-and-retry and recovers; (3) total chain exhaustion → returns `ok=False` with no fabricated data; (4) backoff delay math stays within `[0, min(cap, base·2^attempt)]`. **One real bug was caught and fixed** during this testing: the 413-rechunk path was discarding its own trace entry on recursion (fixed by nesting `sub_attempt` in the trace dict instead of replacing it). |
| `src/entity_resolution/resolver.py` | `EntityResolver` — normalizes names (strip legal suffixes/punctuation, casefold), tries exact/alias match, then fuzzy match (`rapidfuzz.fuzz.token_sort_ratio`, threshold 88/100, with a pure-`difflib` fallback if `rapidfuzz` isn't installed) — **never force-maps** below-threshold names, returns `unresolved_new` instead. Ships `SEED_STARTUPS` with **51** canonical AI startups + aliases (brief asked for ~50). | Verified: exact match, alias-table match, fuzzy match on a real typo ("Antropic" → "Anthropic"), and the critical negative case — an unknown name returns `canonical_name=None` / `match_method="unresolved_new"` rather than being forced onto the nearest neighbor. Empty-string edge case also handled. |
| `src/storage/db.py` | `PipelineDB` — SQLite-backed local store: `records` table with `UNIQUE(record_type, dedup_key)` for dedup-on-insert, `seen_urls` table (+ content-hash index) for freshness/dedup, `entity_mapping_log` table for the "Entity Mapping Log" deliverable tab, `run_log` for basic run/event logging. | Ran directly end-to-end (stdlib `sqlite3`, no external dep): confirmed dedup-on-insert (`upsert_record` returns `False` on the second identical `(record_type, dedup_key)`), seen-URL tracking, entity-mapping-log round trip, and JSON payload round-trip via `iter_records`. |
| `src/scrapers/research_papers.py` | **Phase I flagship pipeline.** Arxiv Atom API (`export.arxiv.org/api/query`) paged across `cs.AI/cs.LG/cs.CL/cs.CV/stat.ML/cs.NE`, respects Arxiv's ≥3s-between-requests guidance. For each paper: regex-scans the abstract for an embedded `github.com` URL, else queries the Papers with Code API (`paperswithcode.com/api/v1/papers/{id}/repositories/`) for the official repo, then hits the GitHub REST API for live `stargazers_count`. `to_record()` drops (logs, doesn't fabricate) any paper missing a required field. | XML parsing tested against a realistic sample Arxiv Atom `<entry>` (multi-author, embedded GitHub link in abstract, whitespace-mangled title) — title cleanup, author list, GitHub URL extraction, and Arxiv-ID extraction from URL all verified. `to_record()` tested for both the happy path and the "missing `paper_url` → returns `None`, never fabricates one" path. **Not yet run against the live Arxiv/GitHub/PapersWithCode APIs.** |
| `src/scrapers/startups.py` | YC Companies directory via their client-side Algolia search index. **Deliberately does NOT hardcode an API key** — a first draft did, and that was caught and corrected mid-build as a hallucination risk (see comment block in the file); it now reads `YC_ALGOLIA_URL`/`YC_ALGOLIA_APP_ID`/`YC_ALGOLIA_API_KEY` from environment variables and raises `RuntimeError` with instructions if unset, rather than sending a guessed request. `to_record()` optionally runs the entity resolver on `entityName` and logs the mapping decision to the DB. | Parsing (`_parse_yc_hit`) and `to_record()` (with and without the entity resolver wired in, including the DB logging path) tested against a realistic synthetic Algolia hit. **Not yet run against the live YC Algolia index — real app ID/API key still need to be captured from a live browser session (instructions in the file and in `README.md`).** |
| `src/scrapers/products.py` | Product Hunt GraphQL API (`api.producthunt.com/v2/api/graphql`), requires a real `PRODUCTHUNT_API_TOKEN` (raises `RuntimeError` if unset — same "don't guess credentials" discipline as the startups module). `pricingModel` is classified via the LLM orchestrator (constrained prompt forcing one of `FREE/FREEMIUM/PAID/ENTERPRISE/UNKNOWN`) since it's rarely a clean structured field on Product Hunt. | Node parsing tested. `classify_pricing_model()` tested against a fake LLM returning a valid classification, an empty-description short-circuit (never calls the LLM, returns `UNKNOWN` immediately), **and critically: total provider failure → returns `PricingModel.UNKNOWN`, confirmed it never fabricates a plausible-sounding pricing tier when the LLM chain is fully down.** `to_record()` tested with the entity resolver wired in. **Not yet run against the live Product Hunt API.** |
| `src/scrapers/jobs.py` | **Phase II, jobs half.** Sources AI job postings directly from AI companies' own Greenhouse/Lever ATS boards (`DEFAULT_JOB_BOARDS`: OpenAI, Anthropic, Scale AI, Cohere on Greenhouse; Hugging Face on Lever) rather than fighting anti-bot on generic aggregator sites — documented design decision in the module docstring. **Board tokens are explicitly flagged UNVERIFIED** (same discipline as `startups.py`) with instructions on how to confirm them live. Deterministic keyword-based `classify_role_family()` (no LLM — cheaper and zero hallucination risk for a field this classifiable). **Layered freshness logic** is the interesting part: Lever's `createdAt` is a genuine original-post timestamp and is trusted alone; Greenhouse only exposes `updated_at` (can be touched by a recruiter without being a new posting), so a Greenhouse job only counts as fresh if its timestamp is within 24h **AND** its URL hasn't been seen by this crawler before (via `PipelineDB.is_seen`) — directly implementing the brief's "intelligent heuristic for sources with unreliable dates" requirement, layered on top of (not instead of) the timestamp check. | **Full test suite in `tests/test_jobs_logic.py`, all passing**: role classification (8 cases incl. an unmatched-title → `"Other"` fallback), remote detection (text heuristic + explicit-flag override), Greenhouse/Lever payload parsing against realistic API response shapes, and — most importantly — the layered freshness logic including the specific regression case "a Greenhouse job with a recent `updated_at` that has ALREADY been seen by this crawler must NOT re-qualify as fresh." **Not yet run against the live Greenhouse/Lever APIs** (and board tokens need live verification, see module docstring). |

### 3.2 Not yet built

- **News scraper** (Phase II, 5 sources) — no code written yet. Plan:
  prefer RSS feeds where available (gives a real timestamp for free,
  solves most of the freshness problem cleanly) via `feedparser`
  (already in `requirements.txt`, unused so far); fall back to
  `httpx`/Playwright + `utils.dates.normalize_date()` for sources
  without RSS. Route full-text extraction through
  `llm/chunking.py` + `llm/orchestrator.py` for structuring into
  `NewsRecord` (schema already exists in `models.py`). The layered
  freshness pattern established in `jobs.py` (trust a strong timestamp
  signal alone; require timestamp-AND-unseen-url for a weak one) should
  carry over directly here.
- **Anti-bot / Playwright module** (Phase V) — `playwright` +
  `playwright-stealth` are in `requirements.txt` but no module has been
  written yet. Plan: build `src/scrapers/antibot_browser.py` wrapping
  Playwright async with stealth patches, realistic header/UA rotation,
  randomized human-like delays between actions, and use it for 1-2
  flagship "hard" sources (e.g. Crunchbase or a Cloudflare-protected
  target) — don't try to make every source go through this, per the
  original plan of reserving anti-bot tooling for where it's actually
  needed.
- **Google Sheets exporter** (`src/export/to_sheets.py`, folder exists,
  empty) — no code written yet. Plan: `gspread` + a service account
  (`google-auth`, already in `requirements.txt`). Pull records from
  `PipelineDB.iter_records()` per `record_type`, write to the 6 required
  tabs. **Important:** write in batches (gspread/Sheets API rate-limits
  aggressively) — do NOT write row-by-row during scraping; export is a
  separate, final step that reads from SQLite once scraping is done.
- **Top-level CLI/orchestrator** (`src/main.py` or similar, not created
  yet) — should run all phases (with concurrency limits per source),
  handle partial-failure gracefully (one phase failing shouldn't kill
  the others), and log progress via `PipelineDB.log_event()`.
- **`architecture.pdf`** — not started. Must answer the exact 4
  questions in §1.1 "Phase VI", max 3 pages. Strong raw material already
  exists in code comments/docstrings across this repo (especially the
  "Design notes" in `chunking.py`, the module docstring in `db.py`
  about the Postgres/Redis production design, and `dates.py`'s
  `SeenUrlStore` docstring which explicitly points to "the Redis-backed
  design used at scale" — that promise needs to actually be written up).
  Suggested outline:
  1. Scale strategy → horizontal worker pool + job queue (Redis/SQS),
     partition by source/category, autoscale workers, this SQLite store
     swapped for Postgres (see `storage/db.py`'s docstring for the
     stated migration path).
  2. 413/429 → exactly what `llm/orchestrator.py` + `llm/chunking.py`
     already do; write up the design, cite the module docstrings.
  3. Freshness/no-double-processing across distributed nodes → promote
     `SeenUrlStore`/`seen_urls` table to a shared Redis `SET` (or
     Postgres table with a unique constraint + `SELECT ... FOR UPDATE
     SKIP LOCKED`-style claiming) so multiple crawler nodes never both
     process the same URL; content-hash fallback for near-duplicate
     content at different URLs.
  4. Storage → Postgres (structured records, ACID, easy to query/export)
     + a graph layer (Neo4j, or lightweight adjacency tables in Postgres
     to start) for entity relationships (startup ↔ product ↔ paper ↔
     job ↔ news mentions). Justify against alternatives (pure
     document store loses relational integrity; pure graph DB is
     overkill/harder to bulk-export to Sheets).
- **Tests for the not-yet-built pieces**, obviously.
- **`.env.example`** file (README references `.env` — a template file
  listing all required variables with placeholder values would help
  whoever runs this).
- **CI / GitHub Actions** — not required by the brief but would
  strengthen "Engineering Rigor" (20% of the grade) if time allows.

---

## 4. Design principles already established — follow these in new code

1. **Never fabricate a field.** Every `to_record()`-style function
   either produces a schema-valid record backed by a real, traceable
   source URL, or returns `None` (logged) and the record is dropped.
2. **LLM failure degrades to "unknown"/`None`, never to a plausible
   guess.** See `classify_pricing_model()` in `products.py`.
3. **Never hardcode a credential/API key you haven't personally
   verified live.** If you don't have network access to verify it,
   read it from an environment variable and raise a clear
   `RuntimeError` if it's missing — see `startups.py`, `products.py`,
   and `jobs.py` (board tokens) for the established pattern. (This
   project already caught and corrected one instance of a
   fabricated-looking API key during development — don't reintroduce
   that mistake.)
4. **Log, don't silently drop.** Every rejection path (`logger.warning`)
   includes the source URL and the reason, so failures are auditable.
5. **Test what you can, honestly flag what you can't.** If you're
   resuming this in a sandbox without network/pip access again, use the
   `tests/_fakes.py` pattern (small, honest stand-ins with a docstring
   explaining exactly what they do and don't verify) rather than
   skipping verification entirely or claiming untested code is tested.

---

## 5. Suggested order to resume in

1. `pip install -r requirements.txt` in a real networked environment;
   run each existing scraper live with `target_count=10` and fix any
   drift between the assumed API shapes and reality. Also verify the
   `DEFAULT_JOB_BOARDS` tokens in `jobs.py` and the YC Algolia
   credentials in `startups.py` per their module docstrings — both are
   explicitly unverified placeholders.
2. ~~Build the Jobs scraper~~ — **done**, see `src/scrapers/jobs.py` +
   `tests/test_jobs_logic.py`.
3. Build the News scraper (RSS-first; reuse the layered-freshness
   pattern from `jobs.py`).
4. Build the Google Sheets exporter and do a first end-to-end export of
   whatever's collected so far, to validate the full pipe before
   scaling up volume.
5. Build the anti-bot Playwright module against 1-2 flagship hard
   sources.
6. Write `architecture.pdf` (raw material already exists in code
   comments, see §3.2 above).
7. Write the top-level CLI orchestrator last, once all phases exist
   independently — it's mostly wiring at that point.
