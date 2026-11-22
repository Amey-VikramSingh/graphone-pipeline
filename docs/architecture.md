# Architecture

The pipeline runs on one machine today: each source is an async task, results
go into SQLite, and an optional step copies them to Google Sheets. This page
covers how the pieces fit, and what would change to run it at a much larger
scale.

## How a run works

```
main.py
 ├─ papers    arXiv API ─► GitHub API (stars)
 ├─ startups  Wikidata SPARQL
 ├─ products  Product Hunt API ─► LLM chain (pricing)
 ├─ jobs      Greenhouse / Lever / Ashby APIs
 └─ news      RSS feeds ─► article pages (headless browser if blocked)
        │
        ▼
  schema validation (pydantic) ─► entity resolution ─► SQLite ─► Google Sheets
```

The phases run concurrently with `asyncio.gather`; a phase that fails is
logged and the others carry on.

## Rate limits and oversized input

Only product pricing needs an LLM, but the calls go through a general
extraction layer (`src/llm/orchestrator.py`):

- **429 Too Many Requests:** retry the same provider with exponential backoff
  and jitter. If the provider says how long to wait ("try again in 11.85s"),
  wait that long instead. After the retries run out, try the next provider.
- **Daily caps and dead keys:** a 429 for a daily limit, a rejected key or an
  empty balance switches that provider off for the rest of the run, so it
  isn't retried for every remaining item. Per-minute limits are ruled out
  first, because Groq's per-minute error also mentions billing.
- **413 / context too long:** text is split by paragraphs to fit each
  provider's budget before the call. If a provider still rejects a chunk, it's
  split again and retried once.

The chain is Gemini Flash, then Groq (gpt-oss-120b, then gpt-oss-20b), then
DeepSeek. Groq's free-tier limits apply per model, so listing two Groq models
doubles the daily allowance.

## Freshness

Jobs and news must be from the last 24 hours.

- Lever's `createdAt` and Ashby's `publishedAt` are real posting times, so the
  timestamp alone decides.
- Greenhouse only exposes `updated_at`, which also changes when an old post is
  edited. A Greenhouse job counts as new only if it's recent *and* its URL
  hasn't been seen before.
- Every processed URL, plus a hash of the content, goes into `seen_urls`, so
  the same item isn't ingested twice across runs.

## Scaling up

None of this is built; it's how I'd take the same design past a few thousand
records per source.

- **Workers:** put source pages (an arXiv page, a feed, a job board) on a
  queue such as Redis or SQS, and run the scrapers as stateless workers that
  pull from it. More throughput means more workers, not code changes.
- **Storage:** move from SQLite to PostgreSQL, which handles concurrent writes
  from many workers and stores the records as JSONB.
- **Shared freshness state:** replace the local `seen_urls` table with a Redis
  set. `SADD` returns whether a URL was new in one atomic step, so two workers
  can't both claim the same item.
- **Graph queries:** the records describe connected things (a company, its
  products, its jobs, news that mentions it). For questions that walk those
  links, a graph database such as Neo4j would sit alongside Postgres, fed from
  the entity mapping log.
