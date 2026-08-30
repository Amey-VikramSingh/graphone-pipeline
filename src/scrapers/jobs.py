"""
Jobs vertical (Phase II).

Design decision (documented, not hidden): rather than scraping generic
"AI job board" aggregator sites -- which are mostly JS-rendered listing
pages with no clean API and a much higher anti-bot burden -- this module
sources AI job postings directly from AI companies' own ATS-hosted
career boards on Greenhouse and Lever. Both platforms expose public,
unauthenticated, structured JSON APIs that the companies' own careers
pages call client-side, so this is the same data the public website
shows, just read directly. This keeps Phase II's job data 100%
structured and traceable to a real posting URL, consistent with this
project's "never fabricate a field" rule, and reserves the anti-bot
tooling (Phase V) for sources that genuinely require it.

IMPORTANT -- board tokens below are UNVERIFIED, same discipline as
startups.py / products.py: this sandbox has no network access, so
these slugs (the `boardToken` in the Greenhouse/Lever URL) have not
been confirmed live. They are my best documented guess for well-known
AI companies' career board slugs, not confirmed fact. Verify each one
before relying on it:
  - Greenhouse: visit the company's careers page, look for a request to
    `boards-api.greenhouse.io/v1/boards/<TOKEN>/jobs` in devtools, or
    check if `https://boards.greenhouse.io/<TOKEN>` redirects/loads.
  - Lever: check if `https://jobs.lever.co/<TOKEN>` loads, or look for
    a request to `api.lever.co/v0/postings/<TOKEN>` in devtools.
A board that 404s simply yields zero jobs for that company (handled
gracefully, logged, does not crash the run) -- but do verify tokens
before trusting the "5 AI job boards" coverage claim in a submission.

Freshness handling (the Phase II "Freshness Challenge"):
  - Lever's `createdAt` field is a genuine original-posting timestamp
    (epoch ms) -- reliable for the 24h freshness check on its own.
  - Greenhouse's public boards API only exposes `updated_at`, which can
    be touched by a recruiter editing an old posting WITHOUT it being a
    new posting -- an unreliable freshness signal on its own. To guard
    against false "fresh" positives from Greenhouse, freshness there is
    the AND of (a) `updated_at` within the last 24h AND (b) the job URL
    has not been seen by this crawler before (via `SeenUrlStore`/
    `PipelineDB.is_seen`). This is exactly the "intelligent heuristic"
    the brief asks for when a source's date signal isn't fully trustworthy.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

from schemas.models import JobRecord, JobContent, Source
from utils.dates import is_within_freshness_window

logger = logging.getLogger("graphone.scrapers.jobs")

GREENHOUSE_JOBS_URL_TMPL = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"
LEVER_JOBS_URL_TMPL = "https://api.lever.co/v0/postings/{token}?mode=json"


@dataclass
class JobBoardConfig:
    platform: str        # "greenhouse" | "lever"
    board_token: str      # UNVERIFIED -- see module docstring
    company_display_name: str


# "5 AI job boards" per the brief, sourced as 5 AI companies' own ATS
# boards. UNVERIFIED tokens -- see module docstring before relying on this.
DEFAULT_JOB_BOARDS: List[JobBoardConfig] = [
    JobBoardConfig(platform="greenhouse", board_token="openai", company_display_name="OpenAI"),
    JobBoardConfig(platform="greenhouse", board_token="anthropic", company_display_name="Anthropic"),
    JobBoardConfig(platform="greenhouse", board_token="scaleai", company_display_name="Scale AI"),
    JobBoardConfig(platform="greenhouse", board_token="cohere", company_display_name="Cohere"),
    JobBoardConfig(platform="lever", board_token="huggingface", company_display_name="Hugging Face"),
]

# --------------------------------------------------------------------------
# Deterministic role-family classification (keyword-based, no LLM needed --
# faster, free, and avoids introducing any hallucination risk into a field
# that's cheaply solvable with a keyword map).
# --------------------------------------------------------------------------

_ROLE_FAMILY_KEYWORDS = [
    ("Research", [r"research scientist", r"research engineer", r"applied scientist", r"\bml researcher\b"]),
    ("Data", [r"data scientist", r"data engineer", r"\bdata analy", r"analytics engineer"]),
    ("Engineering", [r"software engineer", r"\bswe\b", r"backend", r"frontend", r"full.?stack",
                      r"infrastructure", r"platform engineer", r"devops", r"\bsre\b", r"\bengineer\b"]),
    ("Product", [r"product manager", r"product owner", r"\bpm\b"]),
    ("Design", [r"designer", r"\bux\b", r"\bui\b", r"design lead"]),
    ("Sales", [r"account executive", r"sales", r"business development", r"\bbd\b", r"partnerships"]),
    ("Marketing", [r"marketing", r"growth", r"content strategist", r"brand"]),
    ("Legal", [r"counsel", r"legal", r"compliance"]),
    ("Finance", [r"finance", r"accounting", r"controller", r"treasury"]),
    ("Operations", [r"operations", r"program manager", r"project manager", r"recruit", r"people (ops|team)",
                     r"\bhr\b", r"office manager", r"chief of staff"]),
]


def classify_role_family(title: str) -> str:
    """Deterministic keyword classifier. Returns 'Other' rather than
    guessing when no keyword matches -- consistent with the project's
    'never fabricate, degrade to unknown' rule."""
    if not title:
        return "Other"
    lowered = title.lower()
    for family, patterns in _ROLE_FAMILY_KEYWORDS:
        for pattern in patterns:
            if re.search(pattern, lowered):
                return family
    return "Other"


def detect_is_remote(location_text: str, explicit_flag: Optional[bool] = None) -> bool:
    if explicit_flag is not None:
        return explicit_flag
    if not location_text:
        return False
    return bool(re.search(r"\bremote\b", location_text, re.IGNORECASE))


# --------------------------------------------------------------------------
# Greenhouse
# --------------------------------------------------------------------------

def _parse_greenhouse_job(job: dict, board: JobBoardConfig) -> Optional[dict]:
    """Pure parsing of one Greenhouse job object (public boards API shape:
    id, title, updated_at, location.name, absolute_url, ...). Returns None
    if required fields are missing -- never fabricates a title/url."""
    title = job.get("title")
    job_url = job.get("absolute_url")
    if not title or not job_url:
        return None
    location = (job.get("location") or {}).get("name", "")
    return {
        "company": board.company_display_name,
        "title": title.strip(),
        "url": job_url,
        "date_raw": job.get("updated_at"),   # see module docstring: unreliable alone, layered below
        "location_raw": location,
        "is_remote_explicit": None,
        "date_confidence": "weak_updated_at_only",
        "source_name": "boards.greenhouse.io",
    }


async def fetch_greenhouse_jobs(client: "httpx.AsyncClient", board: JobBoardConfig) -> List[dict]:
    url = GREENHOUSE_JOBS_URL_TMPL.format(token=board.board_token)
    try:
        resp = await client.get(url, timeout=20.0)
        if resp.status_code == 404:
            logger.warning("Greenhouse board not found (verify token): %s (%s)", board.board_token, url)
            return []
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        logger.warning("Greenhouse fetch failed for %s: %s", board.board_token, e)
        return []

    data = resp.json()
    parsed = [_parse_greenhouse_job(job, board) for job in data.get("jobs", [])]
    return [p for p in parsed if p is not None]


# --------------------------------------------------------------------------
# Lever
# --------------------------------------------------------------------------

def _parse_lever_job(job: dict, board: JobBoardConfig) -> Optional[dict]:
    """Pure parsing of one Lever posting object (public postings API shape:
    text, hostedUrl, categories.location, workplaceType, createdAt (epoch
    ms), ...). Returns None if required fields are missing."""
    title = job.get("text")
    job_url = job.get("hostedUrl") or job.get("applyUrl")
    if not title or not job_url:
        return None
    categories = job.get("categories", {}) or {}
    location = categories.get("location", "") or ""
    workplace_type = job.get("workplaceType")  # e.g. "remote" | "onsite" | "hybrid", when present
    created_at_ms = job.get("createdAt")
    date_raw = None
    if isinstance(created_at_ms, (int, float)):
        date_raw = datetime.fromtimestamp(created_at_ms / 1000, tz=timezone.utc).isoformat()

    return {
        "company": board.company_display_name,
        "title": title.strip(),
        "url": job_url,
        "date_raw": date_raw,
        "location_raw": location,
        "is_remote_explicit": (workplace_type == "remote") if workplace_type else None,
        "date_confidence": "strong_created_at",
        "source_name": "jobs.lever.co",
    }


async def fetch_lever_jobs(client: "httpx.AsyncClient", board: JobBoardConfig) -> List[dict]:
    url = LEVER_JOBS_URL_TMPL.format(token=board.board_token)
    try:
        resp = await client.get(url, timeout=20.0)
        if resp.status_code == 404:
            logger.warning("Lever board not found (verify token): %s (%s)", board.board_token, url)
            return []
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        logger.warning("Lever fetch failed for %s: %s", board.board_token, e)
        return []

    raw_jobs = resp.json()
    parsed = [_parse_lever_job(job, board) for job in raw_jobs]
    return [p for p in parsed if p is not None]


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def to_record(raw: dict) -> Optional[JobRecord]:
    """Validate+convert. Drops (logs) rather than fabricates a missing
    required field (company/date/url)."""
    from utils.dates import normalize_date

    parsed_date = normalize_date(raw.get("date_raw"))
    if parsed_date is None:
        logger.warning("Dropping job (no parseable date): %s", raw.get("url"))
        return None

    try:
        content = JobContent(
            company=raw["company"],
            date=parsed_date,
            is_remote=detect_is_remote(raw.get("location_raw", ""), raw.get("is_remote_explicit")),
            role_family=classify_role_family(raw["title"]),
        )
        return JobRecord(
            source=Source(name=raw["source_name"], url=raw["url"]),
            content=content,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Dropping job record (schema validation failed): %s | error=%s", raw.get("url"), e)
        return None


def passes_freshness(raw: dict, parsed_date, *, db=None, now=None) -> bool:
    """
    Layered freshness check. Strong-signal sources (Lever `createdAt`)
    are trusted on the timestamp alone. Weak-signal sources (Greenhouse
    `updated_at`) additionally require that the URL has not been seen by
    this crawler before, per the module docstring's rationale.
    """
    if not is_within_freshness_window(parsed_date, now=now):
        return False
    if raw.get("date_confidence") == "weak_updated_at_only" and db is not None:
        if db.is_seen(raw["url"]):
            return False
    return True


async def run_jobs_pipeline(boards: Optional[List[JobBoardConfig]] = None, db=None) -> List[JobRecord]:
    """
    Fetches jobs from all configured boards, applies the layered
    freshness check, marks fresh URLs as seen (so a Greenhouse posting
    that merely gets touched again next run doesn't re-qualify as
    fresh), and returns validated JobRecords for postings within the
    last 24 hours only.
    """
    if httpx is None:
        raise RuntimeError("httpx is not installed; pip install -r requirements.txt")

    boards = boards or DEFAULT_JOB_BOARDS
    from utils.dates import normalize_date

    all_raw: List[dict] = []
    async with httpx.AsyncClient(headers={"User-Agent": "GraphOne-Intelligence-Graph/1.0"}) as client:
        for board in boards:
            if board.platform == "greenhouse":
                jobs = await fetch_greenhouse_jobs(client, board)
            elif board.platform == "lever":
                jobs = await fetch_lever_jobs(client, board)
            else:
                logger.warning("Unknown job board platform: %s (%s)", board.platform, board.board_token)
                continue
            all_raw.extend(jobs)

    records: List[JobRecord] = []
    for raw in all_raw:
        parsed_date = normalize_date(raw.get("date_raw"))
        if parsed_date is None:
            continue
        if not passes_freshness(raw, parsed_date, db=db):
            continue
        record = to_record(raw)
        if record is not None:
            records.append(record)
            if db is not None:
                db.mark_seen(raw["url"])

    logger.info("Jobs pipeline complete: %d fresh, validated records out of %d fetched",
                len(records), len(all_raw))
    return records
