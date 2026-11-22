"""Job postings from AI companies' own careers boards.

Rather than scraping job aggregator sites, this reads the public JSON APIs
behind each company's careers page on Greenhouse, Lever and Ashby. It's the
same data the careers page shows, without fighting bot protection.

Board tokens were checked on 2026-11-14. Companies switch providers now and
then (OpenAI and Cohere moved from Greenhouse to Ashby), so if a board starts
returning 404, look up the new token in the careers page's network requests:
  - Greenhouse: boards-api.greenhouse.io/v1/boards/<token>/jobs
  - Lever:      api.lever.co/v0/postings/<token>
  - Ashby:      api.ashbyhq.com/posting-api/job-board/<token>
A board that 404s just yields no jobs and a log line.

Freshness: Lever's createdAt and Ashby's publishedAt are real posting times.
Greenhouse only exposes updated_at, which also changes when a recruiter edits
an old post, so a Greenhouse job must also have a URL we haven't seen before.
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
ASHBY_JOBS_URL_TMPL = "https://api.ashbyhq.com/posting-api/job-board/{token}"


@dataclass
class JobBoardConfig:
    platform: str        # "greenhouse" | "lever" | "ashby"
    board_token: str      # checked 2026-11-14
    company_display_name: str


# Five AI companies' own applicant-tracking boards, verified live on
# 2026-11-14 (see the module docstring).
DEFAULT_JOB_BOARDS: List[JobBoardConfig] = [
    JobBoardConfig(platform="ashby", board_token="openai", company_display_name="OpenAI"),
    JobBoardConfig(platform="greenhouse", board_token="anthropic", company_display_name="Anthropic"),
    JobBoardConfig(platform="greenhouse", board_token="scaleai", company_display_name="Scale AI"),
    JobBoardConfig(platform="ashby", board_token="cohere", company_display_name="Cohere"),
    JobBoardConfig(platform="ashby", board_token="elevenlabs", company_display_name="ElevenLabs"),
]

# --------------------------------------------------------------------------
# Role family from title keywords. A lookup table is enough here.
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
    """Map a job title to a role family by keyword, or 'Other' if nothing matches."""
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
    """Parse one Greenhouse job. Returns None if the title or URL is missing."""
    title = job.get("title")
    job_url = job.get("absolute_url")
    if not title or not job_url:
        return None
    location = (job.get("location") or {}).get("name", "")
    return {
        "company": board.company_display_name,
        "title": title.strip(),
        "url": job_url,
        "date_raw": job.get("updated_at"),   # also changes on edits; see passes_freshness
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
    """Parse one Lever posting. Returns None if required fields are missing."""
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
# Ashby
# --------------------------------------------------------------------------

def _parse_ashby_job(job: dict, board: JobBoardConfig) -> Optional[dict]:
    """Parse one Ashby posting. Returns None if fields are missing or it's unlisted."""
    title = job.get("title")
    job_url = job.get("jobUrl") or job.get("applyUrl")
    if not title or not job_url or job.get("isListed") is False:
        return None
    # Prefer the specific workplaceType: live data has postings with
    # isRemote=true AND workplaceType="Hybrid" (an SF office role), so the
    # boolean alone over-reports remote roles. Fall back to isRemote, then
    # to the location text.
    workplace_type = (job.get("workplaceType") or "").lower()
    if workplace_type == "remote":
        is_remote: Optional[bool] = True
    elif workplace_type in ("onsite", "hybrid"):
        is_remote = False
    elif isinstance(job.get("isRemote"), bool):
        is_remote = job["isRemote"]
    else:
        is_remote = None
    return {
        "company": board.company_display_name,
        "title": title.strip(),
        "url": job_url,
        "date_raw": job.get("publishedAt"),
        "location_raw": job.get("location") or "",
        "is_remote_explicit": is_remote,
        "date_confidence": "strong_published_at",
        "source_name": "jobs.ashbyhq.com",
    }


async def fetch_ashby_jobs(client: "httpx.AsyncClient", board: JobBoardConfig) -> List[dict]:
    url = ASHBY_JOBS_URL_TMPL.format(token=board.board_token)
    try:
        resp = await client.get(url, timeout=30.0)
        if resp.status_code == 404:
            logger.warning("Ashby board not found (verify token): %s (%s)", board.board_token, url)
            return []
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        logger.warning("Ashby fetch failed for %s: %s", board.board_token, e)
        return []

    parsed = [_parse_ashby_job(job, board) for job in resp.json().get("jobs", [])]
    return [p for p in parsed if p is not None]


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def to_record(raw: dict) -> Optional[JobRecord]:
    """Build a JobRecord, or return None (and log) if company, date or URL is missing."""
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
    """Lever and Ashby: the timestamp is enough. Greenhouse: the timestamp must be
    recent and the URL must be new to this crawler.
    """
    if not is_within_freshness_window(parsed_date, now=now):
        return False
    if raw.get("date_confidence") == "weak_updated_at_only" and db is not None:
        if db.is_seen(raw["url"]):
            return False
    return True


async def run_jobs_pipeline(boards: Optional[List[JobBoardConfig]] = None, db=None) -> List[JobRecord]:
    """Fetch every board and return JobRecords for postings from the last 24 hours.

    Fresh URLs are marked as seen, so a Greenhouse post that gets edited later
    doesn't count as new again.
    """
    if httpx is None:
        raise RuntimeError("httpx is not installed; pip install -r requirements.txt")

    boards = boards or DEFAULT_JOB_BOARDS
    from utils.dates import normalize_date

    all_raw: List[dict] = []
    async with httpx.AsyncClient(headers={"User-Agent": "graphone-pipeline/1.0 (+https://github.com/Amey-VikramSingh/graphone-pipeline)"}) as client:
        for board in boards:
            if board.platform == "greenhouse":
                jobs = await fetch_greenhouse_jobs(client, board)
            elif board.platform == "lever":
                jobs = await fetch_lever_jobs(client, board)
            elif board.platform == "ashby":
                jobs = await fetch_ashby_jobs(client, board)
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
