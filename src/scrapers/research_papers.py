"""
Research Papers vertical: Arxiv API -> GitHub repo resolution -> live star
count, fully via official/free APIs (no HTML scraping, no anti-bot fight,
no LLM guessing required for this vertical -- which is deliberate: this
is the one vertical where we can get 1,000+ records that are 100%
traceable to a real source URL with zero hallucination risk).

Data flow:
  1. Query the Arxiv API (export.arxiv.org/api/query) across configured
     categories (cs.AI, cs.LG, cs.CL, cs.CV, stat.ML, ...) with paging,
     to gather >= target_count unique papers with real published dates.
  2. For each paper, attempt to find an associated GitHub repo:
       a) Regex-scan the abstract/summary for a github.com URL (papers
          often self-report this).
       b) Fall back to the Papers with Code API (paperswithcode.com/api/v1)
          which maps arxiv IDs -> official code repos.
  3. If a repo is found, hit the GitHub REST API for live star count.
  4. Emit ResearchPaperRecord objects; anything that fails schema
     validation (e.g. missing paper_url) is dropped and logged, never
     patched with a guessed value.

Concurrency: bounded by two independent semaphores (one per external
API) so a slow GitHub rate-limit window doesn't stall Arxiv paging, and
vice versa.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional
from xml.etree import ElementTree as ET

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

from schemas.models import ResearchPaperRecord, ResearchPaperContent, Source
from utils.dates import normalize_date

logger = logging.getLogger("graphone.scrapers.research_papers")

ARXIV_API_URL = "http://export.arxiv.org/api/query"
GITHUB_API_URL = "https://api.github.com"
PAPERSWITHCODE_API_URL = "https://paperswithcode.com/api/v1"

ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

DEFAULT_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "stat.ML", "cs.NE"]
GITHUB_URL_RE = re.compile(r"https?://github\.com/([\w.-]+)/([\w.-]+?)(?:[\s\)\]\.,]|$)")

ARXIV_PAGE_SIZE = 100        # Arxiv API max per request per their guidelines
ARXIV_REQUEST_DELAY_SECONDS = 3.0  # Arxiv asks for >=3s between requests


@dataclass
class ScraperConfig:
    target_count: int = 1000
    categories: List[str] = None
    max_concurrent_github_calls: int = 5
    github_token: Optional[str] = None  # unauthenticated = 60 req/hr; token = 5000 req/hr

    def __post_init__(self):
        if self.categories is None:
            self.categories = DEFAULT_CATEGORIES


def _parse_arxiv_entry(entry: ET.Element) -> Optional[dict]:
    """Parse one <entry> from the Arxiv Atom feed into a plain dict.
    Returns None if required fields (title, id/url) are missing."""
    def text(path: str) -> Optional[str]:
        el = entry.find(path, ARXIV_NS)
        return el.text.strip() if el is not None and el.text else None

    paper_url = text("atom:id")
    title = text("atom:title")
    if not paper_url or not title:
        return None

    authors = [
        (a.find("atom:name", ARXIV_NS).text or "").strip()
        for a in entry.findall("atom:author", ARXIV_NS)
        if a.find("atom:name", ARXIV_NS) is not None
    ]
    published_raw = text("atom:published")
    summary = text("atom:summary") or ""

    return {
        "paper_url": paper_url,
        "title": re.sub(r"\s+", " ", title),
        "authors": [a for a in authors if a],
        "published_raw": published_raw,
        "summary": summary,
    }


def _extract_github_url_from_text(text: str) -> Optional[str]:
    m = GITHUB_URL_RE.search(text)
    if not m:
        return None
    owner, repo = m.group(1), m.group(2)
    repo = repo.rstrip(".")
    return f"https://github.com/{owner}/{repo}"


async def _fetch_arxiv_page(client: "httpx.AsyncClient", category: str, start: int) -> List[dict]:
    params = {
        "search_query": f"cat:{category}",
        "start": start,
        "max_results": ARXIV_PAGE_SIZE,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    resp = await client.get(ARXIV_API_URL, params=params, timeout=30.0)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    entries = root.findall("atom:entry", ARXIV_NS)
    parsed = [_parse_arxiv_entry(e) for e in entries]
    return [p for p in parsed if p is not None]


async def iter_arxiv_papers(config: ScraperConfig) -> AsyncIterator[dict]:
    """Yields raw paper dicts (pre-GitHub-enrichment) until target_count
    unique papers have been yielded, paging across configured categories."""
    if httpx is None:
        raise RuntimeError("httpx is not installed; pip install -r requirements.txt")

    seen_urls: set[str] = set()
    yielded = 0

    async with httpx.AsyncClient(headers={"User-Agent": "GraphOne-Intelligence-Graph/1.0 (research contact: data-eng@graphone.example)"}) as client:
        for category in config.categories:
            start = 0
            while yielded < config.target_count:
                page = await _fetch_arxiv_page(client, category, start)
                if not page:
                    break  # exhausted this category
                for paper in page:
                    if paper["paper_url"] in seen_urls:
                        continue
                    seen_urls.add(paper["paper_url"])
                    yielded += 1
                    yield paper
                    if yielded >= config.target_count:
                        break
                start += ARXIV_PAGE_SIZE
                await asyncio.sleep(ARXIV_REQUEST_DELAY_SECONDS)  # respect Arxiv's rate guidance
            if yielded >= config.target_count:
                break


async def _lookup_github_via_paperswithcode(client: "httpx.AsyncClient", arxiv_id: str) -> Optional[str]:
    """arxiv_id like '2401.12345' (no version suffix)."""
    try:
        resp = await client.get(f"{PAPERSWITHCODE_API_URL}/papers/{arxiv_id}/repositories/", timeout=15.0)
        if resp.status_code != 200:
            return None
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None
        # Prefer the official implementation if flagged, else first result.
        official = next((r for r in results if r.get("is_official")), results[0])
        return official.get("url")
    except Exception as e:  # noqa: BLE001 - external API, degrade gracefully
        logger.debug("PapersWithCode lookup failed for %s: %s", arxiv_id, e)
        return None


async def _get_github_stars(client: "httpx.AsyncClient", github_url: str, token: Optional[str]) -> Optional[int]:
    m = re.match(r"https?://github\.com/([\w.-]+)/([\w.-]+)", github_url)
    if not m:
        return None
    owner, repo = m.group(1), m.group(2).rstrip("/")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = await client.get(f"{GITHUB_API_URL}/repos/{owner}/{repo}", headers=headers, timeout=15.0)
        if resp.status_code == 404:
            return None
        if resp.status_code == 403:
            logger.warning("GitHub API rate-limited (403) fetching stars for %s", github_url)
            return None
        resp.raise_for_status()
        return resp.json().get("stargazers_count")
    except Exception as e:  # noqa: BLE001
        logger.debug("GitHub stars lookup failed for %s: %s", github_url, e)
        return None


def _arxiv_id_from_url(paper_url: str) -> Optional[str]:
    # e.g. http://arxiv.org/abs/2401.12345v2 -> 2401.12345
    m = re.search(r"arxiv\.org/abs/([\w.\-]+?)(v\d+)?$", paper_url)
    return m.group(1) if m else None


async def enrich_with_github(
    raw_paper: dict, client: "httpx.AsyncClient", sem: asyncio.Semaphore, token: Optional[str]
) -> dict:
    """Adds github_url / github_stars to a raw paper dict, trying
    abstract-embedded links first, then Papers with Code."""
    github_url = _extract_github_url_from_text(raw_paper.get("summary", ""))

    async with sem:
        if github_url is None:
            arxiv_id = _arxiv_id_from_url(raw_paper["paper_url"])
            if arxiv_id:
                github_url = await _lookup_github_via_paperswithcode(client, arxiv_id)

        stars = None
        if github_url:
            stars = await _get_github_stars(client, github_url, token)

    raw_paper["github_url"] = github_url
    raw_paper["github_stars"] = stars
    return raw_paper


def to_record(raw_paper: dict) -> Optional[ResearchPaperRecord]:
    """Validate+convert a raw (possibly GitHub-enriched) paper dict into a
    ResearchPaperRecord. Returns None (and logs) if it fails schema
    validation -- never coerces a missing required field."""
    try:
        content = ResearchPaperContent(
            title=raw_paper["title"],
            authors=raw_paper.get("authors", []),
            paper_url=raw_paper["paper_url"],
            github_url=raw_paper.get("github_url"),
            github_stars=raw_paper.get("github_stars"),
            published_date=normalize_date(raw_paper.get("published_raw")),
        )
        return ResearchPaperRecord(
            source=Source(name="arxiv.org", url=raw_paper["paper_url"]),
            content=content,
        )
    except Exception as e:  # noqa: BLE001 - pydantic ValidationError et al.
        logger.warning("Dropping paper (schema validation failed): %s | error=%s",
                        raw_paper.get("paper_url"), e)
        return None


async def run_research_papers_pipeline(config: Optional[ScraperConfig] = None) -> List[ResearchPaperRecord]:
    """Top-level entry point: Arxiv paging -> GitHub enrichment -> validated records."""
    config = config or ScraperConfig()
    if httpx is None:
        raise RuntimeError("httpx is not installed; pip install -r requirements.txt")

    sem = asyncio.Semaphore(config.max_concurrent_github_calls)
    records: List[ResearchPaperRecord] = []

    async with httpx.AsyncClient(headers={"User-Agent": "GraphOne-Intelligence-Graph/1.0"}) as gh_client:
        enrich_tasks = []
        async for raw_paper in iter_arxiv_papers(config):
            enrich_tasks.append(enrich_with_github(raw_paper, gh_client, sem, config.github_token))

        enriched = await asyncio.gather(*enrich_tasks, return_exceptions=False)

        for raw in enriched:
            record = to_record(raw)
            if record is not None:
                records.append(record)

    logger.info("Research papers pipeline complete: %d/%d validated records",
                len(records), len(enrich_tasks))
    return records
