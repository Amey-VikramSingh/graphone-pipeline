"""Research papers from arXiv, with GitHub star counts.

  1. Page through the arXiv API for the configured categories (cs.AI, cs.LG,
     cs.CL, cs.CV, stat.ML, ...) until there are enough unique papers.
  2. Look for a github.com link in the abstract and in arXiv's comment
     field. Authors often put "Code: github.com/..." in the comment, so
     checking both roughly doubles the hit rate.
  3. If a repo is found, read its star count from the GitHub API.

arXiv asks for at least 3 seconds between requests, so paging is sequential.
GitHub lookups run concurrently up to a limit. Set GITHUB_TOKEN: without it
GitHub allows only 60 requests an hour.
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

ARXIV_API_URL = "https://export.arxiv.org/api/query"
GITHUB_API_URL = "https://api.github.com"

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
    """Parse one <entry> from arXiv's Atom feed. Returns None if the title or URL is missing."""
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
    comment = text("arxiv:comment") or ""

    return {
        "paper_url": paper_url,
        "title": re.sub(r"\s+", " ", title),
        "authors": [a for a in authors if a],
        "published_raw": published_raw,
        "summary": summary,
        "comment": comment,
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
    """Yield unique papers across the configured categories until target_count is reached."""
    if httpx is None:
        raise RuntimeError("httpx is not installed; pip install -r requirements.txt")

    seen_urls: set[str] = set()
    yielded = 0

    async with httpx.AsyncClient(headers={"User-Agent": "graphone-pipeline/1.0 (+https://github.com/Amey-VikramSingh/graphone-pipeline)"}) as client:
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


_github_token_rejected = False


async def _get_github_stars(client: "httpx.AsyncClient", github_url: str, token: Optional[str]) -> Optional[int]:
    global _github_token_rejected
    m = re.match(r"https?://github\.com/([\w.-]+)/([\w.-]+)", github_url)
    if not m:
        return None
    owner, repo = m.group(1), m.group(2).rstrip("/")
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}"
    headers = {"Accept": "application/vnd.github+json"}
    if token and not _github_token_rejected:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = await client.get(url, headers=headers, timeout=15.0)
        if resp.status_code == 401 and "Authorization" in headers:
            # A revoked token makes every lookup fail, but public repos can be read
            # without one (60 requests an hour), so drop the token for this run.
            if not _github_token_rejected:
                logger.warning("GITHUB_TOKEN was rejected (401); continuing unauthenticated "
                               "(60 requests/hour). Replace the token to restore the higher limit.")
            _github_token_rejected = True
            headers.pop("Authorization")
            resp = await client.get(url, headers=headers, timeout=15.0)
        if resp.status_code == 404:
            return None
        if resp.status_code in (403, 429):
            logger.warning("GitHub API rate-limited (%s) fetching stars for %s", resp.status_code, github_url)
            return None
        resp.raise_for_status()
        return resp.json().get("stargazers_count")
    except Exception as e:  # noqa: BLE001
        logger.debug("GitHub stars lookup failed for %s: %s", github_url, e)
        return None


async def enrich_with_github(
    raw_paper: dict, client: "httpx.AsyncClient", sem: asyncio.Semaphore, token: Optional[str]
) -> dict:
    """Add github_url and github_stars, using a link from the abstract or the comment field."""
    github_url = (
        _extract_github_url_from_text(raw_paper.get("summary", ""))
        or _extract_github_url_from_text(raw_paper.get("comment", ""))
    )

    stars = None
    if github_url:
        async with sem:
            stars = await _get_github_stars(client, github_url, token)

    raw_paper["github_url"] = github_url
    raw_paper["github_stars"] = stars
    return raw_paper


def to_record(raw_paper: dict) -> Optional[ResearchPaperRecord]:
    """Build a ResearchPaperRecord, or return None (and log) if validation fails."""
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
    """arXiv paging, then GitHub lookups, then validated records."""
    config = config or ScraperConfig()
    if httpx is None:
        raise RuntimeError("httpx is not installed; pip install -r requirements.txt")

    sem = asyncio.Semaphore(config.max_concurrent_github_calls)
    records: List[ResearchPaperRecord] = []

    async with httpx.AsyncClient(headers={"User-Agent": "graphone-pipeline/1.0 (+https://github.com/Amey-VikramSingh/graphone-pipeline)"}) as gh_client:
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
