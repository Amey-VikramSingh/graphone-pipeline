"""
Startups vertical.

Primary source: Y Combinator's public company directory, which serves
a JSON payload (Algolia-backed search index) behind its /companies
page. This is a real, structured, high-volume source that does not
require LLM extraction or anti-bot evasion for the bulk of the target
1,000 records -- reserving the LLM extraction engine (Phase III) and
anti-bot tooling (Phase V) for sources that genuinely need them, per
the brief's evaluation criteria (Data Quality rewards precision; a
hallucinated employeeCount from an LLM guess on a page that already
has the real number in a JSON field is a worse outcome than just
reading the field).

Not built yet: a secondary, HTML-only source routed through the LLM
orchestrator (llm/orchestrator.py) to diversify beyond YC. YC is the
only startups source today.
"""
from __future__ import annotations

import logging
from typing import List, Optional

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

from schemas.models import StartupRecord, StartupContent, StartupContentData, Source
from entity_resolution.resolver import EntityResolver

logger = logging.getLogger("graphone.scrapers.startups")

# YC's public /companies directory is backed by an Algolia search index
# called client-side, with a read-only "search-only" API key embedded
# in the page's JS bundle (this is standard Algolia practice -- search
# keys are meant to be public/client-exposed, scoped to read-only search
# on specific indices; this is not a private/authenticated endpoint).
#
# IMPORTANT -- DO NOT TRUST A HARDCODED KEY HERE: Algolia app IDs/search
# keys of this kind can rotate, and I have not verified a live value
# against the real site from this sandbox (no network access). Do not
# ship a guessed key -- that is exactly the kind of unverified value
# this project's "no hallucination" rule exists to prevent. Populate
# these from environment variables after retrieving the real values
# yourself: open ycombinator.com/companies with browser devtools ->
# Network tab -> filter "algolia.net" -> copy the `x-algolia-application-id`
# and `x-algolia-api-key` request headers from a live request.
import json
import os
from dataclasses import dataclass
from urllib.parse import urlencode

YC_COMPANY_PAGE_BASE = "https://www.ycombinator.com/companies/"
YC_INDEX_NAME = os.environ.get("YC_ALGOLIA_INDEX", "YCCompany_production")

PAGE_SIZE = 100


@dataclass(frozen=True)
class AlgoliaConfig:
    url: str
    app_id: str
    api_key: str


def algolia_config_from_env() -> AlgoliaConfig:
    """
    Read the Algolia settings when they're needed, not at import time, so
    values loaded from .env by the entry point are seen. YC_ALGOLIA_URL is
    optional: Algolia's multi-query endpoint is derived from the app id.
    """
    app_id = os.environ.get("YC_ALGOLIA_APP_ID", "").strip()
    api_key = os.environ.get("YC_ALGOLIA_API_KEY", "").strip()
    url = os.environ.get("YC_ALGOLIA_URL", "").strip() or (
        f"https://{app_id.lower()}-dsn.algolia.net/1/indexes/*/queries" if app_id else ""
    )
    return AlgoliaConfig(url=url, app_id=app_id, api_key=api_key)


def build_algolia_params(page: int, tags: Optional[List[str]] = None) -> str:
    """
    Algolia's `params` is a URL-encoded query string. Values must be encoded:
    a tag like "Artificial Intelligence" contains a space, which broke the
    previous hand-built string.
    """
    params = {"query": "", "page": page, "hitsPerPage": PAGE_SIZE}
    if tags:
        params["facetFilters"] = json.dumps([[f"industry:{t}" for t in tags]])
    return urlencode(params)


async def _fetch_yc_page(client: "httpx.AsyncClient", cfg: AlgoliaConfig, page: int,
                         tags: Optional[List[str]] = None) -> dict:
    """
    One page of YC's company search index. `tags` can filter to e.g.
    ["Artificial Intelligence"] to bias the sample toward the AI/venture
    vertical this graph cares about, matching the brief's domain focus.
    """
    body = {"requests": [{"indexName": YC_INDEX_NAME, "params": build_algolia_params(page, tags)}]}
    headers = {
        "X-Algolia-Application-Id": cfg.app_id,
        "X-Algolia-API-Key": cfg.api_key,
        "Content-Type": "application/json",
    }
    resp = await client.post(cfg.url, headers=headers, json=body, timeout=20.0)
    resp.raise_for_status()
    return resp.json()


def _parse_yc_hit(hit: dict) -> Optional[dict]:
    """
    hit is one Algolia record from YC's index. Field names below match
    YC's public index schema (name, slug, team_size, one_liner, etc.).
    Returns None if the record lacks the minimum required fields.
    """
    slug = hit.get("slug")
    name = hit.get("name")
    if not slug or not name:
        return None
    return {
        "entityName": name.strip(),
        "employeeCount": hit.get("team_size"),
        "source_url": f"{YC_COMPANY_PAGE_BASE}{slug}",
        "one_liner": hit.get("one_liner"),
        "industries": hit.get("industries", []),
    }


async def iter_yc_startups(target_count: int = 1000, tags: Optional[List[str]] = None):
    if httpx is None:
        raise RuntimeError("httpx is not installed; pip install -r requirements.txt")
    cfg = algolia_config_from_env()
    if not (cfg.app_id and cfg.api_key):
        raise RuntimeError(
            "YC_ALGOLIA_APP_ID / YC_ALGOLIA_API_KEY are not set. "
            "These must be captured from a live browser session against "
            "ycombinator.com/companies (see comment above) -- refusing to "
            "guess them rather than silently sending a broken/fabricated request."
        )

    yielded = 0
    page = 0
    async with httpx.AsyncClient() as client:
        while yielded < target_count:
            data = await _fetch_yc_page(client, cfg, page, tags)
            hits = data.get("results", [{}])[0].get("hits", [])
            if not hits:
                break
            for hit in hits:
                parsed = _parse_yc_hit(hit)
                if parsed is None:
                    continue
                yielded += 1
                yield parsed
                if yielded >= target_count:
                    break
            page += 1


def to_record(raw: dict, resolver: Optional[EntityResolver] = None, db=None) -> Optional[StartupRecord]:
    """
    Convert + validate a raw startup dict. If `resolver` is provided,
    the entityName is canonicalized and the resolution decision is
    logged to `db` (if provided) for the Entity Mapping Log tab -- the
    STORED entityName is always the canonical form when a confident
    match exists, so downstream Product records referencing
    `content.startupName` line up with Startup records referencing
    `content.entityName`.
    """
    try:
        entity_name = raw["entityName"]
        if resolver is not None:
            result = resolver.resolve(entity_name)
            if db is not None:
                db.log_entity_mapping(
                    raw_name=entity_name,
                    canonical_name=result.canonical_name,
                    match_score=result.match_score,
                    match_method=result.match_method,
                    source_url=raw["source_url"],
                )
            # Use the canonical name only on a confident match; otherwise
            # keep the raw name as-is (still a real, sourced value) and
            # let it stand as a new unresolved entity rather than force-
            # mapping it to the nearest known startup.
            if result.canonical_name:
                entity_name = result.canonical_name

        content = StartupContent(
            entityName=entity_name,
            data=StartupContentData(employeeCount=raw.get("employeeCount")),
        )
        return StartupRecord(
            source=Source(name="ycombinator.com", url=raw["source_url"]),
            content=content,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Dropping startup record (schema validation failed): %s | error=%s",
                        raw.get("source_url"), e)
        return None


async def run_startups_pipeline(target_count: int = 1000, resolver: Optional[EntityResolver] = None,
                                 db=None, tags: Optional[List[str]] = None) -> List[StartupRecord]:
    records: List[StartupRecord] = []
    async for raw in iter_yc_startups(target_count=target_count, tags=tags):
        record = to_record(raw, resolver=resolver, db=db)
        if record is not None:
            records.append(record)
    logger.info("Startups pipeline complete: %d validated records", len(records))
    return records
