"""
Startups vertical.

Source: Wikidata, queried through its public SPARQL endpoint. Wikidata's
data is CC0 (public domain) and the query service is meant for automated
use, so no key or scraping is involved. The query selects organizations
whose industry (P452) is artificial intelligence or machine learning and
whose type (P31, or one subclass step up) is a kind of company. Entries
such as research institutes, associations and software products share
the industry tag but are filtered out by type.

Every field comes straight from a Wikidata statement. Employee counts are
only present for a minority of companies; when the statement is missing,
employeeCount stays None rather than being estimated.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Iterable, List, Optional

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

from schemas.models import StartupRecord, StartupContent, StartupContentData, Source
from entity_resolution.resolver import EntityResolver

logger = logging.getLogger("graphone.scrapers.startups")

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIDATA_ENTITY_BASE = "https://www.wikidata.org/wiki/"
# Wikidata's User-Agent policy asks for a descriptive agent with a contact URL.
USER_AGENT = "graphone-pipeline/1.0 (+https://github.com/Amey-VikramSingh/graphone-pipeline)"

PAGE_SIZE = 200
MAX_RETRIES = 4

# Industry (P452) values: artificial intelligence, machine learning.
AI_INDUSTRIES = ("Q11660", "Q2539")
# Instance-of (P31) values that mean "a company". Matched directly or one
# subclass step up, which covers e.g. "software company" and "startup company"
# without the full subclass walk that times out on the public endpoint.
COMPANY_TYPES = (
    "Q4830453",   # business
    "Q783794",    # company
    "Q6881511",   # enterprise
    "Q20825628",  # startup company
    "Q891723",    # public company
    "Q1589009",   # privately held company
    "Q658255",    # subsidiary
    "Q18388277",  # technology company
    "Q1058914",   # software company
    "Q2995256",   # consulting company
)


def build_company_ids_query(industries: Iterable[str] = AI_INDUSTRIES,
                            company_types: Iterable[str] = COMPANY_TYPES) -> str:
    """Every AI company's entity URI, in one request (about 1,100 rows)."""
    ind = " ".join(f"wd:{q}" for q in industries)
    types = " ".join(f"wd:{q}" for q in company_types)
    return f"""
SELECT DISTINCT ?company WHERE {{
  VALUES ?industry {{ {ind} }}
  VALUES ?type {{ {types} }}
  ?company wdt:P452 ?industry ;
           wdt:P31/wdt:P279? ?type .
}}
""".strip()


def build_details_query(qids: Iterable[str]) -> str:
    """English name, employee count and website for a batch of entity ids."""
    values = " ".join(f"wd:{q}" for q in qids)
    return f"""
SELECT ?company (SAMPLE(?label) AS ?name) (MAX(?employees) AS ?employeeCount) (SAMPLE(?site) AS ?website)
WHERE {{
  VALUES ?company {{ {values} }}
  ?company rdfs:label ?label . FILTER(LANG(?label) = "en")
  OPTIONAL {{ ?company wdt:P1128 ?employees . }}
  OPTIONAL {{ ?company wdt:P856 ?site . }}
}}
GROUP BY ?company
""".strip()


def _qid(uri: str) -> Optional[str]:
    qid = uri.rsplit("/", 1)[-1]
    return qid if qid.startswith("Q") and qid[1:].isdigit() else None


def _binding_value(row: dict, key: str) -> Optional[str]:
    value = row.get(key, {}).get("value")
    return value.strip() if isinstance(value, str) and value.strip() else None


def parse_binding(row: dict) -> Optional[dict]:
    """
    One SPARQL result row -> raw startup dict. Returns None when the row
    lacks an entity URI or an English name (the minimum for a record).
    """
    uri = _binding_value(row, "company")
    name = _binding_value(row, "name")
    if not uri or not name or not uri.startswith("http"):
        return None
    qid = _qid(uri)
    if qid is None:
        return None

    employees: Optional[int] = None
    raw_count = _binding_value(row, "employeeCount")
    if raw_count is not None:
        try:
            employees = int(float(raw_count))
        except ValueError:
            employees = None
        if employees is not None and employees < 0:
            employees = None

    return {
        "entityName": name,
        "employeeCount": employees,
        "source_url": f"{WIKIDATA_ENTITY_BASE}{qid}",
        "website": _binding_value(row, "website"),
    }


async def _run_query(client: "httpx.AsyncClient", query: str) -> List[dict]:
    """POST one query, retrying 429 and 5xx with backoff (honours Retry-After)."""
    delay = 2.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = await client.post(
                WIKIDATA_SPARQL_URL,
                data={"query": query},
                headers={"Accept": "application/sparql-results+json"},
                timeout=90.0,
            )
        except httpx.TransportError as e:
            if attempt == MAX_RETRIES:
                raise
            logger.warning("Wikidata request failed (%s); retrying in %.0fs", e, delay)
        else:
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == MAX_RETRIES:
                    resp.raise_for_status()
                retry_after = resp.headers.get("Retry-After", "")
                wait = float(retry_after) if retry_after.isdigit() else delay
                logger.warning("Wikidata returned %d; retrying in %.0fs", resp.status_code, wait)
                await asyncio.sleep(wait)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp.json().get("results", {}).get("bindings", [])
        await asyncio.sleep(delay)
        delay *= 2
    return []


async def iter_ai_startups(target_count: int = 1000):
    if httpx is None:
        raise RuntimeError("httpx is not installed; pip install -r requirements.txt")

    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}) as client:
        id_rows = await _run_query(client, build_company_ids_query())
        # Sort by numeric id so runs are repeatable and older, better-documented
        # entries come first.
        qids = sorted({q for r in id_rows if (q := _qid(_binding_value(r, "company") or ""))},
                      key=lambda q: int(q[1:]))
        logger.info("Wikidata lists %d AI companies", len(qids))

        yielded = 0
        for i in range(0, len(qids), PAGE_SIZE):
            if yielded >= target_count:
                break
            if i:
                # Wikidata asks clients not to send parallel or rapid-fire queries.
                await asyncio.sleep(1.0)
            rows = await _run_query(client, build_details_query(qids[i:i + PAGE_SIZE]))
            rows.sort(key=lambda r: int((_qid(_binding_value(r, "company") or "") or "Q0")[1:]))
            for row in rows:
                parsed = parse_binding(row)
                if parsed is None:
                    continue
                yielded += 1
                yield parsed
                if yielded >= target_count:
                    break

    if yielded < target_count:
        logger.info("Wikidata returned %d AI companies (target was %d)", yielded, target_count)


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
            source=Source(name="wikidata.org", url=raw["source_url"]),
            content=content,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Dropping startup record (schema validation failed): %s | error=%s",
                        raw.get("source_url"), e)
        return None


async def run_startups_pipeline(target_count: int = 1000, resolver: Optional[EntityResolver] = None,
                                 db=None) -> List[StartupRecord]:
    records: List[StartupRecord] = []
    async for raw in iter_ai_startups(target_count=target_count):
        record = to_record(raw, resolver=resolver, db=db)
        if record is not None:
            records.append(record)
    logger.info("Startups pipeline complete: %d validated records", len(records))
    return records
