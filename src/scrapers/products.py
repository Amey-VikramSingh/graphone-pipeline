"""
Products vertical.

Primary source: Product Hunt's public GraphQL API (developers.producthunt.com).
This requires a free developer API token (OAuth client-credentials flow)
-- unlike YC's directory, Product Hunt does gate this behind auth, so
`PRODUCTHUNT_API_TOKEN` must be supplied via environment variable. This
module refuses to run without it rather than falling back to scraping
the HTML in a way that would violate Product Hunt's terms.

`pricingModel` is frequently NOT present as a clean enum in source data
(a listing might say "Free plan available, paid tiers from $19/mo" in
prose). For that field specifically we route through the LLM extraction
engine (Phase III) with a constrained prompt that forces one of the four
enum values or UNKNOWN -- this is the one place in this vertical where
LLM extraction is actually earning its keep over a direct field read.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

from schemas.models import ProductRecord, ProductContent, PricingModel, Source
from entity_resolution.resolver import EntityResolver
from llm.orchestrator import extract_structured, DEFAULT_CHAIN

logger = logging.getLogger("graphone.scrapers.products")

PRODUCTHUNT_GRAPHQL_URL = "https://api.producthunt.com/v2/api/graphql"

PRODUCTS_QUERY = """
query Products($after: String) {
  posts(order: NEWEST, first: 50, after: $after) {
    edges {
      node {
        id
        name
        tagline
        description
        url
        website
        makers { name }
      }
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""

PRICING_EXTRACTION_PROMPT = """You classify a software product's pricing model from its description.
Return strict JSON: {"pricingModel": "FREE" | "FREEMIUM" | "PAID" | "ENTERPRISE" | "UNKNOWN"}.
Rules:
- FREE: entirely free, no paid tier mentioned.
- FREEMIUM: free tier exists AND a paid tier is mentioned.
- PAID: no free tier; requires payment/subscription/license from the start.
- ENTERPRISE: primarily sold via enterprise/custom/contact-sales pricing.
- UNKNOWN: the text gives no reliable signal. Do not guess -- UNKNOWN is a
  valid and expected answer when the signal is weak or absent."""


class PricingExtraction:
    """Lightweight schema for the pricing-model LLM extraction call.
    (Kept separate from schemas.models since this is an intermediate
    extraction target, not the final ProductRecord shape.)"""
    def __init__(self, pricingModel: str):
        self.pricingModel = pricingModel

    @classmethod
    def model_validate(cls, data: object):
        # A model can return valid JSON that isn't an object (a list, a bare
        # string). Treat that as "no reliable signal" rather than crashing.
        value = data.get("pricingModel", "UNKNOWN") if isinstance(data, dict) else "UNKNOWN"
        if not isinstance(value, str) or value.upper() not in {e.value for e in PricingModel}:
            value = "UNKNOWN"
        return cls(pricingModel=value.upper())


async def _fetch_producthunt_page(client: "httpx.AsyncClient", token: str, after: Optional[str]) -> dict:
    resp = await client.post(
        PRODUCTHUNT_GRAPHQL_URL,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"query": PRODUCTS_QUERY, "variables": {"after": after}},
        timeout=20.0,
    )
    resp.raise_for_status()
    return resp.json()


def _parse_ph_node(node: dict) -> Optional[dict]:
    name = node.get("name")
    url = node.get("url")
    if not name or not url:
        return None
    return {
        "productName": name.strip(),
        "startupName": name.strip(),  # PH doesn't cleanly separate company vs product; refined via entity resolution downstream
        "description": (node.get("description") or node.get("tagline") or "").strip(),
        "source_url": url,
        "website": node.get("website"),
    }


async def iter_producthunt_products(target_count: int = 1000):
    if httpx is None:
        raise RuntimeError("httpx is not installed; pip install -r requirements.txt")
    token = os.environ.get("PRODUCTHUNT_API_TOKEN", "")
    if not token:
        raise RuntimeError(
            "PRODUCTHUNT_API_TOKEN is not set. Register a free developer app at "
            "https://api.producthunt.com/v2/oauth/applications and export the token -- "
            "refusing to proceed without real credentials rather than fabricating requests."
        )

    yielded = 0
    after = None
    async with httpx.AsyncClient() as client:
        while yielded < target_count:
            data = await _fetch_producthunt_page(client, token, after)
            posts = data.get("data", {}).get("posts", {})
            edges = posts.get("edges", [])
            if not edges:
                break
            for edge in edges:
                parsed = _parse_ph_node(edge["node"])
                if parsed is None:
                    continue
                yielded += 1
                yield parsed
                if yielded >= target_count:
                    break
            page_info = posts.get("pageInfo", {})
            if not page_info.get("hasNextPage"):
                break
            after = page_info.get("endCursor")


async def classify_pricing_model(description: str, source_url: str) -> PricingModel:
    """Route through the LLM fallback chain; on total failure, default to
    UNKNOWN rather than guessing FREE/PAID/etc without signal."""
    if not description.strip():
        return PricingModel.UNKNOWN
    result = await extract_structured(
        raw_text=description,
        system_prompt=PRICING_EXTRACTION_PROMPT,
        schema=PricingExtraction,
        source_url=source_url,
        chain=DEFAULT_CHAIN,
    )
    if not result.ok or result.data is None:
        return PricingModel.UNKNOWN
    return PricingModel(result.data.pricingModel)


def to_record(raw: dict, pricing: PricingModel, resolver: Optional[EntityResolver] = None,
              db=None) -> Optional[ProductRecord]:
    try:
        startup_name = raw["startupName"]
        if resolver is not None:
            result = resolver.resolve(startup_name)
            if db is not None:
                db.log_entity_mapping(
                    raw_name=startup_name, canonical_name=result.canonical_name,
                    match_score=result.match_score, match_method=result.match_method,
                    source_url=raw["source_url"],
                )
            if result.canonical_name:
                startup_name = result.canonical_name

        content = ProductContent(startupName=startup_name, pricingModel=pricing)
        return ProductRecord(
            source=Source(name="producthunt.com", url=raw["source_url"]),
            content=content,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Dropping product record (schema validation failed): %s | error=%s",
                        raw.get("source_url"), e)
        return None


async def run_products_pipeline(target_count: int = 1000, resolver: Optional[EntityResolver] = None,
                                 db=None) -> List[ProductRecord]:
    records: List[ProductRecord] = []
    async for raw in iter_producthunt_products(target_count=target_count):
        pricing = await classify_pricing_model(raw["description"], raw["source_url"])
        record = to_record(raw, pricing, resolver=resolver, db=db)
        if record is not None:
            records.append(record)
    logger.info("Products pipeline complete: %d validated records", len(records))
    return records
