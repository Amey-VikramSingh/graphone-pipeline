"""
News vertical (Phase II): fresh AI news from RSS feeds.

For each feed entry published within the last 24 hours (and not seen on a
previous run), fetch the article, extract its readable text, and record
which known entities it mentions.

  - Article text comes from the page's main content (utils/html_text.py),
    not the raw HTML. If the page can't be fetched or yields too little text,
    the feed's own summary/content is used instead, so a record is never
    built from an empty or HTML-soup body.
  - Pages that block plain HTTP clients (403/429/503) are retried once with
    the headless anti-bot browser, if it's installed.
  - `source.name` is the feed's own title (e.g. "TechCrunch").
"""
from __future__ import annotations

import asyncio
from typing import List, Optional

import feedparser
import httpx
from loguru import logger

from entity_resolution.resolver import EntityResolver
from llm.chunking import strip_boilerplate
from schemas.models import NewsContent, NewsRecord, Source
from scrapers.antibot_browser import BLOCKED_STATUSES, LazyAntiBotBrowser
from storage.db import PipelineDB
from utils.dates import is_within_freshness_window, normalize_date
from utils.html_text import html_to_text, strip_tags

AI_NEWS_FEEDS = [
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://cset.georgetown.edu/news/feed/",
    "https://bair.berkeley.edu/blog/feed.xml",
    "https://blogs.nvidia.com/feed/",
    "https://aws.amazon.com/blogs/machine-learning/feed/",
]

# Below this, extracted page text is treated as a failed extraction.
MIN_ARTICLE_CHARS = 300
USER_AGENT = "graphone-pipeline/1.0 (+https://github.com/Amey-VikramSingh/graphone-pipeline)"


def _feed_text(entry: dict) -> str:
    """The feed's own copy of the article: full content if present, else summary."""
    for item in entry.get("content") or []:
        value = item.get("value") if isinstance(item, dict) else None
        if value:
            return strip_tags(value)
    return strip_tags(entry.get("summary") or "")


async def _fetch_article_text(client: httpx.AsyncClient, browser: LazyAntiBotBrowser,
                              url: str) -> Optional[str]:
    html: Optional[str] = None
    try:
        resp = await client.get(url, timeout=15.0)
        if resp.status_code in BLOCKED_STATUSES:
            logger.info(f"News page blocked ({resp.status_code}), trying browser: {url}")
            html = await browser.get_page_content(url)
        else:
            resp.raise_for_status()
            html = resp.text
    except Exception as e:  # noqa: BLE001 - one bad article must not stop the feed
        logger.warning(f"Failed to fetch news article {url}: {type(e).__name__}: {e}")
        return None
    if not html:
        return None
    text = strip_boilerplate(html_to_text(html))
    return text if len(text) >= MIN_ARTICLE_CHARS else None


async def _process_feed_entry(entry: dict, feed_title: str, db: PipelineDB,
                              client: httpx.AsyncClient, browser: LazyAntiBotBrowser,
                              resolver: Optional[EntityResolver]) -> Optional[NewsRecord]:
    article_url = entry.get("link")
    title = (entry.get("title") or "").strip()
    if not article_url or not title or db.is_seen(article_url):
        return None

    parsed_date = normalize_date(entry.get("published") or entry.get("updated"))
    if not parsed_date or not is_within_freshness_window(parsed_date):
        return None

    text = await _fetch_article_text(client, browser, article_url)
    if text is None:
        text = strip_boilerplate(_feed_text(entry))
        if not text:
            logger.warning(f"No usable text for news article, skipping: {article_url}")
            return None
        logger.info(f"Using the feed's own text for {article_url}")

    mentions = resolver.find_mentions(f"{title}\n{text}") if resolver else []
    try:
        return NewsRecord(
            source=Source(name=feed_title, url=article_url),
            content=NewsContent(title=title, published_date=parsed_date,
                                full_text=text, entities_mentioned=mentions),
        )
    except Exception as e:  # noqa: BLE001 - schema validation failure: drop, don't guess
        logger.warning(f"Dropping news record (schema validation failed): {article_url} | {e}")
        return None


async def run_news_pipeline(db: PipelineDB, resolver: Optional[EntityResolver] = None,
                            max_items: Optional[int] = None,
                            feeds: Optional[List[str]] = None) -> List[NewsRecord]:
    """
    Returns fresh NewsRecords. The caller persists them; this function only
    marks article URLs as seen so the next run doesn't re-ingest them.
    """
    logger.info("Starting Phase II: News Pipeline...")
    records: List[NewsRecord] = []
    browser = LazyAntiBotBrowser()
    try:
        async with httpx.AsyncClient(follow_redirects=True,
                                     headers={"User-Agent": USER_AGENT}) as client:
            for feed_url in feeds or AI_NEWS_FEEDS:
                parsed = await asyncio.to_thread(feedparser.parse, feed_url, agent=USER_AGENT)
                if parsed.get("bozo") and not parsed.entries:
                    logger.warning(f"Could not read feed {feed_url}: {parsed.get('bozo_exception')}")
                    continue
                feed_title = (parsed.feed.get("title") or "").strip() or feed_url
                for entry in parsed.entries:
                    if max_items is not None and len(records) >= max_items:
                        break
                    record = await _process_feed_entry(entry, feed_title, db, client, browser, resolver)
                    if record:
                        db.mark_seen(record.source.url)
                        records.append(record)
    finally:
        await browser.close()

    logger.info(f"News Pipeline complete. Extracted {len(records)} fresh records.")
    return records
