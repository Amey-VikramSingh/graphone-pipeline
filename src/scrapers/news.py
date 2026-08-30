import asyncio
import feedparser
import httpx
from loguru import logger
from typing import List, Optional

from schemas.models import NewsRecord, NewsContent, Source
from utils.dates import normalize_date, is_within_freshness_window
from storage.db import PipelineDB
from llm.chunking import strip_boilerplate

AI_NEWS_FEEDS = [
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://cset.georgetown.edu/news/feed/",
    "https://bair.berkeley.edu/blog/feed.xml",
    "https://blogs.nvidia.com/feed/",
    "https://aws.amazon.com/blogs/machine-learning/feed/"
]

async def _process_feed_entry(entry: dict, db: PipelineDB) -> Optional[NewsRecord]:
    article_url = entry.get("link")
    
    if not article_url or db.is_seen(article_url):
        return None

    raw_date = entry.get("published") or entry.get("updated")
    parsed_date = normalize_date(raw_date)
    
    if not parsed_date or not is_within_freshness_window(parsed_date):
        return None

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(article_url, timeout=15.0)
            response.raise_for_status()
            
            clean_text = strip_boilerplate(response.text)
            if not clean_text:
                return None
            title = entry.get("title", "").strip()
            if not title:
                return None
            return NewsRecord(
                source=Source(name=entry.get("source", {}).get("title", "RSS"), url=article_url),
                content=NewsContent(title=title, published_date=parsed_date, full_text=clean_text),
            )
                
    except Exception as e:
        logger.warning(f"Failed to process news article {article_url}: {str(e)}")
        
    return None

async def run_news_pipeline(db: PipelineDB) -> List[NewsRecord]:
    logger.info("Starting Phase II: News Pipeline...")
    records = []
    
    for feed_url in AI_NEWS_FEEDS:
        parsed = await asyncio.to_thread(feedparser.parse, feed_url)
        for entry in parsed.entries:
            record = await _process_feed_entry(entry, db)
            if record:
                db.upsert_record("NEWS", record.source.url, record.model_dump())
                db.mark_seen(record.source.url)
                records.append(record)
                
    logger.info(f"News Pipeline complete. Extracted {len(records)} fresh records.")
    return records
