"""
Top-level entry point: runs every ingestion phase, persists validated
records to SQLite, then (optionally) exports to Google Sheets.

Run from the repository root:

    python src/main.py                         # full run (1,000 per source)
    python src/main.py --target-count 10 --skip-export   # small smoke run
    python src/main.py --phases papers jobs    # only some phases
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Make `python src/main.py` and `python -m src.main` behave the same: the
# modules below import each other as top-level packages (`storage.db`, ...).
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load .env BEFORE importing the scrapers, so anything read from the
# environment sees the values from the file.
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# Use the OS certificate store for HTTPS. On machines where antivirus or a
# corporate proxy re-signs TLS traffic, Python's bundled CA list rejects every
# API call; the OS store accepts it. No-op if truststore isn't installed.
try:  # noqa: SIM105
    import truststore  # noqa: E402

    truststore.inject_into_ssl()
except ImportError:  # pragma: no cover
    pass

from loguru import logger  # noqa: E402

from entity_resolution.resolver import SEED_STARTUPS, EntityResolver  # noqa: E402
from storage.db import PipelineDB  # noqa: E402
from scrapers.research_papers import ScraperConfig, run_research_papers_pipeline  # noqa: E402
from scrapers.startups import run_startups_pipeline  # noqa: E402
from scrapers.products import run_products_pipeline  # noqa: E402
from scrapers.jobs import run_jobs_pipeline  # noqa: E402
from scrapers.news import run_news_pipeline  # noqa: E402

PHASES = ("papers", "startups", "products", "jobs", "news")
RECORD_TYPES = {
    "papers": "RESEARCH_PAPER",
    "startups": "STARTUP",
    "products": "PRODUCT",
    "jobs": "JOB",
    "news": "NEWS",
}


async def run_crawlers(db: PipelineDB, target_count: int, phases: list[str]) -> None:
    logger.info("Initializing GraphOne Ingestion Pipeline...")

    # The resolver needs the seed list of known startups; without it every
    # name is logged as "unresolved_new" and nothing is canonicalized.
    resolver = EntityResolver(seed_entities={k: list(v) for k, v in SEED_STARTUPS.items()})

    coroutines = {
        "papers": lambda: run_research_papers_pipeline(
            ScraperConfig(target_count=target_count, github_token=os.getenv("GITHUB_TOKEN") or None)
        ),
        "startups": lambda: run_startups_pipeline(target_count=target_count, resolver=resolver, db=db),
        "products": lambda: run_products_pipeline(target_count=target_count, resolver=resolver, db=db),
        "jobs": lambda: run_jobs_pipeline(db=db),
        "news": lambda: run_news_pipeline(db, resolver=resolver, max_items=target_count),
    }
    results = await asyncio.gather(*(coroutines[p]() for p in phases), return_exceptions=True)

    for phase, result in zip(phases, results):
        if isinstance(result, Exception):
            logger.error(f"Phase '{phase}' failed: {type(result).__name__}: {result}")
            db.log_event(phase, "error", f"{type(result).__name__}: {result}")
            continue
        new = sum(
            db.upsert_record(RECORD_TYPES[phase], record.source.url, record.model_dump(mode="json"))
            for record in result
        )
        logger.info(f"Phase '{phase}': {len(result)} records, {new} new")
        db.log_event(phase, "completed", f"{len(result)} records, {new} new")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraphOne ingestion pipeline")
    parser.add_argument("--target-count", type=int, default=1000,
                        help="records per source for papers/startups/products/news (default 1000)")
    parser.add_argument("--phases", nargs="+", choices=PHASES, default=list(PHASES),
                        help="which phases to run (default: all)")
    parser.add_argument("--db", default="data/pipeline.sqlite3", help="SQLite database path")
    parser.add_argument("--skip-export", action="store_true", help="don't export to Google Sheets")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    with PipelineDB(args.db) as db:
        asyncio.run(run_crawlers(db, args.target_count, args.phases))

        target_sheet = os.getenv("TARGET_SHEET_URL")
        if args.skip_export:
            logger.info("Skipping Sheets export (--skip-export).")
        elif target_sheet:
            # Imported lazily: gspread is only needed when exporting.
            from export.to_sheets import SheetsExporter

            SheetsExporter(target_sheet, db).export_all()
        else:
            logger.warning("TARGET_SHEET_URL not set. Skipping Sheets export.")


if __name__ == "__main__":
    main()
