import asyncio
import os
from loguru import logger
from storage.db import PipelineDB
from scrapers.research_papers import run_research_papers_pipeline, ScraperConfig
from scrapers.startups import run_startups_pipeline
from scrapers.products import run_products_pipeline
from scrapers.jobs import run_jobs_pipeline
from scrapers.news import run_news_pipeline
from export.to_sheets import SheetsExporter
from entity_resolution.resolver import EntityResolver

async def run_crawlers(db: PipelineDB):
    logger.info("Initializing GraphOne Ingestion Pipeline...")
    
    resolver = EntityResolver()
    results = await asyncio.gather(
        run_research_papers_pipeline(ScraperConfig(target_count=1000)),
        run_startups_pipeline(target_count=1000, resolver=resolver, db=db),
        run_products_pipeline(target_count=1000, resolver=resolver, db=db),
        run_jobs_pipeline(db),
        run_news_pipeline(db),
        return_exceptions=True 
    )
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Pipeline phase {i} failed with error: {str(result)}")
    for record_type, phase_records in zip(("RESEARCH_PAPER", "STARTUP", "PRODUCT", "JOB", "NEWS"), results):
        if isinstance(phase_records, Exception):
            continue
        for record in phase_records:
            db.upsert_record(record_type, record.source.url, record.model_dump(mode="json"))

def main():
    db_path = 'data/pipeline.sqlite3'
    os.makedirs('data', exist_ok=True)
    
    with PipelineDB(db_path) as db:
        asyncio.run(run_crawlers(db))
        
        target_sheet = os.getenv("TARGET_SHEET_URL")
        if target_sheet:
            exporter = SheetsExporter(target_sheet, db)
            exporter.export_all()
        else:
            logger.warning("TARGET_SHEET_URL not set. Skipping Sheets export.")

if __name__ == "__main__":
    main()
