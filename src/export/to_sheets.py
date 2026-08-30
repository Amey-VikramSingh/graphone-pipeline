import gspread
import os
from loguru import logger
from storage.db import PipelineDB

REQUIRED_TABS = [
    "Startups", "Products", "Research Papers", 
    "Jobs", "News", "Entity Mapping Log"
]

class SheetsExporter:
    def __init__(self, spreadsheet_url: str, db: PipelineDB):
        self.spreadsheet_url = spreadsheet_url
        self.db = db
        creds = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        if not creds:
            raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON. Cannot export to sheets.")
        
        self.client = gspread.service_account(filename=creds)
        self.sheet = self.client.open_by_url(self.spreadsheet_url)
        self._ensure_tabs()

    def _ensure_tabs(self):
        existing = [ws.title for ws in self.sheet.worksheets()]
        for tab in REQUIRED_TABS:
            if tab not in existing:
                self.sheet.add_worksheet(title=tab, rows=1000, cols=10)

    def _write_batch(self, tab_name: str, headers: list, records: list, row_mapper: callable):
        worksheet = self.sheet.worksheet(tab_name)
        worksheet.clear()
        
        rows = [headers]
        for r in records:
            rows.append(row_mapper(r))
            
        if len(rows) > 1:
            worksheet.append_rows(rows, value_input_option="USER_ENTERED")
        logger.info(f"Exported {len(rows)-1} records to {tab_name}.")

    def export_all(self):
        logger.info("Starting batch export to Google Sheets...")
        
        startups = list(self.db.iter_records("STARTUP"))
        self._write_batch(
            "Startups", 
            ["Entity Name", "Employee Count", "Source Name", "Source URL", "Collected At"], 
            startups,
            lambda r: [r["content"]["entityName"], r["content"]["data"].get("employeeCount"), r["source"]["name"], r["source"]["url"], r["collectedAt"]]
        )

        products = list(self.db.iter_records("PRODUCT"))
        self._write_batch(
            "Products",
            ["Startup Name", "Pricing Model", "Source Name", "Source URL", "Collected At"],
            products,
            lambda r: [r["content"]["startupName"], r["content"]["pricingModel"], r["source"]["name"], r["source"]["url"], r["collectedAt"]]
        )

        papers = list(self.db.iter_records("RESEARCH_PAPER"))
        self._write_batch(
            "Research Papers",
            ["Title", "Authors", "GitHub Stars", "GitHub URL", "Paper URL", "Published Date"],
            papers,
            lambda r: [r["content"]["title"], ", ".join(r["content"]["authors"]), r["content"].get("github_stars"), r["content"].get("github_url"), r["content"]["paper_url"], r["content"].get("published_date")]
        )

        jobs = list(self.db.iter_records("JOB"))
        self._write_batch(
            "Jobs",
            ["Company", "Role Family", "Is Remote", "Date", "Source URL"],
            jobs,
            lambda r: [r["content"]["company"], r["content"]["role_family"], r["content"]["is_remote"], r["content"]["date"], r["source"]["url"]]
        )
        
        news = list(self.db.iter_records("NEWS"))
        self._write_batch(
            "News",
            ["Title", "Published Date", "Source URL", "Collected At"],
            news,
            lambda r: [r["content"]["title"], r["content"].get("published_date"), r["source"]["url"], r["collectedAt"]]
        )
        mappings = list(self.db.iter_entity_mapping_log())
        self._write_batch(
            "Entity Mapping Log",
            ["Raw Name", "Canonical Name", "Match Score", "Match Method", "Source URL", "Resolved At"],
            mappings,
            lambda r: [r["raw_name"], r["canonical_name"], r["match_score"], r["match_method"], r["source_url"], r["resolved_at"]],
        )
