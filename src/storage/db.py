"""SQLite storage for a single-machine run.

Holds the validated records (as JSON, with a dedup key), the seen-URL list
used by the freshness checks, the entity mapping log and a per-phase run
log. A multi-node setup would use Postgres and Redis instead (see
docs/architecture.md); the tables map onto that design directly.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS seen_urls (
    url TEXT PRIMARY KEY,
    content_hash TEXT,
    seen_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_seen_urls_hash ON seen_urls(content_hash);

CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_type TEXT NOT NULL,          -- STARTUP | PRODUCT | RESEARCH_PAPER | JOB | NEWS
    dedup_key TEXT NOT NULL,            -- e.g. source.url, or paper_url
    payload_json TEXT NOT NULL,
    collected_at TEXT NOT NULL,
    UNIQUE(record_type, dedup_key)
);
CREATE INDEX IF NOT EXISTS idx_records_type ON records(record_type);

CREATE TABLE IF NOT EXISTS entity_mapping_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_name TEXT NOT NULL,
    canonical_name TEXT,
    match_score REAL NOT NULL,
    match_method TEXT NOT NULL,
    source_url TEXT NOT NULL,
    resolved_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phase TEXT NOT NULL,                -- e.g. "research_papers", "startups"
    event TEXT NOT NULL,                -- e.g. "started", "completed", "error"
    detail TEXT,
    ts TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PipelineDB:
    def __init__(self, path: str = "data/pipeline.sqlite3"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL;")  # allows concurrent readers during writes
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # -- seen URLs -----------------------------------------------------

    def mark_seen(self, url: str, content_hash: Optional[str] = None) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO seen_urls (url, content_hash, seen_at) VALUES (?, ?, ?)",
            (url, content_hash, _now()),
        )
        self.conn.commit()

    def is_seen(self, url: str) -> bool:
        cur = self.conn.execute("SELECT 1 FROM seen_urls WHERE url = ? LIMIT 1", (url,))
        return cur.fetchone() is not None

    def is_seen_hash(self, content_hash: str) -> bool:
        cur = self.conn.execute("SELECT 1 FROM seen_urls WHERE content_hash = ? LIMIT 1", (content_hash,))
        return cur.fetchone() is not None

    # -- records ---------------------------------------------------------

    def upsert_record(self, record_type: str, dedup_key: str, payload: dict) -> bool:
        """Insert a record unless one with the same dedup key exists.

        Returns True for a new row. `payload` must be JSON-serialisable, e.g. from
        `model.model_dump(mode="json")`.
        """
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO records (record_type, dedup_key, payload_json, collected_at) "
            "VALUES (?, ?, ?, ?)",
            (record_type, dedup_key, json.dumps(payload, default=str), _now()),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def count(self, record_type: str) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM records WHERE record_type = ?", (record_type,))
        return cur.fetchone()[0]

    def iter_records(self, record_type: str) -> Iterator[dict]:
        cur = self.conn.execute(
            "SELECT payload_json FROM records WHERE record_type = ? ORDER BY id", (record_type,)
        )
        for (payload_json,) in cur:
            yield json.loads(payload_json)

    # -- entity mapping log ----------------------------------------------

    def log_entity_mapping(self, raw_name: str, canonical_name: Optional[str], match_score: float,
                            match_method: str, source_url: str) -> None:
        self.conn.execute(
            "INSERT INTO entity_mapping_log (raw_name, canonical_name, match_score, match_method, "
            "source_url, resolved_at) VALUES (?, ?, ?, ?, ?, ?)",
            (raw_name, canonical_name, match_score, match_method, source_url, _now()),
        )
        self.conn.commit()

    def iter_entity_mapping_log(self) -> Iterator[dict]:
        cur = self.conn.execute(
            "SELECT raw_name, canonical_name, match_score, match_method, source_url, resolved_at "
            "FROM entity_mapping_log ORDER BY id"
        )
        cols = ["raw_name", "canonical_name", "match_score", "match_method", "source_url", "resolved_at"]
        for row in cur:
            yield dict(zip(cols, row))

    # -- run log -----------------------------------------------------------

    def log_event(self, phase: str, event: str, detail: str = "") -> None:
        self.conn.execute(
            "INSERT INTO run_log (phase, event, detail, ts) VALUES (?, ?, ?, ?)",
            (phase, event, detail, _now()),
        )
        self.conn.commit()
