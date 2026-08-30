"""
Date normalization: turns messy real-world date strings ("2 hours ago",
"Jan 3, 2025", ISO 8601, RFC 822 from RSS, bare epoch ints, etc.) into
timezone-aware UTC datetimes, and provides the 24-hour freshness check
used by Phase II.

Design notes:
- We never silently invent a date. If nothing usable is found, the
  caller gets `None` back and must apply the "intelligent heuristic"
  (see `is_probably_new`) rather than fabricating a timestamp.
- All parsing is timezone-aware; naive datetimes are assumed UTC only
  as an explicit, logged fallback (dateparser.settings below).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    import dateparser  # heavy lifting for relative/fuzzy dates
except ImportError:  # pragma: no cover - allows syntax-only checks w/o the dep
    dateparser = None

FRESHNESS_WINDOW_HOURS = 24

_DATEPARSER_SETTINGS = {
    "RETURN_AS_TIMEZONE_AWARE": True,
    "TIMEZONE": "UTC",
    "TO_TIMEZONE": "UTC",
    "PREFER_DATES_FROM": "past",  # "2 hours ago" / "yesterday" style content
}


def normalize_date(raw: Optional[str], *, reference_time: Optional[datetime] = None) -> Optional[datetime]:
    """
    Parse a raw date string (absolute or relative) into a UTC-aware datetime.

    Returns None if the string is empty/unparseable — callers must treat
    that as "unknown", never coerce it to now() or any other guessed value.
    """
    if not raw or not raw.strip():
        return None
    raw = raw.strip()

    # Fast path: already-clean ISO 8601 (the common case for API-sourced
    # dates, e.g. Lever's createdAt or Arxiv's <published>). Skips the
    # heavier dateparser fuzzy-matching entirely when it's not needed,
    # and lets this path work even in environments without dateparser
    # installed (this module degrades gracefully rather than hard-
    # requiring the dependency for the common case).
    iso_candidate = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        dt = datetime.fromisoformat(iso_candidate)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass

    if dateparser is None:
        raise RuntimeError(
            "dateparser is not installed and this string isn't clean ISO 8601 "
            f"({raw!r}); pip install -r requirements.txt for full relative/fuzzy "
            "date parsing (e.g. '2 hours ago')."
        )

    settings = dict(_DATEPARSER_SETTINGS)
    if reference_time is not None:
        settings["RELATIVE_BASE"] = reference_time

    dt = dateparser.parse(raw, settings=settings)
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def is_within_freshness_window(dt: Optional[datetime], *, now: Optional[datetime] = None,
                                hours: int = FRESHNESS_WINDOW_HOURS) -> bool:
    """
    Strict freshness check used to gate News/Jobs ingestion. If dt is None
    (unparseable / missing), this returns False — an item with no
    trustworthy timestamp does NOT get to claim freshness by default.
    Use `is_probably_new` for the heuristic fallback path instead.
    """
    if dt is None:
        return False
    now = now or datetime.now(timezone.utc)
    return (now - dt) <= timedelta(hours=hours) and dt <= now + timedelta(minutes=5)  # small clock-skew tolerance


def is_probably_new(url: str, seen_urls: "SeenUrlStore", *, content_hash: Optional[str] = None) -> bool:
    """
    Intelligent heuristic for sources with no reliable date signal at all.

    Strategy: treat "new" as "not previously seen by this crawler, by URL
    OR by content hash (to catch same content at a different URL/query
    string, e.g. tracking params)". This trades perfect precision for a
    conservative, explainable rule that doesn't fabricate a timestamp.
    Combined with a bounded per-run crawl window (e.g. only the first N
    items on a listing page, which are typically reverse-chronological),
    this keeps false "fresh" positives low without ever inventing a date.
    """
    if seen_urls.contains(url):
        return False
    if content_hash and seen_urls.contains_hash(content_hash):
        return False
    return True


class SeenUrlStore:
    """
    Minimal interface for the "have we processed this before" check that
    must hold across distributed crawler nodes. This in-memory/SQLite
    version is for local runs; see architecture.pdf for the Redis-backed
    design used at scale (shared SETs across worker nodes).
    """

    def __init__(self, db_conn=None):
        self._conn = db_conn
        self._mem_urls: set[str] = set()
        self._mem_hashes: set[str] = set()

    def contains(self, url: str) -> bool:
        if self._conn is None:
            return url in self._mem_urls
        cur = self._conn.execute("SELECT 1 FROM seen_urls WHERE url = ? LIMIT 1", (url,))
        return cur.fetchone() is not None

    def contains_hash(self, content_hash: str) -> bool:
        if self._conn is None:
            return content_hash in self._mem_hashes
        cur = self._conn.execute("SELECT 1 FROM seen_urls WHERE content_hash = ? LIMIT 1", (content_hash,))
        return cur.fetchone() is not None

    def mark_seen(self, url: str, content_hash: Optional[str] = None) -> None:
        if self._conn is None:
            self._mem_urls.add(url)
            if content_hash:
                self._mem_hashes.add(content_hash)
            return
        self._conn.execute(
            "INSERT OR IGNORE INTO seen_urls (url, content_hash, seen_at) VALUES (?, ?, ?)",
            (url, content_hash, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()
