"""Date parsing and the 24-hour freshness check.

normalize_date() turns "2 hours ago", "Jan 3, 2025", RSS (RFC 822) dates, ISO
8601 strings and epoch numbers into UTC datetimes. When a string can't be
parsed it returns None, and callers treat the date as unknown instead of
substituting the current time.
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
    """Parse an absolute or relative date string into a UTC datetime, or return None."""
    if not raw or not raw.strip():
        return None
    raw = raw.strip()

    # Fast path for clean ISO 8601 strings, which is what most APIs return
    # (Lever's createdAt, arXiv's <published>). It skips dateparser, so it
    # also works when dateparser isn't installed.
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
    """True if dt falls inside the window. A missing date (None) is never fresh;
    use is_probably_new() for sources that have no usable dates.
    """
    if dt is None:
        return False
    now = now or datetime.now(timezone.utc)
    return (now - dt) <= timedelta(hours=hours) and dt <= now + timedelta(minutes=5)  # small clock-skew tolerance


def is_probably_new(url: str, seen_urls: "SeenUrlStore", *, content_hash: Optional[str] = None) -> bool:
    """Fallback for sources without reliable dates: an item counts as new if we
    haven't seen its URL or its content hash before. The hash catches the same
    article reposted under a different URL (tracking parameters, for example).
    """
    if seen_urls.contains(url):
        return False
    if content_hash and seen_urls.contains_hash(content_hash):
        return False
    return True


class SeenUrlStore:
    """In-memory record of URLs and content hashes already processed. A multi-node
    setup would keep these in a shared Redis set instead (see docs/architecture.md).
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
