"""
Tests for src/utils/dates.py — ISO fast-path parsing, freshness window
boundaries, and the seen-URL/content-hash store. Run with:
    python3 tests/test_dates_logic.py
"""
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.dates import normalize_date, is_within_freshness_window, SeenUrlStore  # noqa: E402


def test_iso_fast_path():
    d = normalize_date("2024-01-15T18:30:00Z")
    assert d.year == 2024 and d.month == 1 and d.day == 15 and d.tzinfo is not None

    d2 = normalize_date("2024-01-15T18:30:00+00:00")
    assert d2 == d

    assert normalize_date(None) is None
    assert normalize_date("") is None
    assert normalize_date("   ") is None
    print("PASS: ISO 8601 fast-path parsing + None/empty handling")


def test_relative_date_without_dateparser_raises_clearly():
    # Without dateparser installed, confirm the failure mode
    # is a clear, actionable RuntimeError and NOT a silently-wrong date.
    try:
        normalize_date("2 hours ago")
        raised = False
    except RuntimeError:
        raised = True
    # This assertion is conditional: if dateparser IS installed (e.g. in a
    # real deployment), this should succeed instead. Either outcome is
    # correct for its environment; what's never acceptable is a wrong,
    # silently-fabricated date.
    import utils.dates as dates_mod
    if dates_mod.dateparser is None:
        assert raised, "expected a clear RuntimeError when dateparser is unavailable"
        print("PASS: relative date without dateparser raises clearly (no silent fabrication)")
    else:
        result = normalize_date("2 hours ago")
        assert result is not None
        print("PASS: relative date parsed correctly (dateparser available in this environment)")


def test_freshness_window_boundaries():
    now = datetime.now(timezone.utc)
    assert is_within_freshness_window(now - timedelta(hours=1), now=now) is True
    assert is_within_freshness_window(now - timedelta(hours=23, minutes=59), now=now) is True
    assert is_within_freshness_window(now - timedelta(hours=25), now=now) is False
    assert is_within_freshness_window(None, now=now) is False
    assert is_within_freshness_window(now + timedelta(hours=1), now=now) is False  # future date beyond clock-skew tolerance
    assert is_within_freshness_window(now + timedelta(minutes=2), now=now) is True  # small skew tolerated
    print("PASS: freshness window boundaries (23h59m in, 25h out, None out, future-beyond-skew out)")


def test_seen_url_store():
    s = SeenUrlStore()
    assert s.contains("http://a.com/1") is False
    s.mark_seen("http://a.com/1", content_hash="abc123")
    assert s.contains("http://a.com/1") is True
    assert s.contains_hash("abc123") is True
    assert s.contains("http://a.com/2") is False
    assert s.contains_hash("nope") is False
    print("PASS: in-memory SeenUrlStore (url + content-hash tracking)")


if __name__ == "__main__":
    test_iso_fast_path()
    test_relative_date_without_dateparser_raises_clearly()
    test_freshness_window_boundaries()
    test_seen_url_store()
    print("\nALL DATES MODULE TESTS PASSED")
