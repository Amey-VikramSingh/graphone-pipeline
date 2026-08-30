"""
Tests for src/scrapers/jobs.py — role classification, remote detection,
Greenhouse/Lever payload parsing, and the layered freshness logic that
distinguishes Lever's trustworthy createdAt from Greenhouse's weaker
updated_at signal. Run with:
    python3 tests/test_jobs_logic.py
"""
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fakes import install_fake_pydantic
install_fake_pydantic()  # must happen before schemas/jobs are imported

from scrapers.jobs import (  # noqa: E402
    classify_role_family, detect_is_remote, to_record, passes_freshness,
    _parse_greenhouse_job, _parse_lever_job, JobBoardConfig,
)
from utils.dates import normalize_date  # noqa: E402
from storage.db import PipelineDB  # noqa: E402

DB_PATH = "data/_test_jobs.sqlite3"


def _cleanup_db():
    for suffix in ("", "-wal", "-shm"):
        p = DB_PATH + suffix
        if os.path.exists(p):
            os.remove(p)


def test_role_classification():
    cases = [
        ("Senior Software Engineer, Backend", "Engineering"),
        ("Research Scientist - Alignment", "Research"),
        ("Data Engineer", "Data"),
        ("Product Manager, Platform", "Product"),
        ("Account Executive, Enterprise", "Sales"),
        ("Head of People Operations", "Operations"),
        ("Corporate Counsel", "Legal"),
        ("Some Totally Novel Title Xyzzy", "Other"),
    ]
    for title, expected in cases:
        got = classify_role_family(title)
        assert got == expected, f"{title!r} -> {got}, expected {expected}"
    print("PASS: role_family classification")


def test_remote_detection():
    assert detect_is_remote("Remote - US") is True
    assert detect_is_remote("San Francisco, CA") is False
    assert detect_is_remote("anything", explicit_flag=True) is True
    assert detect_is_remote("Remote", explicit_flag=False) is False  # explicit flag always wins
    print("PASS: is_remote detection (text heuristic + explicit-flag override)")


def test_greenhouse_lever_parsing():
    board_gh = JobBoardConfig(platform="greenhouse", board_token="openai", company_display_name="OpenAI")
    gh_job = {
        "id": 12345, "title": "Research Engineer, Alignment",
        "updated_at": "2026-08-29T10:00:00-05:00",
        "location": {"name": "Remote - US"},
        "absolute_url": "https://boards.greenhouse.io/openai/jobs/12345",
    }
    parsed = _parse_greenhouse_job(gh_job, board_gh)
    assert parsed["title"] == "Research Engineer, Alignment"
    assert parsed["date_confidence"] == "weak_updated_at_only"
    assert _parse_greenhouse_job({"title": "X"}, board_gh) is None  # missing url -> dropped

    board_lv = JobBoardConfig(platform="lever", board_token="huggingface", company_display_name="Hugging Face")
    lv_job = {
        "text": "ML Engineer, Inference",
        "hostedUrl": "https://jobs.lever.co/huggingface/abc-123",
        "categories": {"location": "Paris, France"},
        "workplaceType": "remote",
        "createdAt": 1735500000000,
    }
    parsed_lv = _parse_lever_job(lv_job, board_lv)
    assert parsed_lv["date_confidence"] == "strong_created_at"
    assert parsed_lv["is_remote_explicit"] is True
    assert _parse_lever_job({"hostedUrl": "https://x.com"}, board_lv) is None  # missing text -> dropped
    print("PASS: Greenhouse/Lever payload parsing against realistic shapes")


def test_layered_freshness():
    now = datetime.now(timezone.utc)
    recent_iso = (now - timedelta(hours=2)).isoformat()
    old_iso = (now - timedelta(hours=48)).isoformat()

    # Lever (strong signal) trusted on timestamp alone
    lever_recent = {"date_confidence": "strong_created_at", "url": "https://jobs.lever.co/hf/a", "date_raw": recent_iso}
    assert passes_freshness(lever_recent, normalize_date(lever_recent["date_raw"])) is True
    lever_old = {"date_confidence": "strong_created_at", "url": "https://jobs.lever.co/hf/b", "date_raw": old_iso}
    assert passes_freshness(lever_old, normalize_date(lever_old["date_raw"])) is False

    os.makedirs("data", exist_ok=True)
    _cleanup_db()
    db = PipelineDB(DB_PATH)

    # Greenhouse (weak signal): recent + unseen -> fresh
    gh = {"date_confidence": "weak_updated_at_only", "url": "https://boards.greenhouse.io/openai/jobs/1", "date_raw": recent_iso}
    d = normalize_date(gh["date_raw"])
    assert passes_freshness(gh, d, db=db) is True

    # Same job, already seen -> must NOT re-qualify even with a "recent" updated_at
    # (this is the core defense against a recruiter re-touching an old posting)
    db.mark_seen(gh["url"])
    assert passes_freshness(gh, d, db=db) is False

    # Greenhouse job, old + unseen -> not fresh
    gh_old = {"date_confidence": "weak_updated_at_only", "url": "https://boards.greenhouse.io/openai/jobs/2", "date_raw": old_iso}
    assert passes_freshness(gh_old, normalize_date(gh_old["date_raw"]), db=db) is False

    db.close()
    _cleanup_db()
    print("PASS: layered freshness (Lever trusted on timestamp; Greenhouse gated by timestamp AND unseen-url)")


def test_to_record_end_to_end():
    now = datetime.now(timezone.utc)
    raw_job = {
        "company": "OpenAI", "title": "Research Engineer",
        "url": "https://boards.greenhouse.io/openai/jobs/99",
        "date_raw": (now - timedelta(hours=1)).isoformat(),
        "location_raw": "Remote - US", "is_remote_explicit": None,
        "date_confidence": "weak_updated_at_only", "source_name": "boards.greenhouse.io",
    }
    rec = to_record(raw_job)
    assert rec is not None
    assert rec.content.company == "OpenAI"
    assert rec.content.is_remote is True
    assert rec.content.role_family == "Research"

    # missing/unparseable date -> dropped, never fabricated
    bad_job = dict(raw_job)
    bad_job["date_raw"] = None
    assert to_record(bad_job) is None
    print("PASS: to_record end-to-end (valid case + missing-date-is-dropped case)")


if __name__ == "__main__":
    test_role_classification()
    test_remote_detection()
    test_greenhouse_lever_parsing()
    test_layered_freshness()
    test_to_record_end_to_end()
    print("\nALL JOBS SCRAPER TESTS PASSED")
