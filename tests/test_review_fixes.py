"""
Tests that use the real installed libraries (pydantic, litellm, ...)
rather than the fakes in _fakes.py. Run with:
    python tests/test_review_fixes.py
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from entity_resolution.resolver import SEED_STARTUPS, EntityResolver  # noqa: E402
from llm import orchestrator as orch  # noqa: E402
from llm.chunking import chunk_text  # noqa: E402
from main import parse_args  # noqa: E402
from scrapers.jobs import (  # noqa: E402
    DEFAULT_JOB_BOARDS, JobBoardConfig, _parse_ashby_job, passes_freshness, to_record,
)
from scrapers.news import _feed_text  # noqa: E402
from scrapers.products import PricingExtraction  # noqa: E402
from scrapers.research_papers import _extract_github_url_from_text, _parse_arxiv_entry  # noqa: E402
from scrapers.startups import (  # noqa: E402
    build_company_ids_query, build_details_query, parse_binding, to_record as startup_to_record,
)
from utils.dates import normalize_date  # noqa: E402
from utils.html_text import html_to_text, strip_tags  # noqa: E402


def seeded_resolver() -> EntityResolver:
    return EntityResolver(seed_entities={k: list(v) for k, v in SEED_STARTUPS.items()})


def test_resolver_uses_seed_list_and_aliases():
    r = seeded_resolver()
    assert r.resolve("OpenAI, Inc.").canonical_name == "OpenAI"
    fuzzy = r.resolve("Eleven Labz")  # near the alias "Eleven Labs", not the canonical "ElevenLabs"
    assert fuzzy.canonical_name == "ElevenLabs" and fuzzy.match_method == "fuzzy"
    assert r.resolve("Totally New Startup Co").match_method == "unresolved_new"
    # The bug: main.py built EntityResolver() with no seed list, so nothing resolved.
    assert EntityResolver().resolve("OpenAI").match_method == "unresolved_new"
    print("PASS: resolver resolves against the seed list, including fuzzy-on-alias")


def test_find_mentions_precision():
    r = seeded_resolver()
    text = ("OpenAI and Anthropic raised money; Mistral AI too. We need to scale "
            "together. Midjourney, Groq and Hugging Face shipped. An ElevenLabs-backed fund.")
    assert r.find_mentions(text) == [
        "Anthropic", "ElevenLabs", "Groq", "Hugging Face", "Midjourney", "Mistral AI", "OpenAI",
    ]
    # everyday words that are also company names are not counted
    assert r.find_mentions("A notion of scale: work together like a writer.") == []
    assert r.find_mentions("NotOpenAIish") == []
    print("PASS: find_mentions matches known entities and skips ambiguous words")


def test_ashby_parsing_and_freshness():
    board = JobBoardConfig(platform="ashby", board_token="openai", company_display_name="OpenAI")
    job = {
        "title": " Research Engineer ", "jobUrl": "https://jobs.ashbyhq.com/openai/abc",
        "location": "San Francisco", "isRemote": True, "workplaceType": "Hybrid",
        "publishedAt": "2026-11-13T10:00:00.000+00:00", "isListed": True,
    }
    raw = _parse_ashby_job(job, board)
    assert raw["title"] == "Research Engineer" and raw["source_name"] == "jobs.ashbyhq.com"
    # workplaceType wins over the contradictory isRemote flag seen in live data
    assert raw["is_remote_explicit"] is False
    assert _parse_ashby_job({**job, "workplaceType": "Remote", "isRemote": False}, board)["is_remote_explicit"] is True
    assert _parse_ashby_job({**job, "workplaceType": None}, board)["is_remote_explicit"] is True
    assert _parse_ashby_job({**job, "isListed": False}, board) is None
    assert _parse_ashby_job({**job, "jobUrl": None, "applyUrl": None}, board) is None

    record = to_record(raw)
    assert record is not None and record.content.role_family == "Research"
    published = normalize_date(raw["date_raw"])
    now = published.replace(hour=20)
    assert passes_freshness(raw, published, now=now) is True
    print("PASS: Ashby payload parsing (remote precedence, unlisted dropped) + freshness")


def test_default_boards_are_the_verified_set():
    boards = {(b.platform, b.board_token) for b in DEFAULT_JOB_BOARDS}
    assert boards == {
        ("ashby", "openai"), ("greenhouse", "anthropic"), ("greenhouse", "scaleai"),
        ("ashby", "cohere"), ("ashby", "elevenlabs"),
    }
    print("PASS: default job boards are the five verified live")


def test_github_url_found_in_arxiv_comment():
    import xml.etree.ElementTree as ET
    entry = ET.fromstring(
        '<entry xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">'
        "<id>http://arxiv.org/abs/2611.00001v1</id><title>A  Paper</title>"
        "<summary>No link here.</summary>"
        "<arxiv:comment>12 pages. Code: https://github.com/acme/cool-model.</arxiv:comment>"
        "<published>2026-11-01T00:00:00Z</published></entry>"
    )
    raw = _parse_arxiv_entry(entry)
    assert raw["title"] == "A Paper"
    assert _extract_github_url_from_text(raw["summary"]) is None
    assert _extract_github_url_from_text(raw["comment"]) == "https://github.com/acme/cool-model"
    print("PASS: arXiv comment field is parsed and its GitHub link extracted")


def test_wikidata_queries_and_parsing():
    ids_query = build_company_ids_query()
    assert "wdt:P452" in ids_query and "wd:Q11660" in ids_query and "wd:Q2539" in ids_query
    assert "wdt:P31/wdt:P279?" in ids_query  # one subclass step, not the slow full walk
    details = build_details_query(["Q1", "Q42"])
    assert "VALUES ?company { wd:Q1 wd:Q42 }" in details and "wdt:P1128" in details

    def row(uri, name=None, employees=None, site=None):
        r = {"company": {"type": "uri", "value": uri}}
        if name is not None:
            r["name"] = {"type": "literal", "value": name}
        if employees is not None:
            r["employeeCount"] = {"type": "literal", "value": employees}
        if site is not None:
            r["website"] = {"type": "uri", "value": site}
        return r

    full = parse_binding(row("http://www.wikidata.org/entity/Q21708200", "OpenAI", "4500", "https://openai.com/"))
    assert full == {
        "entityName": "OpenAI", "employeeCount": 4500,
        "source_url": "https://www.wikidata.org/wiki/Q21708200", "website": "https://openai.com/",
    }
    # Missing employee count stays None; it is never estimated.
    assert parse_binding(row("http://www.wikidata.org/entity/Q5", "Acme AI"))["employeeCount"] is None
    assert parse_binding(row("http://www.wikidata.org/entity/Q6", "Bad", "lots"))["employeeCount"] is None
    # No English name or no real entity id: dropped.
    assert parse_binding(row("http://www.wikidata.org/entity/Q7")) is None
    assert parse_binding(row("http://www.wikidata.org/entity/P31", "Not an item")) is None

    record = startup_to_record(full, resolver=seeded_resolver())
    assert record is not None and record.content.entityName == "OpenAI"
    assert record.source.name == "wikidata.org" and record.content.data.employeeCount == 4500
    print("PASS: Wikidata queries built; rows parsed without guessing missing fields")


def test_html_to_text_extracts_article():
    html = """<html><head><title>t</title><script>var x = 1;</script></head><body>
      <nav>Home | About | Login</nav>
      <article><h1>Big News</h1><p>First paragraph with <b>bold</b> text &amp; an entity.</p>
      <script>track()</script><p>Second paragraph.</p></article>
      <footer>(c) 2026 Example</footer></body></html>"""
    text = html_to_text(html, min_chars=10)
    assert "Big News" in text and "First paragraph with bold text & an entity." in text
    assert "Second paragraph." in text
    for junk in ("var x", "track()", "Login", "(c) 2026"):
        assert junk not in text, junk
    assert strip_tags("<p>Hello <i>world</i></p>") == "Hello world"
    assert _feed_text({"content": [{"value": "<p>Full body</p>"}], "summary": "short"}) == "Full body"
    assert _feed_text({"summary": "<b>Only</b> summary"}) == "Only summary"
    print("PASS: HTML -> article text (scripts, nav, footer removed); feed text fallback")


def test_pricing_extraction_tolerates_bad_shapes():
    assert PricingExtraction.model_validate({"pricingModel": "freemium"}).pricingModel == "FREEMIUM"
    assert PricingExtraction.model_validate({"pricingModel": "cheap"}).pricingModel == "UNKNOWN"
    assert PricingExtraction.model_validate(["FREE"]).pricingModel == "UNKNOWN"
    assert PricingExtraction.model_validate({"pricingModel": 3}).pricingModel == "UNKNOWN"
    print("PASS: pricing extraction degrades to UNKNOWN on malformed LLM output")


def test_unexpected_provider_error_falls_through_chain():
    class AuthError(Exception):
        pass

    calls = []

    async def fake_call(provider, system_prompt, user_content):
        calls.append(provider.model)
        if provider.model == "fake/bad-key":
            raise AuthError("401 invalid api key")
        return json.dumps({"pricingModel": "PAID"})

    original = orch._call_provider
    orch._call_provider = fake_call
    try:
        result = asyncio.run(orch.extract_structured(
            raw_text="Plans from $10/month.", system_prompt="classify",
            schema=PricingExtraction, source_url="https://example.test",
            chain=[orch.ProviderConfig(model="fake/bad-key"), orch.ProviderConfig(model="fake/good")],
        ))
    finally:
        orch._call_provider = original
    assert result.ok and result.provider_used == "fake/good"
    assert calls == ["fake/bad-key", "fake/good"]  # no retries on a non-retryable error
    assert result.attempts[0]["outcome"] == "provider_failed:AuthError"
    print("PASS: a rejected key / unknown provider error falls through to the next provider")


def test_dead_provider_is_skipped_for_rest_of_run():
    class AuthenticationError(Exception):
        pass

    calls = []

    async def fake_call(provider, system_prompt, user_content):
        calls.append(provider.model)
        if provider.model == "fake/revoked-key":
            raise AuthenticationError("401 invalid api key")
        if provider.model == "fake/no-quota":
            raise orch.RateLimitError(
                message="429: You exceeded your current quota, check your plan and billing",
                llm_provider="fake", model=provider.model)
        return json.dumps({"pricingModel": "FREE"})

    chain = [orch.ProviderConfig(model="fake/revoked-key"),
             orch.ProviderConfig(model="fake/no-quota", base_backoff_seconds=5),
             orch.ProviderConfig(model="fake/good")]
    original = orch._call_provider
    orch._call_provider = fake_call
    orch.reset_disabled_providers()
    try:
        for _ in range(3):  # three products in one run
            result = asyncio.run(orch.extract_structured(
                raw_text="Free forever.", system_prompt="classify", schema=PricingExtraction,
                source_url="https://example.test", chain=chain))
            assert result.ok and result.provider_used == "fake/good"
    finally:
        orch._call_provider = original
        disabled = orch.disabled_providers()
        orch.reset_disabled_providers()
    # each dead provider was called exactly once (no backoff retries on a used-up
    # quota), then skipped for the remaining items
    assert calls == ["fake/revoked-key", "fake/no-quota", "fake/good", "fake/good", "fake/good"], calls
    assert set(disabled) == {"fake/revoked-key", "fake/no-quota"}
    print("PASS: rejected keys and used-up quotas disable a provider for the rest of the run")


def test_default_chain_uses_current_models():
    assert [p.model for p in orch.DEFAULT_CHAIN] == [
        "gemini/gemini-3.8-flash", "groq/llama-3.3-70b-versatile", "deepseek/deepseek-flash",
    ]
    print("PASS: default LLM chain no longer uses retired models")


def test_chunk_strategy_reports_sliding_window():
    plan = chunk_text("word " * 20000, max_tokens_per_chunk=500, strip_noise=False)
    assert plan.strategy == "sliding_window" and len(plan.chunks) > 1
    print("PASS: chunker reports the sliding_window strategy when it uses it")


def test_cli_arguments():
    args = parse_args(["--target-count", "10", "--phases", "papers", "jobs", "--skip-export"])
    assert args.target_count == 10 and args.phases == ["papers", "jobs"] and args.skip_export
    assert parse_args([]).phases == ["papers", "startups", "products", "jobs", "news"]
    print("PASS: CLI arguments (target count, phase selection, skip export)")


if __name__ == "__main__":
    test_resolver_uses_seed_list_and_aliases()
    test_find_mentions_precision()
    test_ashby_parsing_and_freshness()
    test_default_boards_are_the_verified_set()
    test_github_url_found_in_arxiv_comment()
    test_wikidata_queries_and_parsing()
    test_html_to_text_extracts_article()
    test_pricing_extraction_tolerates_bad_shapes()
    test_unexpected_provider_error_falls_through_chain()
    test_dead_provider_is_skipped_for_rest_of_run()
    test_default_chain_uses_current_models()
    test_chunk_strategy_reports_sliding_window()
    test_cli_arguments()
    print("\nALL REVIEW-FIX TESTS PASSED")
