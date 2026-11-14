"""
Exercises orchestrator.py's control flow (fallback chain, 429 backoff,
413 rechunk-and-retry, schema validation gating) against fakes, since
litellm/pydantic can't be installed in this sandbox. Run with:
    python3 tests/test_orchestrator_logic.py
"""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fakes import install_fake_pydantic, FakeBaseModel
install_fake_pydantic()  # must happen before orchestrator is imported

from llm import orchestrator as orch  # noqa: E402


class DummySchema(FakeBaseModel):
    pass


class FakeChoice:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


def make_fake_litellm(call_plan):
    """
    call_plan: dict[model] -> list of behaviors consumed in order per call.
    Each behavior is one of: ("raise", ExceptionInstance) | ("return", json_str)
    """
    call_log = []

    class FakeLiteLLM:
        async def acompletion(self, model, messages, response_format, timeout, temperature, **kwargs):
            call_log.append(model)
            behaviors = call_plan[model]
            behavior = behaviors.pop(0)
            kind, payload = behavior
            if kind == "raise":
                raise payload
            return FakeResponse(payload)

    return FakeLiteLLM(), call_log


async def test_fallback_chain_on_429_exhaustion_then_bad_json_then_success():
    provider1 = orch.ProviderConfig(model="fake/p1", max_retries_on_429=2, base_backoff_seconds=0.01, max_backoff_seconds=0.02)
    provider2 = orch.ProviderConfig(model="fake/p2", max_retries_on_429=1, base_backoff_seconds=0.01, max_backoff_seconds=0.02)
    provider3 = orch.ProviderConfig(model="fake/p3", max_retries_on_429=1, base_backoff_seconds=0.01, max_backoff_seconds=0.02)

    plan = {
        "fake/p1": [("raise", orch.RateLimitError("429")) for _ in range(3)],  # exceeds max_retries_on_429=2 -> exhausted
        "fake/p2": [("return", "not valid json {{{")],  # triggers JSONDecodeError -> fall through
        "fake/p3": [("return", json.dumps({"title": "Real Paper", "paper_url": "https://arxiv.org/abs/1"}))],
    }
    fake_litellm, call_log = make_fake_litellm(plan)
    orch.litellm = fake_litellm

    result = await orch.extract_structured(
        raw_text="short text",
        system_prompt="extract",
        schema=DummySchema,
        source_url="https://arxiv.org/abs/1",
        chain=[provider1, provider2, provider3],
    )

    assert result.ok is True, f"expected success, got {result.error}"
    assert result.provider_used == "fake/p3"
    assert call_log.count("fake/p1") == 3  # 2 retries + initial = 3 attempts before exhaustion
    assert call_log.count("fake/p2") == 1
    assert call_log.count("fake/p3") == 1
    outcomes = [a["outcome"] for a in result.attempts]
    assert "429_exhausted_after_3_attempts" in outcomes
    assert any(o.startswith("invalid_response") for o in outcomes)
    assert "success" in outcomes
    print("PASS: fallback chain (429 exhaustion -> bad json -> success)")


async def test_413_triggers_rechunk_and_retry_succeeds_on_smaller_chunk():
    # This exercises _extract_from_chunk_with_retry directly with a chunk
    # that is deliberately larger than the provider's stated budget --
    # simulating a provider that unexpectedly 413s on something the
    # top-level chunker sized optimistically (token-count estimation
    # mismatches between our estimator and the provider's real tokenizer
    # are exactly the real-world case this path exists for).
    provider = orch.ProviderConfig(model="fake/p1", max_context_tokens=2000, max_retries_on_429=1,
                                    base_backoff_seconds=0.01, max_backoff_seconds=0.02)

    plan = {
        "fake/p1": [
            ("raise", orch.ContextWindowExceededError("413")),
            ("return", json.dumps({"title": "Small Chunk Paper", "paper_url": "https://arxiv.org/abs/2"})),
        ],
    }
    fake_litellm, call_log = make_fake_litellm(plan)
    orch.litellm = fake_litellm

    oversized_chunk = "word " * 4000  # ~1000+ tokens by approx counter, well above smaller_budget after halving

    outcome = await orch._extract_from_chunk_with_retry(
        provider=provider, chunk=oversized_chunk, chunk_idx=0, total_chunks=1,
        system_prompt="extract", schema=DummySchema, source_url="https://arxiv.org/abs/2",
    )

    assert outcome["result"] is not None, "expected success after rechunk"
    assert call_log.count("fake/p1") == 2
    assert outcome["trace"]["outcome"].startswith("413_rechunked_into_")
    assert outcome["trace"]["sub_attempt"]["outcome"] == "success"
    print("PASS: 413 triggers rechunk-and-retry, recovers on smaller chunk")


async def test_all_providers_exhausted_returns_ok_false():
    provider1 = orch.ProviderConfig(model="fake/x1", max_retries_on_429=0, base_backoff_seconds=0.01, max_backoff_seconds=0.02)
    provider2 = orch.ProviderConfig(model="fake/x2", max_retries_on_429=0, base_backoff_seconds=0.01, max_backoff_seconds=0.02)

    plan = {
        "fake/x1": [("raise", orch.RateLimitError("429"))],
        "fake/x2": [("raise", orch.RateLimitError("429"))],
    }
    fake_litellm, call_log = make_fake_litellm(plan)
    orch.litellm = fake_litellm

    result = await orch.extract_structured(
        raw_text="short",
        system_prompt="extract",
        schema=DummySchema,
        source_url="https://example.com/x",
        chain=[provider1, provider2],
    )
    assert result.ok is False
    assert result.data is None
    assert "exhausted" in result.error
    print("PASS: full chain exhaustion returns ok=False with no fabricated data")


async def test_backoff_timing_grows_and_respects_cap():
    delays = [orch._backoff_delay(a, base=1.0, cap=8.0) for a in range(6)]
    # Each delay must be within [0, min(cap, base*2^attempt)]
    for attempt, d in enumerate(delays):
        upper = min(8.0, 1.0 * (2 ** attempt))
        assert 0 <= d <= upper, f"attempt {attempt}: delay {d} exceeds upper bound {upper}"
    assert max(delays) <= 8.0
    print("PASS: exponential backoff respects per-attempt bound and cap")


async def main():
    await test_fallback_chain_on_429_exhaustion_then_bad_json_then_success()
    await test_413_triggers_rechunk_and_retry_succeeds_on_smaller_chunk()
    await test_all_providers_exhausted_returns_ok_false()
    await test_backoff_timing_grows_and_respects_cap()
    print("\nALL ORCHESTRATOR LOGIC TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
