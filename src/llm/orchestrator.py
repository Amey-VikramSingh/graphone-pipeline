"""Structured extraction with a chain of LLM providers.

Providers are tried in order (see DEFAULT_CHAIN). For each one:
  - 429 rate limits are retried with exponential backoff, waiting as long as
    the provider asks when it says. Once retries run out, the next provider
    is tried.
  - 413 / context-length errors make the chunk smaller and retry it.
  - Every response must parse as JSON and validate against the schema; one
    that doesn't counts as a failure and the next provider is tried.
  - A provider that can't recover during this run (rejected key, no balance,
    daily cap reached) is switched off for the rest of the run.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel, ValidationError

from .chunking import chunk_text, count_tokens

try:
    import litellm
    from litellm.exceptions import (
        RateLimitError,
        ContextWindowExceededError,
        APIError,
        Timeout,
        ServiceUnavailableError,
    )
except ImportError:  # pragma: no cover - allows syntax-only checks w/o the dep
    litellm = None
    class RateLimitError(Exception): ...
    class ContextWindowExceededError(Exception): ...
    class APIError(Exception): ...
    class Timeout(Exception): ...
    class ServiceUnavailableError(Exception): ...

logger = logging.getLogger("graphone.llm")


@dataclass
class ProviderConfig:
    model: str                      # litellm model string, e.g. "gemini/gemini-3.8-flash"
    max_context_tokens: int = 8000  # conservative payload-safe budget, not the model's true max
    max_retries_on_429: int = 4
    base_backoff_seconds: float = 1.5
    max_backoff_seconds: float = 30.0
    timeout_seconds: float = 30.0


@dataclass
class ExtractionResult:
    ok: bool
    data: Optional[BaseModel] = None
    provider_used: Optional[str] = None
    attempts: list = field(default_factory=list)  # trace of (provider, outcome) for logging/debugging
    error: Optional[str] = None


# Model ids checked against the providers' live model lists on 2026-11-22.
# Groq retired its Llama 3.3 models; gpt-oss-120b and gpt-oss-20b replace
# them. Groq's free tier allows 1,000 requests a day and 8,000 tokens a
# minute *per model*, so listing both doubles the daily allowance, and the
# extra 429 retries ride out the per-minute token window.
# max_context_tokens is a payload-safe budget, not the model's true maximum.
DEFAULT_CHAIN = [
    ProviderConfig(model="gemini/gemini-3.8-flash", max_context_tokens=100_000),
    ProviderConfig(model="groq/openai/gpt-oss-120b", max_context_tokens=6_000,
                   max_retries_on_429=6, max_backoff_seconds=60.0),
    ProviderConfig(model="groq/openai/gpt-oss-20b", max_context_tokens=6_000,
                   max_retries_on_429=6, max_backoff_seconds=60.0),
    ProviderConfig(model="deepseek/deepseek-flash", max_context_tokens=28_000),
]


# Providers that failed in a way retrying can't fix (rejected key, no
# balance, quota used up, retired model) are skipped for the rest of the
# run. Otherwise a dead provider would be retried for every single item.
_DISABLED_PROVIDERS: dict = {}  # model -> reason

# Groq reports its daily caps as a 429 naming "requests per day (RPD)" or
# "tokens per day (TPD)"; those reset in hours, so treat them as exhausted.
_QUOTA_MARKERS = ("exceeded your current quota", "insufficient_quota", "quota exceeded",
                  "insufficient balance", "insufficient_balance", "billing",
                  "requests per day", "tokens per day", "(rpd)", "(tpd)")


def disabled_providers() -> dict:
    """model -> reason, for providers switched off during this run."""
    return dict(_DISABLED_PROVIDERS)


def reset_disabled_providers() -> None:
    _DISABLED_PROVIDERS.clear()


def _disable(provider: "ProviderConfig", reason: str) -> None:
    if provider.model not in _DISABLED_PROVIDERS:
        _DISABLED_PROVIDERS[provider.model] = reason
        logger.warning("Disabling %s for the rest of this run: %s", provider.model, reason)


# Per-minute limits clear within seconds. Groq's TPM/RPM errors also end with an
# upgrade link to ".../settings/billing", so they must be ruled out before the
# quota markers are checked, or a routine wait would disable the provider.
_PER_MINUTE_MARKERS = ("per minute", "(tpm)", "(rpm)")
_RETRY_AFTER_RE = re.compile(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", re.IGNORECASE)


def _is_quota_exhausted(err: Exception) -> bool:
    text = str(err).lower()
    if any(marker in text for marker in _PER_MINUTE_MARKERS):
        return False
    return any(marker in text for marker in _QUOTA_MARKERS)


def _suggested_wait(err: Exception) -> Optional[float]:
    """Seconds the provider asked us to wait ("Please try again in 11.85s"), if it said."""
    match = _RETRY_AFTER_RE.search(str(err))
    if not match:
        return None
    minutes = float(match.group(1) or 0)
    return minutes * 60 + float(match.group(2))


def _backoff_delay(attempt: int, base: float, cap: float) -> float:
    """Exponential backoff with full jitter (AWS-style), attempt is 0-indexed."""
    upper = min(cap, base * (2 ** attempt))
    return random.uniform(0, upper)


async def _call_provider(
    provider: ProviderConfig,
    system_prompt: str,
    user_content: str,
) -> str:
    """Single call to one provider. Raises on failure; caller handles retry/fallback."""
    if litellm is None:
        raise RuntimeError("litellm is not installed; pip install -r requirements.txt")

    response = await litellm.acompletion(
        model=provider.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        timeout=provider.timeout_seconds,
        temperature=0,
        # Some newer models reject parameters (gemini-3.8-flash rejects
        # `temperature`). Drop what a model doesn't support instead of
        # failing the call; JSON-mode and schema validation still apply.
        drop_params=True,
    )
    return response.choices[0].message.content


async def extract_structured(
    *,
    raw_text: str,
    system_prompt: str,
    schema: Type[BaseModel],
    source_url: str,
    chain: Optional[list] = None,
) -> ExtractionResult:
    """Extract `schema`-validated data from `raw_text` using the provider chain.

    `source_url` is only used in logs, to trace a result back to its page. It is
    never taken from the model's output.
    """
    chain = chain or DEFAULT_CHAIN
    attempts = []

    for provider in chain:
        if provider.model in _DISABLED_PROVIDERS:
            attempts.append({"provider": provider.model, "source_url": source_url,
                             "outcome": f"skipped_disabled:{_DISABLED_PROVIDERS[provider.model]}"})
            continue

        # Chunk to fit this provider's payload-safe budget. If the raw
        # text already fits, chunk_text returns a single chunk (no-op).
        plan = chunk_text(raw_text, max_tokens_per_chunk=provider.max_context_tokens)

        # Use the first chunk that gives a valid result. The facts we extract
        # (title, price, company) are usually near the top of a page, so this
        # saves calls; pages with scattered facts would need merging instead.
        for chunk_idx, chunk in enumerate(plan.chunks):
            outcome = await _extract_from_chunk_with_retry(
                provider=provider,
                chunk=chunk,
                chunk_idx=chunk_idx,
                total_chunks=len(plan.chunks),
                system_prompt=system_prompt,
                schema=schema,
                source_url=source_url,
            )
            attempts.append(outcome["trace"])

            if outcome["result"] is not None:
                return ExtractionResult(
                    ok=True,
                    data=outcome["result"],
                    provider_used=provider.model,
                    attempts=attempts,
                )
            # Still too large after re-chunking: skip this chunk on this provider.
            if provider.model in _DISABLED_PROVIDERS:
                break  # provider died mid-document; don't try its other chunks

        logger.warning("Provider exhausted without a valid extraction: %s (source=%s)",
                        provider.model, source_url)

    return ExtractionResult(
        ok=False,
        attempts=attempts,
        error="All providers in fallback chain exhausted without a schema-valid extraction.",
    )


async def _extract_from_chunk_with_retry(
    *, provider: ProviderConfig, chunk: str, chunk_idx: int, total_chunks: int,
    system_prompt: str, schema: Type[BaseModel], source_url: str,
) -> dict:
    trace = {"provider": provider.model, "chunk": f"{chunk_idx+1}/{total_chunks}", "source_url": source_url}

    for attempt in range(provider.max_retries_on_429 + 1):
        try:
            raw_response = await _call_provider(provider, system_prompt, chunk)
            parsed = json.loads(raw_response)
            validated = schema.model_validate(parsed)
            trace["outcome"] = "success"
            return {"result": validated, "trace": trace}

        except RateLimitError as e:
            if _is_quota_exhausted(e):
                # A used-up quota/balance won't recover within backoff range.
                trace["outcome"] = "quota_exhausted"
                _disable(provider, "quota or balance exhausted")
                return {"result": None, "trace": trace}
            if attempt >= provider.max_retries_on_429:
                trace["outcome"] = f"429_exhausted_after_{attempt+1}_attempts"
                logger.warning("429 exhausted on %s for %s", provider.model, source_url)
                return {"result": None, "trace": trace}
            delay = _backoff_delay(attempt, provider.base_backoff_seconds, provider.max_backoff_seconds)
            suggested = _suggested_wait(e)
            if suggested is not None:
                # Wait as long as the provider asked (plus a little), capped.
                delay = min(max(delay, suggested + 0.5), provider.max_backoff_seconds)
            logger.info("429 from %s (attempt %d), backing off %.2fs", provider.model, attempt + 1, delay)
            await asyncio.sleep(delay)
            continue

        except ContextWindowExceededError:
            # Re-chunk this specific piece smaller and retry once, in-place.
            smaller_budget = max(500, provider.max_context_tokens // 2)
            if count_tokens(chunk) <= smaller_budget:
                # Already small and still too large; give up on this provider.
                trace["outcome"] = "413_unresolvable_at_min_chunk_size"
                return {"result": None, "trace": trace}
            sub_plan = chunk_text(chunk, max_tokens_per_chunk=smaller_budget)
            trace["outcome"] = f"413_rechunked_into_{len(sub_plan.chunks)}"
            # Retry with the first, smaller piece only (same reasoning as the
            # chunk selection above). The nested trace keeps the 413 visible
            # in the logs.
            sub_outcome = await _extract_from_chunk_with_retry(
                provider=provider, chunk=sub_plan.chunks[0], chunk_idx=chunk_idx,
                total_chunks=total_chunks, system_prompt=system_prompt,
                schema=schema, source_url=source_url,
            )
            trace["sub_attempt"] = sub_outcome["trace"]
            return {"result": sub_outcome["result"], "trace": trace}

        except (json.JSONDecodeError, ValidationError) as e:
            # Invalid JSON or a schema mismatch: count it as a failure and move on
            # to the next provider rather than trying to repair the output.
            trace["outcome"] = f"invalid_response:{type(e).__name__}"
            logger.warning("Schema validation failed for %s on %s: %s", source_url, provider.model, e)
            return {"result": None, "trace": trace}

        except (APIError, Timeout, ServiceUnavailableError) as e:
            if attempt >= provider.max_retries_on_429:
                trace["outcome"] = f"provider_error_exhausted:{type(e).__name__}"
                return {"result": None, "trace": trace}
            delay = _backoff_delay(attempt, provider.base_backoff_seconds, provider.max_backoff_seconds)
            await asyncio.sleep(delay)
            continue

        except Exception as e:  # noqa: BLE001
            # Anything else (rejected key, retired model, bad request) won't be
            # fixed by retrying this provider, but the next one may work.
            trace["outcome"] = f"provider_failed:{type(e).__name__}"
            logger.warning("Provider %s failed for %s: %s: %s",
                           provider.model, source_url, type(e).__name__, str(e)[:200])
            # Auth failures, missing balance and unknown models won't fix
            # themselves within this run, so stop calling this provider.
            name = type(e).__name__
            if name in ("AuthenticationError", "PermissionDeniedError", "NotFoundError")                     or _is_quota_exhausted(e):
                _disable(provider, name if not _is_quota_exhausted(e) else "quota or balance exhausted")
            return {"result": None, "trace": trace}

    trace["outcome"] = "unknown_exit"
    return {"result": None, "trace": trace}
