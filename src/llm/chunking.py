"""Split text into pieces that fit a model's token budget.

Paragraphs are kept together where possible, so a fact isn't cut in half.
A paragraph that is too big on its own is split with a sliding window that
overlaps neighbouring pieces. Page boilerplate (navigation, cookie banners,
footers) is removed first.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None

_ENCODER = None
if tiktoken is not None:
    try:
        # The first call downloads the encoding file. If that fails (offline,
        # blocked, TLS interception) fall back to the approximation below
        # rather than crashing every module that imports this one.
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:  # noqa: BLE001
        _ENCODER = None


def count_tokens(text: str) -> int:
    if _ENCODER is not None:
        return len(_ENCODER.encode(text, disallowed_special=()))
    # Rough estimate (about 4 characters per token) when tiktoken isn't available.
    return max(1, len(text) // 4)


BOILERPLATE_PATTERNS = [
    re.compile(r"(?i)^\s*(cookie|privacy) (policy|notice|settings).*$", re.MULTILINE),
    re.compile(r"(?i)^\s*(subscribe|sign up) (to|for) our newsletter.*$", re.MULTILINE),
    re.compile(r"(?i)^\s*(all rights reserved|©\s*\d{4}).*$", re.MULTILINE),
    re.compile(r"(?i)^\s*(share this|follow us on).*$", re.MULTILINE),
]


def strip_boilerplate(text: str) -> str:
    for pattern in BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)
    # Collapse runs of blank lines left behind.
    return re.sub(r"\n{3,}", "\n\n", text).strip()


@dataclass
class ChunkPlan:
    chunks: List[str]
    strategy: str  # "single" | "structural" | "sliding_window"


def chunk_text(
    text: str,
    *,
    max_tokens_per_chunk: int = 6000,
    overlap_tokens: int = 200,
    strip_noise: bool = True,
) -> ChunkPlan:
    """Split `text` into chunks of at most `max_tokens_per_chunk` tokens.

    Keep the budget below both the model's context window and the provider's
    request-size limit; they differ, which is why the orchestrator sets the
    budget per provider.
    """
    if strip_noise:
        text = strip_boilerplate(text)

    total_tokens = count_tokens(text)
    if total_tokens <= max_tokens_per_chunk:
        return ChunkPlan(chunks=[text], strategy="single")

    # Split on paragraphs, then merge neighbours until the budget is reached.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0
    used_sliding_window = False

    def flush():
        nonlocal current, current_tokens
        if current:
            chunks.append("\n\n".join(current))
        current, current_tokens = [], 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if para_tokens > max_tokens_per_chunk:
            # A single paragraph is too big on its own (e.g. dumped raw
            # HTML text blob): sliding-window it with overlap.
            flush()
            used_sliding_window = True
            words = para.split()
            approx_words_per_chunk = max(50, int(max_tokens_per_chunk / 1.3))
            approx_overlap_words = max(10, int(overlap_tokens / 1.3))
            i = 0
            while i < len(words):
                window = words[i: i + approx_words_per_chunk]
                chunks.append(" ".join(window))
                if i + approx_words_per_chunk >= len(words):
                    break
                i += approx_words_per_chunk - approx_overlap_words
            continue

        if current_tokens + para_tokens > max_tokens_per_chunk:
            flush()

        current.append(para)
        current_tokens += para_tokens

    flush()
    if used_sliding_window:
        strategy = "sliding_window"
    else:
        strategy = "structural" if len(chunks) > 1 else "single"
    return ChunkPlan(chunks=chunks, strategy=strategy)
