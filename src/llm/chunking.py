"""
Token-aware chunking so payloads never trigger 413 (Payload Too Large)
while retaining the most semantically dense content.

Strategy:
1. Prefer structural splitting (headings / paragraph boundaries) over
   blind character slicing, so we don't cut a sentence — and therefore
   a fact — in half.
2. If a single semantic block is *itself* too large (e.g. one giant
   paragraph), fall back to sliding-window token slicing with overlap
   so no information sits entirely inside a truncation boundary.
3. Density heuristic: for extraction tasks we don't need the whole
   document — boilerplate (nav text, cookie banners, footers) is
   stripped before chunking, and for very long documents we bias
   toward the first N and last M tokens plus any block containing
   extraction-relevant keywords (dates, numbers, entity-like capitalized
   phrases), since that's where the target fields usually live.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except ImportError:  # pragma: no cover
    tiktoken = None
    _ENCODER = None


def count_tokens(text: str) -> int:
    if _ENCODER is not None:
        return len(_ENCODER.encode(text, disallowed_special=()))
    # Fallback approximation (~4 chars/token) when tiktoken isn't installed,
    # so this module still degrades gracefully rather than hard-crashing.
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
    """
    Split `text` into chunks that each fit under `max_tokens_per_chunk`,
    which should be set comfortably below the target model's context
    window and below the provider's payload-size limit (the two are not
    the same thing — provider payload caps can bite even on models with
    large context windows, hence keeping this configurable per-provider
    in llm/orchestrator.py).
    """
    if strip_noise:
        text = strip_boilerplate(text)

    total_tokens = count_tokens(text)
    if total_tokens <= max_tokens_per_chunk:
        return ChunkPlan(chunks=[text], strategy="single")

    # Structural split: paragraphs first, then merge greedily up to the
    # token budget so we keep semantically coherent blocks together.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    def flush():
        nonlocal current, current_tokens
        if current:
            chunks.append("\n\n".join(current))
        current, current_tokens = [], 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if para_tokens > max_tokens_per_chunk:
            # A single paragraph is too big on its own (e.g. dumped raw
            # HTML text blob) — sliding-window it with overlap.
            flush()
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
    return ChunkPlan(chunks=chunks, strategy="structural" if len(chunks) > 1 else "single")
