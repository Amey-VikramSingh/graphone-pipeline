"""Match company names written different ways ("OpenAI, Inc.", "Open AI",
"openai") to one canonical name from a seed list.

Steps, cheapest first:
  1. Normalise: lowercase, drop legal suffixes and punctuation.
  2. Exact match on the normalised name or a known alias.
  3. Fuzzy match (token-sort ratio) above a conservative threshold.
  4. Otherwise log it as "unresolved_new" instead of forcing the closest
     match. A wrong merge is much harder to spot later than a missed one.

Every decision goes into the entity mapping log with its method and score.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from rapidfuzz import fuzz as _rf_fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - fall back to difflib without rapidfuzz
    import difflib
    _HAS_RAPIDFUZZ = False

FUZZY_MATCH_THRESHOLD = 88.0  # 0-100 scale; conservative to avoid false merges

# Seed names/aliases that are also everyday words. They still resolve when a
# scraper hands over a company name, but find_mentions() ignores them in
# free text, where "Scale" or "Together" usually aren't the company.
AMBIGUOUS_MENTIONS = {
    "Scale", "Writer", "Notion", "Gamma", "Together", "Lambda", "Glean", "Cursor",
    "Modular", "Contextual", "Inflection", "Adept", "Sierra", "Harvey", "Runway",
    "Luma", "Pika", "Jasper", "Imbue", "Perplexity", "Mistral", "Pinecone", "Snorkel",
}

_LEGAL_SUFFIXES = re.compile(
    r"(?i)\b(inc\.?|incorporated|corp\.?|corporation|co\.?|llc\.?|ltd\.?|"
    r"limited|plc\.?|gmbh|s\.?a\.?|pte\.?\s*ltd\.?)\s*$"
)
_PUNCT = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    """Lowercase, strip legal suffixes and punctuation, collapse whitespace."""
    if not name:
        return ""
    n = name.strip()
    n = _LEGAL_SUFFIXES.sub("", n)
    n = _PUNCT.sub(" ", n)
    n = _WHITESPACE.sub(" ", n).strip().lower()
    return n


def _similarity(a: str, b: str) -> float:
    """Token-sort similarity on a 0-100 scale, handling word-order variance
    ('AI Labs OpenAI' vs 'OpenAI AI Labs')."""
    if _HAS_RAPIDFUZZ:
        return _rf_fuzz.token_sort_ratio(a, b)
    # difflib fallback: sort tokens then compare with SequenceMatcher,
    # scaled to 0-100 to match rapidfuzz's convention.
    a_sorted = " ".join(sorted(a.split()))
    b_sorted = " ".join(sorted(b.split()))
    return difflib.SequenceMatcher(None, a_sorted, b_sorted).ratio() * 100


@dataclass
class ResolutionResult:
    raw_name: str
    canonical_name: Optional[str]
    match_score: float          # 0.0-1.0
    match_method: str           # "exact" | "fuzzy" | "alias_table" | "unresolved_new"
    is_new_entity: bool = False


@dataclass
class EntityResolver:
    """seed_entities maps each canonical name to its known aliases (about 50 AI companies)."""
    seed_entities: Dict[str, List[str]] = field(default_factory=dict)
    threshold: float = FUZZY_MATCH_THRESHOLD

    _normalized_index: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _canonical_norm_list: List[tuple] = field(default_factory=list, init=False, repr=False)
    _dynamic_aliases: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _mention_patterns: List[tuple] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        self._rebuild_index()

    def _rebuild_index(self):
        self._normalized_index = {}
        self._canonical_norm_list = []
        self._mention_patterns = []
        for canonical, aliases in self.seed_entities.items():
            canon_norm = normalize_name(canonical)
            self._normalized_index[canon_norm] = canonical
            # Fuzzy candidates include aliases, not just canonical names
            # ("Eleven Labz" should land near the alias "Eleven Labs").
            self._canonical_norm_list.append((canon_norm, canonical))
            for alias in aliases:
                alias_norm = normalize_name(alias)
                self._normalized_index[alias_norm] = canonical
                if alias_norm and alias_norm != canon_norm:
                    self._canonical_norm_list.append((alias_norm, canonical))
            for surface in [canonical, *aliases]:
                if surface in AMBIGUOUS_MENTIONS:
                    continue
                self._mention_patterns.append(
                    (re.compile(r"(?<!\w)" + re.escape(surface) + r"(?!\w)"), canonical)
                )

    def find_mentions(self, text: str) -> List[str]:
        """Canonical names of known companies mentioned in free text, such as a news
        article. Matches whole words, case-sensitively, and skips names that are also
        everyday words ("Scale", "Writer", "Notion") to avoid false hits.
        """
        if not text:
            return []
        found = {canonical for pattern, canonical in self._mention_patterns if pattern.search(text)}
        return sorted(found)

    def add_canonical_entity(self, canonical_name: str, aliases: Optional[List[str]] = None) -> None:
        """Register a new canonical company at runtime, e.g. one promoted from unresolved_new."""
        self.seed_entities.setdefault(canonical_name, [])
        if aliases:
            self.seed_entities[canonical_name].extend(aliases)
        self._rebuild_index()

    def resolve(self, raw_name: str) -> ResolutionResult:
        if not raw_name or not raw_name.strip():
            return ResolutionResult(raw_name=raw_name, canonical_name=None, match_score=0.0,
                                     match_method="unresolved_new", is_new_entity=True)

        norm = normalize_name(raw_name)

        # 1. Exact match on normalized form or known alias.
        if norm in self._normalized_index:
            return ResolutionResult(
                raw_name=raw_name,
                canonical_name=self._normalized_index[norm],
                match_score=1.0,
                match_method="exact" if self._normalized_index[norm].lower().strip() == raw_name.lower().strip()
                             else "alias_table",
            )

        # 2. Fuzzy match against all known canonical names (+ registered aliases).
        best_score = 0.0
        best_canonical = None
        for canon_norm, canonical in self._canonical_norm_list:
            score = _similarity(norm, canon_norm)
            if score > best_score:
                best_score = score
                best_canonical = canonical

        if best_canonical is not None and best_score >= self.threshold:
            return ResolutionResult(
                raw_name=raw_name,
                canonical_name=best_canonical,
                match_score=round(best_score / 100.0, 4),
                match_method="fuzzy",
            )

        # 3. Nothing cleared the bar -- do not guess.
        return ResolutionResult(
            raw_name=raw_name,
            canonical_name=None,
            match_score=round(best_score / 100.0, 4) if best_canonical else 0.0,
            match_method="unresolved_new",
            is_new_entity=True,
        )


# ---------------------------------------------------------------------------
# Seed data: well-known AI companies and their common aliases
# ---------------------------------------------------------------------------

SEED_STARTUPS: Dict[str, List[str]] = {
    "OpenAI": ["OpenAI, Inc.", "Open AI", "OpenAI Inc"],
    "Anthropic": ["Anthropic PBC", "Anthropic, Inc."],
    "Google DeepMind": ["DeepMind", "DeepMind Technologies", "Google Deepmind"],
    "Mistral AI": ["Mistral", "Mistral AI SAS"],
    "Cohere": ["Cohere Inc", "Cohere AI"],
    "Stability AI": ["Stability AI Ltd", "StabilityAI"],
    "Hugging Face": ["HuggingFace", "Hugging Face Inc"],
    "Scale AI": ["Scale", "Scale AI Inc"],
    "Perplexity AI": ["Perplexity", "Perplexity AI Inc"],
    "Runway": ["Runway ML", "Runway AI Inc"],
    "Character.AI": ["Character AI", "CharacterAI"],
    "Inflection AI": ["Inflection", "Inflection AI Inc"],
    "Adept AI": ["Adept", "Adept AI Labs"],
    "xAI": ["X.AI", "xAI Corp"],
    "Together AI": ["Together", "Together Computer"],
    "Databricks": ["Databricks Inc"],
    "Snorkel AI": ["Snorkel", "Snorkel AI Inc"],
    "Weights & Biases": ["Weights and Biases", "WandB"],
    "LangChain": ["LangChain Inc"],
    "Pinecone": ["Pinecone Systems Inc"],
    "Weaviate": ["Weaviate B.V."],
    "Glean": ["Glean Technologies Inc"],
    "Harvey AI": ["Harvey", "Harvey AI Inc"],
    "Sierra": ["Sierra AI", "Sierra Platform Inc"],
    "Cursor (Anysphere)": ["Anysphere", "Cursor", "Anysphere Inc"],
    "Replit": ["Replit Inc"],
    "Vercel": ["Vercel Inc"],
    "ElevenLabs": ["Eleven Labs", "ElevenLabs Inc"],
    "Synthesia": ["Synthesia Ltd"],
    "Suno": ["Suno AI", "Suno Inc"],
    "Midjourney": ["Midjourney Inc"],
    "Ideogram": ["Ideogram AI"],
    "Luma AI": ["Luma", "Luma Labs"],
    "Pika": ["Pika Labs", "Pika AI"],
    "Groq": ["Groq Inc"],
    "Cerebras": ["Cerebras Systems"],
    "SambaNova Systems": ["SambaNova"],
    "Modular": ["Modular AI", "Modular Inc"],
    "Fireworks AI": ["Fireworks", "Fireworks AI Inc"],
    "Baseten": ["Baseten Labs"],
    "Contextual AI": ["Contextual", "Contextual AI Inc"],
    "Imbue": ["Imbue AI", "Generally Intelligent"],
    "Reka AI": ["Reka", "Reka AI Inc"],
    "AI21 Labs": ["AI21", "AI21 Labs Ltd"],
    "Writer": ["Writer Inc", "Writer AI"],
    "Jasper": ["Jasper AI", "Jasper.ai"],
    "Copy.ai": ["CopyAI", "Copy AI"],
    "Gamma": ["Gamma App", "Gamma Software"],
    "Notion AI": ["Notion Labs", "Notion"],
    "Vectara": ["Vectara Inc"],
    "Lambda Labs": ["Lambda", "Lambda Inc"],
}
