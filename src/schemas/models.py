"""
Canonical output schemas for the ingestion pipeline.

Every scraper/LLM-extractor in this repo must produce records that validate
against one of these models before they are persisted or exported.
Validation failure = the record is dropped and logged, never guessed at.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "1.0"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Shared sub-objects
# ---------------------------------------------------------------------------

class Source(BaseModel):
    name: str = Field(..., description="Name of the source site, e.g. 'arxiv.org'")
    url: str = Field(..., description="Original source URL this record was extracted from")

    @field_validator("url")
    @classmethod
    def url_must_be_http(cls, v: str) -> str:
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError(f"source.url must be a real http(s) URL, got: {v!r}")
        return v


class PricingModel(str, Enum):
    FREE = "FREE"
    FREEMIUM = "FREEMIUM"
    PAID = "PAID"
    ENTERPRISE = "ENTERPRISE"
    UNKNOWN = "UNKNOWN"  # explicit "we don't know" rather than a guessed value


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

class StartupContentData(BaseModel):
    employeeCount: Optional[int] = None


class StartupContent(BaseModel):
    entityName: str
    data: StartupContentData = Field(default_factory=StartupContentData)


class StartupRecord(BaseModel):
    schemaVersion: str = SCHEMA_VERSION
    recordType: str = "STARTUP"
    source: Source
    content: StartupContent
    collectedAt: datetime = Field(default_factory=utc_now)

    @field_validator("recordType")
    @classmethod
    def lock_record_type(cls, v: str) -> str:
        if v != "STARTUP":
            raise ValueError("recordType must be 'STARTUP'")
        return v


# ---------------------------------------------------------------------------
# Product
# ---------------------------------------------------------------------------

class ProductContent(BaseModel):
    startupName: str
    pricingModel: PricingModel = PricingModel.UNKNOWN


class ProductRecord(BaseModel):
    schemaVersion: str = SCHEMA_VERSION
    recordType: str = "PRODUCT"
    source: Source
    content: ProductContent
    collectedAt: datetime = Field(default_factory=utc_now)

    @field_validator("recordType")
    @classmethod
    def lock_record_type(cls, v: str) -> str:
        if v != "PRODUCT":
            raise ValueError("recordType must be 'PRODUCT'")
        return v


# ---------------------------------------------------------------------------
# Research Paper
# ---------------------------------------------------------------------------

class ResearchPaperContent(BaseModel):
    title: str
    authors: List[str] = Field(default_factory=list)
    paper_url: str
    github_url: Optional[str] = None
    github_stars: Optional[int] = None
    published_date: Optional[datetime] = None

    @field_validator("paper_url")
    @classmethod
    def paper_url_valid(cls, v: str) -> str:
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError(f"paper_url must be a real URL, got: {v!r}")
        return v


class ResearchPaperRecord(BaseModel):
    schemaVersion: str = SCHEMA_VERSION
    recordType: str = "RESEARCH_PAPER"
    source: Source
    content: ResearchPaperContent
    collectedAt: datetime = Field(default_factory=utc_now)

    @field_validator("recordType")
    @classmethod
    def lock_record_type(cls, v: str) -> str:
        if v != "RESEARCH_PAPER":
            raise ValueError("recordType must be 'RESEARCH_PAPER'")
        return v


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

class JobContent(BaseModel):
    company: str
    date: datetime
    is_remote: bool = False
    role_family: str = "Unknown"


class JobRecord(BaseModel):
    schemaVersion: str = SCHEMA_VERSION
    recordType: str = "JOB"
    source: Source
    content: JobContent
    collectedAt: datetime = Field(default_factory=utc_now)

    @field_validator("recordType")
    @classmethod
    def lock_record_type(cls, v: str) -> str:
        if v != "JOB":
            raise ValueError("recordType must be 'JOB'")
        return v


# ---------------------------------------------------------------------------
# News (modeled consistently with the Job and ResearchPaper conventions)
# ---------------------------------------------------------------------------

class NewsContent(BaseModel):
    title: str
    published_date: datetime
    full_text: str
    entities_mentioned: List[str] = Field(default_factory=list)


class NewsRecord(BaseModel):
    schemaVersion: str = SCHEMA_VERSION
    recordType: str = "NEWS"
    source: Source
    content: NewsContent
    collectedAt: datetime = Field(default_factory=utc_now)

    @field_validator("recordType")
    @classmethod
    def lock_record_type(cls, v: str) -> str:
        if v != "NEWS":
            raise ValueError("recordType must be 'NEWS'")
        return v


# ---------------------------------------------------------------------------
# Entity mapping log (raw -> canonical), for the "Entity Mapping Log" tab
# ---------------------------------------------------------------------------

class EntityMappingLogEntry(BaseModel):
    raw_name: str
    canonical_name: str
    match_score: float  # 0.0-1.0 confidence from the resolver
    match_method: str   # "exact" | "fuzzy" | "alias_table" | "unresolved_new"
    source_url: str
    resolved_at: datetime = Field(default_factory=utc_now)
