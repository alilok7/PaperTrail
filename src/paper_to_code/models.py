"""Core domain models.

The headline output is :class:`Verdict` — a validated, structured judgement of how a
paper concept relates to its implementation. Field descriptions are intentionally rich
because they are surfaced to the LLM as the JSON schema for structured output.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class VerdictType(str, Enum):
    """The relationship between a paper concept and the code."""

    MATCH = "MATCH"
    MATCH_SIMPLIFIED = "MATCH_SIMPLIFIED"
    EXTRA_IN_CODE = "EXTRA_IN_CODE"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    UNCERTAIN = "UNCERTAIN"


class Direction(str, Enum):
    """Which way the question was asked."""

    PAPER_TO_CODE = "paper_to_code"
    CODE_TO_PAPER = "code_to_paper"


SourceType = Literal["paper", "code"]


class PaperCitation(BaseModel):
    """A pointer into the paper supporting the verdict."""

    section: str | None = Field(
        None, description="Section number or title, e.g. '3.2' or 'Scaled Dot-Product Attention'."
    )
    page: int | None = Field(None, description="1-based page number in the PDF.")
    equations: list[str] = Field(
        default_factory=list, description="Equation labels referenced, e.g. ['Eq. 1']."
    )
    figures: list[str] = Field(
        default_factory=list, description="Figure or table labels referenced, e.g. ['Figure 2']."
    )
    quote: str = Field("", description="Short verbatim excerpt from the paper that supports the verdict.")


class CodeCitation(BaseModel):
    """A pointer into the code supporting the verdict."""

    file: str = Field(..., description="Repository-relative file path.")
    symbol: str | None = Field(None, description="Function or class name.")
    language: str | None = Field(None, description="Programming language of the file.")
    start_line: int | None = Field(None, description="1-based start line of the symbol.")
    end_line: int | None = Field(None, description="1-based end line of the symbol.")
    quote: str = Field("", description="Short verbatim code excerpt that supports the verdict.")


class Verdict(BaseModel):
    """The reconciliation verdict — PaperTrail's headline output."""

    verdict: VerdictType = Field(
        ..., description="The relationship between the paper concept and the code."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence in the verdict from 0 to 1. Use a low value when evidence is thin; "
        "prefer UNCERTAIN over guessing.",
    )
    direction: Direction = Field(..., description="Whether the question went paper->code or code->paper.")
    explanation: str = Field(
        ..., description="Concise, evidence-grounded justification that references the paper and the code."
    )
    paper_citation: PaperCitation | None = Field(
        None, description="Where in the paper the concept is described. Null if no relevant passage was found."
    )
    code_citation: CodeCitation | None = Field(
        None, description="Where in the code the concept is implemented. Null if it is not implemented."
    )


class Chunk(BaseModel):
    """A unit of text stored in a vector collection.

    Metadata values are restricted to Chroma-compatible scalars; ingestion joins any
    lists (e.g. section headings) into strings before they land here.
    """

    id: str
    text: str
    source: SourceType
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval, with its fused relevance score."""

    chunk: Chunk
    score: float = Field(..., description="Fused relevance score; higher is more relevant.")
