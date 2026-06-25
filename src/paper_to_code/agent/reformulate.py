"""Query planning (direction + concept) and cross-side query reformulation.

This is the LLM "bridge" between the two collections: to search the code from a paper
concept (or vice versa) we rewrite the query for the target side, then embed it with the
target side's model. We never compare a paper vector to a code vector.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from paper_to_code.agent.prompts import (
    SYSTEM_PLAN,
    SYSTEM_REFORMULATE_CODE,
    SYSTEM_REFORMULATE_PAPER,
    format_code_evidence,
    format_paper_evidence,
)
from paper_to_code.config import Settings
from paper_to_code.llm import structured_model
from paper_to_code.models import Direction, RetrievedChunk


class QueryPlan(BaseModel):
    direction: Direction = Field(..., description="Which way the question is asked.")
    concept: str = Field(..., description="Short phrase naming the core concept or symbol.")


class TargetQuery(BaseModel):
    query: str = Field(..., description="A concise search query for the target side.")


def plan_query(question: str, settings: Settings | None = None) -> QueryPlan:
    """Detect direction and extract the core concept from the question."""
    model = structured_model(QueryPlan, settings)
    out = model.invoke([("system", SYSTEM_PLAN), ("human", question)])
    parsed = out.get("parsed")
    if parsed is None:
        return QueryPlan(direction=Direction.PAPER_TO_CODE, concept=question)
    return parsed


def reformulate(
    question: str,
    concept: str,
    source_hits: list[RetrievedChunk],
    target: Literal["code", "paper"],
    settings: Settings | None = None,
) -> str:
    """Rewrite the question into a query for the ``target`` side, using source-side evidence."""
    if target == "code":
        system = SYSTEM_REFORMULATE_CODE
        evidence = format_paper_evidence(source_hits)
    else:
        system = SYSTEM_REFORMULATE_PAPER
        evidence = format_code_evidence(source_hits)

    user = (
        f"Concept: {concept}\n"
        f"Original question: {question}\n"
        f"Source-side evidence (for context):\n{evidence}\n\n"
        f"Produce the {target}-search query."
    )
    model = structured_model(TargetQuery, settings)
    out = model.invoke([("system", system), ("human", user)])
    parsed = out.get("parsed")
    if parsed is None or not parsed.query.strip():
        return concept or question
    return parsed.query
