"""Reconciliation: turn paper + code evidence into a validated Verdict."""

from __future__ import annotations

from paper_to_code.agent.prompts import SYSTEM_RECONCILE, reconcile_user_prompt
from paper_to_code.config import Settings
from paper_to_code.llm import structured_model
from paper_to_code.models import Direction, RetrievedChunk, Verdict, VerdictType


def reconcile(
    question: str,
    direction: Direction,
    paper_hits: list[RetrievedChunk],
    code_hits: list[RetrievedChunk],
    settings: Settings | None = None,
) -> Verdict:
    """Ask the LLM for a structured verdict, with a safe UNCERTAIN fallback."""
    model = structured_model(Verdict, settings)
    out = model.invoke(
        [
            ("system", SYSTEM_RECONCILE),
            ("human", reconcile_user_prompt(question, direction, paper_hits, code_hits)),
        ]
    )
    parsed = out.get("parsed")
    if parsed is not None:
        # The model occasionally omits direction; pin it to what the loop decided.
        if parsed.direction != direction:
            parsed.direction = direction
        return parsed
    return Verdict(
        verdict=VerdictType.UNCERTAIN,
        confidence=0.0,
        direction=direction,
        explanation="Could not produce a structured verdict from the retrieved evidence.",
        paper_citation=None,
        code_citation=None,
    )
