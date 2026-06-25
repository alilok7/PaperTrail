"""The agent loop: a deterministic, explainable pipeline (no free-form ReAct).

    decompose (direction + concept)
      -> retrieve source side (hybrid)
      -> [single-hop reference following on the paper side]
      -> reformulate query for the target side
      -> retrieve target side (hybrid)
      -> reconcile -> structured Verdict

Running at temperature 0, the same inputs give the same verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from paper_to_code.agent.reconcile import reconcile
from paper_to_code.agent.reformulate import QueryPlan, plan_query, reformulate
from paper_to_code.config import Settings, get_settings
from paper_to_code.models import Direction, RetrievedChunk, Verdict
from paper_to_code.retrieval.reference import follow_references


class _Retriever(Protocol):
    def search(self, query: str, k: int) -> list[RetrievedChunk]: ...


@dataclass
class AgentResult:
    verdict: Verdict
    direction: Direction
    concept: str
    reformulated_query: str
    paper_evidence: list[RetrievedChunk]
    code_evidence: list[RetrievedChunk]


def run_agent(
    question: str,
    *,
    paper_retriever: _Retriever,
    code_retriever: _Retriever,
    settings: Settings | None = None,
    direction: Direction | None = None,
    k: int = 8,
    follow_refs: bool = True,
) -> AgentResult:
    settings = settings or get_settings()
    plan = QueryPlan(direction=direction, concept=question) if direction else plan_query(question, settings)

    if plan.direction == Direction.PAPER_TO_CODE:
        paper_hits = paper_retriever.search(plan.concept or question, k)
        if follow_refs:
            paper_hits = follow_references(paper_hits, paper_retriever)
        reformulated = reformulate(question, plan.concept, paper_hits, "code", settings)
        code_hits = code_retriever.search(reformulated, k)
    else:
        code_hits = code_retriever.search(plan.concept or question, k)
        reformulated = reformulate(question, plan.concept, code_hits, "paper", settings)
        paper_hits = paper_retriever.search(reformulated, k)
        if follow_refs:
            paper_hits = follow_references(paper_hits, paper_retriever)

    verdict = reconcile(question, plan.direction, paper_hits, code_hits, settings)
    return AgentResult(
        verdict=verdict,
        direction=plan.direction,
        concept=plan.concept,
        reformulated_query=reformulated,
        paper_evidence=paper_hits,
        code_evidence=code_hits,
    )
