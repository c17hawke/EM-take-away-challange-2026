"""
Response formatter and audit trail generator — Tasks 8 & 9.

Formats the structured agent response into the mandatory three-part output:
  Reasoning / Thinking → Answer → Citations

Also produces a machine-readable audit trail JSON object.
"""
from __future__ import annotations

import datetime
import json
from typing import Any, Dict, List, Optional


def format_response(
    reasoning: str,
    answer: str,
    citations: List[Dict[str, Any]],
) -> str:
    """
    Render the mandatory Reasoning / Thinking → Answer → Citations format.
    """
    lines: list[str] = []

    lines.append("## Reasoning / Thinking")
    lines.append(reasoning.strip())
    lines.append("")

    lines.append("## Answer")
    lines.append(answer.strip())
    lines.append("")

    lines.append("## Citations")
    if citations:
        for i, cit in enumerate(citations, start=1):
            doc = cit.get("document", "Unknown")
            identifier = cit.get("article_or_section", "")
            page = cit.get("page_number")
            snippet = cit.get("snippet", "")
            page_info = f", p. {page}" if page else ""
            lines.append(f"{i}. **{doc}** — {identifier}{page_info}")
            if snippet:
                lines.append(f"   > {snippet[:200]}")
    else:
        lines.append("_No citations available._")

    return "\n".join(lines)


def build_audit_trail(
    query: str,
    sub_queries: List[str],
    retrieved_chunks: List[Dict[str, Any]],
    cited_sources: List[str],
    grounding_score: float,
    hallucination_risk: bool,
    context_tokens_used: int,
    guardrail_triggered: bool,
    llm_used: str,
) -> Dict[str, Any]:
    """
    Build a machine-readable audit trail object (Task 9).
    """
    return {
        "query": query,
        "sub_queries": sub_queries,
        "retrieved_chunks": [
            {
                "text": ch.get("text", "")[:300],
                "source": ch.get("source", ""),
                "article": ch.get("article_id") or ch.get("section", ""),
                "page": ch.get("page_number"),
                "score": ch.get("score", 0.0),
            }
            for ch in retrieved_chunks
        ],
        "cited_sources": cited_sources,
        "grounding_score": round(grounding_score, 4),
        "hallucination_risk": hallucination_risk,
        "context_tokens_used": context_tokens_used,
        "guardrail_triggered": guardrail_triggered,
        "llm_used": llm_used,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
