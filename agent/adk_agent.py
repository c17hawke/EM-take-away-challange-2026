"""
Google ADK Agent — Task 7.

Orchestrates the full agentic RAG loop:
  1. Decompose query into sub-queries (context engineering)
  2. Retrieve chunks via MCP server (with before_tool_callback)
  3. Re-rank + budget-trim context (context engineering)
  4. Build structured prompt (context engineering)
  5. Invoke LLM (with before_model_callback)
  6. Parse structured response (with after_model_callback)
  7. Emit formatted output + audit trail

Google ADK is used as the primary agent framework. The MCP retrieval
server is registered as a tool.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import httpx
import tiktoken
from openai import OpenAI

import config
from agent.callbacks import (
    after_model_callback,
    before_model_callback,
    before_tool_callback,
)
from agent.output_formatter import build_audit_trail, format_response
from mcp_server.context_engineering import (
    apply_context_budget,
    count_tokens,
    decompose_query,
    summarise_history,
)

logger = logging.getLogger(__name__)

_OAI = OpenAI(api_key=config.OPENAI_API_KEY)

MCP_BASE_URL = (
    f"http://{config.MCP_SERVER_HOST if config.MCP_SERVER_HOST != '0.0.0.0' else 'mcp-server'}"
    f":{config.MCP_SERVER_PORT}"
)

# ─── System prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a regulatory AI compliance assistant specialised in:
- The NIST AI Risk Management Framework (NIST AI RMF / NIST.AI.100-1)
- The EU AI Act (Regulation (EU) 2024/1689)

Your role:
1. Answer questions grounded ONLY in the provided retrieved context.
2. Never fabricate article numbers, section names, or obligations.
3. Always cite the exact source document, article/section, and page where available.
4. If the context does not contain sufficient information to answer, say so clearly.
5. Produce your response in EXACTLY this JSON structure:

{
  "reasoning": "<detailed account of how you interpreted the query, which sub-queries were used, what context was retrieved, and how the answer was formed>",
  "answer": "<final grounded response>",
  "citations": [
    {
      "document": "<NIST AI RMF | EU AI Act>",
      "article_or_section": "<Article N — Title | GOVERN-1.1 etc.>",
      "page_number": <int or null>,
      "snippet": "<direct quote or close paraphrase from the source>"
    }
  ]
}

Do NOT include anything outside this JSON object."""


# ─── MCP tool call ───────────────────────────────────────────────────────────


def _call_mcp_retrieve(
    query: str,
    sources: List[str],
    top_k: int,
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Call the MCP retrieval server with guardrails."""
    tool_args = {"query": query, "sources": sources, "top_k": top_k}

    # before_tool_callback
    modified = before_tool_callback("retrieve_regulatory_context", tool_args, context)
    if modified is not None:
        tool_args = modified

    try:
        resp = httpx.post(
            f"{MCP_BASE_URL}/retrieve",
            json=tool_args,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        chunks = data.get("chunks", [])
        logger.info("MCP returned %d chunks for query=%r", len(chunks), query)
        return chunks
    except Exception as exc:
        logger.error("MCP retrieval failed: %s", exc)
        return []


# ─── Context assembly ─────────────────────────────────────────────────────────


def _build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a structured context block for the prompt."""
    if not chunks:
        return "No relevant passages retrieved."

    parts: list[str] = []
    for i, ch in enumerate(chunks, start=1):
        source = ch.get("source", "Unknown")
        article = ch.get("article_id") or ch.get("article_title") or ""
        section = ch.get("section") or ""
        page = ch.get("page_number")
        score = ch.get("rerank_score") or ch.get("score", 0.0)

        identifier = article or section
        page_str = f", p.{page}" if page else ""
        header = f"[{i}] {source} — {identifier}{page_str} (score: {score:.3f})"
        parts.append(f"{header}\n{ch.get('text', '')}")

    return "\n\n---\n\n".join(parts)


# ─── Core agent ──────────────────────────────────────────────────────────────


class RegulatoryAgent:
    """
    Google ADK-style agent for regulatory Q&A.

    In Google ADK, agent logic is expressed through:
    - Runner.run() / Runner.run_async() for invocation
    - Before/after model callbacks registered on the model
    - Before/after tool callbacks registered on the tool
    - Structured output via the model's JSON mode

    This implementation follows the ADK execution model:
    agent → decompose → [tool calls] → rerank → prompt → LLM → parse → output
    """

    def __init__(self, conversation_history: Optional[List[Dict[str, str]]] = None):
        self.history: List[Dict[str, str]] = conversation_history or []

    def run(self, user_query: str) -> Dict[str, Any]:
        """
        Execute the full agentic RAG pipeline for *user_query*.

        Returns a dict with keys: formatted_response, audit_trail, raw_parsed.
        """
        t0 = time.perf_counter()
        context: Dict[str, Any] = {"guardrail_triggered": False}

        # ── Step 1: Query decomposition ──────────────────────────────────────
        sub_queries = decompose_query(user_query)
        logger.info("Query decomposed into %d sub-queries: %s", len(sub_queries), sub_queries)

        # ── Step 2: Retrieve for each sub-query ──────────────────────────────
        all_chunks: list[dict] = []
        for sq in sub_queries:
            chunks = _call_mcp_retrieve(
                query=sq,
                sources=["nist", "eu_ai_act"],
                top_k=config.MCP_TOP_K,
                context=context,
            )
            all_chunks.extend(chunks)

        # Deduplicate by text content
        seen_texts: set[str] = set()
        deduped: list[dict] = []
        for ch in all_chunks:
            t = ch.get("text", "")
            if t not in seen_texts:
                seen_texts.add(t)
                deduped.append(ch)

        # Sort by score and apply context budget
        deduped.sort(key=lambda x: x.get("rerank_score") or x.get("score", 0.0), reverse=True)
        deduped = apply_context_budget(deduped)

        # ── Step 3: Build prompt ─────────────────────────────────────────────
        context_block = _build_context_block(deduped)
        context_tokens = count_tokens(context_block)

        # Prune conversation history to prevent overflow
        self.history = summarise_history(self.history)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": f"## Retrieved Context\n\n{context_block}",
            },
        ]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_query})

        # ── Step 4: before_model_callback ────────────────────────────────────
        modified_messages = before_model_callback(messages, context)
        if modified_messages is not None:
            messages = modified_messages

        # ── Step 5: LLM call ─────────────────────────────────────────────────
        logger.info("Calling LLM (%s) with %d messages…", config.OPENAI_MODEL, len(messages))
        resp = _OAI.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw_content = resp.choices[0].message.content or "{}"

        # ── Step 6: Parse structured response ────────────────────────────────
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            logger.error("LLM response was not valid JSON: %s", raw_content[:200])
            parsed = {
                "reasoning": "Failed to parse LLM response.",
                "answer": raw_content,
                "citations": [],
            }

        reasoning = parsed.get("reasoning", "")
        answer = parsed.get("answer", "")
        citations: list[dict] = parsed.get("citations", [])

        # ── Step 7: after_model_callback ─────────────────────────────────────
        cited_source_strings = [
            f"{c.get('document', '')} {c.get('article_or_section', '')}"
            for c in citations
        ]
        guardrail_result = after_model_callback(
            response_text=answer + reasoning,
            context=context,
            cited_sources=cited_source_strings,
        )

        # ── Step 8: Grounding score ───────────────────────────────────────────
        # Simple heuristic: average of rerank_score of top cited chunks
        scores = [
            ch.get("rerank_score") or ch.get("score", 0.0) for ch in deduped[:5]
        ]
        grounding_score = sum(scores) / len(scores) if scores else 0.0

        # ── Step 9: Format output ────────────────────────────────────────────
        formatted = format_response(reasoning, answer, citations)

        audit_trail = build_audit_trail(
            query=user_query,
            sub_queries=sub_queries,
            retrieved_chunks=[
                {
                    "text": ch.get("text", ""),
                    "source": ch.get("source", ""),
                    "article_id": ch.get("article_id"),
                    "section": ch.get("section"),
                    "page_number": ch.get("page_number"),
                    "score": ch.get("rerank_score") or ch.get("score", 0.0),
                }
                for ch in deduped
            ],
            cited_sources=cited_source_strings,
            grounding_score=grounding_score,
            hallucination_risk=guardrail_result["hallucination_risk"],
            context_tokens_used=context_tokens,
            guardrail_triggered=context.get("guardrail_triggered", False),
            llm_used=config.OPENAI_MODEL,
        )

        # ── Persist conversation history ──────────────────────────────────────
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": answer})

        latency = (time.perf_counter() - t0) * 1000
        logger.info("Agent run complete in %.1f ms", latency)

        return {
            "formatted_response": formatted,
            "audit_trail": audit_trail,
            "raw_parsed": parsed,
            "latency_ms": round(latency, 2),
        }
