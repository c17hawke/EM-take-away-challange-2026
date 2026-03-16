"""
Context engineering utilities — Task 6.

Implements:
  - Context budget enforcement (token counting + trimming)
  - Relevance re-ranking (cross-encoder scoring via OpenAI)
  - Query decomposition for multi-part / cross-document questions
  - Conversation memory summarisation / pruning
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import tiktoken
from openai import OpenAI

import config

logger = logging.getLogger(__name__)

_ENCODING = tiktoken.encoding_for_model("gpt-4o")
_OAI = OpenAI(api_key=config.OPENAI_API_KEY)


# ─── Token counting ──────────────────────────────────────────────────────────


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


# ─── Relevance re-ranking ────────────────────────────────────────────────────


def rerank_chunks(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Re-rank *chunks* by relevance to *query* using an LLM-based pointwise scorer.

    Strategy: ask the LLM to score each chunk 0–10 for relevance to the query.
    Chunks are then sorted descending by their re-rank score.
    This is a lightweight alternative to a full cross-encoder; it is
    proportional to the PoC scope.
    """
    if not chunks:
        return chunks

    scored: list[tuple[float, dict]] = []
    for chunk in chunks:
        snippet = chunk["text"][:400]  # score on first 400 chars for speed
        prompt = (
            f"Rate the relevance of the following passage to the question on a scale "
            f"of 0 to 10 (integer only, no explanation).\n\n"
            f"Question: {query}\n\nPassage: {snippet}\n\nScore:"
        )
        try:
            resp = _OAI.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3,
                temperature=0,
            )
            score_str = resp.choices[0].message.content.strip()
            score = float(score_str)
        except Exception:
            score = chunk.get("score", 0.0) * 10  # fall back to retrieval score

        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    reranked = []
    for score_val, ch in scored:
        ch = dict(ch)
        ch["rerank_score"] = round(score_val / 10.0, 4)
        reranked.append(ch)
    return reranked


# ─── Context budget ──────────────────────────────────────────────────────────


def apply_context_budget(
    chunks: List[Dict[str, Any]],
    budget_tokens: int = config.MCP_CONTEXT_TOKEN_BUDGET,
) -> List[Dict[str, Any]]:
    """
    Trim *chunks* so that their combined token count stays within *budget_tokens*.

    Chunks are processed in order (highest relevance first) and included until
    the budget is exhausted. Oversized individual chunks are truncated rather
    than dropped entirely.
    """
    selected: list[dict] = []
    used = 0
    for chunk in chunks:
        chunk_tokens = count_tokens(chunk["text"])
        if used + chunk_tokens <= budget_tokens:
            selected.append(chunk)
            used += chunk_tokens
        elif used < budget_tokens:
            # Partial inclusion — truncate to remaining budget
            remaining = budget_tokens - used
            approx_chars = remaining * 4  # tokens ≈ chars / 4
            truncated = dict(chunk)
            truncated["text"] = chunk["text"][:approx_chars] + " [truncated]"
            truncated["truncated"] = True
            selected.append(truncated)
            used = budget_tokens
            break
    logger.debug("Context budget: %d / %d tokens used, %d chunks selected",
                 used, budget_tokens, len(selected))
    return selected


# ─── Query decomposition ─────────────────────────────────────────────────────


def decompose_query(query: str) -> List[str]:
    """
    Decompose a complex or multi-part query into focused sub-queries.

    Uses the LLM to split the query. Returns a list of sub-queries (at least
    the original if no decomposition is needed).
    """
    prompt = (
        "You are a query planning assistant. Given a user question that may span "
        "multiple topics or documents, decompose it into focused, atomic sub-queries "
        "suitable for retrieval from a regulatory document database. "
        "Return ONLY a JSON array of strings. If the question is already simple and "
        "focused, return a JSON array with just that one question.\n\n"
        f"Question: {query}\n\nSub-queries (JSON array):"
    )
    try:
        resp = _OAI.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0,
        )
        import json
        content = resp.choices[0].message.content.strip()
        sub_queries = json.loads(content)
        if isinstance(sub_queries, list) and all(
            isinstance(q, str) for q in sub_queries
        ):
            logger.info("Decomposed query into %d sub-queries", len(sub_queries))
            return sub_queries
    except Exception as exc:
        logger.warning("Query decomposition failed (%s); using original query", exc)
    return [query]


# ─── Conversation memory ─────────────────────────────────────────────────────


def summarise_history(
    history: List[Dict[str, str]],
    max_turns: int = 6,
) -> List[Dict[str, str]]:
    """
    Prune conversation history to at most *max_turns* turns.

    If the history exceeds *max_turns* pairs (user/assistant), older turns are
    summarised by the LLM into a single system message prepended to the kept turns.
    """
    pairs: list[tuple[dict, dict]] = []
    i = 0
    while i + 1 < len(history):
        if history[i]["role"] == "user" and history[i + 1]["role"] == "assistant":
            pairs.append((history[i], history[i + 1]))
            i += 2
        else:
            i += 1

    if len(pairs) <= max_turns:
        return history

    to_summarise = pairs[: len(pairs) - max_turns]
    kept = pairs[len(pairs) - max_turns :]

    summary_input = "\n".join(
        f"User: {u['content']}\nAssistant: {a['content']}"
        for u, a in to_summarise
    )
    summary_prompt = (
        "Summarise the following conversation turns concisely, preserving any "
        "important regulatory facts or decisions:\n\n" + summary_input
    )
    try:
        resp = _OAI.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=300,
            temperature=0,
        )
        summary_text = resp.choices[0].message.content.strip()
    except Exception:
        summary_text = "(Earlier conversation summarisation failed.)"

    pruned: list[dict] = [
        {"role": "system", "content": f"[Earlier conversation summary]: {summary_text}"}
    ]
    for u, a in kept:
        pruned.append(u)
        pruned.append(a)
    return pruned
