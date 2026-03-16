"""
Google ADK callback guardrails — Task 7.

Three lightweight callbacks are implemented:
  1. before_model_callback  — prompt injection detection + token overflow guard
  2. before_tool_callback   — MCP query validation
  3. after_model_callback   — hallucination signal detection

All callback decisions are logged.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ─── Injection / overflow patterns ───────────────────────────────────────────

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"you\s+are\s+now\s+a", re.I),
    re.compile(r"disregard\s+(your\s+)?(system\s+)?prompt", re.I),
    re.compile(r"<\s*script", re.I),
    re.compile(r"system\s*:\s*you\s+must", re.I),
    re.compile(r"pretend\s+you\s+are", re.I),
]

# Fabricated article patterns — valid EU AI Act articles are 1–113
_INVALID_ARTICLE_RE = re.compile(r"Article\s+(\d+)", re.I)

# Phrases that signal hallucination
_HALLUCINATION_PHRASES = [
    "as stated in Article 200",
    "per NIST section XYZ",
    "according to the regulation",  # vague, non-specific
]

MAX_PROMPT_TOKENS = 12_000  # guard against context overflow


def _count_tokens_approximate(text: str) -> int:
    """Approximate token count: chars / 4."""
    return len(text) // 4


# ─── Callback 1: before_model_callback ───────────────────────────────────────


def before_model_callback(
    prompt_messages: list[dict[str, str]],
    context: Dict[str, Any],
) -> Optional[list[dict[str, str]]]:
    """
    Inspect the constructed prompt before it is sent to the LLM.

    Checks:
    - Prompt injection attempts in any message
    - Context token overflow

    Returns:
    - None  → allow the call to proceed unchanged
    - list  → modified messages (with injected content sanitised)
    - Raises ValueError if the prompt should be blocked entirely.
    """
    full_text = " ".join(m.get("content", "") for m in prompt_messages)

    # 1. Injection detection
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(full_text):
            logger.warning(
                "[GUARDRAIL][before_model] Prompt injection detected. Pattern: %s",
                pattern.pattern,
            )
            context["guardrail_triggered"] = True
            # Sanitise: remove only the offending message parts
            sanitised = []
            for msg in prompt_messages:
                content = msg.get("content", "")
                for pat in _INJECTION_PATTERNS:
                    content = pat.sub("[SANITISED]", content)
                sanitised.append({**msg, "content": content})
            logger.info("[GUARDRAIL][before_model] Prompt sanitised and allowed.")
            return sanitised

    # 2. Token overflow guard
    approx_tokens = _count_tokens_approximate(full_text)
    if approx_tokens > MAX_PROMPT_TOKENS:
        logger.warning(
            "[GUARDRAIL][before_model] Prompt token overflow: ~%d tokens (limit %d). "
            "Truncating oldest user messages.",
            approx_tokens,
            MAX_PROMPT_TOKENS,
        )
        context["guardrail_triggered"] = True
        # Keep system messages and most-recent user/assistant turns
        system_msgs = [m for m in prompt_messages if m.get("role") == "system"]
        other_msgs = [m for m in prompt_messages if m.get("role") != "system"]
        # Drop oldest non-system messages until within budget
        while _count_tokens_approximate(
            " ".join(m.get("content", "") for m in system_msgs + other_msgs)
        ) > MAX_PROMPT_TOKENS and len(other_msgs) > 1:
            other_msgs.pop(0)
        return system_msgs + other_msgs

    logger.debug("[GUARDRAIL][before_model] Prompt passed. ~%d tokens.", approx_tokens)
    return None  # no modification needed


# ─── Callback 2: before_tool_callback ────────────────────────────────────────


def before_tool_callback(
    tool_name: str,
    tool_args: Dict[str, Any],
    context: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Validate the tool call before it is executed.

    Checks:
    - Tool name is the expected MCP retrieval tool
    - Query is a non-empty string within a reasonable length
    - No injection content in the query

    Returns:
    - None → allow the call unchanged
    - dict → modified tool arguments
    - Raises ValueError to block the call
    """
    logger.debug("[GUARDRAIL][before_tool] Tool call: %s | args: %s", tool_name, tool_args)

    if tool_name != "retrieve_regulatory_context":
        # Unexpected tool — allow but log
        logger.warning("[GUARDRAIL][before_tool] Unexpected tool: %s", tool_name)
        return None

    query = tool_args.get("query", "")
    if not isinstance(query, str) or not query.strip():
        logger.error("[GUARDRAIL][before_tool] Empty or invalid query. Blocking.")
        context["guardrail_triggered"] = True
        raise ValueError("Tool call blocked: query must be a non-empty string.")

    if len(query) > 2000:
        logger.warning(
            "[GUARDRAIL][before_tool] Query too long (%d chars); truncating.", len(query)
        )
        context["guardrail_triggered"] = True
        tool_args = dict(tool_args)
        tool_args["query"] = query[:2000]
        return tool_args

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(query):
            logger.error(
                "[GUARDRAIL][before_tool] Injection in query. Pattern: %s. Blocking.",
                pattern.pattern,
            )
            context["guardrail_triggered"] = True
            raise ValueError("Tool call blocked: injection pattern detected in query.")

    logger.debug("[GUARDRAIL][before_tool] Query validated OK.")
    return None


# ─── Callback 3: after_model_callback ────────────────────────────────────────


def after_model_callback(
    response_text: str,
    context: Dict[str, Any],
    cited_sources: list[str] | None = None,
) -> Dict[str, Any]:
    """
    Inspect the raw model response for hallucination signals.

    Checks:
    - References to EU AI Act articles beyond the real scope (Art 1–113)
    - Vague unsupported attribution phrases
    - Claims not grounded in retrieved citations

    Returns a dict with hallucination_risk flag and details.
    """
    issues: list[str] = []

    # Check for out-of-range article numbers
    for m in _INVALID_ARTICLE_RE.finditer(response_text):
        art_num = int(m.group(1))
        if art_num > 113:
            issues.append(f"Fabricated EU AI Act article number: Article {art_num}")

    # Check for vague attribution phrases
    for phrase in _HALLUCINATION_PHRASES:
        if phrase.lower() in response_text.lower():
            issues.append(f"Vague unsupported attribution phrase: '{phrase}'")

    # Check that any NIST section references look valid
    nist_invalid = re.findall(r"§\s*([A-Z]+-\d+\.\d+)", response_text)
    if cited_sources:
        cited_set = {s.lower() for s in cited_sources}
        for ref in nist_invalid:
            full_ref = f"nist ai rmf §{ref.lower()}"
            if full_ref not in cited_set and ref not in " ".join(cited_sources):
                issues.append(f"NIST reference not in citations: §{ref}")

    hallucination_risk = len(issues) > 0
    if hallucination_risk:
        logger.warning(
            "[GUARDRAIL][after_model] Hallucination signals detected: %s", issues
        )
        context["guardrail_triggered"] = True
    else:
        logger.debug("[GUARDRAIL][after_model] No hallucination signals detected.")

    return {
        "hallucination_risk": hallucination_risk,
        "hallucination_details": issues,
    }
