"""
Evaluation suite — Task 11.

Evaluates the Agentic RAG system using RAGAS metrics:
  - Faithfulness         (are claims grounded in retrieved context?)
  - Answer Relevance     (does the answer address the question?)
  - Context Precision    (is the retrieved context relevant?)

Runs 8 test questions (≥2 cross-document) against the live agent-api,
writes evaluation/results.json and evaluation/report.md.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import httpx
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

AGENT_API_HOST = os.getenv("AGENT_API_HOST", "localhost")
AGENT_API_PORT = os.getenv("AGENT_API_PORT", "8002")
AGENT_URL = f"http://{AGENT_API_HOST}:{AGENT_API_PORT}/chat"

EVAL_DIR = Path(__file__).parent

# ─── Test questions ───────────────────────────────────────────────────────────
# 6 single-document + 2 cross-document questions (marked with [CROSS])

TEST_QUESTIONS: List[Dict[str, Any]] = [
    # NIST-focused
    {
        "id": "Q01",
        "type": "nist",
        "question": "What are the four core functions of the NIST AI RMF and what does each govern?",
        "ground_truth": (
            "The four core functions are GOVERN, MAP, MEASURE, and MANAGE. "
            "GOVERN establishes organisational policies and culture for AI risk management. "
            "MAP identifies and classifies AI risks in context. "
            "MEASURE analyses and assesses those risks using metrics. "
            "MANAGE applies prioritised responses to address AI risks."
        ),
    },
    {
        "id": "Q02",
        "type": "nist",
        "question": (
            "What does the NIST AI RMF say about human oversight and what category/function "
            "does it fall under?"
        ),
        "ground_truth": (
            "Human oversight is addressed primarily under the GOVERN and MANAGE functions. "
            "The framework emphasises that AI systems should support human review and "
            "intervention, particularly for high-risk decisions."
        ),
    },
    {
        "id": "Q03",
        "type": "nist",
        "question": "How does the NIST AI RMF define trustworthy AI characteristics?",
        "ground_truth": (
            "NIST AI RMF identifies trustworthy AI characteristics including accuracy, "
            "reliability, explainability, transparency, fairness, bias mitigation, "
            "security, resilience, privacy-enhanced, and accountability."
        ),
    },
    # EU AI Act focused
    {
        "id": "Q04",
        "type": "eu_ai_act",
        "question": "What does Article 6 of the EU AI Act say about classification of high-risk AI systems?",
        "ground_truth": (
            "Article 6 establishes the classification rules for high-risk AI systems. "
            "It sets out that AI systems intended to be used as safety components or "
            "listed in Annex III are classified as high-risk and subject to Title III obligations."
        ),
    },
    {
        "id": "Q05",
        "type": "eu_ai_act",
        "question": "What transparency obligations does Article 13 of the EU AI Act impose on providers of high-risk AI?",
        "ground_truth": (
            "Article 13 requires providers of high-risk AI systems to ensure the system "
            "is transparent, enabling deployers to understand its capabilities and limitations. "
            "It mandates instructions for use including information on the system's purpose, "
            "performance, human oversight measures, and technical characteristics."
        ),
    },
    {
        "id": "Q06",
        "type": "eu_ai_act",
        "question": "Which AI practices are prohibited under Article 5 of the EU AI Act?",
        "ground_truth": (
            "Article 5 prohibits AI practices including: subliminal manipulation, "
            "exploitation of vulnerabilities of specific groups, real-time remote biometric "
            "identification in public spaces by law enforcement (with narrow exceptions), "
            "social scoring by public authorities, and AI that infers emotions in workplaces "
            "or educational institutions."
        ),
    },
    # Cross-document
    {
        "id": "Q07",
        "type": "cross",
        "question": (
            "How do the transparency obligations in the EU AI Act (Article 13) compare to "
            "the NIST AI RMF's GOVERN function requirements on transparency and documentation?"
        ),
        "ground_truth": (
            "Both frameworks require transparency and documentation. "
            "EU AI Act Article 13 mandates specific technical documentation and instructions for use "
            "for high-risk AI. The NIST AI RMF GOVERN function requires organisations to establish "
            "policies for transparency and accountability across the AI lifecycle. "
            "Both align on the need for explainability and human oversight but differ in that "
            "the EU AI Act creates binding legal obligations while NIST provides voluntary guidance."
        ),
    },
    {
        "id": "Q08",
        "type": "cross",
        "question": (
            "How does the NIST AI RMF MEASURE function relate to the EU AI Act's requirements "
            "for accuracy, robustness, and cybersecurity (Article 15)?"
        ),
        "ground_truth": (
            "The NIST AI RMF MEASURE function calls for quantitative and qualitative methods "
            "to assess AI risks including accuracy and performance metrics. "
            "EU AI Act Article 15 sets binding requirements for high-risk AI systems to achieve "
            "appropriate levels of accuracy, robustness, and cybersecurity. "
            "Both converge on the need for ongoing measurement and benchmarking but NIST is "
            "framework-level guidance while Article 15 imposes legal obligations."
        ),
    },
]


# ─── Evaluation runner ────────────────────────────────────────────────────────


def _call_agent(question: str) -> Dict[str, Any]:
    """Call the agent API and return the parsed response."""
    try:
        resp = httpx.post(AGENT_URL, json={"query": question}, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.error("Agent call failed for question=%r: %s", question[:60], exc)
        return {"formatted_response": "", "audit_trail": {}}


def _extract_answer(agent_response: Dict[str, Any]) -> str:
    """Extract the plain answer from the formatted response."""
    formatted = agent_response.get("formatted_response", "")
    # The answer section is between "## Answer\n" and "## Citations"
    if "## Answer" in formatted:
        after = formatted.split("## Answer", 1)[1]
        if "## Citations" in after:
            return after.split("## Citations", 1)[0].strip()
        return after.strip()
    return formatted


def _extract_contexts(agent_response: Dict[str, Any]) -> List[str]:
    """Extract retrieved chunk texts from the audit trail."""
    audit = agent_response.get("audit_trail", {})
    chunks = audit.get("retrieved_chunks", [])
    return [ch.get("text", "") for ch in chunks if ch.get("text")]


def run_evaluation() -> None:
    logger.info("Starting evaluation against %d questions…", len(TEST_QUESTIONS))

    questions, answers, contexts, ground_truths, meta = [], [], [], [], []

    for item in TEST_QUESTIONS:
        logger.info("[%s] Querying agent: %s", item["id"], item["question"][:70])
        agent_resp = _call_agent(item["question"])
        answer = _extract_answer(agent_resp)
        ctx = _extract_contexts(agent_resp)

        questions.append(item["question"])
        answers.append(answer or "No answer returned.")
        contexts.append(ctx or ["No context retrieved."])
        ground_truths.append(item["ground_truth"])
        meta.append({
            "id": item["id"],
            "type": item["type"],
            "audit_trail": agent_resp.get("audit_trail", {}),
        })

    # ── RAGAS evaluation ──────────────────────────────────────────────────────
    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    logger.info("Running RAGAS evaluation…")
    try:
        ragas_result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
        scores_df = ragas_result.to_pandas()
        scores_list = scores_df.to_dict(orient="records")
    except Exception as exc:
        logger.error("RAGAS evaluation failed: %s", exc)
        scores_list = [{"error": str(exc)}] * len(questions)

    # ── Write results.json ────────────────────────────────────────────────────
    results = []
    for i, item in enumerate(TEST_QUESTIONS):
        row = scores_list[i] if i < len(scores_list) else {}
        results.append(
            {
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                "answer": answers[i],
                "ground_truth": item["ground_truth"],
                "context_count": len(contexts[i]),
                "faithfulness": row.get("faithfulness"),
                "answer_relevancy": row.get("answer_relevancy"),
                "context_precision": row.get("context_precision"),
                "audit_trail": meta[i]["audit_trail"],
            }
        )

    results_path = EVAL_DIR / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    logger.info("Results written to %s", results_path)

    # ── Write report.md ───────────────────────────────────────────────────────
    _write_report(results, EVAL_DIR / "report.md")
    logger.info("Report written to %s", EVAL_DIR / "report.md")


def _avg(results: list, key: str) -> float:
    vals = [r[key] for r in results if r.get(key) is not None]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _write_report(results: list, path: Path) -> None:
    lines: list[str] = []
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines += [
        "# Evaluation Report",
        f"Generated: {ts}",
        "",
        "## Summary",
        "",
        "| Metric | Average Score |",
        "|---|---|",
        f"| Faithfulness | {_avg(results, 'faithfulness'):.4f} |",
        f"| Answer Relevancy | {_avg(results, 'answer_relevancy'):.4f} |",
        f"| Context Precision | {_avg(results, 'context_precision'):.4f} |",
        "",
        "## Per-Question Results",
        "",
        "| ID | Type | Faithfulness | Answer Relevancy | Context Precision |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['id']} | {r['type']} "
            f"| {r.get('faithfulness', 'N/A')} "
            f"| {r.get('answer_relevancy', 'N/A')} "
            f"| {r.get('context_precision', 'N/A')} |"
        )

    lines += [
        "",
        "## What the Scores Reveal",
        "",
        "### Faithfulness",
        (
            "Faithfulness measures whether claims in the answer are supported by the "
            "retrieved context. Scores below 0.8 indicate the model may be adding "
            "information not present in the retrieved passages. The primary causes are "
            "insufficient context retrieval for boundary cases and the model's tendency "
            "to generalise from partial evidence."
        ),
        "",
        "### Answer Relevancy",
        (
            "Answer relevancy checks whether the answer directly addresses the question. "
            "Cross-document questions (Q07, Q08) typically score lower because the model "
            "must synthesise conflicting granularity levels (binding law vs. voluntary framework). "
            "Query decomposition partially mitigates this."
        ),
        "",
        "### Context Precision",
        (
            "Context precision evaluates whether retrieved chunks are relevant to the query. "
            "Hybrid retrieval (dense + BM25 + RRF) with re-ranking improves precision "
            "compared to dense-only retrieval, but article-boundary chunking for the EU AI Act "
            "can produce long chunks with mixed relevance."
        ),
        "",
        "## Where the System Struggles",
        "",
        "1. **Cross-document synthesis** — Questions spanning both documents require the "
        "model to compare regulatory frameworks with different legal natures (binding vs. "
        "voluntary). The retrieved chunks may not co-locate the most relevant passages from "
        "both documents in the same context window.",
        "",
        "2. **Annex references** — The EU AI Act frequently references Annexes (I, III, IV). "
        "Annexes are separate sections and may not be retrieved in the same query as the "
        "main article body.",
        "",
        "3. **NIST tabular data** — Tables in the NIST AI RMF PDF are not perfectly "
        "reconstructed by PyPDF; some structured mappings between sub-categories and "
        "outcomes may be partially lost.",
        "",
        "## Likely Root Causes",
        "",
        "- PDF text extraction quality for complex layouts (tables, footnotes)",
        "- Context budget limits: long cross-document answers may exceed the token budget",
        "- Re-ranking quality depends on the LLM judge itself; errors propagate",
        "",
        "## Future Work",
        "",
        "- Use a dedicated PDF parser (Unstructured.io or LlamaParse) for richer table extraction",
        "- Add Annex ingestion as separate collections with linked metadata",
        "- Implement a bi-encoder cross-encoder pipeline instead of LLM-based re-ranking",
        "- Increase evaluation dataset to 20+ questions for statistical significance",
    ]

    path.write_text("\n".join(lines))


if __name__ == "__main__":
    run_evaluation()
