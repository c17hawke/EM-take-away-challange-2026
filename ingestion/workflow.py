"""
LangGraph ingestion workflow — Task 4.

Graph shape:
  load_sources → parse_and_chunk → enrich_metadata → embed_and_index

Each node is independently testable. The state dict threads data between nodes.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, TypedDict

from langchain.schema import Document
from langgraph.graph import END, StateGraph

from ingestion.eu_ai_act_loader import load_eu_ai_act_chunks
from ingestion.nist_loader import load_nist_chunks
from ingestion.vector_store import index_documents

logger = logging.getLogger(__name__)


# ─── State ───────────────────────────────────────────────────────────────────


class IngestionState(TypedDict, total=False):
    nist_chunks: List[Document]
    eu_chunks: List[Document]
    all_chunks: List[Document]
    errors: List[str]


# ─── Nodes ───────────────────────────────────────────────────────────────────


def node_load_sources(state: IngestionState) -> Dict[str, Any]:
    """Load raw chunks from both source documents."""
    errors: list[str] = []
    nist_chunks: list[Document] = []
    eu_chunks: list[Document] = []

    try:
        logger.info("[load_sources] Loading NIST AI RMF PDF…")
        nist_chunks = load_nist_chunks()
        logger.info("[load_sources] NIST: %d raw chunks", len(nist_chunks))
    except Exception as exc:
        msg = f"NIST load failed: {exc}"
        logger.error(msg)
        errors.append(msg)

    try:
        logger.info("[load_sources] Loading EU AI Act website…")
        eu_chunks = load_eu_ai_act_chunks()
        logger.info("[load_sources] EU AI Act: %d raw chunks", len(eu_chunks))
    except Exception as exc:
        msg = f"EU AI Act load failed: {exc}"
        logger.error(msg)
        errors.append(msg)

    return {"nist_chunks": nist_chunks, "eu_chunks": eu_chunks, "errors": errors}


def node_parse_and_chunk(state: IngestionState) -> Dict[str, Any]:
    """
    Combine chunks from both sources.

    For this pipeline the heavy parsing/chunking is already done inside the
    source-specific loaders (Task 1 & 2). This node validates and merges them.
    """
    nist = state.get("nist_chunks", [])
    eu = state.get("eu_chunks", [])
    errors = list(state.get("errors", []))

    # Basic validation
    empty_sources = []
    if not nist:
        empty_sources.append("NIST")
    if not eu:
        empty_sources.append("EU AI Act")
    if empty_sources:
        errors.append(f"No chunks loaded for: {', '.join(empty_sources)}")

    all_chunks = nist + eu
    logger.info("[parse_and_chunk] Total chunks to index: %d", len(all_chunks))
    return {"all_chunks": all_chunks, "errors": errors}


def node_enrich_metadata(state: IngestionState) -> Dict[str, Any]:
    """
    Ensure every chunk has the mandatory metadata fields.

    Missing fields are filled with sensible defaults so downstream indexing
    never fails on a KeyError.
    """
    chunks = state.get("all_chunks", [])
    errors = list(state.get("errors", []))
    enriched: list[Document] = []

    for chunk in chunks:
        meta = dict(chunk.metadata)
        source = meta.get("source", "Unknown")

        if source == "NIST AI RMF":
            meta.setdefault("section", "General")
            meta.setdefault("page_number", 0)
            meta.setdefault("article_id", None)
            meta.setdefault("article_title", None)
            meta.setdefault("chapter", None)
        elif source == "EU AI Act":
            meta.setdefault("article_id", "Unknown")
            meta.setdefault("article_title", "Unknown")
            meta.setdefault("chapter", "General")
            meta.setdefault("section", None)
            meta.setdefault("page_number", None)

        enriched.append(Document(page_content=chunk.page_content, metadata=meta))

    logger.info("[enrich_metadata] Enriched %d chunks", len(enriched))
    return {"all_chunks": enriched, "errors": errors}


def node_embed_and_index(state: IngestionState) -> Dict[str, Any]:
    """Embed chunks and upsert into ChromaDB."""
    chunks = state.get("all_chunks", [])
    errors = list(state.get("errors", []))

    nist_chunks = [c for c in chunks if c.metadata.get("source") == "NIST AI RMF"]
    eu_chunks = [c for c in chunks if c.metadata.get("source") == "EU AI Act"]

    try:
        logger.info("[embed_and_index] Indexing %d NIST + %d EU chunks…",
                    len(nist_chunks), len(eu_chunks))
        index_documents(nist_docs=nist_chunks, eu_docs=eu_chunks)
        logger.info("[embed_and_index] Indexing complete.")
    except Exception as exc:
        msg = f"Indexing failed: {exc}"
        logger.error(msg)
        errors.append(msg)

    return {"errors": errors}


# ─── Graph ───────────────────────────────────────────────────────────────────


def build_ingestion_graph() -> Any:
    """Build and compile the LangGraph ingestion workflow."""
    g = StateGraph(IngestionState)

    g.add_node("load_sources", node_load_sources)
    g.add_node("parse_and_chunk", node_parse_and_chunk)
    g.add_node("enrich_metadata", node_enrich_metadata)
    g.add_node("embed_and_index", node_embed_and_index)

    g.set_entry_point("load_sources")
    g.add_edge("load_sources", "parse_and_chunk")
    g.add_edge("parse_and_chunk", "enrich_metadata")
    g.add_edge("enrich_metadata", "embed_and_index")
    g.add_edge("embed_and_index", END)

    return g.compile()


def run_ingestion() -> IngestionState:
    """Execute the ingestion pipeline and return the final state."""
    graph = build_ingestion_graph()
    final_state: IngestionState = graph.invoke({})
    errors = final_state.get("errors", [])
    if errors:
        logger.warning("Ingestion finished with %d error(s): %s", len(errors), errors)
    else:
        logger.info("Ingestion finished successfully.")
    return final_state
