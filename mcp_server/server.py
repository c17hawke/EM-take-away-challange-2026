"""
MCP Server — Task 5.

Exposes hybrid retrieval as an HTTP/JSON endpoint compatible with the
Model Context Protocol (MCP) tool-calling interface.

Endpoints
---------
POST /retrieve
    Body:  {"query": "...", "top_k": 10}
    Returns structured retrieval results including text, metadata, and scores.

GET /health
    Returns {"status": "ok"}.

The server runs as its own Docker service (`mcp-server`) and is registered
as a tool by the Google ADK agent.
"""
from __future__ import annotations

import logging
import sys
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
from ingestion.vector_store import HybridRetriever
from mcp_server.context_engineering import (
    apply_context_budget,
    rerank_chunks,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Regulatory RAG MCP Server",
    description="MCP-compatible retrieval tool for NIST AI RMF and EU AI Act.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Lazy-initialised retrievers ─────────────────────────────────────────────

_nist_retriever: HybridRetriever | None = None
_eu_retriever: HybridRetriever | None = None


def _get_retrievers() -> tuple[HybridRetriever, HybridRetriever]:
    global _nist_retriever, _eu_retriever
    if _nist_retriever is None:
        _nist_retriever = HybridRetriever(
            collection_name=config.CHROMA_COLLECTION_NIST,
            top_k=config.MCP_TOP_K,
        )
    if _eu_retriever is None:
        _eu_retriever = HybridRetriever(
            collection_name=config.CHROMA_COLLECTION_EUAI,
            top_k=config.MCP_TOP_K,
        )
    return _nist_retriever, _eu_retriever


# ─── Request / Response models ───────────────────────────────────────────────


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="The retrieval query string.")
    top_k: int = Field(default=10, ge=1, le=50, description="Max chunks to return.")
    sources: List[str] = Field(
        default=["nist", "eu_ai_act"],
        description="Which source(s) to query: 'nist', 'eu_ai_act', or both.",
    )
    apply_reranking: bool = Field(default=True, description="Whether to re-rank results.")
    apply_budget: bool = Field(
        default=True, description="Whether to enforce context token budget."
    )


class ChunkResult(BaseModel):
    text: str
    source: str
    article_id: Optional[str] = None
    article_title: Optional[str] = None
    section: Optional[str] = None
    chapter: Optional[str] = None
    page_number: Optional[int] = None
    score: float
    rerank_score: Optional[float] = None
    truncated: bool = False


class RetrieveResponse(BaseModel):
    query: str
    chunks: List[ChunkResult]
    total_chunks: int
    latency_ms: float


# ─── MCP Tool Schema (exposed for agent discovery) ───────────────────────────

MCP_TOOL_SCHEMA: Dict[str, Any] = {
    "name": "retrieve_regulatory_context",
    "description": (
        "Retrieve relevant passages from the NIST AI Risk Management Framework "
        "and/or the EU AI Act, given a natural-language query. Returns grounded "
        "text chunks with source attribution and relevance scores."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The retrieval query."},
            "top_k": {"type": "integer", "default": 10},
            "sources": {
                "type": "array",
                "items": {"type": "string", "enum": ["nist", "eu_ai_act"]},
                "default": ["nist", "eu_ai_act"],
            },
            "apply_reranking": {"type": "boolean", "default": True},
            "apply_budget": {"type": "boolean", "default": True},
        },
        "required": ["query"],
    },
}


# ─── Routes ──────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/mcp/schema")
async def mcp_schema() -> Dict[str, Any]:
    """Return the MCP tool schema for agent discovery."""
    return MCP_TOOL_SCHEMA


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    t0 = time.perf_counter()

    nist_retriever, eu_retriever = _get_retrievers()
    raw_chunks: list[dict] = []

    # Validate sources to prevent injection
    allowed_sources = {"nist", "eu_ai_act"}
    sources = [s for s in req.sources if s in allowed_sources]
    if not sources:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sources. Allowed: {allowed_sources}",
        )

    if "nist" in sources:
        try:
            nist_results = nist_retriever.retrieve(req.query)
            raw_chunks.extend(nist_results)
        except Exception as exc:
            logger.error("NIST retrieval error: %s", exc)

    if "eu_ai_act" in sources:
        try:
            eu_results = eu_retriever.retrieve(req.query)
            raw_chunks.extend(eu_results)
        except Exception as exc:
            logger.error("EU AI Act retrieval error: %s", exc)

    # Sort by score before re-ranking
    raw_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    raw_chunks = raw_chunks[: req.top_k * 2]  # widen pool for re-ranking

    if req.apply_reranking and raw_chunks:
        raw_chunks = rerank_chunks(req.query, raw_chunks)

    if req.apply_budget and raw_chunks:
        raw_chunks = apply_context_budget(raw_chunks)

    raw_chunks = raw_chunks[: req.top_k]

    chunk_results: list[ChunkResult] = []
    for ch in raw_chunks:
        meta = ch.get("metadata", {})
        chunk_results.append(
            ChunkResult(
                text=ch["text"],
                source=meta.get("source", "Unknown"),
                article_id=str(meta["article_id"]) if meta.get("article_id") else None,
                article_title=meta.get("article_title"),
                section=meta.get("section"),
                chapter=meta.get("chapter"),
                page_number=int(meta["page_number"]) if meta.get("page_number") else None,
                score=ch.get("score", 0.0),
                rerank_score=ch.get("rerank_score"),
                truncated=ch.get("truncated", False),
            )
        )

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Retrieved %d chunks for query=%r in %.1f ms", len(chunk_results), req.query, latency_ms
    )

    return RetrieveResponse(
        query=req.query,
        chunks=chunk_results,
        total_chunks=len(chunk_results),
        latency_ms=round(latency_ms, 2),
    )


# ─── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "mcp_server.server:app",
        host=config.MCP_SERVER_HOST,
        port=config.MCP_SERVER_PORT,
        reload=False,
    )
