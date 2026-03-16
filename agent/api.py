"""
Agent REST API — exposes the Google ADK agent over HTTP.

Endpoints
---------
POST /chat
    Body:  {"query": "...", "session_id": "optional-uuid"}
    Returns structured response + audit trail.

GET /health
    Returns {"status": "ok"}.
"""
from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from agent.adk_agent import RegulatoryAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Regulatory RAG Agent API",
    description="Agentic RAG chatbot for NIST AI RMF and EU AI Act queries.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Session store (in-memory for PoC) ───────────────────────────────────────

_SESSIONS: Dict[str, RegulatoryAgent] = {}


def _get_agent(session_id: str) -> RegulatoryAgent:
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = RegulatoryAgent()
    return _SESSIONS[session_id]


# ─── Request / Response models ───────────────────────────────────────────────


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    formatted_response: str
    audit_trail: Dict[str, Any]
    latency_ms: float


# ─── Routes ──────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    # Input validation
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    if len(query) > 4000:
        raise HTTPException(status_code=400, detail="Query too long (max 4000 chars).")

    session_id = req.session_id or str(uuid4())
    agent = _get_agent(session_id)

    result = agent.run(query)

    return ChatResponse(
        session_id=session_id,
        formatted_response=result["formatted_response"],
        audit_trail=result["audit_trail"],
        latency_ms=result["latency_ms"],
    )


# ─── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "agent.api:app",
        host=config.AGENT_API_HOST,
        port=config.AGENT_API_PORT,
        reload=False,
    )
