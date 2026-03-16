"""
Streamlit chatbot UI — bonus deliverable.

Talks to the agent-api service and renders the structured
Reasoning / Thinking → Answer → Citations output.
"""
from __future__ import annotations

import json
import os
from uuid import uuid4

import requests
import streamlit as st

AGENT_API_HOST = os.getenv("AGENT_API_HOST", "localhost")
AGENT_API_PORT = os.getenv("AGENT_API_PORT", "8002")
AGENT_API_URL = f"http://{AGENT_API_HOST}:{AGENT_API_PORT}"

st.set_page_config(
    page_title="Regulatory RAG Chatbot",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Regulatory AI Chatbot")
st.caption(
    "Powered by Google ADK · NIST AI RMF · EU AI Act · ChromaDB · OpenAI GPT-4o-mini"
)

# ─── Session state ────────────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "audit_trails" not in st.session_state:
    st.session_state.audit_trails = []

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Session")
    st.code(st.session_state.session_id, language=None)
    if st.button("New session"):
        st.session_state.session_id = str(uuid4())
        st.session_state.messages = []
        st.session_state.audit_trails = []
        st.rerun()

    st.divider()
    st.header("Sample Questions")
    SAMPLES = [
        "What are the four core functions of the NIST AI RMF?",
        "Which EU AI Act article defines 'high-risk AI systems'?",
        "How does the NIST AI RMF GOVERN function relate to EU AI Act transparency obligations?",
        "What obligations does Article 13 of the EU AI Act impose on providers?",
        "What is the role of the AI risk taxonomy in the NIST AI RMF MAP function?",
    ]
    for q in SAMPLES:
        if st.button(q, key=q):
            st.session_state._pending_query = q

    st.divider()
    st.markdown("**Agent API:** `http://agent-api:8002`")
    st.markdown("**MCP Server:** `http://mcp-server:8001`")

# ─── Chat history display ─────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── Input ───────────────────────────────────────────────────────────────────

pending = st.session_state.pop("_pending_query", None)
user_input = st.chat_input("Ask about NIST AI RMF or EU AI Act…") or pending

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call agent
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reasoning…"):
            try:
                resp = requests.post(
                    f"{AGENT_API_URL}/chat",
                    json={
                        "query": user_input,
                        "session_id": st.session_state.session_id,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()

                formatted = data["formatted_response"]
                audit = data["audit_trail"]
                latency = data["latency_ms"]

                st.markdown(formatted)
                st.caption(f"⏱ {latency:.0f} ms · grounding: {audit.get('grounding_score', 0):.2f}")

                if audit.get("guardrail_triggered"):
                    st.warning("⚠️ A guardrail was triggered during this response.")

                if audit.get("hallucination_risk"):
                    st.error("🚨 Potential hallucination signal detected.")

                with st.expander("🔍 Audit Trail"):
                    st.json(audit)

                st.session_state.messages.append(
                    {"role": "assistant", "content": formatted}
                )
                st.session_state.audit_trails.append(audit)

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to agent API. Is the stack running?")
            except Exception as exc:
                st.error(f"Error: {exc}")
