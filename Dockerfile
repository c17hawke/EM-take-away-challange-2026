# ─── Base image used by ingestion, mcp-server, and agent-api ─────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System deps for PDF parsing and HTTP
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py .
COPY ingestion/ ingestion/
COPY mcp_server/ mcp_server/
COPY agent/ agent/
COPY ui/ ui/
COPY evaluation/ evaluation/

# ─── Ingestion image ──────────────────────────────────────────────────────────
FROM base AS ingestion
CMD ["python", "-m", "ingestion.main"]

# ─── MCP server image ─────────────────────────────────────────────────────────
FROM base AS mcp-server
EXPOSE 8001
CMD ["uvicorn", "mcp_server.server:app", "--host", "0.0.0.0", "--port", "8001"]

# ─── Agent API image ──────────────────────────────────────────────────────────
FROM base AS agent-api
EXPOSE 8002
CMD ["uvicorn", "agent.api:app", "--host", "0.0.0.0", "--port", "8002"]
