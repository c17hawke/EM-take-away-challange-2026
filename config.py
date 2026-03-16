"""
Shared configuration loaded from environment variables.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── LLM ────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ─── ChromaDB ───────────────────────────────────────────────────────
CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION_NIST: str = os.getenv("CHROMA_COLLECTION_NIST", "nist_rmf")
CHROMA_COLLECTION_EUAI: str = os.getenv("CHROMA_COLLECTION_EUAI", "eu_ai_act")

# ─── MCP Server ─────────────────────────────────────────────────────
MCP_SERVER_HOST: str = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
MCP_SERVER_PORT: int = int(os.getenv("MCP_SERVER_PORT", "8001"))
MCP_TOP_K: int = int(os.getenv("MCP_TOP_K", "10"))
MCP_CONTEXT_TOKEN_BUDGET: int = int(os.getenv("MCP_CONTEXT_TOKEN_BUDGET", "4096"))

# ─── Agent API ──────────────────────────────────────────────────────
AGENT_API_HOST: str = os.getenv("AGENT_API_HOST", "0.0.0.0")
AGENT_API_PORT: int = int(os.getenv("AGENT_API_PORT", "8002"))

# ─── LangSmith ──────────────────────────────────────────────────────
LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "em-agentic-rag-2026")

# ─── Ingestion ──────────────────────────────────────────────────────
NIST_PDF_URL: str = os.getenv(
    "NIST_PDF_URL", "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf"
)
EU_AI_ACT_URL: str = os.getenv(
    "EU_AI_ACT_URL",
    "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689",
)
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
