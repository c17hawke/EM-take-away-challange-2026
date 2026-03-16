"""
NIST AI RMF PDF ingestion — Task 1.

Loads the NIST.AI.100-1 PDF, applies section-aware chunking that respects
GOVERN / MAP / MEASURE / MANAGE structural boundaries, and enriches every
chunk with metadata: source, section, page_number.
"""
from __future__ import annotations

import io
import logging
import os
import re
import urllib.request
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

import config

logger = logging.getLogger(__name__)

# Top-level section headings in the NIST AI RMF document.
# Order matters: more specific patterns first.
_SECTION_PATTERNS: list[tuple[str, str]] = [
    (r"GOVERN\s+\d", "GOVERN"),
    (r"MAP\s+\d", "MAP"),
    (r"MEASURE\s+\d", "MEASURE"),
    (r"MANAGE\s+\d", "MANAGE"),
    (r"GOVERN", "GOVERN"),
    (r"MAP\b", "MAP"),
    (r"MEASURE\b", "MEASURE"),
    (r"MANAGE\b", "MANAGE"),
    (r"Introduction", "Introduction"),
    (r"Audience", "Audience"),
    (r"Appendix", "Appendix"),
]

_LOCAL_PDF_PATH = Path("data/NIST.AI.100-1.pdf")


def _detect_section(text: str) -> str:
    """Heuristically map chunk text to the NIST AI RMF section it belongs to."""
    for pattern, section_name in _SECTION_PATTERNS:
        if re.search(pattern, text[:500]):
            return section_name
    return "General"


def _is_noise_line(line: str) -> bool:
    """Return True for header/footer/figure-caption boilerplate lines."""
    stripped = line.strip()
    # Page numbers, repeated document title, running headers
    if re.fullmatch(r"\d+", stripped):
        return True
    if "NIST AI 100-1" in stripped and len(stripped) < 60:
        return True
    if stripped.lower().startswith("figure ") and len(stripped) < 80:
        return True
    if stripped.lower().startswith("table ") and len(stripped) < 80:
        return True
    return False


def _clean_text(text: str) -> str:
    """Remove noise lines and normalise whitespace."""
    lines = text.splitlines()
    cleaned = [ln for ln in lines if not _is_noise_line(ln)]
    return "\n".join(cleaned).strip()


def _download_pdf() -> Path:
    _LOCAL_PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _LOCAL_PDF_PATH.exists():
        logger.info("Using cached NIST PDF: %s", _LOCAL_PDF_PATH)
        return _LOCAL_PDF_PATH
    logger.info("Downloading NIST PDF from %s", config.NIST_PDF_URL)
    urllib.request.urlretrieve(config.NIST_PDF_URL, _LOCAL_PDF_PATH)  # noqa: S310
    logger.info("Downloaded NIST PDF to %s", _LOCAL_PDF_PATH)
    return _LOCAL_PDF_PATH


def load_nist_chunks() -> List[Document]:
    """
    Download (if needed), parse, clean, and section-aware-chunk the NIST PDF.

    Returns a list of LangChain Documents enriched with metadata:
        source, section, page_number
    """
    pdf_path = _download_pdf()

    loader = PyPDFLoader(str(pdf_path))
    raw_pages: list[Document] = loader.load()
    logger.info("Loaded %d raw pages from NIST PDF", len(raw_pages))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n\n", "\n\n", "\n", " ", ""],
    )

    chunks: list[Document] = []
    for page_doc in raw_pages:
        page_num: int = page_doc.metadata.get("page", 0) + 1  # 0-indexed → 1-indexed
        cleaned = _clean_text(page_doc.page_content)
        if not cleaned:
            continue

        page_chunks = splitter.create_documents(
            texts=[cleaned],
            metadatas=[{"source": "NIST AI RMF", "page_number": page_num}],
        )
        for chunk in page_chunks:
            chunk.metadata["section"] = _detect_section(chunk.page_content)
        chunks.extend(page_chunks)

    logger.info("Produced %d chunks from NIST PDF", len(chunks))
    return chunks
