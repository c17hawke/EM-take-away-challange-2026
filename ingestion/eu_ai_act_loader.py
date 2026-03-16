"""
EU AI Act website ingestion — Task 2.

Fetches the EU AI Act HTML page, strips navigation / boilerplate, and
produces article-boundary chunks with metadata: source, article_id,
article_title, chapter.
"""
from __future__ import annotations

import logging
import re
from typing import List

import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import config

logger = logging.getLogger(__name__)

# Patterns that mark content we want to discard
_NOISE_SELECTORS = [
    "nav", "header", "footer",
    ".cookie-consent", "#cookie-banner",
    ".navigation", ".breadcrumb",
    "script", "style", "noscript",
]

_ARTICLE_RE = re.compile(
    r"Article\s+(\d+)\s*[–\-—]\s*(.+?)(?=Article\s+\d+|Chapter\s+[IVXLC]+|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_CHAPTER_RE = re.compile(
    r"Chapter\s+([IVXLC]+)\s*[–\-—]\s*(.+?)(?=Chapter\s+[IVXLC]+|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _fetch_html(url: str) -> str:
    """Download and return the raw HTML from *url*."""
    logger.info("Fetching EU AI Act from %s", url)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; Agentic-RAG-Ingestion/1.0; "
            "+https://github.com/c17hawke/EM-take-away-challange-2026)"
        )
    }
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.text


def _clean_html(html: str) -> tuple[str, BeautifulSoup]:
    """Strip boilerplate and return (clean_text, soup)."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise elements
    for selector in _NOISE_SELECTORS:
        for el in soup.select(selector):
            el.decompose()

    # The EUR-Lex page wraps the legal text in <div class="eli-main-title"> and
    # the body in generic divs — grab the main content area.
    main = soup.find("div", {"id": "text"}) or soup.find("body")
    text = main.get_text(separator="\n") if main else soup.get_text(separator="\n")

    # Normalise whitespace
    lines = [ln.rstrip() for ln in text.splitlines()]
    cleaned_lines: list[str] = []
    blank_count = 0
    for ln in lines:
        if ln == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append(ln)
        else:
            blank_count = 0
            cleaned_lines.append(ln)

    return "\n".join(cleaned_lines), soup


def _extract_chapters(text: str) -> dict[str, str]:
    """Build a mapping from article_id → chapter label from the full text."""
    chapter_map: dict[str, str] = {}
    current_chapter = "General"
    for line in text.splitlines():
        chapter_m = re.match(r"^Chapter\s+([IVXLC]+)", line, re.IGNORECASE)
        if chapter_m:
            current_chapter = f"Chapter {chapter_m.group(1)}"
        art_m = re.match(r"^Article\s+(\d+)", line, re.IGNORECASE)
        if art_m:
            chapter_map[art_m.group(1)] = current_chapter
    return chapter_map


def load_eu_ai_act_chunks() -> List[Document]:
    """
    Fetch, clean, and article-boundary-chunk the EU AI Act webpage.

    Returns a list of LangChain Documents enriched with metadata:
        source, article_id, article_title, chapter
    """
    html = _fetch_html(config.EU_AI_ACT_URL)
    clean_text, _soup = _clean_html(html)

    chapter_map = _extract_chapters(clean_text)

    # Split by article boundaries
    # We look for lines that begin "Article N" as split points
    article_split_re = re.compile(r"(?=^Article\s+\d+)", re.MULTILINE)
    raw_articles = article_split_re.split(clean_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: list[Document] = []
    for article_text in raw_articles:
        article_text = article_text.strip()
        if len(article_text) < 50:
            continue

        # Extract article id and title from leading line
        first_line = article_text.splitlines()[0].strip()
        art_m = re.match(r"Article\s+(\d+)\s*[–\-—]?\s*(.*)", first_line, re.IGNORECASE)
        if art_m:
            article_id = art_m.group(1)
            article_title = art_m.group(2).strip() or f"Article {article_id}"
        else:
            article_id = "Unknown"
            article_title = first_line[:80]

        chapter = chapter_map.get(article_id, "General")

        sub_chunks = splitter.create_documents(
            texts=[article_text],
            metadatas=[{
                "source": "EU AI Act",
                "article_id": article_id,
                "article_title": article_title,
                "chapter": chapter,
            }],
        )
        chunks.extend(sub_chunks)

    logger.info("Produced %d chunks from EU AI Act", len(chunks))
    return chunks
