"""
Embedding + ChromaDB indexing — Task 3.

Uses OpenAI text-embedding-3-small (strong on legal/technical text, low cost)
and stores vectors in a ChromaDB instance running as a Docker service.

Hybrid retrieval is implemented via:
  - dense  : ChromaDB cosine-similarity vector search
  - sparse : BM25 keyword retrieval via rank_bm25
Results are merged with Reciprocal Rank Fusion (RRF).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import chromadb
from chromadb.config import Settings
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

import config

logger = logging.getLogger(__name__)

# ─── Embeddings ──────────────────────────────────────────────────────────────


def get_embeddings() -> OpenAIEmbeddings:
    """
    Return an OpenAI embedding model.

    Chosen model: text-embedding-3-small
    Justification: strong semantic alignment on legal/technical language,
    1536-dimensional output, cost-effective for this PoC scale, officially
    supported by OpenAI and LangChain.
    """
    return OpenAIEmbeddings(
        model=config.OPENAI_EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
    )


# ─── ChromaDB client ─────────────────────────────────────────────────────────


def get_chroma_client() -> chromadb.HttpClient:
    return chromadb.HttpClient(
        host=config.CHROMA_HOST,
        port=config.CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False),
    )


# ─── Indexing ────────────────────────────────────────────────────────────────


def _upsert_collection(
    client: chromadb.HttpClient,
    collection_name: str,
    docs: List[Document],
    embeddings_model: OpenAIEmbeddings,
    batch_size: int = 100,
) -> None:
    """Embed *docs* in batches and upsert into ChromaDB collection."""
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    total = len(docs)
    for start in range(0, total, batch_size):
        batch = docs[start : start + batch_size]
        texts = [d.page_content for d in batch]
        metas = [d.metadata for d in batch]
        ids = [f"{collection_name}_{start + i}" for i in range(len(batch))]

        vectors = embeddings_model.embed_documents(texts)

        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metas,
        )
        logger.info(
            "Upserted batch %d–%d into '%s'",
            start,
            start + len(batch) - 1,
            collection_name,
        )


def index_documents(
    nist_docs: List[Document],
    eu_docs: List[Document],
) -> None:
    """Index both document sets into their respective ChromaDB collections."""
    client = get_chroma_client()
    emb = get_embeddings()

    logger.info("Indexing %d NIST chunks …", len(nist_docs))
    _upsert_collection(client, config.CHROMA_COLLECTION_NIST, nist_docs, emb)

    logger.info("Indexing %d EU AI Act chunks …", len(eu_docs))
    _upsert_collection(client, config.CHROMA_COLLECTION_EUAI, eu_docs, emb)

    logger.info("Indexing complete.")


# ─── Hybrid Retrieval (RRF) ──────────────────────────────────────────────────


def _reciprocal_rank_fusion(
    ranked_lists: List[List[str]], k: int = 60
) -> List[Tuple[str, float]]:
    """
    Merge multiple ranked lists via Reciprocal Rank Fusion.

    Returns a sorted list of (doc_id, rrf_score) tuples, highest score first.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """
    Combines ChromaDB dense retrieval with BM25 sparse retrieval via RRF.

    The BM25 index is built lazily from the same chunks stored in ChromaDB
    so no extra storage is needed.
    """

    def __init__(
        self,
        collection_name: str,
        embeddings_model: OpenAIEmbeddings | None = None,
        top_k: int = 10,
    ) -> None:
        self.collection_name = collection_name
        self.top_k = top_k
        self._emb = embeddings_model or get_embeddings()
        self._client = get_chroma_client()
        self._collection = self._client.get_collection(collection_name)
        self._bm25: BM25Okapi | None = None
        self._corpus_ids: list[str] = []
        self._corpus_texts: list[str] = []
        self._corpus_metas: list[dict] = []

    def _build_bm25_index(self) -> None:
        """Fetch all documents from ChromaDB and build a BM25 index."""
        if self._bm25 is not None:
            return
        logger.info("Building BM25 index for '%s'…", self.collection_name)
        result = self._collection.get(include=["documents", "metadatas"])
        self._corpus_ids = result["ids"]
        self._corpus_texts = result["documents"]
        self._corpus_metas = result["metadatas"]
        tokenized = [t.lower().split() for t in self._corpus_texts]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built (%d docs).", len(self._corpus_ids))

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: dense (ChromaDB) + sparse (BM25) fused with RRF.

        Returns a list of result dicts including text, metadata, and score.
        """
        self._build_bm25_index()

        # Dense retrieval
        query_vec = self._emb.embed_query(query)
        dense_result = self._collection.query(
            query_embeddings=[query_vec],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )
        dense_ids: list[str] = dense_result["ids"][0]
        dense_docs = dense_result["documents"][0]
        dense_metas = dense_result["metadatas"][0]
        # ChromaDB returns L2 or cosine distance; convert to similarity
        dense_distances = dense_result["distances"][0]

        # Sparse retrieval via BM25
        tokenized_query = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_query)
        top_bm25_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[: self.top_k]
        sparse_ids = [self._corpus_ids[i] for i in top_bm25_indices]

        # RRF merge
        rrf_ranked = _reciprocal_rank_fusion([dense_ids, sparse_ids])

        # Build an id → data lookup from both sources
        id_to_data: dict[str, dict] = {}
        for idx, doc_id in enumerate(dense_ids):
            id_to_data[doc_id] = {
                "text": dense_docs[idx],
                "metadata": dense_metas[idx],
                "dense_score": 1.0 - dense_distances[idx],  # cosine sim
            }
        for i in top_bm25_indices:
            did = self._corpus_ids[i]
            if did not in id_to_data:
                id_to_data[did] = {
                    "text": self._corpus_texts[i],
                    "metadata": self._corpus_metas[i],
                    "dense_score": 0.0,
                }
            id_to_data[did]["bm25_score"] = float(bm25_scores[i])

        results: list[dict] = []
        for doc_id, rrf_score in rrf_ranked[: self.top_k]:
            if doc_id not in id_to_data:
                continue
            data = id_to_data[doc_id]
            results.append(
                {
                    "text": data["text"],
                    "metadata": data["metadata"],
                    "score": round(rrf_score, 6),
                    "dense_score": round(data.get("dense_score", 0.0), 6),
                    "bm25_score": round(data.get("bm25_score", 0.0), 6),
                }
            )

        return results
