"""
retriever.py
============

Query-time half of the RAG pipeline for the Kenya Climate Risk agent.

Responsibilities
----------------
1. Load the persisted index built by build_index.py (passages + embeddings,
   and a FAISS index when available).
2. Embed an incoming query using the SAME mode the index was built with
   (Gemini embeddings or TF-IDF), so query and passage vectors are comparable.
3. Return the top-k most similar passages, each with a similarity score and
   source attribution.

Public function
---------------
- retrieve(query, top_k, config) -> list[dict]

Each returned item:
    {"text": <passage>, "source": <filename>, "score": <float 0..1>}

This is consumed by tools/knowledge.py, which prefers this retriever and only
falls back to keyword search if loading or retrieval fails.

The index is loaded once and cached, so repeated queries are fast.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False

try:
    from google import genai

    _GENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    genai = None  # type: ignore
    _GENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


def _index_dir(config: dict[str, Any] | None) -> str:
    """Resolve the index directory (must match build_index.index_dir)."""
    if config:
        configured = config.get("rag", {}).get("index_dir")
        if configured:
            return configured if os.path.isabs(configured) else os.path.join(
                _project_root(), configured
            )
    return os.path.join(_project_root(), "rag_index")


# ---------------------------------------------------------------------------
# Index loading (cached)
# ---------------------------------------------------------------------------
class _LoadedIndex:
    """Container for a loaded index and everything needed to query it."""

    def __init__(
        self,
        passages: list[dict[str, Any]],
        embeddings: np.ndarray,
        meta: dict[str, Any],
        faiss_index: Any | None,
    ) -> None:
        self.passages = passages
        self.embeddings = embeddings
        self.meta = meta
        self.faiss_index = faiss_index


@lru_cache(maxsize=2)
def _load_index(index_path: str) -> _LoadedIndex | None:
    """
    Load passages, embeddings, meta and (optionally) the FAISS index from disk.

    Cached by path so repeated retrievals don't re-read files. Returns None if
    the index is missing or corrupt (caller then falls back to keyword search).
    """
    passages_path = os.path.join(index_path, "passages.json")
    embeddings_path = os.path.join(index_path, "embeddings.npy")
    meta_path = os.path.join(index_path, "meta.json")

    if not (os.path.exists(passages_path) and os.path.exists(embeddings_path)):
        logger.info("No RAG index found at %s.", index_path)
        return None

    try:
        with open(passages_path, "r", encoding="utf-8") as fh:
            passages = json.load(fh)
        embeddings = np.load(embeddings_path).astype("float32")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)

        faiss_index = None
        faiss_path = os.path.join(index_path, "faiss.index")
        if _FAISS_AVAILABLE and meta.get("faiss") and os.path.exists(faiss_path):
            try:
                faiss_index = faiss.read_index(faiss_path)
            except Exception as exc:
                logger.warning("Failed to read FAISS index (%s); using NumPy.", exc)
                faiss_index = None

        return _LoadedIndex(passages, embeddings, meta, faiss_index)
    except Exception as exc:
        logger.warning("Failed to load RAG index at %s: %s", index_path, exc)
        return None


# ---------------------------------------------------------------------------
# Query embedding (must mirror build_index)
# ---------------------------------------------------------------------------
def _l2_normalize_vector(vector: np.ndarray) -> np.ndarray:
    """L2-normalize a single vector."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.astype("float32")
    return (vector / norm).astype("float32")


def _embed_query_gemini(
    query: str, model: str, config: dict[str, Any] | None
) -> np.ndarray | None:
    """Embed a query with the Gemini embeddings API; None on failure."""
    if not _GENAI_AVAILABLE:
        return None
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.embed_content(model=model, contents=[query])
        embeddings = getattr(response, "embeddings", None)
        if not embeddings:
            return None
        values = getattr(embeddings[0], "values", None) or embeddings[0]
        return _l2_normalize_vector(np.array(list(values), dtype="float32"))
    except Exception as exc:
        logger.warning("Gemini query embedding failed: %s", exc)
        return None


def _embed_query_tfidf(query: str, meta: dict[str, Any]) -> np.ndarray | None:
    """
    Embed a query using the TF-IDF vocabulary/idf stored at build time.

    Reconstructs the same vector space so cosine similarity is valid.
    """
    vocabulary = meta.get("vocabulary")
    idf = meta.get("idf")
    if not vocabulary or not idf:
        return None
    try:
        import re
        from collections import Counter

        dim = len(idf)
        vector = np.zeros(dim, dtype="float32")
        tokens = re.findall(r"[a-z0-9']+", query.lower())
        counts = Counter(t for t in tokens if t in vocabulary)
        if not counts:
            return vector  # zero vector -> no matches
        max_count = max(counts.values())
        for term, count in counts.items():
            idx = vocabulary[term]
            tf = count / max_count
            vector[idx] = tf * idf[idx]
        return _l2_normalize_vector(vector)
    except Exception as exc:
        logger.warning("TF-IDF query embedding failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------
def _search(
    loaded: _LoadedIndex, query_vec: np.ndarray, top_k: int
) -> list[tuple[int, float]]:
    """Return [(passage_index, score)] for the top_k matches."""
    if loaded.faiss_index is not None:
        try:
            scores, indices = loaded.faiss_index.search(
                query_vec.reshape(1, -1), top_k
            )
            return [
                (int(i), float(s))
                for i, s in zip(indices[0], scores[0])
                if i >= 0
            ]
        except Exception as exc:
            logger.debug("FAISS search failed (%s); using NumPy.", exc)

    # NumPy cosine similarity (vectors are already L2-normalized).
    sims = loaded.embeddings @ query_vec
    if sims.size == 0:
        return []
    top_idx = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def retrieve(
    query: str,
    top_k: int = 4,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve the top_k most relevant passages for a query.

    Parameters
    ----------
    query:
        Natural-language query string.
    top_k:
        Number of passages to return (minimum 1).
    config:
        Optional configuration dict (rag.index_dir, rag.embedding_model).

    Returns
    -------
    list[dict]
        Each {"text", "source", "score"}; empty list if the index is missing
        or the query cannot be embedded (caller then falls back to keyword).
    """
    if not isinstance(query, str) or not query.strip():
        return []
    top_k = max(1, int(top_k)) if isinstance(top_k, int) else 4

    loaded = _load_index(_index_dir(config))
    if loaded is None or not loaded.passages:
        return []

    mode = loaded.meta.get("mode", "tfidf")
    if mode == "gemini":
        model = loaded.meta.get("model", "gemini-embedding-001")
        query_vec = _embed_query_gemini(query, model, config)
        if query_vec is None:
            logger.info("Gemini query embedding unavailable; cannot use this index.")
            return []
    else:
        query_vec = _embed_query_tfidf(query, loaded.meta)
        if query_vec is None:
            return []

    # Guard against dimension mismatch between query and index.
    if query_vec.shape[0] != loaded.embeddings.shape[1]:
        logger.warning(
            "Query/index dimension mismatch (%d vs %d).",
            query_vec.shape[0], loaded.embeddings.shape[1],
        )
        return []

    matches = _search(loaded, query_vec, top_k)

    results: list[dict[str, Any]] = []
    for idx, score in matches:
        if idx < 0 or idx >= len(loaded.passages):
            continue
        passage = loaded.passages[idx]
        results.append(
            {
                "text": passage.get("text", ""),
                "source": passage.get("source", "unknown"),
                "score": round(max(0.0, min(1.0, float(score))), 4),
            }
        )
    return results


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(
        retrieve("which counties flood during the long rains", top_k=3),
        indent=2,
    ))
