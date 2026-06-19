"""
knowledge.py
============

RAG (retrieval-augmented generation) tool for the Kenya Climate Risk agent.

This is the "grounded retrieval" sub-agent. It exposes a single tool function,
``search_knowledge(query, top_k)``, that returns the most relevant passages from
the local Kenyan knowledge base, each with source attribution so the fusion
layer can cite it.

Public function
---------------
- search_knowledge(query, top_k, config) -> dict   (orchestrator contract)

Output contract (consumed by orchestrator/fusion):

    {
        "passages": [
            {"text": "<passage>", "source": "<filename>", "score": <float>},
            ...
        ],
        "meta": { "mode": "faiss" | "keyword", "count": <int> }
    }

Design
------
Primary path: delegate to rag.retriever (FAISS + embeddings) when an index has
been built. Fallback path: if the index is missing or the retriever cannot load
(e.g. FAISS not installed, embeddings unavailable offline), perform a simple but
effective keyword/overlap search directly over the markdown + PDF-derived text
in the knowledge/ directory. Either way the tool returns useful, attributed
passages so the agent is never left without context.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# Common English stop-words excluded from keyword overlap scoring.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "is", "are", "was", "were", "be", "been", "being", "this",
    "that", "these", "those", "it", "its", "as", "from", "which", "who", "what",
    "where", "when", "how", "why", "will", "would", "can", "could", "should",
    "do", "does", "did", "has", "have", "had", "i", "you", "we", "they", "most",
}


def _knowledge_dir() -> str:
    """Resolve the absolute path to the project's knowledge/ directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    return os.path.join(root, "knowledge")


# ---------------------------------------------------------------------------
# Keyword-search fallback over knowledge/ documents
# ---------------------------------------------------------------------------
def _read_text_files(directory: str) -> list[tuple[str, str]]:
    """
    Read all text-like documents under a directory.

    Returns a list of (source_filename, text) tuples. Reads .md/.txt directly;
    attempts PDF text extraction if pypdf is available. Scanned/binary content
    is skipped gracefully.
    """
    documents: list[tuple[str, str]] = []
    if not os.path.isdir(directory):
        logger.warning("Knowledge directory not found: %s", directory)
        return documents

    for root, _dirs, files in os.walk(directory):
        for fname in sorted(files):
            path = os.path.join(root, fname)
            lower = fname.lower()
            try:
                if lower.endswith((".md", ".txt")):
                    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                        documents.append((fname, fh.read()))
                elif lower.endswith(".pdf"):
                    text = _extract_pdf_text(path)
                    if text.strip():
                        documents.append((fname, text))
            except Exception as exc:
                logger.debug("Could not read %s: %s", path, exc)
    return documents


def _extract_pdf_text(path: str) -> str:
    """Extract text from a PDF using pypdf if available; empty string if not."""
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        logger.debug("pypdf not installed; skipping PDF %s", path)
        return ""
    try:
        reader = PdfReader(path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:
        logger.debug("PDF extraction failed for %s: %s", path, exc)
        return ""


def _split_passages(text: str, max_chars: int = 600) -> list[str]:
    """
    Split a document into passages on blank lines, capping passage length.

    Keeps passages reasonably sized so keyword overlap is meaningful and the
    returned context is focused.
    """
    blocks = re.split(r"\n\s*\n", text)
    passages: list[str] = []
    for block in blocks:
        cleaned = " ".join(block.split())
        if not cleaned:
            continue
        if len(cleaned) <= max_chars:
            passages.append(cleaned)
        else:
            # Hard-wrap overly long blocks into max_chars-sized chunks.
            for i in range(0, len(cleaned), max_chars):
                chunk = cleaned[i : i + max_chars].strip()
                if chunk:
                    passages.append(chunk)
    return passages


@lru_cache(maxsize=1)
def _load_passages() -> tuple[tuple[str, str], ...]:
    """Load and cache (source, passage) pairs from the knowledge directory."""
    directory = _knowledge_dir()
    docs = _read_text_files(directory)
    pairs: list[tuple[str, str]] = []
    for source, text in docs:
        for passage in _split_passages(text):
            pairs.append((source, passage))
    logger.info("Loaded %d knowledge passages from %s.", len(pairs), directory)
    return tuple(pairs)


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens with stop-words removed."""
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def _keyword_search(query: str, top_k: int) -> list[dict[str, Any]]:
    """
    Score passages by token overlap with the query and return the top_k.

    A lightweight Jaccard-style overlap: robust, dependency-free, and good
    enough for a small curated knowledge base when embeddings are unavailable.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scored: list[tuple[float, str, str]] = []
    for source, passage in _load_passages():
        passage_tokens = _tokenize(passage)
        if not passage_tokens:
            continue
        overlap = query_tokens & passage_tokens
        if not overlap:
            continue
        score = len(overlap) / len(query_tokens | passage_tokens)
        scored.append((score, source, passage))

    scored.sort(key=lambda x: (-x[0], x[1]))
    results: list[dict[str, Any]] = []
    for score, source, passage in scored[: max(1, top_k)]:
        results.append(
            {"text": passage, "source": source, "score": round(float(score), 4)}
        )
    return results


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------
def search_knowledge(
    query: str,
    top_k: int = 4,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Retrieve the most relevant Kenyan knowledge passages for a query.

    Tries the FAISS-backed retriever first; falls back to keyword search over
    the knowledge/ documents if the index or embeddings are unavailable.

    Parameters
    ----------
    query:
        Natural-language search query.
    top_k:
        Number of passages to return (default 4, minimum 1).
    config:
        Optional configuration dict (passed to the retriever).

    Returns
    -------
    dict
        See module docstring for the output contract.
    """
    if not isinstance(query, str) or not query.strip():
        return {"passages": [], "meta": {"mode": "none", "count": 0}}

    query = query.strip()
    top_k = max(1, int(top_k) if isinstance(top_k, (int, float, str)) and str(top_k).isdigit() else 4)

    # Primary path: FAISS retriever.
    try:
        from ..rag import retriever

        passages = retriever.retrieve(query, top_k=top_k, config=config)
        if passages:
            return {
                "passages": passages,
                "meta": {"mode": "faiss", "count": len(passages)},
            }
        logger.info("Retriever returned no results; falling back to keyword search.")
    except Exception as exc:
        logger.info("FAISS retriever unavailable (%s); using keyword search.", exc)

    # Fallback path: keyword search.
    passages = _keyword_search(query, top_k)
    return {"passages": passages, "meta": {"mode": "keyword", "count": len(passages)}}


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    import json

    print(json.dumps(
        search_knowledge("which counties flood during the long rains", top_k=3),
        indent=2,
    ))
