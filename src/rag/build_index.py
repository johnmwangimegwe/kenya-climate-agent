"""
build_index.py
==============

Builds the RAG vector index for the Kenya Climate Risk agent.

Pipeline
--------
1. Load every document under knowledge/ (.md, .txt, .pdf).
2. Split each document into overlapping passages, keeping source + chunk id.
3. Embed every passage:
     - primary:  Gemini embeddings (google-genai) when GOOGLE_API_KEY is set
     - fallback: scikit-learn TF-IDF vectors when offline / no key
4. Build a FAISS index over the embeddings (inner-product on L2-normalized
   vectors == cosine similarity). If FAISS is unavailable, persist a plain
   NumPy matrix instead — retriever.py understands both.
5. Persist the index + passage metadata to disk so retriever.py can load it.

Artifacts written to the index directory (default: rag_index/ at project root):
    - passages.json   : list of {id, text, source, chunk}
    - embeddings.npy  : float32 matrix [n_passages, dim] (L2-normalized)
    - faiss.index     : FAISS index (only if faiss is installed)
    - meta.json       : {mode: 'gemini'|'tfidf', dim, count, model, vocab?}

This is the "ingest" half of grounded retrieval. Run it once (or whenever the
knowledge base changes). The agent's knowledge.py prefers this index and only
falls back to keyword search if it is missing.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Optional dependencies handled gracefully.
try:
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False

try:
    from google import genai
    from google.genai import types as genai_types  # noqa: F401

    _GENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    genai = None  # type: ignore
    _GENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _project_root() -> str:
    """Absolute path to the project root (two levels up from src/rag/)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


def knowledge_dir() -> str:
    """Default knowledge directory path."""
    return os.path.join(_project_root(), "knowledge")


def index_dir(config: dict[str, Any] | None = None) -> str:
    """Resolve the index output directory from config or default."""
    if config:
        configured = config.get("rag", {}).get("index_dir")
        if configured:
            return configured if os.path.isabs(configured) else os.path.join(
                _project_root(), configured
            )
    return os.path.join(_project_root(), "rag_index")


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------
def _extract_pdf_text(path: str) -> str:
    """Extract text from a PDF using pypdf if available, else empty string."""
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        logger.warning("pypdf not installed; skipping PDF %s", os.path.basename(path))
        return ""
    try:
        reader = PdfReader(path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:
        logger.warning("PDF extraction failed for %s: %s", path, exc)
        return ""


def load_documents(directory: str) -> list[tuple[str, str]]:
    """
    Load all text-like documents under a directory.

    Returns a list of (source_filename, full_text) tuples.
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
                        text = fh.read()
                elif lower.endswith(".pdf"):
                    text = _extract_pdf_text(path)
                else:
                    continue
                if text and text.strip():
                    documents.append((fname, text))
            except Exception as exc:
                logger.warning("Could not read %s: %s", path, exc)
    logger.info("Loaded %d documents from %s.", len(documents), directory)
    return documents


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(
    text: str, chunk_size: int = 600, overlap: int = 100
) -> list[str]:
    """
    Split text into overlapping, sentence-aware chunks of ~chunk_size chars.

    Splits first on blank lines, then packs sentences into chunks, adding a
    character overlap between consecutive chunks to preserve context across
    boundaries.
    """
    # Normalize whitespace within paragraphs, keep paragraph breaks.
    paragraphs = [
        " ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()
    ]
    if not paragraphs:
        return []

    # Sentence-ish split for finer packing.
    sentences: list[str] = []
    for para in paragraphs:
        sentences.extend(re.split(r"(?<=[.!?])\s+", para))

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            # Start new chunk with an overlap tail from the previous chunk.
            if overlap > 0 and chunks:
                tail = chunks[-1][-overlap:]
                current = f"{tail} {sentence}".strip()
            else:
                current = sentence
    if current:
        chunks.append(current)
    return chunks


def build_passages(
    documents: list[tuple[str, str]],
    chunk_size: int = 600,
    overlap: int = 100,
) -> list[dict[str, Any]]:
    """Turn loaded documents into a flat list of passage records."""
    passages: list[dict[str, Any]] = []
    pid = 0
    for source, text in documents:
        for chunk_index, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
            passages.append(
                {"id": pid, "text": chunk, "source": source, "chunk": chunk_index}
            )
            pid += 1
    logger.info("Built %d passages.", len(passages))
    return passages


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows so inner product equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype("float32")


def _embed_with_gemini(
    texts: list[str], model: str, client: Any
) -> np.ndarray | None:
    """Embed texts with the Gemini embeddings API; None on failure."""
    try:
        vectors: list[list[float]] = []
        # Batch to stay within request limits.
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = client.models.embed_content(model=model, contents=batch)
            # google-genai returns an object with `.embeddings` (list of values).
            embeddings = getattr(response, "embeddings", None)
            if embeddings is None:
                return None
            for emb in embeddings:
                values = getattr(emb, "values", None) or emb
                vectors.append(list(values))
        return _l2_normalize(np.array(vectors, dtype="float32"))
    except Exception as exc:
        logger.warning("Gemini embedding failed (%s); will fall back to TF-IDF.", exc)
        return None


def _embed_with_tfidf(texts: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Embed texts with scikit-learn TF-IDF as an offline fallback.

    Returns the L2-normalized dense matrix and a meta dict containing the
    fitted vocabulary + idf so the retriever can transform queries identically.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(stop_words="english", max_features=4096)
    matrix = vectorizer.fit_transform(texts).toarray().astype("float32")
    matrix = _l2_normalize(matrix)
    vocab = {term: int(idx) for term, idx in vectorizer.vocabulary_.items()}
    meta = {
        "vocabulary": vocab,
        "idf": vectorizer.idf_.tolist(),
    }
    return matrix, meta


def _get_embedding_model(config: dict[str, Any] | None) -> str:
    """Resolve the Gemini embedding model name."""
    if config:
        configured = config.get("rag", {}).get("embedding_model")
        if configured:
            return str(configured)
    return os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _persist(
    out_dir: str,
    passages: list[dict[str, Any]],
    embeddings: np.ndarray,
    meta: dict[str, Any],
) -> None:
    """Write passages, embeddings, FAISS index (if available) and meta to disk."""
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "passages.json"), "w", encoding="utf-8") as fh:
        json.dump(passages, fh, ensure_ascii=False, indent=2)

    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)

    if _FAISS_AVAILABLE and embeddings.size:
        try:
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
            faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
            meta["faiss"] = True
        except Exception as exc:
            logger.warning("Failed to write FAISS index (%s); NumPy matrix only.", exc)
            meta["faiss"] = False
    else:
        meta["faiss"] = False

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    logger.info("Index persisted to %s (mode=%s, count=%d).",
                out_dir, meta.get("mode"), meta.get("count"))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_index(
    source_dir: str | None = None,
    config: dict[str, Any] | None = None,
    chunk_size: int = 600,
    overlap: int = 100,
) -> dict[str, Any]:
    """
    Build and persist the RAG index from the knowledge directory.

    Parameters
    ----------
    source_dir:
        Directory of knowledge documents. Defaults to knowledge/.
    config:
        Optional configuration dict (rag.index_dir, rag.embedding_model).
    chunk_size, overlap:
        Chunking parameters.

    Returns
    -------
    dict
        Summary: {mode, count, dim, index_dir}. Raises ValueError if no
        documents/passages were found.
    """
    source_dir = source_dir or knowledge_dir()
    out_dir = index_dir(config)

    documents = load_documents(source_dir)
    if not documents:
        raise ValueError(
            f"No documents found in {source_dir!r}. Add .md/.txt/.pdf files first."
        )

    passages = build_passages(documents, chunk_size, overlap)
    if not passages:
        raise ValueError("Documents produced no passages after chunking.")

    texts = [p["text"] for p in passages]

    embeddings: np.ndarray | None = None
    mode = "tfidf"
    model_name = _get_embedding_model(config)
    extra_meta: dict[str, Any] = {}

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if _GENAI_AVAILABLE and api_key:
        try:
            client = genai.Client(api_key=api_key)
            embeddings = _embed_with_gemini(texts, model_name, client)
            if embeddings is not None:
                mode = "gemini"
        except Exception as exc:
            logger.warning("Gemini client init failed (%s); using TF-IDF.", exc)

    if embeddings is None:
        embeddings, extra_meta = _embed_with_tfidf(texts)
        mode = "tfidf"

    meta: dict[str, Any] = {
        "mode": mode,
        "dim": int(embeddings.shape[1]) if embeddings.size else 0,
        "count": len(passages),
        "model": model_name if mode == "gemini" else "tfidf",
        **extra_meta,
    }

    _persist(out_dir, passages, embeddings, meta)
    return {
        "mode": mode,
        "count": len(passages),
        "dim": meta["dim"],
        "index_dir": out_dir,
    }


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    summary = build_index()
    print(json.dumps(summary, indent=2))
