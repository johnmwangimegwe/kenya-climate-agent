"""
fusion.py
=========

The synthesis stage of the Kenya Climate Risk agent.

Responsibilities
----------------
1. Receive the consolidated outputs from all tools:
   - search_knowledge        (RAG passages with sources)
   - get_earth_engine_layers (flood / rainfall / elevation signals)
   - get_population_exposure  (people exposed per county)
   - compute_risk_score       (ranked, transparent risk scores)
2. Build a grounding context and send it to Gemini 3.5 Flash.
3. Produce a final answer containing:
   - ranked counties
   - explanations grounded in the data and retrieved documents
   - confidence indicators
   - actionable recommendations
4. Degrade gracefully: if the model is unavailable, compose a deterministic
   answer directly from the collected data so the agent still responds.

This is the "grounded retrieval" payoff from the I/O 2026 pattern: the answer
is constructed from satellite signals AND retrieved Kenyan documents, with
explicit source attribution, rather than from the model's parametric memory.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "The 'google-genai' package is required for fusion.py. "
        "Install it with: pip install google-genai"
    ) from exc


# ---------------------------------------------------------------------------
# Configuration helpers (kept local so this module is self-contained)
# ---------------------------------------------------------------------------
def _get_model_name(config: dict[str, Any] | None) -> str:
    """Resolve the Gemini model name from config, env, or a default."""
    if config:
        model = config.get("model") or config.get("gemini", {}).get("model")
        if model:
            return str(model)
    return os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")


def _get_api_key() -> str:
    """Read the Gemini API key, raising a clear error if missing."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Add it to your environment or .env file."
        )
    return api_key


# ---------------------------------------------------------------------------
# Context extraction from collected tool outputs
# ---------------------------------------------------------------------------
def _as_single(value: Any) -> Any:
    """If a tool ran multiple times the orchestrator stores a list; take last."""
    if isinstance(value, list) and value:
        return value[-1]
    return value


def _extract_knowledge(collected: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull retrieved passages (with sources) from the knowledge tool output."""
    block = _as_single(collected.get("search_knowledge"))
    if not isinstance(block, dict):
        return []
    passages = block.get("passages", [])
    if not isinstance(passages, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for p in passages:
        if isinstance(p, dict) and p.get("text"):
            cleaned.append(
                {
                    "text": str(p.get("text", "")).strip(),
                    "source": str(p.get("source", "unknown")),
                    "score": p.get("score"),
                }
            )
    return cleaned


def _extract_ranking(collected: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull the ranked county risk scores from the risk-scoring tool output."""
    block = _as_single(collected.get("compute_risk_score"))
    if not isinstance(block, dict):
        return []
    ranking = block.get("ranking", [])
    if not isinstance(ranking, list):
        return []
    return [r for r in ranking if isinstance(r, dict)]


def _extract_population(collected: dict[str, Any]) -> dict[str, Any]:
    """Pull per-county exposed population from the population tool output."""
    block = _as_single(collected.get("get_population_exposure"))
    if not isinstance(block, dict):
        return {}
    counties = block.get("counties", {})
    return counties if isinstance(counties, dict) else {}


def _extract_layers(collected: dict[str, Any]) -> dict[str, Any]:
    """Pull per-county geophysical layer values from the Earth Engine output."""
    block = _as_single(collected.get("get_earth_engine_layers"))
    if not isinstance(block, dict):
        return {}
    counties = block.get("counties", {})
    return counties if isinstance(counties, dict) else {}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
FUSION_SYSTEM_PROMPT = (
    "You are the synthesis component of an AI agent that assesses climate risk "
    "across the 47 counties of Kenya. You are given structured data gathered by "
    "tools: retrieved Kenyan documents (RAG), satellite-derived layers, "
    "population exposure, and a transparent risk ranking.\n\n"
    "Write a clear, grounded answer to the user's question. Rules:\n"
    "1. Base every factual claim on the provided data or retrieved passages. "
    "Do NOT invent numbers or county names.\n"
    "2. When you use a retrieved passage, attribute it by its source "
    "(e.g. 'according to <source>').\n"
    "3. Lead with the ranked counties most relevant to the question.\n"
    "4. Briefly explain WHY each top county ranks where it does, referencing "
    "the contributing factors (flood signal, rainfall, population, elevation).\n"
    "5. State a confidence level (high / medium / low) based on how complete "
    "the data is. If a data source was missing, say so plainly.\n"
    "6. End with 2-4 concrete, actionable recommendations for decision-makers.\n"
    "7. Be concise and direct. Use short paragraphs or a short list. Avoid "
    "filler.\n"
)


def _build_context_block(
    question: str,
    knowledge: list[dict[str, Any]],
    ranking: list[dict[str, Any]],
    population: dict[str, Any],
    layers: dict[str, Any],
) -> str:
    """Render all collected data into a compact text context for the model."""
    parts: list[str] = [f"USER QUESTION:\n{question}\n"]

    if ranking:
        parts.append(
            "RANKED RISK SCORES (per county, higher = more at risk):\n"
            + json.dumps(ranking, indent=2)
        )
    else:
        parts.append("RANKED RISK SCORES: (none computed)")

    if population:
        parts.append(
            "POPULATION EXPOSURE (per county):\n" + json.dumps(population, indent=2)
        )
    else:
        parts.append("POPULATION EXPOSURE: (none available)")

    if layers:
        parts.append(
            "SATELLITE / GEOPHYSICAL LAYERS (per county):\n"
            + json.dumps(layers, indent=2)
        )
    else:
        parts.append("SATELLITE / GEOPHYSICAL LAYERS: (none available)")

    if knowledge:
        kb_lines = ["RETRIEVED KENYAN DOCUMENTS (use these for context & cite them):"]
        for i, p in enumerate(knowledge, start=1):
            kb_lines.append(f"[{i}] (source: {p['source']}) {p['text']}")
        parts.append("\n".join(kb_lines))
    else:
        parts.append("RETRIEVED KENYAN DOCUMENTS: (none retrieved)")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Deterministic fallback answer (no model required)
# ---------------------------------------------------------------------------
def _deterministic_answer(
    question: str,
    knowledge: list[dict[str, Any]],
    ranking: list[dict[str, Any]],
    population: dict[str, Any],
) -> str:
    """
    Compose a readable answer directly from collected data, used when Gemini
    is unavailable. Deterministic and citation-aware so the agent still works
    offline or under API failure.
    """
    lines: list[str] = []

    if ranking:
        top = ranking[: min(5, len(ranking))]
        lines.append("Counties ranked by climate-risk score (highest first):")
        for entry in top:
            county = entry.get("county", "Unknown")
            score = entry.get("risk_score")
            pop = population.get(county, {})
            exposed = (
                pop.get("exposed_population")
                if isinstance(pop, dict)
                else None
            )
            piece = f"- {county}: risk score {score}"
            if exposed is not None:
                piece += f", ~{int(exposed):,} people exposed"
            lines.append(piece)
    else:
        lines.append(
            "No ranked risk scores were produced, so a ranking is unavailable."
        )

    if knowledge:
        lines.append("\nSupporting context from Kenyan sources:")
        for p in knowledge[:3]:
            snippet = p["text"]
            if len(snippet) > 240:
                snippet = snippet[:240].rstrip() + "..."
            lines.append(f"- ({p['source']}) {snippet}")

    confidence = "medium" if ranking else "low"
    lines.append(f"\nConfidence: {confidence} (composed without the language "
                 "model; based directly on tool outputs).")

    lines.append(
        "\nRecommendation: prioritize the top-ranked counties for monitoring "
        "and pre-positioning of resources, and validate these findings against "
        "the latest county-level field reports."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fuse_answer(
    question: str,
    collected: dict[str, Any],
    config: dict[str, Any] | None = None,
    client: Any | None = None,
) -> str:
    """
    Fuse all collected tool outputs into a final, grounded answer.

    Parameters
    ----------
    question:
        The original plain-English question.
    collected:
        The orchestrator's merged tool outputs, keyed by tool name.
    config:
        Optional configuration dict (expects an optional 'model' key).
    client:
        Optional pre-constructed google-genai Client for reuse.

    Returns
    -------
    str
        The final natural-language answer. Falls back to a deterministic,
        data-derived answer if the model call fails.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")
    if not isinstance(collected, dict):
        raise ValueError("collected must be a dict of tool outputs.")

    question = question.strip()

    knowledge = _extract_knowledge(collected)
    ranking = _extract_ranking(collected)
    population = _extract_population(collected)
    layers = _extract_layers(collected)

    context_block = _build_context_block(
        question, knowledge, ranking, population, layers
    )
    model_name = _get_model_name(config)

    try:
        if client is None:
            client = genai.Client(api_key=_get_api_key())

        response = client.models.generate_content(
            model=model_name,
            contents=context_block,
            config=genai_types.GenerateContentConfig(
                system_instruction=FUSION_SYSTEM_PROMPT,
                temperature=0.3,
            ),
        )

        answer = getattr(response, "text", None)
        if not answer or not answer.strip():
            raise ValueError("Empty response from Gemini fusion stage.")

        logger.info("Fusion produced an answer of %d characters.", len(answer))
        return answer.strip()

    except Exception as exc:
        logger.error("Fusion model call failed (%s); using deterministic answer.", exc)
        return _deterministic_answer(question, knowledge, ranking, population)


if __name__ == "__main__":
    # Manual smoke test with synthetic collected data (no services needed
    # unless GOOGLE_API_KEY is set, in which case the real model is used).
    logging.basicConfig(level=logging.INFO)
    demo_collected = {
        "search_knowledge": {
            "passages": [
                {
                    "text": "Tana River and Garissa counties experience recurrent "
                            "riverine flooding during the long rains.",
                    "source": "kenya_met_flood_notes.md",
                    "score": 0.88,
                }
            ]
        },
        "get_earth_engine_layers": {
            "counties": {
                "Tana River": {"flood": 0.82, "rainfall": 0.70, "elevation": 0.20},
                "Garissa": {"flood": 0.75, "rainfall": 0.66, "elevation": 0.25},
            }
        },
        "get_population_exposure": {
            "counties": {
                "Tana River": {"exposed_population": 120000},
                "Garissa": {"exposed_population": 95000},
            }
        },
        "compute_risk_score": {
            "ranking": [
                {"county": "Tana River", "risk_score": 0.74},
                {"county": "Garissa", "risk_score": 0.68},
            ]
        },
    }
    print(
        fuse_answer(
            "Which counties face the highest flood risk and who is most exposed?",
            demo_collected,
        )
    )
