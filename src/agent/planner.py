"""
planner.py
==========

The planning stage of the Kenya Climate Risk agent.

Responsibilities
----------------
1. Receive a plain-English user question.
2. Build a system prompt describing the agent's role and the tools it can use.
3. Call Gemini 3.5 Flash to decompose the question into an ordered plan.
4. Parse and validate the model's JSON response.
5. Recover gracefully from malformed responses (markdown fences, partial JSON).
6. Return a clean, validated execution plan.

The plan format returned is:

    {
        "steps": [
            {
                "tool": "<tool_name>",
                "reason": "<why this tool is called>",
                "parameters": { ... }
            },
            ...
        ]
    }

This mirrors the I/O 2026 agent pattern: Gemini 3.5 Flash acts as the
planner that drives a set of tools (expert sub-agents). The tool registry
passed in is the single source of truth for which tools exist.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gemini SDK import (google-genai). Imported lazily-friendly at module load so
# that a missing dependency produces a clear, early error message.
# ---------------------------------------------------------------------------
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "The 'google-genai' package is required for planner.py. "
        "Install it with: pip install google-genai"
    ) from exc


# ---------------------------------------------------------------------------
# Tool descriptions
# ---------------------------------------------------------------------------
# These human-readable descriptions are injected into the system prompt so the
# planner knows what each tool does and what parameters it accepts. They are
# intentionally kept here (not in the tool modules) so the planner has one
# compact, prompt-friendly catalogue. Tool *names* must match the keys of the
# registry the orchestrator dispatches against.
TOOL_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "search_knowledge": {
        "description": (
            "Retrieve authoritative Kenyan climate context from a local "
            "knowledge base of reports and publications (RAG). Use this for "
            "background facts, historical floods/droughts, county profiles, "
            "or anything the satellite data alone cannot answer."
        ),
        "parameters": {
            "query": "string - the natural-language search query",
            "top_k": "integer (optional) - number of passages to return, default 4",
        },
    },
    "get_earth_engine_layers": {
        "description": (
            "Query Google Earth Engine for geospatial risk layers (flood mask "
            "from Sentinel-1 SAR, elevation from SRTM, rainfall from CHIRPS) "
            "for one or more Kenyan counties. Use this whenever physical/"
            "environmental measurements are needed."
        ),
        "parameters": {
            "counties": "list[str] (optional) - county names; empty means all 47 counties",
            "start_date": "string (optional) - ISO date, e.g. '2026-03-01'",
            "end_date": "string (optional) - ISO date, e.g. '2026-05-31'",
        },
    },
    "get_population_exposure": {
        "description": (
            "Estimate how many people live in the at-risk areas using WorldPop "
            "population data, aggregated by county. Use this to answer 'who is "
            "exposed' or 'how many people are affected'."
        ),
        "parameters": {
            "counties": "list[str] (optional) - county names; empty means all 47 counties",
        },
    },
    "compute_risk_score": {
        "description": (
            "Combine flood, rainfall, population exposure and elevation into a "
            "transparent, explainable climate-risk score per county and rank "
            "them. Use this as the final analytical step before answering."
        ),
        "parameters": {
            "counties": "list[str] (optional) - county names; empty means all 47 counties",
        },
    },
}

# Valid tool names derived from the catalogue above.
VALID_TOOLS: set[str] = set(TOOL_DESCRIPTIONS.keys())


# ---------------------------------------------------------------------------
# Configuration helper
# ---------------------------------------------------------------------------
def _get_model_name(config: dict[str, Any] | None) -> str:
    """
    Resolve the Gemini model name.

    Priority: explicit config value -> GEMINI_MODEL env var -> sensible default.
    """
    if config:
        model = config.get("model") or config.get("gemini", {}).get("model")
        if model:
            return str(model)
    return os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")


def _get_api_key() -> str:
    """Read the Gemini API key from the environment, or raise a clear error."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Add it to your environment or .env file."
        )
    return api_key


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def _build_tool_catalogue() -> str:
    """Render the tool catalogue into a compact text block for the prompt."""
    lines: list[str] = []
    for name, spec in TOOL_DESCRIPTIONS.items():
        lines.append(f"- {name}: {spec['description']}")
        params = spec.get("parameters", {})
        if params:
            param_text = "; ".join(f"{k}: {v}" for k, v in params.items())
            lines.append(f"    parameters -> {param_text}")
    return "\n".join(lines)


def build_system_prompt() -> str:
    """
    Build the system prompt that instructs Gemini how to plan.

    The prompt is deliberately strict about output format because the response
    is machine-parsed. It also encodes the sub-agent reasoning pattern.
    """
    catalogue = _build_tool_catalogue()
    return (
        "You are the planning component of an AI agent that assesses climate "
        "risk (floods, drought, rainfall) across the 47 counties of Kenya.\n\n"
        "Your job is NOT to answer the question. Your job is to produce an "
        "ordered plan of tool calls that will gather everything needed to "
        "answer it. Another component will execute your plan and synthesize "
        "the final answer.\n\n"
        "AVAILABLE TOOLS:\n"
        f"{catalogue}\n\n"
        "PLANNING RULES:\n"
        "1. Use only the tools listed above. Never invent tool names.\n"
        "2. Order steps logically: gather context and raw data first, then "
        "compute_risk_score last when a ranking or risk judgement is needed.\n"
        "3. Call search_knowledge when Kenyan context, history, or "
        "documentation would strengthen the answer.\n"
        "4. Only include steps that are necessary for THIS question. A simple "
        "factual/context question may need only search_knowledge.\n"
        "5. Pass parameters explicitly. If the user names specific counties, "
        "put them in the 'counties' parameter; otherwise omit it to mean all "
        "counties.\n\n"
        "OUTPUT FORMAT (STRICT):\n"
        "Respond with a SINGLE JSON object and nothing else. No markdown, no "
        "code fences, no commentary. The schema is:\n"
        "{\n"
        '  "steps": [\n'
        '    {"tool": "<tool_name>", "reason": "<short reason>", '
        '"parameters": {<key>: <value>}}\n'
        "  ]\n"
        "}\n"
    )


# ---------------------------------------------------------------------------
# JSON extraction / validation
# ---------------------------------------------------------------------------
def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences if the model added them."""
    fence_pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
    match = fence_pattern.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_json_object(text: str) -> str:
    """
    Best-effort extraction of the first balanced top-level JSON object.

    Handles cases where the model wraps the JSON in prose despite instructions.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response.")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("Unbalanced JSON object in model response.")


def _parse_plan_text(raw_text: str) -> dict[str, Any]:
    """
    Parse raw model text into a plan dict, recovering from common issues.

    Tries, in order: direct json.loads -> strip fences -> extract object.
    """
    candidates: list[str] = []
    cleaned = raw_text.strip()
    candidates.append(cleaned)
    candidates.append(_strip_code_fences(cleaned))
    try:
        candidates.append(_extract_json_object(cleaned))
    except ValueError:
        pass

    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as err:
            last_error = err
            continue

    raise ValueError(
        f"Could not parse a valid JSON plan from model output. "
        f"Last error: {last_error}. Raw output: {raw_text[:500]!r}"
    )


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and normalize a parsed plan.

    Ensures:
    - 'steps' exists and is a list.
    - Each step has a valid tool name, a reason, and a dict of parameters.
    - Unknown tools are dropped (with a warning) rather than crashing.

    Returns a cleaned plan. Raises ValueError only if nothing usable remains.
    """
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a JSON object.")

    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Plan must contain a non-empty 'steps' list.")

    cleaned_steps: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            logger.warning("Skipping step %d: not an object.", index)
            continue

        tool = step.get("tool")
        if tool not in VALID_TOOLS:
            logger.warning(
                "Skipping step %d: unknown tool %r (valid: %s).",
                index,
                tool,
                ", ".join(sorted(VALID_TOOLS)),
            )
            continue

        parameters = step.get("parameters", {})
        if not isinstance(parameters, dict):
            logger.warning(
                "Step %d: parameters not a dict; coercing to empty dict.", index
            )
            parameters = {}

        reason = step.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            reason = f"Call {tool} to gather information for the question."

        cleaned_steps.append(
            {"tool": tool, "reason": reason.strip(), "parameters": parameters}
        )

    if not cleaned_steps:
        raise ValueError(
            "Plan contained no valid steps after validation. "
            "All tools referenced were unknown."
        )

    return {"steps": cleaned_steps}


# ---------------------------------------------------------------------------
# Fallback plan
# ---------------------------------------------------------------------------
def _fallback_plan(question: str) -> dict[str, Any]:
    """
    Build a safe default plan used when the model is unavailable or its output
    cannot be salvaged. Covers the most common intent: assess and rank risk.
    """
    logger.warning("Using fallback plan for question: %s", question)
    return {
        "steps": [
            {
                "tool": "search_knowledge",
                "reason": "Retrieve Kenyan climate context relevant to the question.",
                "parameters": {"query": question, "top_k": 4},
            },
            {
                "tool": "get_earth_engine_layers",
                "reason": "Fetch flood, rainfall and elevation layers for all counties.",
                "parameters": {},
            },
            {
                "tool": "get_population_exposure",
                "reason": "Estimate population exposed to the hazard by county.",
                "parameters": {},
            },
            {
                "tool": "compute_risk_score",
                "reason": "Combine signals into a transparent, ranked risk score.",
                "parameters": {},
            },
        ]
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def create_plan(
    question: str,
    config: dict[str, Any] | None = None,
    client: Any | None = None,
    use_fallback_on_error: bool = True,
) -> dict[str, Any]:
    """
    Generate a validated execution plan for a user question.

    Parameters
    ----------
    question:
        The plain-English user question.
    config:
        Optional configuration dict (expects an optional 'model' key).
    client:
        Optional pre-constructed google-genai Client (useful for testing or
        reuse). If None, a client is created from GOOGLE_API_KEY.
    use_fallback_on_error:
        If True, return a safe default plan when planning fails instead of
        raising. If False, re-raise the underlying error.

    Returns
    -------
    dict
        A validated plan: {"steps": [...]}.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")

    question = question.strip()
    model_name = _get_model_name(config)

    try:
        if client is None:
            client = genai.Client(api_key=_get_api_key())

        system_prompt = build_system_prompt()
        response = client.models.generate_content(
            model=model_name,
            contents=question,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

        raw_text = getattr(response, "text", None)
        if not raw_text:
            raise ValueError("Empty response from Gemini planner.")

        parsed = _parse_plan_text(raw_text)
        plan = validate_plan(parsed)
        logger.info("Planner produced %d step(s).", len(plan["steps"]))
        return plan

    except Exception as exc:  # broad: planning must not hard-crash the app
        logger.error("Planning failed: %s", exc)
        if use_fallback_on_error:
            return _fallback_plan(question)
        raise


if __name__ == "__main__":
    # Simple manual smoke test (requires GOOGLE_API_KEY).
    logging.basicConfig(level=logging.INFO)
    demo_question = (
        "Which counties in Kenya face the highest flood risk this season, "
        "and which communities are most exposed?"
    )
    try:
        result = create_plan(demo_question)
    except Exception:  # pragma: no cover
        result = _fallback_plan(demo_question)
    print(json.dumps(result, indent=2))
