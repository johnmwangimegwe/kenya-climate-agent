"""
orchestrator.py
===============

The execution stage of the Kenya Climate Risk agent.

Responsibilities
----------------
1. Take a validated plan from planner.create_plan().
2. Execute each step by dispatching to the correct tool.
3. Handle individual tool failures without aborting the whole run.
4. Log execution for transparency (surfaced in the Streamlit UI).
5. Collect all tool outputs.
6. Pass the consolidated context to the fusion layer.
7. Return a final, structured answer.

Design notes
------------
- Each tool is a callable: ``fn(parameters: dict) -> dict``. This uniform
  contract is what makes the orchestration a clean "sub-agent" pattern and is
  also WebMCP-friendly (each tool is a self-describing structured function).
- The tool registry maps tool names (as used by the planner) to callables.
  A default registry is built lazily from the project's tool modules, but a
  custom registry can be injected for testing or for swapping implementations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .planner import create_plan

logger = logging.getLogger(__name__)

# A tool is any callable taking a params dict and returning a result dict.
Tool = Callable[[dict[str, Any]], dict[str, Any]]


# ---------------------------------------------------------------------------
# Execution records (for logging / UI transparency)
# ---------------------------------------------------------------------------
@dataclass
class StepResult:
    """Outcome of a single executed plan step."""

    tool: str
    reason: str
    parameters: dict[str, Any]
    success: bool
    output: dict[str, Any] | None = None
    error: str | None = None
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging or JSON display."""
        return {
            "tool": self.tool,
            "reason": self.reason,
            "parameters": self.parameters,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 3),
        }


@dataclass
class AgentResult:
    """Full result of an agent run, including the answer and the audit trail."""

    question: str
    answer: str
    plan: dict[str, Any]
    step_results: list[StepResult] = field(default_factory=list)
    collected: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None

    @property
    def execution_log(self) -> list[dict[str, Any]]:
        """Human-readable log of each step (used by the Streamlit UI)."""
        return [sr.to_dict() for sr in self.step_results]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the whole result."""
        return {
            "question": self.question,
            "answer": self.answer,
            "plan": self.plan,
            "execution_log": self.execution_log,
            "collected": self.collected,
            "success": self.success,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Default tool registry
# ---------------------------------------------------------------------------
def build_default_registry(config: dict[str, Any] | None = None) -> dict[str, Tool]:
    """
    Build the default tool registry from the project's tool modules.

    Imports are done inside the function so that importing the orchestrator
    does not require every heavy dependency (Earth Engine, FAISS, ...) to be
    present — useful for unit-testing the planner/orchestrator in isolation.

    Each wrapper adapts the underlying tool function to the uniform
    ``fn(parameters: dict) -> dict`` contract.
    """
    from ..tools import cache, knowledge, risk_score

    def _knowledge_tool(params: dict[str, Any]) -> dict[str, Any]:
        query = params.get("query", "")
        top_k = int(params.get("top_k", 4) or 4)
        return knowledge.search_knowledge(query=query, top_k=top_k)

    def _earth_engine_tool(params: dict[str, Any]) -> dict[str, Any]:
        # Uses the disk cache (data/cache.json) when available, else live EE.
        return cache.cached_earth_engine_layers(
            counties=params.get("counties"),
            start_date=params.get("start_date"),
            end_date=params.get("end_date"),
            config=config,
        )

    def _population_tool(params: dict[str, Any]) -> dict[str, Any]:
        # Uses the disk cache when available, else live WorldPop.
        return cache.cached_population_exposure(
            counties=params.get("counties"),
            config=config,
        )

    def _risk_tool(params: dict[str, Any]) -> dict[str, Any]:
        # Feed the risk scorer the CACHED layers + population so it does not
        # re-query Earth Engine — this is the main speed win for the demo.
        counties = params.get("counties")
        layers = cache.cached_earth_engine_layers(counties=counties, config=config)
        pop = cache.cached_population_exposure(counties=counties, config=config)
        return risk_score.compute_risk_score(
            counties=counties,
            config=config,
            layers=layers,
            population_data=pop,
        )

    return {
        "search_knowledge": _knowledge_tool,
        "get_earth_engine_layers": _earth_engine_tool,
        "get_population_exposure": _population_tool,
        "compute_risk_score": _risk_tool,
    }


# ---------------------------------------------------------------------------
# Core execution
# ---------------------------------------------------------------------------
def _execute_step(
    step: dict[str, Any],
    registry: dict[str, Tool],
) -> StepResult:
    """Execute a single plan step, capturing timing, output and any error."""
    tool_name = step["tool"]
    reason = step.get("reason", "")
    parameters = step.get("parameters", {}) or {}

    tool = registry.get(tool_name)
    if tool is None:
        msg = f"No implementation registered for tool '{tool_name}'."
        logger.error(msg)
        return StepResult(
            tool=tool_name,
            reason=reason,
            parameters=parameters,
            success=False,
            error=msg,
        )

    start = time.perf_counter()
    try:
        output = tool(parameters)
        if not isinstance(output, dict):
            output = {"result": output}
        duration = time.perf_counter() - start
        logger.info("Tool '%s' completed in %.2fs.", tool_name, duration)
        return StepResult(
            tool=tool_name,
            reason=reason,
            parameters=parameters,
            success=True,
            output=output,
            duration_seconds=duration,
        )
    except Exception as exc:  # a single tool failing must not abort the run
        duration = time.perf_counter() - start
        logger.exception("Tool '%s' failed: %s", tool_name, exc)
        return StepResult(
            tool=tool_name,
            reason=reason,
            parameters=parameters,
            success=False,
            error=str(exc),
            duration_seconds=duration,
        )


def _collect_outputs(step_results: list[StepResult]) -> dict[str, Any]:
    """
    Merge successful step outputs into a single context dict keyed by tool.

    If a tool runs more than once, later results are stored in a list so no
    data is silently lost.
    """
    collected: dict[str, Any] = {}
    for sr in step_results:
        if not sr.success or sr.output is None:
            continue
        if sr.tool in collected:
            existing = collected[sr.tool]
            if isinstance(existing, list):
                existing.append(sr.output)
            else:
                collected[sr.tool] = [existing, sr.output]
        else:
            collected[sr.tool] = sr.output
    return collected


def run_agent(
    question: str,
    config: dict[str, Any] | None = None,
    registry: dict[str, Tool] | None = None,
    client: Any | None = None,
) -> AgentResult:
    """
    Run the full agent loop for a question and return a structured result.

    Parameters
    ----------
    question:
        The plain-English question.
    config:
        Optional configuration dict shared with planner, tools and fusion.
    registry:
        Optional tool registry. Defaults to build_default_registry(config).
    client:
        Optional google-genai client reused for both planning and fusion.

    Returns
    -------
    AgentResult
        Contains the final answer, the plan, per-step logs and collected data.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")
    question = question.strip()

    # 1. Plan -----------------------------------------------------------------
    try:
        plan = create_plan(question, config=config, client=client)
    except Exception as exc:
        logger.error("Planning stage failed irrecoverably: %s", exc)
        return AgentResult(
            question=question,
            answer=(
                "I could not build a plan to answer that question. "
                "Please rephrase and try again."
            ),
            plan={"steps": []},
            success=False,
            error=str(exc),
        )

    # 2. Build / accept registry ---------------------------------------------
    if registry is None:
        try:
            registry = build_default_registry(config)
        except Exception as exc:
            logger.error("Failed to build tool registry: %s", exc)
            return AgentResult(
                question=question,
                answer=(
                    "The analysis tools are unavailable in this environment. "
                    "Check that Earth Engine and dependencies are installed."
                ),
                plan=plan,
                success=False,
                error=str(exc),
            )

    # 3. Execute steps --------------------------------------------------------
    step_results: list[StepResult] = []
    for step in plan["steps"]:
        result = _execute_step(step, registry)
        step_results.append(result)

    collected = _collect_outputs(step_results)
    any_success = any(sr.success for sr in step_results)

    # 4. Fuse -----------------------------------------------------------------
    # Imported here to avoid a circular import at module load time.
    from .fusion import fuse_answer

    if not collected:
        logger.warning("No tool produced usable output; answering defensively.")
        answer = (
            "I was unable to gather the data needed to answer that question. "
            "Every analysis step failed — please check Earth Engine "
            "authentication, the knowledge index, and your network connection."
        )
        return AgentResult(
            question=question,
            answer=answer,
            plan=plan,
            step_results=step_results,
            collected=collected,
            success=False,
            error="All tool steps failed.",
        )

    try:
        answer = fuse_answer(
            question=question,
            collected=collected,
            config=config,
            client=client,
        )
    except Exception as exc:
        logger.error("Fusion stage failed: %s", exc)
        answer = (
            "I gathered the data but could not compose a final summary. "
            "Raw results are available in the execution log below."
        )
        return AgentResult(
            question=question,
            answer=answer,
            plan=plan,
            step_results=step_results,
            collected=collected,
            success=False,
            error=str(exc),
        )

    return AgentResult(
        question=question,
        answer=answer,
        plan=plan,
        step_results=step_results,
        collected=collected,
        success=any_success,
        error=None if any_success else "All tool steps failed.",
    )


if __name__ == "__main__":
    # Manual smoke test with a stub registry (no external services needed).
    logging.basicConfig(level=logging.INFO)

    def _stub_knowledge(params: dict[str, Any]) -> dict[str, Any]:
        return {"passages": [{"text": "Tana River County floods seasonally.",
                              "source": "demo.md", "score": 0.9}]}

    def _stub_layers(params: dict[str, Any]) -> dict[str, Any]:
        return {"counties": {"Tana River": {"flood": 0.8, "rainfall": 0.7,
                                            "elevation": 0.2}}}

    def _stub_pop(params: dict[str, Any]) -> dict[str, Any]:
        return {"counties": {"Tana River": {"exposed_population": 120000}}}

    def _stub_risk(params: dict[str, Any]) -> dict[str, Any]:
        return {"ranking": [{"county": "Tana River", "risk_score": 0.74}]}

    stub_registry: dict[str, Tool] = {
        "search_knowledge": _stub_knowledge,
        "get_earth_engine_layers": _stub_layers,
        "get_population_exposure": _stub_pop,
        "compute_risk_score": _stub_risk,
    }

    # Use the fallback plan path by forcing a stub registry; planner still runs
    # if GOOGLE_API_KEY is present, otherwise it falls back automatically.
    res = run_agent(
        "Which counties face the highest flood risk?",
        registry=stub_registry,
    )
    import json

    print(json.dumps(res.to_dict(), indent=2))