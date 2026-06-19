"""
streamlit_app.py
================

Streamlit user interface for the Kenya Climate Risk agent.

Provides:
- a plain-English question input box
- a Run button that invokes the agent (planner -> orchestrator -> fusion)
- the grounded natural-language answer
- a ranked county risk table
- an interactive map (folium) of county risk
- a risk bar chart
- an expandable execution log (per-step plan, tools, timings, errors)
- robust error handling and clear status messages

Run locally:
    streamlit run app/streamlit_app.py

The app degrades gracefully: a missing GOOGLE_API_KEY, missing Earth Engine
authentication, a missing RAG index, or a missing county GeoJSON each produce a
clear message rather than a crash, so the app always responds.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Make the project importable whether run from repo root or app/ directory.
# ---------------------------------------------------------------------------
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_APP_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st  # noqa: E402

# Load environment variables from a .env file if python-dotenv is present.
try:  # noqa: SIM105
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
except Exception:  # pragma: no cover
    pass

from src.agent.orchestrator import run_agent  # noqa: E402
from src.utils import viz  # noqa: E402

logger = logging.getLogger(__name__)

# Example questions to seed the demo.
EXAMPLE_QUESTIONS = [
    "Which counties in Kenya face the highest flood risk this season, and who is most exposed?",
    "What are the most drought-vulnerable counties in northern Kenya?",
    "Compare flood risk between Tana River and Garissa counties.",
    "Which densely populated counties are most exposed to flooding?",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_config() -> dict[str, Any]:
    """Load config.yaml from the project root if available, else return {}."""
    path = os.path.join(_PROJECT_ROOT, "config.yaml")
    if not os.path.exists(path):
        return {}
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to load config.yaml: %s", exc)
        return {}


def ensure_rag_index(config: dict[str, Any]) -> str:
    """
    Ensure a RAG index exists; build it on first run if missing.

    Returns a short human-readable status string for display.
    """
    try:
        from src.rag.build_index import build_index, index_dir

        out_dir = index_dir(config)
        if os.path.exists(os.path.join(out_dir, "passages.json")):
            return "Knowledge index: ready."
        summary = build_index(config=config)
        return (
            f"Knowledge index: built ({summary['count']} passages, "
            f"mode={summary['mode']})."
        )
    except Exception as exc:
        logger.warning("Could not build RAG index: %s", exc)
        return f"Knowledge index: unavailable ({exc}). Using keyword fallback."


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_answer(result: Any) -> None:
    """Render the agent's final answer prominently."""
    st.subheader("Answer")
    if result.answer:
        st.markdown(result.answer)
    else:
        st.info("No answer was produced.")


def render_ranking_table(result: Any) -> Any | None:
    """Render the ranked county risk table; return the ranking list if present."""
    ranking = (
        result.collected.get("compute_risk_score", {}).get("ranking")
        if isinstance(result.collected, dict)
        else None
    )
    if not ranking:
        st.info("No risk ranking was produced for this question.")
        return None

    st.subheader("County risk ranking")
    df = viz.risk_dataframe(ranking)
    if df is not None and not df.empty:
        # Show the most relevant columns first.
        display_cols = [
            c
            for c in [
                "county",
                "risk_score",
                "exposed_population",
                "flood",
                "rainfall",
                "population",
                "elevation",
            ]
            if c in df.columns
        ]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.write(ranking)
    return ranking


def render_map(ranking: list[dict[str, Any]], config: dict[str, Any]) -> None:
    """Render the interactive folium risk map, if it can be built."""
    st.subheader("Risk map")
    try:
        top_n = int(config.get("ui", {}).get("map_top_n", 15)) if config else 15
        risk_map = viz.build_risk_map(ranking, top_n=top_n, config=config)
        if risk_map is None:
            st.info(
                "Map unavailable in this environment "
                "(mapping library or county boundaries missing)."
            )
            return
        # Render the folium map via its HTML representation.
        html = risk_map._repr_html_()  # folium Map supports this
        st.components.v1.html(html, height=520, scrolling=False)
    except Exception as exc:
        logger.warning("Map rendering failed: %s", exc)
        st.info(f"Map could not be rendered: {exc}")


def render_bar_chart(ranking: list[dict[str, Any]], config: dict[str, Any]) -> None:
    """Render the top-N risk bar chart, if matplotlib is available."""
    try:
        top_n = int(config.get("ui", {}).get("chart_top_n", 10)) if config else 10
        fig = viz.build_risk_bar_chart(ranking, top_n=top_n)
        if fig is not None:
            st.subheader("Top counties by risk score")
            st.pyplot(fig)
    except Exception as exc:
        logger.debug("Bar chart rendering failed: %s", exc)


def render_execution_log(result: Any) -> None:
    """Render the per-step execution log inside an expander."""
    with st.expander("Execution log (plan, tools, timings)", expanded=False):
        st.markdown("**Plan**")
        st.json(result.plan)
        st.markdown("**Steps**")
        if not result.step_results:
            st.write("No steps were executed.")
            return
        for i, step in enumerate(result.step_results, start=1):
            status = "✅" if step.success else "❌"
            st.markdown(
                f"{status} **{i}. {step.tool}** "
                f"({step.duration_seconds:.2f}s) — {step.reason}"
            )
            if step.parameters:
                st.caption(f"parameters: {step.parameters}")
            if not step.success and step.error:
                st.error(step.error)


def render_status_sidebar(config: dict[str, Any]) -> None:
    """Show environment/setup status in the sidebar."""
    with st.sidebar:
        st.header("Status")

        has_key = bool(
            os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        )
        st.write(
            "Gemini API key:",
            "✅ detected" if has_key else "⚠️ not set (deterministic fallback)",
        )

        ee_project = (
            config.get("earth_engine", {}).get("project")
            if config
            else None
        ) or os.environ.get("EE_PROJECT_ID")
        st.write(
            "Earth Engine project:",
            f"✅ {ee_project}" if ee_project else "⚠️ not set (fallback data)",
        )

        st.write(ensure_rag_index(config))

        st.divider()
        st.caption(
            "This is an open replica of the I/O 2026 Geospatial Reasoning "
            "pattern, running on Gemini 3.5 Flash and the free Earth Engine "
            "tier — not Google's gated agent."
        )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for the Streamlit application."""
    st.set_page_config(
        page_title="Kenya Climate Risk Agent",
        page_icon="🌍",
        layout="wide",
    )

    config = load_config()

    st.markdown(
            "<h1 style='text-align:center;'>🌍 Reasoning Over Earth Engine</h1>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<p style='text-align:center; color:gray;'>"
        "An AI agent for Kenya's climate risk.</p>",
        unsafe_allow_html=True,
    )

    render_status_sidebar(config)

    # Question input.
    # Initialize the question state once.
    if "question_input" not in st.session_state:
        st.session_state["question_input"] = EXAMPLE_QUESTIONS[0]

    # Example-question quick buttons (above the box so they feel like presets).
    st.caption("Or try an example:")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for i, (col, example) in enumerate(zip(cols, EXAMPLE_QUESTIONS)):
        if col.button(example[:32] + "…", help=example, key=f"ex_{i}",
                    use_container_width=True):
            st.session_state["question_input"] = example  # write to the widget's own key
            st.rerun()

    # Question input — reads/writes the same session key the buttons set.
    question = st.text_area(
        "Your question",
        height=80,
        key="question_input",
    )

    run_clicked = st.button("Run analysis", type="primary")

    if run_clicked:
        if not question or not question.strip():
            st.warning("Please enter a question first.")
            return

        with st.spinner("Reasoning over Earth Engine… planning, querying, fusing."):
            try:
                result = run_agent(question.strip(), config=config)
            except Exception as exc:
                logger.exception("Agent run failed.")
                st.error(f"The agent encountered an unexpected error: {exc}")
                return

        # Surface a soft warning if the run only partially succeeded.
        if not result.success and result.error:
            st.warning(f"Partial result: {result.error}")

        render_answer(result)

        ranking = render_ranking_table(result)
        if ranking:
            map_col, chart_col = st.columns([3, 2])
            with map_col:
                render_map(ranking, config)
            with chart_col:
                render_bar_chart(ranking, config)

        render_execution_log(result)
    else:
        st.info("Enter a question and click **Run analysis** to begin.")


if __name__ == "__main__":
    main()
