"""
viz.py
======

Visualization helpers for the Kenya Climate Risk agent.

Turns the risk-scoring output (the 'ranking' list from tools.risk_score) into:
- an interactive county risk map (geemap/folium), choropleth-style by score
- a ranked map highlighting the top-N highest-risk counties
- a risk bar chart (matplotlib) of the top-N counties

Public functions
----------------
- build_risk_map(ranking, top_n, config)        -> map object | None
- build_ranked_map(ranking, top_n, config)      -> map object | None
- build_risk_bar_chart(ranking, top_n)          -> matplotlib Figure | None
- risk_dataframe(ranking)                        -> pandas.DataFrame

Reliability
-----------
geemap/folium and matplotlib are optional. Every function degrades gracefully:
if a plotting library or the county GeoJSON is unavailable, it logs a warning
and returns None (the Streamlit UI then shows the table instead of the map).
County geometries come from utils.geo so this module stays presentation-only.
"""

from __future__ import annotations

import logging
from typing import Any

from . import geo

logger = logging.getLogger(__name__)

try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover
    pd = None  # type: ignore
    _PANDAS_AVAILABLE = False


# Color stops for risk (low -> high). Used for both maps and chart.
_RISK_COLORS = [
    (0.0, "#1a9850"),   # green  - low
    (0.25, "#91cf60"),
    (0.5, "#fee08b"),   # yellow - medium
    (0.75, "#fc8d59"),
    (1.0, "#d73027"),   # red    - high
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _color_for_score(score: float) -> str:
    """Map a 0..1 risk score to a hex color via the stops above."""
    score = max(0.0, min(1.0, float(score)))
    chosen = _RISK_COLORS[0][1]
    for threshold, color in _RISK_COLORS:
        if score >= threshold:
            chosen = color
    return chosen


def _ranking_to_lookup(ranking: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index a ranking list by county name for quick lookup."""
    lookup: dict[str, dict[str, Any]] = {}
    for row in ranking:
        if isinstance(row, dict) and "county" in row:
            lookup[str(row["county"])] = row
    return lookup


def risk_dataframe(ranking: list[dict[str, Any]]) -> Any:
    """
    Build a tidy pandas DataFrame from the ranking for tables/charts.

    Columns: county, risk_score, exposed_population, flood, rainfall,
    population, elevation. Returns an empty DataFrame if pandas is missing.
    """
    if not _PANDAS_AVAILABLE:
        logger.warning("pandas unavailable; cannot build DataFrame.")
        return None

    rows: list[dict[str, Any]] = []
    for r in ranking:
        if not isinstance(r, dict):
            continue
        components = r.get("components", {})
        rows.append(
            {
                "county": r.get("county"),
                "risk_score": r.get("risk_score"),
                "exposed_population": r.get("exposed_population"),
                "flood": components.get("flood"),
                "rainfall": components.get("rainfall"),
                "population": components.get("population"),
                "elevation": components.get("elevation"),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------
def _new_map(center: tuple[float, float] = (0.5, 37.8), zoom: int = 6) -> Any | None:
    """
    Create a base interactive map.

    Prefers folium, which needs no authentication and is ideal for these static
    county choropleths. geemap is only used when Earth Engine has ALREADY been
    initialized in this process — importing/instantiating geemap can otherwise
    trigger a blocking Earth Engine auth prompt, which would hang the Streamlit
    app and the live demo. Returns None if no mapping library is available.
    """
    # 1. folium first: lightweight, no auth, perfect for choropleths.
    try:
        import folium  # type: ignore

        return folium.Map(location=list(center), zoom_start=zoom)
    except Exception as exc:
        logger.debug("folium unavailable (%s); considering geemap.", exc)

    # 2. geemap only if Earth Engine is already initialized (never trigger auth).
    try:
        from ..tools import earth_engine as _ee_tool

        if getattr(_ee_tool, "_EE_INITIALIZED", False):
            import geemap  # type: ignore

            return geemap.Map(center=center, zoom=zoom)
        logger.debug("Earth Engine not initialized; skipping geemap to avoid auth.")
    except Exception as exc:
        logger.warning("No mapping library available (%s).", exc)

    return None


def _add_county_polygons(
    map_obj: Any,
    scored_counties: dict[str, dict[str, Any]],
    only: set[str] | None = None,
) -> int:
    """
    Add county polygons colored by risk score to a folium/geemap map.

    Returns the number of counties drawn. Uses utils.geo for geometries.
    """
    try:
        import folium  # type: ignore
    except ImportError:
        logger.warning("folium not available; cannot draw polygons.")
        return 0

    drawn = 0
    for name, row in scored_counties.items():
        if only is not None and name not in only:
            continue
        geometry = geo.get_county_geometry(name)
        if geometry is None:
            continue
        score = float(row.get("risk_score", 0.0))
        exposed = row.get("exposed_population", 0)
        color = _color_for_score(score)
        tooltip = f"{name}: risk {score:.2f}, ~{int(exposed):,} exposed"
        try:
            gj = folium.GeoJson(
                data=geometry.__geo_interface__,
                style_function=(lambda _f, c=color: {
                    "fillColor": c,
                    "color": "#333333",
                    "weight": 1,
                    "fillOpacity": 0.65,
                }),
                tooltip=tooltip,
            )
            # geemap.Map exposes add_child via the underlying folium map.
            add_child = getattr(map_obj, "add_child", None)
            if callable(add_child):
                map_obj.add_child(gj)
            else:
                gj.add_to(map_obj)
            drawn += 1
        except Exception as exc:
            logger.debug("Failed to draw %s: %s", name, exc)
    return drawn


def build_risk_map(
    ranking: list[dict[str, Any]],
    top_n: int | None = None,
    config: dict[str, Any] | None = None,
) -> Any | None:
    """
    Build an interactive choropleth map of county risk scores.

    Parameters
    ----------
    ranking:
        The 'ranking' list from tools.risk_score.compute_risk_score.
    top_n:
        If given, only color the top_n counties (others omitted).
    config:
        Reserved for future styling configuration.

    Returns
    -------
    map object or None
        A geemap/folium map, or None if mapping is unavailable.
    """
    if not ranking:
        logger.warning("Empty ranking; no map produced.")
        return None

    map_obj = _new_map()
    if map_obj is None:
        return None

    lookup = _ranking_to_lookup(ranking)
    only = None
    if top_n:
        only = {row["county"] for row in ranking[: max(1, top_n)] if "county" in row}

    drawn = _add_county_polygons(map_obj, lookup, only=only)
    logger.info("Risk map drew %d counties.", drawn)
    return map_obj


def build_ranked_map(
    ranking: list[dict[str, Any]],
    top_n: int = 10,
    config: dict[str, Any] | None = None,
) -> Any | None:
    """
    Build a map highlighting only the top_n highest-risk counties, with
    numbered markers at their centroids.
    """
    if not ranking:
        return None

    map_obj = _new_map()
    if map_obj is None:
        return None

    top = ranking[: max(1, top_n)]
    lookup = _ranking_to_lookup(top)
    only = {row["county"] for row in top if "county" in row}
    _add_county_polygons(map_obj, lookup, only=only)

    # Add ranked markers.
    try:
        import folium  # type: ignore

        for rank, row in enumerate(top, start=1):
            name = row.get("county")
            if not name:
                continue
            c = geo.county_centroid(name)
            if c is None:
                continue
            lon, lat = c
            marker = folium.Marker(
                location=[lat, lon],
                tooltip=f"#{rank} {name} (risk {float(row.get('risk_score', 0)):.2f})",
                icon=folium.DivIcon(
                    html=(
                        f'<div style="font-size:12px;font-weight:bold;color:#fff;'
                        f'background:#222;border-radius:50%;width:22px;height:22px;'
                        f'text-align:center;line-height:22px;">{rank}</div>'
                    )
                ),
            )
            add_child = getattr(map_obj, "add_child", None)
            if callable(add_child):
                map_obj.add_child(marker)
            else:
                marker.add_to(map_obj)
    except Exception as exc:
        logger.debug("Failed to add ranked markers: %s", exc)

    return map_obj


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------
def build_risk_bar_chart(
    ranking: list[dict[str, Any]],
    top_n: int = 10,
) -> Any | None:
    """
    Build a horizontal bar chart of the top_n counties by risk score.

    Returns a matplotlib Figure, or None if matplotlib is unavailable or the
    ranking is empty.
    """
    if not ranking:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")  # safe for headless / Streamlit environments
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable; cannot build bar chart.")
        return None

    top = ranking[: max(1, top_n)]
    names = [str(r.get("county", "?")) for r in top][::-1]
    scores = [float(r.get("risk_score", 0.0)) for r in top][::-1]
    colors = [_color_for_score(s) for s in scores]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(top))))
    ax.barh(names, scores, color=colors, edgecolor="#333333")
    ax.set_xlabel("Climate-risk score (0–1)")
    ax.set_xlim(0, 1)
    ax.set_title(f"Top {len(top)} Kenyan counties by climate-risk score")
    for i, score in enumerate(scores):
        ax.text(score + 0.01, i, f"{score:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    from ..tools import risk_score

    result = risk_score.compute_risk_score(
        ["Tana River", "Garissa", "Nairobi", "Turkana", "Busia"]
    )
    df = risk_dataframe(result["ranking"])
    if df is not None:
        print(df.to_string(index=False))
    fig = build_risk_bar_chart(result["ranking"], top_n=5)
    print("Bar chart built:", fig is not None)
    m = build_risk_map(result["ranking"], top_n=5)
    print("Risk map built:", m is not None)
