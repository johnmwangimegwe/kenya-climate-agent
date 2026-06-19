"""
risk_score.py
=============

Transparent climate-risk scoring tool for the Kenya Climate Risk agent.

This is the analytical "decision" sub-agent. It combines the geophysical signals
(from earth_engine) and population exposure (from population) into a single,
explainable risk score per county, then ranks the counties.

Scoring model (fixed, documented, deterministic — NO machine learning):

    risk_score = 0.4 * flood
               + 0.3 * rainfall
               + 0.2 * population_exposure
               + 0.1 * elevation_vulnerability

All four inputs are normalized to [0, 1], so risk_score is also in [0, 1].
Because the weights are fixed and the inputs deterministic, the same data always
yields the same score — which is exactly what a fairness/auditability story
needs: every number can be traced back to its contributing factors.

Public function
---------------
- compute_risk_score(counties, config) -> dict   (orchestrator contract)

Output contract (consumed by orchestrator/fusion/viz):

    {
        "ranking": [
            {
                "county": "<name>",
                "risk_score": <float 0..1>,
                "components": {
                    "flood": <float>, "rainfall": <float>,
                    "population": <float>, "elevation": <float>
                },
                "contributions": {   # weight * value, sums to risk_score
                    "flood": <float>, "rainfall": <float>,
                    "population": <float>, "elevation": <float>
                },
                "exposed_population": <int>
            },
            ...
        ],
        "weights": { ... },
        "meta": { "sources": {...} }
    }

The tool fetches its own inputs (calling earth_engine and population) so the
planner can invoke it as a single final step.
"""

from __future__ import annotations

import logging
from typing import Any

from ..utils import geo
from . import earth_engine, population

logger = logging.getLogger(__name__)

# Fixed, documented weights. Overridable via config.yaml -> risk.weights.
DEFAULT_WEIGHTS: dict[str, float] = {
    "flood": 0.4,
    "rainfall": 0.3,
    "population": 0.2,
    "elevation": 0.1,
}


def _resolve_weights(config: dict[str, Any] | None) -> dict[str, float]:
    """
    Resolve scoring weights from config, validating they are non-negative and
    normalizing them to sum to 1.0 so the score stays in [0, 1].
    """
    weights = dict(DEFAULT_WEIGHTS)
    if config:
        configured = config.get("risk", {}).get("weights")
        if isinstance(configured, dict):
            for key in weights:
                if key in configured:
                    try:
                        val = float(configured[key])
                        if val < 0:
                            raise ValueError
                        weights[key] = val
                    except (TypeError, ValueError):
                        logger.warning(
                            "Invalid weight for %r; keeping default %.2f.",
                            key, weights[key],
                        )
    total = sum(weights.values())
    if total <= 0:
        logger.warning("Weights sum to zero; reverting to defaults.")
        return dict(DEFAULT_WEIGHTS)
    return {k: v / total for k, v in weights.items()}


def _clamp_unit(value: Any) -> float:
    """Coerce a value into a float in [0, 1], defaulting to 0.0 on bad input."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


def compute_risk_score(
    counties: list[str] | None = None,
    config: dict[str, Any] | None = None,
    layers: dict[str, Any] | None = None,
    population_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute and rank transparent climate-risk scores per county.

    Parameters
    ----------
    counties:
        County names to score. None or empty means all 47 counties.
    config:
        Optional configuration dict (risk.weights overrides, plus passed
        through to the data tools).
    layers:
        Optional pre-fetched Earth Engine layers (output of
        earth_engine.get_county_risk_layers). If None, fetched here.
    population_data:
        Optional pre-fetched population output. If None, fetched here.

    Returns
    -------
    dict
        See module docstring for the output contract.
    """
    target_counties = geo.resolve_counties(counties)
    if not target_counties:
        return {"ranking": [], "weights": _resolve_weights(config), "meta": {}}

    weights = _resolve_weights(config)

    # Fetch inputs if not supplied (lets the planner call this as one step).
    if layers is None:
        layers = earth_engine.get_county_risk_layers(counties=target_counties, config=config)
    if population_data is None:
        population_data = population.get_population_exposure(
            counties=target_counties, config=config
        )

    layer_counties = layers.get("counties", {}) if isinstance(layers, dict) else {}
    pop_counties = (
        population_data.get("counties", {}) if isinstance(population_data, dict) else {}
    )

    ranking: list[dict[str, Any]] = []
    for name in target_counties:
        layer = layer_counties.get(name, {})
        pop = pop_counties.get(name, {})

        flood = _clamp_unit(layer.get("flood"))
        rainfall = _clamp_unit(layer.get("rainfall"))
        elevation = _clamp_unit(layer.get("elevation"))
        pop_norm = _clamp_unit(pop.get("population_normalized"))
        exposed = int(pop.get("exposed_population", 0) or 0)

        contributions = {
            "flood": round(weights["flood"] * flood, 4),
            "rainfall": round(weights["rainfall"] * rainfall, 4),
            "population": round(weights["population"] * pop_norm, 4),
            "elevation": round(weights["elevation"] * elevation, 4),
        }
        score = round(sum(contributions.values()), 4)

        ranking.append(
            {
                "county": name,
                "risk_score": score,
                "components": {
                    "flood": flood,
                    "rainfall": rainfall,
                    "population": pop_norm,
                    "elevation": elevation,
                },
                "contributions": contributions,
                "exposed_population": exposed,
            }
        )

    # Sort high-risk first; tie-break alphabetically for stable output.
    ranking.sort(key=lambda r: (-r["risk_score"], r["county"]))

    return {
        "ranking": ranking,
        "weights": weights,
        "meta": {
            "sources": {
                "layers": layers.get("meta", {}) if isinstance(layers, dict) else {},
                "population": population_data.get("meta", {})
                if isinstance(population_data, dict)
                else {},
            }
        },
    }


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    import json

    out = compute_risk_score(["Tana River", "Garissa", "Nairobi", "Turkana"])
    print(json.dumps(out["ranking"], indent=2))
