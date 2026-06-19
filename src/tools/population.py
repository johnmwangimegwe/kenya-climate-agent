"""
population.py
=============

Population-exposure tool for the Kenya Climate Risk agent.

Estimates how many people live in each county (and, by extension, are exposed
to the climate hazard being analyzed) using the WorldPop gridded population
dataset on Earth Engine, aggregated per county.

Public function
---------------
- get_population_exposure(counties, config) -> dict   (orchestrator contract)

Output contract (consumed by orchestrator/fusion/risk_score):

    {
        "counties": {
            "<County Name>": {
                "exposed_population": <int>,
                "population_normalized": <float 0..1>,
            },
            ...
        },
        "meta": { "source": "worldpop" | "fallback", "year": <int> }
    }

Reliability
-----------
If Earth Engine or WorldPop is unavailable, the tool falls back to a built-in
table of approximate county populations (2019 census order of magnitude) so the
agent still returns plausible, deterministic exposure figures during a demo.
The ``population_normalized`` field scales each county's population into [0, 1]
across the returned set, which the risk-scoring model consumes directly.
"""

from __future__ import annotations

import logging
from typing import Any

from ..utils import geo
from . import earth_engine

logger = logging.getLogger(__name__)

try:
    import ee  # type: ignore

    _EE_AVAILABLE = True
except ImportError:  # pragma: no cover
    ee = None  # type: ignore
    _EE_AVAILABLE = False

# WorldPop population count collection on Earth Engine.
DEFAULT_WORLDPOP = "WorldPop/GP/100m/pop"
DEFAULT_YEAR = 2020

# ---------------------------------------------------------------------------
# Built-in fallback populations (approximate, 2019 Kenya census magnitude).
# Used only when Earth Engine / WorldPop is unavailable.
# ---------------------------------------------------------------------------
FALLBACK_POPULATION: dict[str, int] = {
    "Mombasa": 1208333, "Kwale": 866820, "Kilifi": 1453787, "Tana River": 315943,
    "Lamu": 143920, "Taita Taveta": 340671, "Garissa": 841353, "Wajir": 781263,
    "Mandera": 867457, "Marsabit": 459785, "Isiolo": 268002, "Meru": 1545714,
    "Tharaka Nithi": 393177, "Embu": 608599, "Kitui": 1136187, "Machakos": 1421932,
    "Makueni": 987653, "Nyandarua": 638289, "Nyeri": 759164, "Kirinyaga": 610411,
    "Murang'a": 1056640, "Kiambu": 2417735, "Turkana": 926976, "West Pokot": 621241,
    "Samburu": 310327, "Trans Nzoia": 990341, "Uasin Gishu": 1163186,
    "Elgeyo Marakwet": 454480, "Nandi": 885711, "Baringo": 666763, "Laikipia": 518560,
    "Nakuru": 2162202, "Narok": 1157873, "Kajiado": 1117840, "Kericho": 901777,
    "Bomet": 875689, "Kakamega": 1867579, "Vihiga": 590013, "Bungoma": 1670570,
    "Busia": 893681, "Siaya": 993183, "Kisumu": 1155574, "Homa Bay": 1131950,
    "Migori": 1116436, "Kisii": 1266860, "Nyamira": 605576, "Nairobi": 4397073,
}


def _worldpop_id(config: dict[str, Any] | None) -> str:
    """Resolve the WorldPop collection id from config or default."""
    if config:
        configured = config.get("population", {}).get("worldpop")
        if configured:
            return str(configured)
    return DEFAULT_WORLDPOP


def _population_year(config: dict[str, Any] | None) -> int:
    """Resolve the population year from config or default."""
    if config:
        configured = config.get("population", {}).get("year")
        if configured:
            try:
                return int(configured)
            except (TypeError, ValueError):
                pass
    return DEFAULT_YEAR


def _ee_population_for_geometry(
    ee_geometry: Any, worldpop_id: str, year: int
) -> int | None:
    """Sum WorldPop population over a geometry for the given year."""
    try:
        collection = (
            ee.ImageCollection(worldpop_id)
            .filter(ee.Filter.eq("year", year))
            .filterBounds(ee_geometry)
        )
        size = collection.size().getInfo()
        if size == 0:
            # Fall back to the most recent available image in the collection.
            collection = ee.ImageCollection(worldpop_id).filterBounds(ee_geometry)
            if collection.size().getInfo() == 0:
                return None
        image = collection.mosaic()
        stats = image.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=ee_geometry,
            scale=100,
            maxPixels=1e10,
            bestEffort=True,
        )
        info = stats.getInfo() or {}
        for value in info.values():
            if value is not None:
                return int(round(float(value)))
        return None
    except Exception as exc:
        logger.debug("WorldPop reduction failed: %s", exc)
        return None


def _normalize(values: dict[str, int]) -> dict[str, float]:
    """Scale a dict of populations into [0, 1] by the max value present."""
    if not values:
        return {}
    max_val = max(values.values()) or 1
    return {name: round(val / max_val, 3) for name, val in values.items()}


def get_population_exposure(
    counties: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Estimate exposed population per county.

    Parameters
    ----------
    counties:
        County names to analyze. None or empty means all 47 counties.
    config:
        Optional configuration dict (population.worldpop, population.year).

    Returns
    -------
    dict
        See module docstring for the output contract.
    """
    target_counties = geo.resolve_counties(counties)
    if not target_counties:
        return {"counties": {}, "meta": {"source": "none", "year": _population_year(config)}}

    worldpop_id = _worldpop_id(config)
    year = _population_year(config)
    ee_ready = earth_engine.initialize_earth_engine(config)

    raw_population: dict[str, int] = {}
    used_live = False

    for name in target_counties:
        value: int | None = None
        if ee_ready:
            try:
                ee_geometry = geo.county_to_ee_geometry(name)
                value = _ee_population_for_geometry(ee_geometry, worldpop_id, year)
                if value is not None and value > 0:
                    used_live = True
            except Exception as exc:
                logger.debug("Live population failed for %s: %s", name, exc)
                value = None

        if value is None or value <= 0:
            value = FALLBACK_POPULATION.get(name, 100000)

        raw_population[name] = int(value)

    normalized = _normalize(raw_population)

    counties_out: dict[str, dict[str, Any]] = {}
    for name, pop in raw_population.items():
        counties_out[name] = {
            "exposed_population": pop,
            "population_normalized": normalized.get(name, 0.0),
        }

    source = "worldpop" if used_live else "fallback"
    return {"counties": counties_out, "meta": {"source": source, "year": year}}


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    import json

    print(json.dumps(
        get_population_exposure(["Nairobi", "Tana River", "Garissa"]), indent=2
    ))
