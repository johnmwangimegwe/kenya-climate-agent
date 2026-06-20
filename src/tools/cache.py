"""
cache.py
========

Disk cache for the two slow tools (Earth Engine layers and population exposure).

Why this exists
---------------
Querying Earth Engine live for all 47 counties takes minutes — too slow for a
live demo. This module lets you PRE-COMPUTE those results once (see
precompute.py) and store them in data/cache.json. At demo time the tools read
from that file in milliseconds instead of calling Google.

How it works
------------
- `cached_earth_engine_layers(...)` and `cached_population_exposure(...)` first
  look for a matching entry in the cache file. On a hit, they return it
  instantly. On a miss (or if caching is disabled), they call the real tool and
  — if `write=True` — store the fresh result for next time.
- The cache key is the sorted list of counties (plus dates for layers), so a
  scoped 2-county question and the all-47 question are cached separately.

This is purely a speed layer. The real tools are unchanged and remain the source
of truth; delete data/cache.json to force fresh live queries.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from . import earth_engine, population

logger = logging.getLogger(__name__)


def _cache_path() -> str:
    """Absolute path to the cache file (data/cache.json at project root)."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    return os.path.join(root, "data", "cache.json")


def _load_cache() -> dict[str, Any]:
    """Load the whole cache file, or an empty structure if it doesn't exist."""
    path = _cache_path()
    if not os.path.exists(path):
        return {"layers": {}, "population": {}}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        data.setdefault("layers", {})
        data.setdefault("population", {})
        return data
    except Exception as exc:
        logger.warning("Could not read cache (%s); ignoring.", exc)
        return {"layers": {}, "population": {}}


def _save_cache(cache: dict[str, Any]) -> None:
    """Persist the cache file, creating the data/ directory if needed."""
    path = _cache_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Could not write cache (%s).", exc)


def _normalize_counties(counties: list[str] | None) -> str:
    """Build a stable cache key fragment from a county list (None -> 'ALL')."""
    if not counties:
        return "ALL"
    return ",".join(sorted(str(c).strip() for c in counties))


def _layers_key(counties: list[str] | None, start: str | None, end: str | None) -> str:
    """Cache key for an Earth Engine layers request."""
    return f"{_normalize_counties(counties)}|{start or 'def'}|{end or 'def'}"


def cached_earth_engine_layers(
    counties: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    config: dict[str, Any] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """
    Return Earth Engine layers, using the disk cache when available.

    On a cache miss, calls the real tool and stores the result (if write=True).
    """
    cache = _load_cache()
    key = _layers_key(counties, start_date, end_date)

    cached = cache.get("layers", {}).get(key)
    if cached is not None:
        logger.info("Earth Engine layers: cache HIT (%s).", key)
        return cached

    logger.info("Earth Engine layers: cache MISS (%s); querying live.", key)
    result = earth_engine.get_county_risk_layers(
        counties=counties, start_date=start_date, end_date=end_date, config=config
    )
    if write:
        cache["layers"][key] = result
        _save_cache(cache)
    return result


def cached_population_exposure(
    counties: list[str] | None = None,
    config: dict[str, Any] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """
    Return population exposure, using the disk cache when available.

    On a cache miss, calls the real tool and stores the result (if write=True).
    """
    cache = _load_cache()
    key = _normalize_counties(counties)

    cached = cache.get("population", {}).get(key)
    if cached is not None:
        logger.info("Population exposure: cache HIT (%s).", key)
        return cached

    logger.info("Population exposure: cache MISS (%s); querying live.", key)
    result = population.get_population_exposure(counties=counties, config=config)
    if write:
        cache["population"][key] = result
        _save_cache(cache)
    return result


def cache_exists() -> bool:
    """True if a cache file is present (used by the UI status line)."""
    return os.path.exists(_cache_path())
