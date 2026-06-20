"""
precompute.py
=============

Run this ONCE before your demo to pre-bake the slow Earth Engine and population
results into data/cache.json. After this, the app reads from cache in
milliseconds instead of querying Google live (which can take minutes for all 47
counties).

Usage (from the project root, with your .env / credentials set up):

    python precompute.py

It pre-computes:
  - all 47 counties (the "whole Kenya" queries), and
  - each scoped county-pair used in the example questions,
so every demo question is a cache hit.

Re-run it whenever you change the analysis dates or want fresh data. Delete
data/cache.json to force the app back to live queries.
"""

from __future__ import annotations

import logging
import os
import sys

# Make the project importable when run from the root.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import yaml  # noqa: E402

from src.tools import cache  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("precompute")


def load_config() -> dict:
    """Load config.yaml, plus .env for credentials."""
    try:
        from dotenv import load_dotenv

        load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    except Exception:
        pass

    path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


# County scopes to warm. None = all 47 counties. Add any pairs your demo uses.
SCOPES: list[list[str] | None] = [
    None,                                  # whole Kenya
    ["Tana River", "Garissa"],             # "compare flood risk" example
    ["Turkana", "Marsabit", "Mandera", "Wajir", "Samburu"],  # northern drought
]


def main() -> None:
    config = load_config()

    has_key = bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    ee_project = (config.get("earth_engine", {}) or {}).get("project") or os.environ.get(
        "EE_PROJECT_ID"
    )
    logger.info("Gemini key: %s", "set" if has_key else "NOT set (fallback values)")
    logger.info("Earth Engine project: %s", ee_project or "NOT set (fallback values)")
    logger.info("Pre-baking %d scope(s) into data/cache.json...\n", len(SCOPES))

    for i, scope in enumerate(SCOPES, start=1):
        label = "ALL 47 counties" if scope is None else ", ".join(scope)
        logger.info("[%d/%d] %s", i, len(SCOPES), label)

        logger.info("    - Earth Engine layers...")
        cache.cached_earth_engine_layers(counties=scope, config=config, write=True)

        logger.info("    - population exposure...")
        cache.cached_population_exposure(counties=scope, config=config, write=True)

    path = cache._cache_path()
    size_kb = round(os.path.getsize(path) / 1024, 1) if os.path.exists(path) else 0
    logger.info("\nDone. Cache written to %s (%s KB).", path, size_kb)
    logger.info("Your demo will now load from cache instead of querying live.")


if __name__ == "__main__":
    main()
