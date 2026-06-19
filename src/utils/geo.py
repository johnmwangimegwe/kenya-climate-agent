"""
geo.py
======

Geospatial helper utilities shared by every tool in the Kenya Climate Risk
agent. This is the single place that knows about Kenyan county geometries, so
the tool modules stay focused on their own logic.

Responsibilities
----------------
- Load the county boundaries from data/kenya_counties.geojson (GeoPandas).
- Resolve a user-supplied county list (case-insensitive) to canonical names,
  defaulting to all 47 counties.
- Provide per-county geometry, centroid and bounding box.
- Convert GeoPandas/shapely geometries to Earth Engine geometries.
- Perform a simple point-in-county spatial join for settlements.

Reliability
-----------
If the GeoDataFrame or GeoPandas itself is unavailable, the module falls back
to a built-in list of the 47 county names (with approximate centroids) so the
rest of the agent can still resolve counties and run in fallback mode.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# GeoPandas / shapely are optional at import time.
try:
    import geopandas as gpd  # type: ignore

    _GPD_AVAILABLE = True
except ImportError:  # pragma: no cover
    gpd = None  # type: ignore
    _GPD_AVAILABLE = False

try:
    import ee  # type: ignore

    _EE_AVAILABLE = True
except ImportError:  # pragma: no cover
    ee = None  # type: ignore
    _EE_AVAILABLE = False


# Path to the counties GeoJSON (relative to project root).
def _counties_path() -> str:
    """Resolve the path to kenya_counties.geojson under the project data dir."""
    here = os.path.dirname(os.path.abspath(__file__))
    # src/utils/ -> project root is two levels up.
    root = os.path.abspath(os.path.join(here, "..", ".."))
    return os.path.join(root, "data", "kenya_counties.geojson")


# Candidate property names that may hold the county name in the GeoJSON.
_NAME_FIELDS = ("COUNTY", "county", "NAME_1", "name", "Name", "ADM1_EN", "shapeName")


# ---------------------------------------------------------------------------
# Built-in fallback: 47 counties with approximate (lon, lat) centroids.
# Used only when the GeoJSON / GeoPandas is unavailable.
# ---------------------------------------------------------------------------
KENYA_COUNTIES: dict[str, tuple[float, float]] = {
    "Mombasa": (39.66, -4.05), "Kwale": (39.22, -4.18), "Kilifi": (39.85, -3.51),
    "Tana River": (39.80, -1.50), "Lamu": (40.90, -2.27), "Taita Taveta": (38.36, -3.40),
    "Garissa": (39.66, -0.45), "Wajir": (40.06, 1.75), "Mandera": (40.96, 3.94),
    "Marsabit": (37.99, 2.33), "Isiolo": (38.49, 0.35), "Meru": (37.65, 0.05),
    "Tharaka Nithi": (37.95, -0.30), "Embu": (37.46, -0.53), "Kitui": (38.01, -1.37),
    "Machakos": (37.26, -1.52), "Makueni": (37.62, -2.18), "Nyandarua": (36.52, -0.18),
    "Nyeri": (36.95, -0.42), "Kirinyaga": (37.30, -0.50), "Murang'a": (37.15, -0.78),
    "Kiambu": (36.83, -1.17), "Turkana": (35.60, 3.12), "West Pokot": (35.20, 1.62),
    "Samburu": (37.11, 1.17), "Trans Nzoia": (34.95, 1.02), "Uasin Gishu": (35.30, 0.55),
    "Elgeyo Marakwet": (35.50, 0.80), "Nandi": (35.10, 0.18), "Baringo": (35.97, 0.47),
    "Laikipia": (36.78, 0.20), "Nakuru": (36.07, -0.30), "Narok": (35.87, -1.08),
    "Kajiado": (36.78, -2.10), "Kericho": (35.28, -0.37), "Bomet": (35.34, -0.78),
    "Kakamega": (34.75, 0.28), "Vihiga": (34.72, 0.07), "Bungoma": (34.56, 0.57),
    "Busia": (34.11, 0.46), "Siaya": (34.29, 0.06), "Kisumu": (34.77, -0.09),
    "Homa Bay": (34.46, -0.53), "Migori": (34.47, -1.06), "Kisii": (34.78, -0.68),
    "Nyamira": (34.94, -0.57), "Nairobi": (36.82, -1.29),
}


# ---------------------------------------------------------------------------
# County boundary loading
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_counties_gdf() -> Any | None:
    """
    Load the counties GeoDataFrame once and cache it.

    Returns None if GeoPandas is unavailable or the file cannot be read, in
    which case callers fall back to the built-in KENYA_COUNTIES table.
    """
    if not _GPD_AVAILABLE:
        return None
    path = _counties_path()
    if not os.path.exists(path):
        logger.warning("Counties GeoJSON not found at %s; using fallback table.", path)
        return None
    try:
        gdf = gpd.read_file(path)
        # Normalize the county-name column to a canonical 'county' column.
        name_field = next((f for f in _NAME_FIELDS if f in gdf.columns), None)
        if name_field is None:
            logger.warning(
                "No recognized county-name field in GeoJSON (cols=%s); fallback.",
                list(gdf.columns),
            )
            return None
        gdf = gdf.rename(columns={name_field: "county"})
        gdf["county"] = gdf["county"].astype(str).str.strip()
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gdf.to_crs("EPSG:4326")
        return gdf
    except Exception as exc:
        logger.warning("Failed to load counties GeoJSON (%s); using fallback.", exc)
        return None


@lru_cache(maxsize=1)
def all_county_names() -> tuple[str, ...]:
    """Return the canonical list of county names (from GeoJSON or fallback)."""
    gdf = _load_counties_gdf()
    if gdf is not None and "county" in gdf.columns:
        names = sorted({str(n).strip() for n in gdf["county"].tolist() if str(n).strip()})
        if names:
            return tuple(names)
    return tuple(sorted(KENYA_COUNTIES.keys()))


# ---------------------------------------------------------------------------
# County resolution
# ---------------------------------------------------------------------------
def resolve_counties(counties: list[str] | None) -> list[str]:
    """
    Resolve a user-supplied county list to canonical names (case-insensitive).

    None or an empty list means "all counties". Unknown names are dropped with
    a warning. Always returns at least the full list when input is empty.
    """
    canonical = all_county_names()
    if not counties:
        return list(canonical)

    lookup = {name.lower(): name for name in canonical}
    resolved: list[str] = []
    for raw in counties:
        if not isinstance(raw, str):
            continue
        key = raw.strip().lower()
        if key in lookup:
            resolved.append(lookup[key])
        else:
            logger.warning("Unknown county %r ignored.", raw)
    return resolved or list(canonical)


# ---------------------------------------------------------------------------
# Geometry access
# ---------------------------------------------------------------------------
def get_county_geometry(name: str) -> Any | None:
    """
    Return the shapely geometry for a county, or None if unavailable.

    When the GeoJSON is missing, returns None (callers then rely on centroid
    based fallbacks).
    """
    gdf = _load_counties_gdf()
    if gdf is None:
        return None
    try:
        match = gdf[gdf["county"].str.lower() == name.strip().lower()]
        if match.empty:
            return None
        return match.geometry.iloc[0]
    except Exception as exc:
        logger.debug("get_county_geometry failed for %s: %s", name, exc)
        return None


def centroid(geometry: Any | None) -> tuple[float, float] | None:
    """Return (lon, lat) centroid of a shapely geometry, or None."""
    if geometry is None:
        return None
    try:
        c = geometry.centroid
        return (float(c.x), float(c.y))
    except Exception:
        return None


def county_centroid(name: str) -> tuple[float, float] | None:
    """Return (lon, lat) centroid for a county by name, with fallback table."""
    geometry = get_county_geometry(name)
    c = centroid(geometry)
    if c is not None:
        return c
    return KENYA_COUNTIES.get(name)


def bounding_box(geometry: Any | None) -> tuple[float, float, float, float] | None:
    """Return (minx, miny, maxx, maxy) bounds of a geometry, or None."""
    if geometry is None:
        return None
    try:
        bounds = geometry.bounds  # (minx, miny, maxx, maxy)
        return (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Earth Engine conversion
# ---------------------------------------------------------------------------
def to_ee_geometry(geometry: Any) -> Any:
    """
    Convert a shapely geometry to an ee.Geometry.

    Raises RuntimeError if Earth Engine is unavailable. Callers should only use
    this after confirming EE is initialized.
    """
    if not _EE_AVAILABLE:
        raise RuntimeError("Earth Engine is not available.")
    try:
        mapping = geometry.__geo_interface__
        return ee.Geometry(mapping)
    except Exception as exc:
        raise RuntimeError(f"Failed to convert geometry to ee.Geometry: {exc}") from exc


def county_to_ee_geometry(name: str) -> Any:
    """
    Build an ee.Geometry for a county, falling back to a small bbox around the
    centroid if no polygon is available.
    """
    geometry = get_county_geometry(name)
    if geometry is not None:
        return to_ee_geometry(geometry)
    # Fallback: a ~0.5 degree box around the centroid.
    c = county_centroid(name)
    if c is None:
        raise RuntimeError(f"No geometry or centroid for county {name!r}.")
    if not _EE_AVAILABLE:
        raise RuntimeError("Earth Engine is not available.")
    lon, lat = c
    d = 0.25
    return ee.Geometry.Rectangle([lon - d, lat - d, lon + d, lat + d])


# ---------------------------------------------------------------------------
# Spatial join (settlements -> county)
# ---------------------------------------------------------------------------
def assign_points_to_counties(points_gdf: Any) -> Any | None:
    """
    Spatially join a GeoDataFrame of point settlements to their counties.

    Returns the joined GeoDataFrame (with a 'county' column) or None if the
    operation cannot be performed.
    """
    gdf = _load_counties_gdf()
    if gdf is None or not _GPD_AVAILABLE or points_gdf is None:
        return None
    try:
        if points_gdf.crs is None:
            points_gdf = points_gdf.set_crs("EPSG:4326")
        else:
            points_gdf = points_gdf.to_crs("EPSG:4326")
        joined = gpd.sjoin(
            points_gdf, gdf[["county", "geometry"]], how="left", predicate="within"
        )
        return joined
    except Exception as exc:
        logger.debug("Spatial join failed: %s", exc)
        return None


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    print("Total counties:", len(all_county_names()))
    print("Resolve ['tana river','NAIROBI','atlantis']:",
          resolve_counties(["tana river", "NAIROBI", "atlantis"]))
    print("Nairobi centroid:", county_centroid("Nairobi"))
