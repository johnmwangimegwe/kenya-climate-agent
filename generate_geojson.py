"""
Generate kenya_counties.geojson and settlements.geojson.

Counties: build a Voronoi tessellation from the 47 county centroids (defined in
src/utils/geo.py), clipped to an approximate Kenya national boundary. This
yields adjacent, realistic-looking county polygons (not squares) that are valid
GeoJSON and good enough for choropleths and Earth Engine region clipping in a
demo. For publication-grade boundaries, replace with an official GADM/HDX file.

Settlements: a set of real major Kenyan towns with coordinates and population,
as point features, each tagged with its county.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point, mapping
from shapely.ops import unary_union

# Import the canonical county centroids from geo.py.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.utils.geo import KENYA_COUNTIES  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")

# Approximate Kenya national boundary (simplified polygon, lon/lat).
KENYA_BOUNDARY = Polygon([
    (33.9, -4.7), (37.7, -3.1), (39.2, -4.7), (41.6, -1.7), (41.0, 2.0),
    (41.9, 4.0), (39.5, 3.6), (38.1, 3.6), (36.0, 4.5), (35.9, 5.0),
    (34.5, 4.2), (34.1, 1.2), (33.9, 0.1), (34.0, -1.0), (33.9, -4.7),
])


def _bounded_voronoi(points: np.ndarray, boundary: Polygon) -> list[Polygon]:
    """
    Build Voronoi regions for points, clipped to a boundary polygon.

    Distant mirror points are added so every real point gets a finite cell.
    """
    minx, miny, maxx, maxy = boundary.bounds
    span = max(maxx - minx, maxy - miny) * 3
    # Four far-away points to bound all finite regions.
    far = np.array([
        [minx - span, miny - span],
        [minx - span, maxy + span],
        [maxx + span, miny - span],
        [maxx + span, maxy + span],
    ])
    all_points = np.vstack([points, far])
    vor = Voronoi(all_points)

    polygons: list[Polygon] = []
    for i in range(len(points)):
        region_index = vor.point_region[i]
        vertices = vor.regions[region_index]
        if not vertices or -1 in vertices:
            polygons.append(None)  # type: ignore
            continue
        poly = Polygon([vor.vertices[v] for v in vertices])
        clipped = poly.intersection(boundary)
        polygons.append(clipped if not clipped.is_empty else None)  # type: ignore
    return polygons


def build_counties_geojson() -> dict:
    """Build the counties FeatureCollection."""
    names = list(KENYA_COUNTIES.keys())
    coords = np.array([KENYA_COUNTIES[n] for n in names])  # (lon, lat)

    polygons = _bounded_voronoi(coords, KENYA_BOUNDARY)

    features = []
    for name, (lon, lat), poly in zip(names, coords, polygons):
        if poly is None or poly.is_empty:
            # Fallback: small box around the centroid clipped to boundary.
            d = 0.3
            box = Polygon([
                (lon - d, lat - d), (lon + d, lat - d),
                (lon + d, lat + d), (lon - d, lat + d),
            ]).intersection(KENYA_BOUNDARY)
            poly = box if not box.is_empty else Polygon([
                (lon - d, lat - d), (lon + d, lat - d),
                (lon + d, lat + d), (lon - d, lat + d),
            ])
        features.append({
            "type": "Feature",
            "properties": {"COUNTY": name},
            "geometry": mapping(poly),
        })

    return {"type": "FeatureCollection", "features": features}


# Major Kenyan towns: (name, county, lon, lat, approx_population).
SETTLEMENTS = [
    ("Nairobi", "Nairobi", 36.8172, -1.2864, 4397073),
    ("Mombasa", "Mombasa", 39.6682, -4.0435, 1208333),
    ("Kisumu", "Kisumu", 34.7617, -0.0917, 610082),
    ("Nakuru", "Nakuru", 36.0667, -0.3031, 570674),
    ("Eldoret", "Uasin Gishu", 35.2698, 0.5143, 475716),
    ("Garissa", "Garissa", 39.6583, -0.4536, 163399),
    ("Hola", "Tana River", 40.0300, -1.5000, 6000),
    ("Garsen", "Tana River", 40.1167, -2.2667, 5000),
    ("Lodwar", "Turkana", 35.5966, 3.1191, 82970),
    ("Mandera", "Mandera", 41.8569, 3.9366, 87371),
    ("Wajir", "Wajir", 40.0573, 1.7471, 90116),
    ("Marsabit", "Marsabit", 37.9899, 2.3284, 19196),
    ("Kakamega", "Kakamega", 34.7519, 0.2827, 91778),
    ("Kitui", "Kitui", 38.0106, -1.3667, 155896),
    ("Malindi", "Kilifi", 40.1169, -3.2192, 119859),
    ("Nyeri", "Nyeri", 36.9476, -0.4201, 119273),
    ("Machakos", "Machakos", 37.2634, -1.5177, 150041),
    ("Busia", "Busia", 34.1117, 0.4608, 51981),
    ("Homa Bay", "Homa Bay", 34.4571, -0.5273, 56000),
]


def build_settlements_geojson() -> dict:
    """Build the settlements FeatureCollection."""
    features = []
    for name, county, lon, lat, pop in SETTLEMENTS:
        features.append({
            "type": "Feature",
            "properties": {"name": name, "county": county, "population": pop},
            "geometry": mapping(Point(lon, lat)),
        })
    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    counties = build_counties_geojson()
    with open(os.path.join(OUT_DIR, "kenya_counties.geojson"), "w", encoding="utf-8") as fh:
        json.dump(counties, fh)
    print(f"Wrote kenya_counties.geojson with {len(counties['features'])} counties.")

    settlements = build_settlements_geojson()
    with open(os.path.join(OUT_DIR, "settlements.geojson"), "w", encoding="utf-8") as fh:
        json.dump(settlements, fh)
    print(f"Wrote settlements.geojson with {len(settlements['features'])} settlements.")


if __name__ == "__main__":
    main()
