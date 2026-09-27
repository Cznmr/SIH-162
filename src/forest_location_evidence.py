import json
import os

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, shape
from shapely.ops import unary_union


# ============================================================
# CONFIG
# ============================================================

FOREST_FILE = r"data\raw\forest_boundary_telangana.json"

MASTER_FILE = r"data\training\master_feature_table_final.csv"

OUTPUT_FILE = r"data\training\forest_location_evidence.csv"

FOREST_CRS = "EPSG:32644"
WGS84_CRS = "EPSG:4326"

# Distance thresholds in kilometres
NEAR_1KM = 1.0
NEAR_2KM = 2.0
NEAR_5KM = 5.0
NEAR_10KM = 10.0


# ============================================================
# LOAD FOREST POLYGONS
# ============================================================

print("=" * 70)
print("FOREST LOCATION EVIDENCE")
print("=" * 70)

print("\nLoading forest boundary data...")

with open(FOREST_FILE, "r", encoding="utf-8") as f:
    forest_data = json.load(f)

forest_features = forest_data["features"]

print("Forest polygons:", len(forest_features))


# ============================================================
# CONVERT ARCGIS FEATURES TO GEOPANDAS
# ============================================================

records = []
geometries = []

for feature in forest_features:

    attrs = feature.get("attributes", {})
    geometry = feature.get("geometry")

    if not geometry:
        continue

    rings = geometry.get("rings")

    if not rings:
        continue

    try:
        polygon = shape({
            "type": "Polygon",
            "coordinates": rings
        })

        # Repair invalid geometry if necessary
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        if polygon.is_empty:
            continue

        records.append({
            "FID": attrs.get("FID"),
            "Forest_Nam": attrs.get("Forest_Nam"),
            "Forest_typ": attrs.get("Forest_typ"),
            "FB_Type": attrs.get("FB_Type"),
            "Descript": attrs.get("Descript"),
            "remarks": attrs.get("remarks"),
        })

        geometries.append(polygon)

    except Exception as e:
        print("Skipping invalid geometry:", e)


forest_gdf = gpd.GeoDataFrame(
    records,
    geometry=geometries,
    crs=FOREST_CRS
)

print("Valid forest polygons:", len(forest_gdf))


# ============================================================
# CONVERT TO WGS84
# ============================================================

print("\nConverting forest polygons to WGS84...")

forest_gdf = forest_gdf.to_crs(WGS84_CRS)

print("Forest CRS:", forest_gdf.crs)


# ============================================================
# LOAD MASTER GRID
# ============================================================

print("\nLoading master feature table...")

master = pd.read_csv(MASTER_FILE)

print("Master rows:", len(master))

required_columns = [
    "lat_grid",
    "lon_grid"
]

missing = [
    col for col in required_columns
    if col not in master.columns
]

if missing:
    raise ValueError(
        f"Missing required columns in master table: {missing}"
    )


# ============================================================
# CREATE POINTS
# ============================================================

print("\nCreating FIRMS grid points...")

points = gpd.GeoDataFrame(
    master.copy(),
    geometry=[
        Point(lon, lat)
        for lat, lon in zip(
            master["lat_grid"],
            master["lon_grid"]
        )
    ],
    crs=WGS84_CRS
)

print("Points:", len(points))


# ============================================================
# SPATIAL JOIN — INSIDE FOREST
# ============================================================

print("\nChecking which locations are inside forest polygons...")

# Keep only required forest attributes
forest_for_join = forest_gdf[
    [
        "FID",
        "Forest_Nam",
        "Forest_typ",
        "FB_Type",
        "Descript",
        "remarks",
        "geometry"
    ]
].copy()

inside = gpd.sjoin(
    points,
    forest_for_join,
    how="left",
    predicate="within"
)

# If a point somehow falls into multiple polygons,
# keep the first match.
inside = inside[
    ~inside.index.duplicated(keep="first")
]

# Restore original point index
inside = inside.reindex(points.index)


# ============================================================
# INITIAL EVIDENCE
# ============================================================

evidence = master[
    [
        "lat_grid",
        "lon_grid"
    ]
].copy()

evidence["inside_forest"] = (
    inside["FID"].notna().astype(int)
)

evidence["forest_fid"] = inside["FID"]

evidence["forest_name"] = inside["Forest_Nam"]

evidence["forest_type"] = inside["Forest_typ"]

evidence["forest_fb_type"] = inside["FB_Type"]

evidence["forest_description"] = inside["Descript"]

evidence["forest_remarks"] = inside["remarks"]


# ============================================================
# DISTANCE TO NEAREST FOREST POLYGON
# ============================================================

print("\nCalculating distance to nearest forest polygon...")

# For accurate distance calculations,
# project both points and forests to UTM 44N.

points_utm = points.to_crs(FOREST_CRS)

forest_utm = forest_gdf.to_crs(FOREST_CRS)

# Combine all forest polygons into one geometry.
# This avoids repeatedly searching all 1,020 polygons.
forest_union = unary_union(
    forest_utm.geometry
)

print("Calculating distances...")

distances_m = []

for geom in points_utm.geometry:

    distance = geom.distance(forest_union)

    distances_m.append(distance)


evidence["nearest_forest_km"] = [
    d / 1000.0
    for d in distances_m
]


# ============================================================
# DISTANCE FLAGS
# ============================================================

evidence["near_forest_1km"] = (
    evidence["nearest_forest_km"] <= NEAR_1KM
).astype(int)

evidence["near_forest_2km"] = (
    evidence["nearest_forest_km"] <= NEAR_2KM
).astype(int)

evidence["near_forest_5km"] = (
    evidence["nearest_forest_km"] <= NEAR_5KM
).astype(int)

evidence["near_forest_10km"] = (
    evidence["nearest_forest_km"] <= NEAR_10KM
).astype(int)


# ============================================================
# FOREST CONTEXT CLASS
# ============================================================

def classify_forest_context(row):

    if row["inside_forest"] == 1:
        return "INSIDE_FOREST"

    distance = row["nearest_forest_km"]

    if distance <= 1:
        return "VERY_NEAR_FOREST"

    if distance <= 2:
        return "NEAR_FOREST"

    if distance <= 5:
        return "WITHIN_5KM_FOREST"

    if distance <= 10:
        return "WITHIN_10KM_FOREST"

    return "FAR_FROM_FOREST"


evidence["forest_context"] = evidence.apply(
    classify_forest_context,
    axis=1
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

evidence.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FOREST EVIDENCE COMPLETE")
print("=" * 70)

print("\nOutput:")
print(OUTPUT_FILE)

print("\nRows:", len(evidence))

print("\nInside forest:")
print(
    evidence["inside_forest"].value_counts(
        dropna=False
    )
)

print("\nForest context:")
print(
    evidence["forest_context"].value_counts(
        dropna=False
    )
)

print("\nDistance statistics (km):")

print(
    evidence["nearest_forest_km"].describe()
)

print("\nForest name examples:")

print(
    evidence.loc[
        evidence["inside_forest"] == 1,
        "forest_name"
    ]
    .dropna()
    .value_counts()
    .head(20)
)

print("\nSaved successfully.")