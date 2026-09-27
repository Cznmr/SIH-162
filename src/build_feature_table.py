import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

FIRMS_FILE = (
    "data/processed/noaa20_telangana_12months.csv"
)

PERSISTENCE_FILE = (
    "data/processed/noaa20_persistence_features.csv"
)

OSM_FILE = (
    "data/processed/persistent_locations_osm_distance_features.csv"
)

OUTPUT_FILE = (
    "data/training/master_feature_table.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading FIRMS data...")

firms = pd.read_csv(
    FIRMS_FILE,
    low_memory=False
)

print(f"FIRMS detections: {len(firms):,}")


print("\nLoading persistence data...")

persistence = pd.read_csv(
    PERSISTENCE_FILE
)

print(f"Persistence locations: {len(persistence):,}")


print("\nLoading OSM feature data...")

osm = pd.read_csv(
    OSM_FILE
)

print(f"OSM/persistent locations: {len(osm):,}")


# ============================================================
# CLEAN FIRMS DATA
# ============================================================

firms["acq_date"] = pd.to_datetime(
    firms["acq_date"],
    errors="coerce"
)


# Actual VIIRS column names
numeric_columns = [
    "latitude",
    "longitude",
    "bright_ti4",
    "bright_ti5",
    "scan",
    "track",
    "frp"
]

for column in numeric_columns:

    if column in firms.columns:

        firms[column] = pd.to_numeric(
            firms[column],
            errors="coerce"
        )


# ============================================================
# CONFIDENCE
# ============================================================

# Keep original confidence
firms["confidence_raw"] = (
    firms["confidence"].astype(str).str.lower()
)


# VIIRS confidence:
# l = low
# n = nominal
# h = high
confidence_mapping = {
    "l": 1,
    "n": 2,
    "h": 3
}

firms["confidence_numeric"] = (
    firms["confidence_raw"]
    .map(confidence_mapping)
)


# ============================================================
# CREATE SAME GRID USED BY PERSISTENCE ANALYSIS
# ============================================================

firms["lat_grid"] = (
    np.floor(firms["latitude"] / 0.01) * 0.01
).round(2)

firms["lon_grid"] = (
    np.floor(firms["longitude"] / 0.01) * 0.01
).round(2)


# ============================================================
# FIRMS THERMAL FEATURES
# ============================================================

group_columns = [
    "lat_grid",
    "lon_grid"
]


thermal = (
    firms
    .groupby(group_columns)
    .agg(
        detection_count=("latitude", "size"),

        mean_bright_ti4=("bright_ti4", "mean"),
        max_bright_ti4=("bright_ti4", "max"),

        mean_bright_ti5=("bright_ti5", "mean"),
        max_bright_ti5=("bright_ti5", "max"),

        mean_frp=("frp", "mean"),
        max_frp=("frp", "max"),

        mean_scan=("scan", "mean"),
        mean_track=("track", "mean"),

        first_detection_date=("acq_date", "min"),
        last_detection_date=("acq_date", "max"),

        day_detections=(
            "daynight",
            lambda x: (
                x.astype(str)
                .str.upper() == "D"
            ).sum()
        ),

        night_detections=(
            "daynight",
            lambda x: (
                x.astype(str)
                .str.upper() == "N"
            ).sum()
        )
    )
    .reset_index()
)


# ============================================================
# CONFIDENCE FEATURES
# ============================================================

confidence = (
    firms
    .groupby(group_columns)
    ["confidence_numeric"]
    .agg(
        mean_confidence="mean",
        max_confidence="max"
    )
    .reset_index()
)


thermal = thermal.merge(
    confidence,
    on=group_columns,
    how="left"
)


# ============================================================
# DAY/NIGHT RATIO
# ============================================================

thermal["night_ratio"] = (
    thermal["night_detections"]
    /
    thermal["detection_count"].replace(0, np.nan)
)


# ============================================================
# TEMPORAL FEATURES
# ============================================================

thermal["observation_span_days"] = (
    thermal["last_detection_date"]
    -
    thermal["first_detection_date"]
).dt.days


thermal["observation_span_days"] = (
    thermal["observation_span_days"]
    .fillna(0)
)


# ============================================================
# LOAD PERSISTENCE FEATURES
# ============================================================

persistence_columns = [
    "lat_grid",
    "lon_grid",
    "total_detections",
    "active_days",
    "first_detection",
    "last_detection",
    "active_days_30d",
    "active_days_90d",
    "active_days_180d",
    "score_30",
    "score_90",
    "score_180",
    "persistence_score",
    "persistent_30d",
    "persistent_90d",
    "persistent_180d"
]


persistence_columns = [
    column
    for column in persistence_columns
    if column in persistence.columns
]


persistence_selected = persistence[
    persistence_columns
].copy()


# ============================================================
# MERGE FIRMS + PERSISTENCE
# ============================================================

features = persistence_selected.merge(
    thermal,
    on=[
        "lat_grid",
        "lon_grid"
    ],
    how="left",
    suffixes=(
        "",
        "_thermal"
    )
)


# ============================================================
# LOAD OSM FEATURES
# ============================================================

# These already exist in persistence table,
# so don't duplicate them from OSM.
osm_exclude = [
    "total_detections",
    "active_days",
    "active_days_30d",
    "active_days_90d",
    "active_days_180d",
    "persistence_score"
]


osm_feature_columns = [
    column
    for column in osm.columns
    if column not in osm_exclude
]


osm_selected = osm[
    osm_feature_columns
].copy()


# ============================================================
# MERGE OSM
# ============================================================

features = features.merge(
    osm_selected,
    on=[
        "lat_grid",
        "lon_grid"
    ],
    how="left",
    suffixes=(
        "",
        "_osm"
    )
)


# ============================================================
# CREATE OSM CONTEXT FLAGS
# ============================================================

osm_flag_features = [
    (
        "mine_quarry_within_1km",
        "near_mine_1km"
    ),
    (
        "mine_quarry_within_2km",
        "near_mine_2km"
    ),
    (
        "power_plant_within_1km",
        "near_power_1km"
    ),
    (
        "power_plant_within_2km",
        "near_power_2km"
    ),
    (
        "industrial_area_within_1km",
        "near_industrial_area_1km"
    ),
    (
        "industrial_area_within_2km",
        "near_industrial_area_2km"
    ),
    (
        "industrial_works_within_1km",
        "near_industrial_works_1km"
    ),
    (
        "industrial_works_within_2km",
        "near_industrial_works_2km"
    )
]


for source_column, new_column in osm_flag_features:

    if source_column in features.columns:

        features[new_column] = (
            features[source_column]
            .fillna(0)
            .gt(0)
            .astype(int)
        )


# ============================================================
# CLEAN NUMERIC FEATURES
# ============================================================

for column in features.columns:

    if (
        column.endswith("_km")
        or "within_" in column
    ):

        features[column] = pd.to_numeric(
            features[column],
            errors="coerce"
        )


# ============================================================
# SORT
# ============================================================

features = features.sort_values(
    [
        "persistence_score",
        "detection_count"
    ],
    ascending=[
        False,
        False
    ]
)


# ============================================================
# SAVE
# ============================================================

features.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("MASTER FEATURE TABLE CREATED")
print("========================================")

print(f"\nOutput:")
print(OUTPUT_FILE)

print(f"\nRows: {len(features):,}")
print(f"Columns: {len(features.columns):,}")


print("\n----------------------------------------")
print("THERMAL FEATURES")
print("----------------------------------------")

thermal_columns = [
    "detection_count",
    "mean_bright_ti4",
    "max_bright_ti4",
    "mean_bright_ti5",
    "max_bright_ti5",
    "mean_frp",
    "max_frp",
    "mean_confidence",
    "max_confidence",
    "night_ratio"
]

for column in thermal_columns:

    if column in features.columns:
        print(" ", column)


print("\n----------------------------------------")
print("PERSISTENCE FEATURES")
print("----------------------------------------")

persistence_display = [
    "active_days",
    "active_days_30d",
    "active_days_90d",
    "active_days_180d",
    "persistence_score"
]

for column in persistence_display:

    if column in features.columns:
        print(" ", column)


print("\n----------------------------------------")
print("OSM FEATURES")
print("----------------------------------------")

osm_display = [
    "nearest_mine_quarry_km",
    "nearest_power_plant_km",
    "nearest_industrial_area_km",
    "nearest_industrial_works_km",
    "nearest_oil_gas_km"
]

for column in osm_display:

    if column in features.columns:
        print(" ", column)


print("\n----------------------------------------")
print("MASTER TABLE PREVIEW")
print("----------------------------------------")

preview_columns = [
    "lat_grid",
    "lon_grid",

    "detection_count",

    "mean_bright_ti4",
    "max_bright_ti4",

    "mean_bright_ti5",
    "max_bright_ti5",

    "mean_frp",
    "max_frp",

    "active_days",
    "active_days_30d",
    "active_days_90d",
    "active_days_180d",

    "persistence_score",

    "nearest_mine_quarry_km",
    "nearest_power_plant_km",
    "nearest_industrial_area_km",
    "nearest_industrial_works_km"
]


preview_columns = [
    column
    for column in preview_columns
    if column in features.columns
]


print(
    features[
        preview_columns
    ]
    .head(21)
    .round(3)
    .to_string(index=False)
)


print("\nDone.")