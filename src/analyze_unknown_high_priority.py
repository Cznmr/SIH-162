import os
import pandas as pd


ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PREDICTIONS_PATH = os.path.join(
    ROOT,
    "data",
    "training",
    "unknown_high_priority_review.csv"
)

ML_PATH = os.path.join(
    ROOT,
    "data",
    "training",
    "ml_feature_table.csv"
)

OUTPUT_PATH = os.path.join(
    ROOT,
    "data",
    "training",
    "unknown_high_priority_evidence.csv"
)


print("=" * 70)
print("UNKNOWN HIGH-PRIORITY EVIDENCE ANALYSIS")
print("=" * 70)


# ---------------------------------------------------------
# 1. Load predictions
# ---------------------------------------------------------
print("\n[1/4] Loading high-priority predictions...")

pred = pd.read_csv(PREDICTIONS_PATH)

print("Prediction rows:", len(pred))


# ---------------------------------------------------------
# 2. Load feature table
# ---------------------------------------------------------
print("\n[2/4] Loading ML feature table...")

df = pd.read_csv(ML_PATH)

print("Feature-table rows:", len(df))


# ---------------------------------------------------------
# 3. Merge using exact grid coordinates
# ---------------------------------------------------------
print("\n[3/4] Merging evidence...")

if not {"lat_grid", "lon_grid"}.issubset(pred.columns):
    raise ValueError(
        "Prediction file must contain lat_grid and lon_grid."
    )

if not {"lat_grid", "lon_grid"}.issubset(df.columns):
    raise ValueError(
        "ML feature table must contain lat_grid and lon_grid."
    )


# Only keep useful evidence columns
evidence_columns = [
    "lat_grid",
    "lon_grid",

    # FIRMS / thermal
    "detection_count",
    "mean_frp",
    "max_frp",
    "mean_bright_ti4",
    "max_bright_ti4",
    "mean_bright_ti5",
    "max_bright_ti5",
    "day_detections",
    "night_detections",
    "night_ratio",

    # Persistence
    "active_days",
    "active_days_30d",
    "active_days_90d",
    "active_days_180d",
    "persistence_score",
    "persistent_30d",
    "persistent_90d",
    "persistent_180d",
    "observation_span_days",

    # OSM
    "nearest_industrial_area_km",
    "nearest_industrial_works_km",
    "nearest_power_plant_km",
    "nearest_power_infrastructure_km",
    "nearest_storage_tank_km",
    "nearest_relevant_osm_km",

    # Dynamic World
    "dw_vegetation_score",
    "dw_nonvegetation_score",
    "dw_built",
    "dw_trees",
    "dw_grass",
    "dw_crops",
    "dw_shrub_and_scrub",
    "dw_bare",

    # Spatial behavior
    "spatial_observation_days",
    "expansion_days_3km",
    "expansion_days_5km",
    "new_neighbor_days_3km",
    "new_neighbor_days_5km",
    "persistent_local_days_5km",
    "strong_persistence_days_5km",
    "directional_days_5km",
    "expansion_day_ratio_5km",
    "new_neighbor_day_ratio_5km",
    "persistent_day_ratio_5km",
]


available_columns = [
    c for c in evidence_columns
    if c in df.columns
]


evidence = df[
    available_columns
].copy()


result = pred.merge(
    evidence,
    on=["lat_grid", "lon_grid"],
    how="left",
    validate="one_to_one"
)


# ---------------------------------------------------------
# 4. Create simple evidence indicators
# ---------------------------------------------------------
print("\n[4/4] Creating evidence indicators...")


# Thermal strength
result["thermal_frp_high"] = (
    result["max_frp"] >= 10
)

result["thermal_frp_very_high"] = (
    result["max_frp"] >= 15
)

result["repeated_detections"] = (
    result["detection_count"] >= 3
)


# Vegetation context
result["vegetation_context"] = (
    result["dw_vegetation_score"] >= 0.70
)


# Industrial context
result["industrial_context"] = (
    (result["nearest_industrial_area_km"] <= 5) |
    (result["nearest_industrial_works_km"] <= 5) |
    (result["nearest_power_infrastructure_km"] <= 5) |
    (result["nearest_storage_tank_km"] <= 5)
)


# Strong persistence
result["persistent_context"] = (
    (result["active_days_180d"] >= 5) &
    (result["observation_span_days"] >= 60)
)


# Spatial persistence
result["spatial_persistence"] = (
    result["persistent_local_days_5km"] >= 5
)


# ---------------------------------------------------------
# Evidence scores
# ---------------------------------------------------------

result["wildfire_evidence_score"] = (
    result["thermal_frp_high"].astype(int)
    + result["repeated_detections"].astype(int)
    + result["vegetation_context"].astype(int)
    + (~result["industrial_context"]).astype(int)
)


result["persistent_evidence_score"] = (
    (result["active_days_180d"] >= 5).astype(int)
    + (result["observation_span_days"] >= 60).astype(int)
    + result["spatial_persistence"].astype(int)
    + (result["persistent_90d"]).astype(int)
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
result.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print("Rows analyzed:", len(result))

print("\nPredicted classes:")
print(
    result["predicted_class"].value_counts()
)

print("\nWildfire evidence score:")
print(
    result["wildfire_evidence_score"].value_counts()
    .sort_index()
)

print("\nPersistent evidence score:")
print(
    result["persistent_evidence_score"].value_counts()
    .sort_index()
)

print("\nStrong wildfire evidence:")
print(
    (
        (result["predicted_class"] == "WILDFIRE") &
        (result["wildfire_evidence_score"] >= 3)
    ).sum()
)

print("\nStrong persistent evidence:")
print(
    (
        (result["predicted_class"] == "PERSISTENT_THERMAL_SOURCE") &
        (result["persistent_evidence_score"] >= 3)
    ).sum()
)

print("\nSaved:")
print(OUTPUT_PATH)

print("=" * 70)