import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ML_FILE = PROJECT_ROOT / "data" / "training" / "ml_feature_table.csv"
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "persistent_candidates_analysis.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading ML feature table...")

df = pd.read_csv(ML_FILE)

print(f"Loaded rows: {len(df):,}")
print(f"Loaded columns: {len(df.columns)}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "lat_grid",
    "lon_grid",
    "seed_label",

    "max_frp",
    "mean_frp",
    "detection_count",

    "active_days",
    "active_days_180d",
    "observation_span_days",
    "persistence_score",

    "nearest_relevant_osm_km",
    "nearest_industrial_area_km",
    "nearest_industrial_works_km",
    "nearest_power_infrastructure_km",
    "nearest_storage_tank_km",

    "dw_vegetation_score",
    "dw_built",

    "spatial_observation_days",
    "expansion_days_5km",
    "new_neighbor_days_5km",
    "persistent_local_days_5km",
    "strong_persistence_days_5km",
    "directional_days_5km",
    "expansion_day_ratio_5km",
    "new_neighbor_day_ratio_5km",
    "persistent_day_ratio_5km",
]


missing = [c for c in required_columns if c not in df.columns]

if missing:
    print("\nERROR: Missing required columns:")
    for c in missing:
        print(f"  - {c}")
    raise SystemExit(1)


# ============================================================
# IDENTIFY PERSISTENT CANDIDATES
# ============================================================

# Existing confirmed-by-seed-rule persistent locations
known_persistent = df[
    df["seed_label"] == "PERSISTENT_THERMAL_SOURCE"
].copy()


# UNKNOWN locations satisfying the proposed persistent rule
#
# Same rule tested earlier:
# active_days_180d >= 5
# observation_span_days >= 60
unknown_candidates = df[
    (df["seed_label"] == "UNKNOWN")
    & (df["active_days_180d"] >= 5)
    & (df["observation_span_days"] >= 60)
].copy()


print("\n" + "=" * 70)
print("PERSISTENT THERMAL SOURCE ANALYSIS")
print("=" * 70)

print(f"\nKnown persistent seeds     : {len(known_persistent)}")
print(f"UNKNOWN persistent candidates: {len(unknown_candidates)}")


# ============================================================
# COLUMNS TO INSPECT
# ============================================================

inspection_columns = [
    "lat_grid",
    "lon_grid",

    "max_frp",
    "mean_frp",
    "detection_count",

    "active_days",
    "active_days_180d",
    "observation_span_days",
    "persistence_score",

    "nearest_relevant_osm_km",
    "nearest_industrial_area_km",
    "nearest_industrial_works_km",
    "nearest_power_infrastructure_km",
    "nearest_storage_tank_km",

    "dw_vegetation_score",
    "dw_built",

    "spatial_observation_days",
    "expansion_days_5km",
    "new_neighbor_days_5km",
    "persistent_local_days_5km",
    "strong_persistence_days_5km",
    "directional_days_5km",
    "expansion_day_ratio_5km",
    "new_neighbor_day_ratio_5km",
    "persistent_day_ratio_5km",
]


# ============================================================
# PRINT KNOWN PERSISTENT SEEDS
# ============================================================

print("\n" + "=" * 70)
print("KNOWN PERSISTENT THERMAL SOURCE SEEDS")
print("=" * 70)

print(
    known_persistent[inspection_columns]
    .sort_values(
        ["active_days_180d", "observation_span_days"],
        ascending=False
    )
    .to_string(index=False)
)


# ============================================================
# PRINT UNKNOWN CANDIDATES
# ============================================================

print("\n" + "=" * 70)
print("UNKNOWN PERSISTENT CANDIDATES")
print("=" * 70)

print(
    unknown_candidates[inspection_columns]
    .sort_values(
        ["active_days_180d", "observation_span_days"],
        ascending=False
    )
    .to_string(index=False)
)


# ============================================================
# SUMMARY COMPARISON
# ============================================================

comparison_features = [
    "max_frp",
    "mean_frp",
    "detection_count",
    "active_days",
    "active_days_180d",
    "observation_span_days",
    "persistence_score",

    "nearest_relevant_osm_km",
    "nearest_industrial_area_km",
    "nearest_industrial_works_km",
    "nearest_power_infrastructure_km",
    "nearest_storage_tank_km",

    "dw_vegetation_score",
    "dw_built",

    "spatial_observation_days",
    "expansion_days_5km",
    "new_neighbor_days_5km",
    "persistent_local_days_5km",
    "strong_persistence_days_5km",
    "directional_days_5km",
    "expansion_day_ratio_5km",
    "new_neighbor_day_ratio_5km",
    "persistent_day_ratio_5km",
]


print("\n" + "=" * 70)
print("DISTRIBUTION COMPARISON")
print("=" * 70)

summary_rows = []

for feature in comparison_features:

    known_values = pd.to_numeric(
        known_persistent[feature],
        errors="coerce"
    ).dropna()

    candidate_values = pd.to_numeric(
        unknown_candidates[feature],
        errors="coerce"
    ).dropna()

    if len(known_values) == 0 or len(candidate_values) == 0:
        continue

    summary_rows.append({
        "feature": feature,

        "known_persistent_mean": known_values.mean(),
        "known_persistent_median": known_values.median(),

        "candidate_mean": candidate_values.mean(),
        "candidate_median": candidate_values.median(),

        "candidate_min": candidate_values.min(),
        "candidate_max": candidate_values.max(),
    })


comparison_df = pd.DataFrame(summary_rows)

print(
    comparison_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SIMPLE PROFILE DISTANCE
# ============================================================

print("\n" + "=" * 70)
print("CANDIDATE PROFILE SIMILARITY")
print("=" * 70)

# We compare candidates against the median profile of the
# existing persistent seeds.
#
# This is NOT a classifier.
# It is only a diagnostic similarity measure.

profile_features = [
    "max_frp",
    "mean_frp",
    "detection_count",
    "active_days",
    "active_days_180d",
    "observation_span_days",
    "persistence_score",
    "nearest_relevant_osm_km",
    "nearest_industrial_area_km",
    "nearest_industrial_works_km",
    "nearest_power_infrastructure_km",
    "nearest_storage_tank_km",
    "dw_vegetation_score",
    "dw_built",
    "spatial_observation_days",
    "expansion_days_5km",
    "new_neighbor_days_5km",
    "persistent_local_days_5km",
    "strong_persistence_days_5km",
    "directional_days_5km",
    "expansion_day_ratio_5km",
    "new_neighbor_day_ratio_5km",
    "persistent_day_ratio_5km",
]


# Convert to numeric
known_numeric = known_persistent[profile_features].apply(
    pd.to_numeric,
    errors="coerce"
)

candidate_numeric = unknown_candidates[profile_features].apply(
    pd.to_numeric,
    errors="coerce"
)


# Median profile from known persistent seeds
known_median = known_numeric.median()

# IQR used as a robust scale
q25 = known_numeric.quantile(0.25)
q75 = known_numeric.quantile(0.75)

iqr = q75 - q25

# Prevent division by zero
iqr = iqr.replace(0, 1)


# Calculate standardized distance
candidate_scores = []

for idx, row in candidate_numeric.iterrows():

    differences = (row - known_median).abs() / iqr

    differences = differences.replace(
        [np.inf, -np.inf],
        np.nan
    )

    score = differences.mean()

    candidate_scores.append({
        "original_index": idx,
        "profile_distance": score
    })


distance_df = pd.DataFrame(candidate_scores)


# ============================================================
# COMBINE RESULTS
# ============================================================

result = unknown_candidates[inspection_columns].copy()

result["profile_distance"] = result.index.map(
    distance_df.set_index("original_index")["profile_distance"]
)


# Smaller distance = more similar to known persistent seeds
result = result.sort_values(
    "profile_distance",
    ascending=True
)


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("TOP CANDIDATES MOST SIMILAR TO KNOWN PERSISTENT SEEDS")
print("=" * 70)

display_columns = [
    "lat_grid",
    "lon_grid",
    "max_frp",
    "mean_frp",
    "detection_count",
    "active_days",
    "active_days_180d",
    "observation_span_days",
    "persistence_score",
    "nearest_relevant_osm_km",
    "nearest_industrial_area_km",
    "nearest_industrial_works_km",
    "nearest_power_infrastructure_km",
    "nearest_storage_tank_km",
    "dw_vegetation_score",
    "dw_built",
    "profile_distance",
]

print(
    result[display_columns]
    .head(14)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\n" + "=" * 70)
print("OUTPUT")
print("=" * 70)

print(f"\nSaved analysis to:")
print(OUTPUT_FILE)

print("\nIMPORTANT:")
print("These candidates have NOT been relabeled.")
print("profile_distance is only a similarity diagnostic.")
print("Smaller profile_distance = more similar to known persistent seeds.")