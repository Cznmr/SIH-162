import pandas as pd
import numpy as np

INPUT = "data/training/master_feature_table_final.csv"
OUTPUT = "data/training/seed_labels.csv"

df = pd.read_csv(INPUT, low_memory=False)

# ---------------------------------------------------------
# Helper conditions
# ---------------------------------------------------------

# Thermal intensity
high_frp = df["max_frp"] >= 30
moderate_frp = df["max_frp"] >= 15

# Repeated / persistent activity
high_persistence = df["active_days_180d"] >= 10
moderate_persistence = df["active_days_180d"] >= 5

# Long-term observation
long_span = df["observation_span_days"] >= 60

# Industrial / infrastructure context
near_industrial = df["nearest_industrial_area_km"] <= 5
near_industrial_works = df["nearest_industrial_works_km"] <= 5
near_power = df["nearest_power_plant_km"] <= 5
near_storage = df["nearest_storage_tank_km"] <= 5

industrial_context = (
    near_industrial
    | near_industrial_works
    | near_power
    | near_storage
)

# Built-up context
built_context = df["dw_built"] >= 0.20

# Vegetation context
vegetated_context = df["dw_vegetation_score"] >= 0.70

# ---------------------------------------------------------
# Initialize
# ---------------------------------------------------------

df["seed_label"] = "UNKNOWN"
df["seed_reason"] = ""

# ---------------------------------------------------------
# 1. PERSISTENT THERMAL SOURCE
# ---------------------------------------------------------
#
# Very strong recurrence + long observation + infrastructure
# context.
#
# We deliberately keep this strict because only 21 locations
# have active_days_180d >= 10.
# ---------------------------------------------------------

persistent_condition = (
    high_persistence
    & long_span
    & industrial_context
)

df.loc[persistent_condition, "seed_label"] = "PERSISTENT_THERMAL_SOURCE"
df.loc[persistent_condition, "seed_reason"] = (
    "high_180d_persistence + long_observation + infrastructure_context"
)

# ---------------------------------------------------------
# 2. INDUSTRIAL FIRE
# ---------------------------------------------------------
#
# High thermal intensity + industrial/infrastructure context.
#
# We also require the event NOT to look strongly persistent,
# because a continuously recurring source is better represented
# by PERSISTENT_THERMAL_SOURCE.
# ---------------------------------------------------------

industrial_fire_condition = (
    high_frp
    & industrial_context
    & ~high_persistence
)

df.loc[industrial_fire_condition, "seed_label"] = "INDUSTRIAL_FIRE"
df.loc[industrial_fire_condition, "seed_reason"] = (
    "high_frp + infrastructure_context + nonpersistent"
)

# ---------------------------------------------------------
# 3. WILDFIRE
# ---------------------------------------------------------
#
# Strong thermal event + vegetation context + weak industrial
# context.
#
# This is deliberately conservative.
# ---------------------------------------------------------

wildfire_condition = (
    high_frp
    & vegetated_context
    & ~industrial_context
    & ~high_persistence
)

df.loc[wildfire_condition, "seed_label"] = "WILDFIRE"
df.loc[wildfire_condition, "seed_reason"] = (
    "high_frp + strong_vegetation_context + weak_industrial_context"
)

# ---------------------------------------------------------
# 4. OTHER THERMAL ACTIVITY
# ---------------------------------------------------------
#
# Moderate thermal activity that doesn't fit the stronger
# categories above.
# ---------------------------------------------------------

other_condition = (
    moderate_frp
    & ~high_frp
    & ~persistent_condition
    & ~industrial_fire_condition
    & ~wildfire_condition
)

df.loc[other_condition, "seed_label"] = "OTHER"
df.loc[other_condition, "seed_reason"] = (
    "moderate_thermal_activity_without_strong_class_context"
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

output_cols = [
    "lat_grid",
    "lon_grid",
    "seed_label",
    "seed_reason",
    "detection_count",
    "max_frp",
    "mean_frp",
    "max_bright_ti4",
    "mean_bright_ti4",
    "night_ratio",
    "observation_span_days",
    "active_days",
    "active_days_30d",
    "active_days_90d",
    "active_days_180d",
    "persistence_score",
    "nearest_industrial_area_km",
    "nearest_industrial_works_km",
    "nearest_power_plant_km",
    "nearest_storage_tank_km",
    "nearest_relevant_osm_km",
    "dw_built",
    "dw_vegetation_score",
    "dw_nonvegetation_score",
]

# Keep only columns that exist
output_cols = [c for c in output_cols if c in df.columns]

df[output_cols].to_csv(OUTPUT, index=False)

print("\nSeed labeling complete.")
print(f"Input rows: {len(df):,}")
print(f"Output: {OUTPUT}")

print("\nSeed label counts:")
print(df["seed_label"].value_counts().to_string())

print("\nSeed label percentages:")
print(
    (df["seed_label"].value_counts(normalize=True) * 100)
    .round(2)
    .to_string()
)

print("\nLabeled rows:")
print(f"INDUSTRIAL_FIRE:          {(df.seed_label == 'INDUSTRIAL_FIRE').sum():,}")
print(f"PERSISTENT_THERMAL_SOURCE:{(df.seed_label == 'PERSISTENT_THERMAL_SOURCE').sum():,}")
print(f"WILDFIRE:                 {(df.seed_label == 'WILDFIRE').sum():,}")
print(f"OTHER:                    {(df.seed_label == 'OTHER').sum():,}")
print(f"UNKNOWN:                  {(df.seed_label == 'UNKNOWN').sum():,}")                                                              