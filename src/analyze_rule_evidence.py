from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PS 26162 — Rule Evidence Distribution Analysis
#
# PURPOSE:
# Analyze the existing 47,677 locations BEFORE changing
# seed_labeling.py.
#
# This script does NOT create labels.
# It only measures how much evidence exists in the dataset.
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\ENTRO-26162"
)

MASTER_FILE = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "master_feature_table_final.csv"
)

SPATIAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spatial_fire_behavior_v2.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "rule_evidence_distribution.csv"
)


# ============================================================
# LOAD MASTER DATA
# ============================================================

print("=" * 70)
print("PS 26162 — RULE EVIDENCE DISTRIBUTION ANALYSIS")
print("=" * 70)

print("\n[1/5] Loading master feature table...")

master = pd.read_csv(MASTER_FILE)

print(f"Master rows: {len(master):,}")
print(f"Master columns: {len(master.columns):,}")


# ============================================================
# VERIFY CORE FEATURES
# ============================================================

required = [
    "lat_grid",
    "lon_grid",

    # Thermal
    "max_frp",
    "mean_frp",
    "detection_count",

    # Temporal
    "active_days",
    "active_days_30d",
    "active_days_90d",
    "active_days_180d",
    "observation_span_days",

    # Persistence
    "persistence_score",

    # Dynamic World
    "dw_vegetation_score",
    "dw_built",

    # OSM
    "nearest_industrial_area_km",
    "nearest_industrial_works_km",
    "nearest_power_infrastructure_km",
    "nearest_storage_tank_km",
    "nearest_relevant_osm_km",
]


missing = [
    col
    for col in required
    if col not in master.columns
]

if missing:
    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(missing)
    )

print("\nCore feature check: PASSED")


# ============================================================
# LOAD SPATIAL FEATURES
# ============================================================

print("\n[2/5] Loading spatial behavior...")

spatial = pd.read_csv(SPATIAL_FILE)

print(f"Spatial rows: {len(spatial):,}")
print(f"Spatial columns: {len(spatial.columns):,}")


# ============================================================
# FIND SPATIAL FEATURES
# ============================================================

spatial_candidates = [
    "spatial_observation_days",
    "new_cell_days",
    "continuing_cell_days",
    "expansion_days_5km",
    "persistent_local_days_5km",
    "strong_persistence_days",
    "directional_days_5km",
]


available_spatial = [
    col
    for col in spatial_candidates
    if col in spatial.columns
]

print("\nAvailable spatial evidence:")

for col in available_spatial:
    print(f"  - {col}")


# ============================================================
# MERGE SPATIAL FEATURES
# ============================================================

print("\n[3/5] Merging spatial evidence...")

if "lat_idx" in spatial.columns and "lon_idx" in spatial.columns:

    spatial["lat_grid"] = (
        np.floor(
            spatial["lat_idx"] * 0.01
        )
        .round(2)
    )

    spatial["lon_grid"] = (
        np.floor(
            spatial["lon_idx"] * 0.01
        )
        .round(2)
    )

elif "lat_grid" in spatial.columns and "lon_grid" in spatial.columns:

    pass

else:

    raise ValueError(
        "Spatial file has neither "
        "lat_idx/lon_idx nor lat_grid/lon_grid."
    )


# Keep only useful columns
spatial_keep = [
    "lat_grid",
    "lon_grid",
] + available_spatial

spatial_small = (
    spatial[spatial_keep]
    .drop_duplicates(
        subset=["lat_grid", "lon_grid"]
    )
)


data = master.merge(
    spatial_small,
    on=["lat_grid", "lon_grid"],
    how="left"
)


print(
    f"Merged rows: {len(data):,}"
)


# ============================================================
# EVIDENCE FLAGS
# ============================================================

print("\n[4/5] Calculating evidence thresholds...")


# ------------------------------------------------------------
# Thermal intensity
# ------------------------------------------------------------

data["frp_ge_3"] = (
    data["max_frp"] >= 3
)

data["frp_ge_5"] = (
    data["max_frp"] >= 5
)

data["frp_ge_7"] = (
    data["max_frp"] >= 7
)

data["frp_ge_10"] = (
    data["max_frp"] >= 10
)

data["frp_ge_15"] = (
    data["max_frp"] >= 15
)

data["frp_ge_20"] = (
    data["max_frp"] >= 20
)

data["frp_ge_30"] = (
    data["max_frp"] >= 30
)


# ------------------------------------------------------------
# Repeated detections
# ------------------------------------------------------------

data["detections_ge_2"] = (
    data["detection_count"] >= 2
)

data["detections_ge_3"] = (
    data["detection_count"] >= 3
)

data["detections_ge_5"] = (
    data["detection_count"] >= 5
)

data["detections_ge_10"] = (
    data["detection_count"] >= 10
)


# ------------------------------------------------------------
# Active days
# ------------------------------------------------------------

data["active_ge_2"] = (
    data["active_days"] >= 2
)

data["active_ge_3"] = (
    data["active_days"] >= 3
)

data["active_ge_5"] = (
    data["active_days"] >= 5
)

data["active_ge_10"] = (
    data["active_days"] >= 10
)

data["active_ge_20"] = (
    data["active_days"] >= 20
)


# ------------------------------------------------------------
# 180-day persistence
# ------------------------------------------------------------

data["active180_ge_2"] = (
    data["active_days_180d"] >= 2
)

data["active180_ge_3"] = (
    data["active_days_180d"] >= 3
)

data["active180_ge_5"] = (
    data["active_days_180d"] >= 5
)

data["active180_ge_10"] = (
    data["active_days_180d"] >= 10
)


# ------------------------------------------------------------
# Observation span
# ------------------------------------------------------------

data["span_ge_30"] = (
    data["observation_span_days"] >= 30
)

data["span_ge_60"] = (
    data["observation_span_days"] >= 60
)

data["span_ge_90"] = (
    data["observation_span_days"] >= 90
)

data["span_ge_180"] = (
    data["observation_span_days"] >= 180
)


# ------------------------------------------------------------
# Vegetation
# ------------------------------------------------------------

data["vegetation_ge_0_50"] = (
    data["dw_vegetation_score"] >= 0.50
)

data["vegetation_ge_0_60"] = (
    data["dw_vegetation_score"] >= 0.60
)

data["vegetation_ge_0_70"] = (
    data["dw_vegetation_score"] >= 0.70
)

data["vegetation_ge_0_80"] = (
    data["dw_vegetation_score"] >= 0.80
)


# ------------------------------------------------------------
# Built environment
# ------------------------------------------------------------

data["built_ge_0_10"] = (
    data["dw_built"] >= 0.10
)

data["built_ge_0_20"] = (
    data["dw_built"] >= 0.20
)

data["built_ge_0_30"] = (
    data["dw_built"] >= 0.30
)

data["built_ge_0_50"] = (
    data["dw_built"] >= 0.50
)


# ------------------------------------------------------------
# Industrial context
# ------------------------------------------------------------

industrial_distances = [
    "nearest_industrial_area_km",
    "nearest_industrial_works_km",
    "nearest_power_infrastructure_km",
    "nearest_storage_tank_km",
    "nearest_relevant_osm_km",
]


for distance in industrial_distances:

    short_name = distance.replace(
        "nearest_", ""
    ).replace("_km", "")

    data[f"{short_name}_le_1km"] = (
        data[distance] <= 1
    )

    data[f"{short_name}_le_2km"] = (
        data[distance] <= 2
    )

    data[f"{short_name}_le_5km"] = (
        data[distance] <= 5
    )

    data[f"{short_name}_le_10km"] = (
        data[distance] <= 10
    )


# Any relevant industrial context
data["industrial_context_5km"] = (
    data[
        [
            "nearest_industrial_area_km",
            "nearest_industrial_works_km",
            "nearest_power_infrastructure_km",
            "nearest_storage_tank_km",
        ]
    ]
    .le(5)
    .any(axis=1)
)


data["industrial_context_10km"] = (
    data[
        [
            "nearest_industrial_area_km",
            "nearest_industrial_works_km",
            "nearest_power_infrastructure_km",
            "nearest_storage_tank_km",
        ]
    ]
    .le(10)
    .any(axis=1)
)


# ============================================================
# SPATIAL EVIDENCE
# ============================================================

if "persistent_local_days_5km" in data.columns:

    data["persistent_local_ge_2"] = (
        data["persistent_local_days_5km"] >= 2
    )

    data["persistent_local_ge_5"] = (
        data["persistent_local_days_5km"] >= 5
    )

    data["persistent_local_ge_10"] = (
        data["persistent_local_days_5km"] >= 10
    )


if "expansion_days_5km" in data.columns:

    data["expansion_ge_2"] = (
        data["expansion_days_5km"] >= 2
    )

    data["expansion_ge_5"] = (
        data["expansion_days_5km"] >= 5
    )


# ============================================================
# COMBINATIONS
# ============================================================

print("\nCalculating evidence combinations...")


# ------------------------------------------------------------
# Thermal + repetition
# ------------------------------------------------------------

data["frp5_repeat2"] = (
    data["frp_ge_5"]
    & data["detections_ge_2"]
)

data["frp10_repeat2"] = (
    data["frp_ge_10"]
    & data["detections_ge_2"]
)

data["frp10_repeat3"] = (
    data["frp_ge_10"]
    & data["detections_ge_3"]
)

data["frp15_repeat2"] = (
    data["frp_ge_15"]
    & data["detections_ge_2"]
)


# ------------------------------------------------------------
# Potential wildfire evidence
# ------------------------------------------------------------

data["wildfire_combo_1"] = (
    data["frp_ge_5"]
    & data["detections_ge_2"]
    & data["vegetation_ge_0_60"]
)

data["wildfire_combo_2"] = (
    data["frp_ge_5"]
    & data["detections_ge_2"]
    & data["vegetation_ge_0_70"]
)

data["wildfire_combo_3"] = (
    data["frp_ge_10"]
    & data["detections_ge_2"]
    & data["vegetation_ge_0_60"]
)

data["wildfire_combo_4"] = (
    data["frp_ge_10"]
    & data["detections_ge_2"]
    & data["vegetation_ge_0_70"]
)

data["wildfire_combo_5"] = (
    data["frp_ge_10"]
    & data["detections_ge_3"]
    & data["vegetation_ge_0_70"]
    & ~data["industrial_context_5km"]
)


# ------------------------------------------------------------
# Potential industrial-fire evidence
# ------------------------------------------------------------

data["industrial_combo_1"] = (
    data["frp_ge_5"]
    & data["detections_ge_2"]
    & data["industrial_context_5km"]
)

data["industrial_combo_2"] = (
    data["frp_ge_10"]
    & data["detections_ge_2"]
    & data["industrial_context_5km"]
)

data["industrial_combo_3"] = (
    data["frp_ge_10"]
    & data["detections_ge_3"]
    & data["industrial_context_5km"]
)

data["industrial_combo_4"] = (
    data["frp_ge_5"]
    & data["detections_ge_2"]
    & data["industrial_context_5km"]
    & data["built_ge_0_20"]
)


# ------------------------------------------------------------
# Potential persistent-source evidence
# ------------------------------------------------------------

data["persistent_combo_1"] = (
    data["active180_ge_3"]
    & data["span_ge_30"]
)

data["persistent_combo_2"] = (
    data["active180_ge_5"]
    & data["span_ge_60"]
)

data["persistent_combo_3"] = (
    data["active_ge_5"]
    & data["span_ge_60"]
)

data["persistent_combo_4"] = (
    data["active180_ge_5"]
    & data["span_ge_60"]
    & data["industrial_context_5km"]
)


# ============================================================
# PRINT DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 70)
print("EVIDENCE DISTRIBUTION")
print("=" * 70)


def print_counts(columns):

    for column in columns:

        if column not in data.columns:
            continue

        count = int(
            data[column].fillna(False).sum()
        )

        percentage = (
            count
            / len(data)
            * 100
        )

        print(
            f"{column:<45} "
            f"{count:>7,} "
            f"({percentage:>6.2f}%)"
        )


print("\n--- FRP ---")

print_counts([
    "frp_ge_3",
    "frp_ge_5",
    "frp_ge_7",
    "frp_ge_10",
    "frp_ge_15",
    "frp_ge_20",
    "frp_ge_30",
])


print("\n--- DETECTIONS ---")

print_counts([
    "detections_ge_2",
    "detections_ge_3",
    "detections_ge_5",
    "detections_ge_10",
])


print("\n--- ACTIVE DAYS ---")

print_counts([
    "active_ge_2",
    "active_ge_3",
    "active_ge_5",
    "active_ge_10",
    "active_ge_20",
])


print("\n--- 180-DAY ACTIVITY ---")

print_counts([
    "active180_ge_2",
    "active180_ge_3",
    "active180_ge_5",
    "active180_ge_10",
])


print("\n--- OBSERVATION SPAN ---")

print_counts([
    "span_ge_30",
    "span_ge_60",
    "span_ge_90",
    "span_ge_180",
])


print("\n--- VEGETATION ---")

print_counts([
    "vegetation_ge_0_50",
    "vegetation_ge_0_60",
    "vegetation_ge_0_70",
    "vegetation_ge_0_80",
])


print("\n--- BUILT ---")

print_counts([
    "built_ge_0_10",
    "built_ge_0_20",
    "built_ge_0_30",
    "built_ge_0_50",
])


print("\n--- INDUSTRIAL CONTEXT ---")

print_counts([
    "industrial_context_5km",
    "industrial_context_10km",
])


print("\n--- BASIC THERMAL COMBINATIONS ---")

print_counts([
    "frp5_repeat2",
    "frp10_repeat2",
    "frp10_repeat3",
    "frp15_repeat2",
])


print("\n--- WILDFIRE COMBINATIONS ---")

print_counts([
    "wildfire_combo_1",
    "wildfire_combo_2",
    "wildfire_combo_3",
    "wildfire_combo_4",
    "wildfire_combo_5",
])


print("\n--- INDUSTRIAL COMBINATIONS ---")

print_counts([
    "industrial_combo_1",
    "industrial_combo_2",
    "industrial_combo_3",
    "industrial_combo_4",
])


print("\n--- PERSISTENT COMBINATIONS ---")

print_counts([
    "persistent_combo_1",
    "persistent_combo_2",
    "persistent_combo_3",
    "persistent_combo_4",
])


# ============================================================
# SAVE
# ============================================================

print("\n[5/5] Saving evidence table...")

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

data.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nSaved:\n{OUTPUT_FILE}"
)

print("\n" + "=" * 70)
print("EVIDENCE ANALYSIS COMPLETE")
print("=" * 70)