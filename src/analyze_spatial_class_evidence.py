from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PS 26162
# SPATIAL / CLASS EVIDENCE ANALYSIS — CORRECTED VERSION
# ============================================================
#
# PURPOSE:
# Analyze whether spatial fire behavior helps distinguish:
#
#   WILDFIRE
#   INDUSTRIAL_FIRE
#   PERSISTENT_THERMAL_SOURCE
#   OTHER
#   UNKNOWN
#
# IMPORTANT:
# The spatial file contains DAILY observations.
# We therefore aggregate different features according to
# what they actually represent.
#
# This script DOES NOT modify seed labels.
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\ENTRO-26162"
)

SEED_FILE = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "seed_labels.csv"
)

SPATIAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spatial_fire_behavior_v2.csv"
)

OUTPUT_FEATURES = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "spatial_location_evidence.csv"
)

OUTPUT_STATS = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "spatial_class_evidence_analysis.csv"
)

OUTPUT_THRESHOLDS = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "spatial_class_threshold_analysis.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

RADII = ["1.5km", "3km", "5km"]

CLASSES = [
    "WILDFIRE",
    "INDUSTRIAL_FIRE",
    "PERSISTENT_THERMAL_SOURCE",
    "OTHER",
    "UNKNOWN",
]


print("=" * 75)
print("PS 26162 — SPATIAL / CLASS EVIDENCE ANALYSIS")
print("CORRECTED LOCATION-LEVEL AGGREGATION")
print("=" * 75)


# ============================================================
# 1. LOAD
# ============================================================

print("\n[1/8] Loading data...")

seed = pd.read_csv(
    SEED_FILE,
    low_memory=False
)

spatial = pd.read_csv(
    SPATIAL_FILE,
    low_memory=False
)

print(f"Seeds:   {len(seed):,}")
print(f"Spatial: {len(spatial):,}")


# ============================================================
# 2. VALIDATE SOURCE SCHEMA
# ============================================================

print("\n[2/8] Validating source schema...")


required_seed = [
    "lat_grid",
    "lon_grid",
    "seed_label",
]

missing_seed = [
    c for c in required_seed
    if c not in seed.columns
]

if missing_seed:
    raise ValueError(
        f"Seed file missing columns: {missing_seed}"
    )


required_spatial = [
    "acq_date",
    "latitude",
    "longitude",
    "lat_idx",
    "lon_idx",
]

missing_spatial = [
    c for c in required_spatial
    if c not in spatial.columns
]

if missing_spatial:
    raise ValueError(
        f"Spatial file missing columns: {missing_spatial}"
    )


# ============================================================
# 3. USE VERIFIED GRID
# ============================================================

print("\n[3/8] Creating verified spatial grid keys...")


# IMPORTANT:
# lat_idx and lon_idx are already integer representations
# of the 0.01-degree grid.
#
# Example:
# lat_idx = 1714
# -> grid latitude = 17.14
#
# Do NOT floor them again.

spatial["lat_grid"] = (
    spatial["lat_idx"] / 100.0
).round(2)

spatial["lon_grid"] = (
    spatial["lon_idx"] / 100.0
).round(2)


# Validate against actual coordinates.

lat_difference = (
    spatial["latitude"] - spatial["lat_grid"]
).abs()

lon_difference = (
    spatial["longitude"] - spatial["lon_grid"]
).abs()


print(
    "Maximum latitude grid difference:",
    f"{lat_difference.max():.6f}"
)

print(
    "Maximum longitude grid difference:",
    f"{lon_difference.max():.6f}"
)


spatial_locations = (
    spatial[
        ["lat_grid", "lon_grid"]
    ]
    .drop_duplicates()
)


seed_locations = (
    seed[
        ["lat_grid", "lon_grid"]
    ]
    .drop_duplicates()
)


print(
    f"Spatial unique locations: "
    f"{len(spatial_locations):,}"
)

print(
    f"Seed unique locations: "
    f"{len(seed_locations):,}"
)


# Exact location overlap.

spatial_keys = set(
    zip(
        spatial_locations["lat_grid"],
        spatial_locations["lon_grid"]
    )
)

seed_keys = set(
    zip(
        seed_locations["lat_grid"],
        seed_locations["lon_grid"]
    )
)


print(
    f"Common locations: "
    f"{len(spatial_keys & seed_keys):,}"
)

print(
    f"Spatial-only locations: "
    f"{len(spatial_keys - seed_keys):,}"
)

print(
    f"Seed-only locations: "
    f"{len(seed_keys - spatial_keys):,}"
)


if spatial_keys != seed_keys:

    print(
        "\nWARNING: Spatial and seed location sets "
        "are not identical."
    )


# ============================================================
# 4. CREATE LOCATION-LEVEL SPATIAL FEATURES
# ============================================================

print("\n[4/8] Building location-level spatial evidence...")


grid = [
    "lat_grid",
    "lon_grid",
]


location = (
    spatial[
        grid
    ]
    .drop_duplicates()
    .copy()
)


# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------

def add_feature(name, values):
    location[name] = (
        values
        .reindex(
            pd.MultiIndex.from_frame(
                location[grid]
            )
        )
        .values
    )


spatial_index = spatial.set_index(
    grid
)


# ============================================================
# BASIC SPATIAL ACTIVITY
# ============================================================

print("  Creating activity features...")


# Number of daily spatial observations.
obs_days = (
    spatial
    .groupby(grid)
    .size()
)

add_feature(
    "spatial_observation_days",
    obs_days
)


# Number of days where spatial expansion occurred.
for radius in RADII:

    suffix = radius

    expansion_col = (
        f"local_population_change_{suffix}"
    )

    new_col = (
        f"new_neighbor_count_{suffix}"
    )

    if expansion_col in spatial.columns:

        values = (
            spatial
            .assign(
                _active_expansion=
                spatial[expansion_col] > 0
            )
            .groupby(grid)["_active_expansion"]
            .sum()
        )

        add_feature(
            f"expansion_days_{suffix}",
            values
        )

    if new_col in spatial.columns:

        values = (
            spatial
            .assign(
                _new_activity=
                spatial[new_col] > 0
            )
            .groupby(grid)["_new_activity"]
            .sum()
        )

        add_feature(
            f"new_activity_days_{suffix}",
            values
        )


# ============================================================
# PERSISTENCE
# ============================================================

print("  Creating persistence features...")


for radius in RADII:

    suffix = radius

    col = (
        f"local_persistence_{suffix}"
    )

    if col not in spatial.columns:
        continue


    values_max = (
        spatial
        .groupby(grid)[col]
        .max()
    )

    values_mean = (
        spatial
        .groupby(grid)[col]
        .mean()
    )


    add_feature(
        f"max_local_persistence_{suffix}",
        values_max
    )

    add_feature(
        f"mean_local_persistence_{suffix}",
        values_mean
    )


# ============================================================
# NEIGHBOR ACTIVITY
# ============================================================

print("  Creating neighbor activity features...")


for radius in RADII:

    suffix = radius

    for base in [
        "new_neighbor_count",
        "continuing_neighbor_count",
        "local_population",
        "local_population_change",
    ]:

        col = f"{base}_{suffix}"

        if col not in spatial.columns:
            continue


        values_sum = (
            spatial
            .groupby(grid)[col]
            .sum()
        )

        values_max = (
            spatial
            .groupby(grid)[col]
            .max()
        )

        values_mean = (
            spatial
            .groupby(grid)[col]
            .mean()
        )


        add_feature(
            f"sum_{base}_{suffix}",
            values_sum
        )

        add_feature(
            f"max_{base}_{suffix}",
            values_max
        )

        add_feature(
            f"mean_{base}_{suffix}",
            values_mean
        )


# ============================================================
# EXPANSION RATE
# ============================================================

print("  Creating expansion-rate features...")


for radius in RADII:

    suffix = radius

    col = (
        f"local_expansion_rate_{suffix}"
    )

    if col not in spatial.columns:
        continue


    values_max = (
        spatial
        .groupby(grid)[col]
        .max()
    )

    values_mean = (
        spatial
        .groupby(grid)[col]
        .mean()
    )


    add_feature(
        f"max_expansion_rate_{suffix}",
        values_max
    )

    add_feature(
        f"mean_expansion_rate_{suffix}",
        values_mean
    )


# ============================================================
# DISTANCE OF NEW ACTIVITY
# ============================================================

print("  Creating spread-distance features...")


for radius in RADII:

    suffix = radius

    for base in [
        "mean_new_distance",
        "max_new_distance",
    ]:

        col = f"{base}_{suffix}"

        if col not in spatial.columns:
            continue


        values_max = (
            spatial
            .groupby(grid)[col]
            .max()
        )

        values_mean = (
            spatial
            .groupby(grid)[col]
            .mean()
        )


        add_feature(
            f"max_{base}_{suffix}",
            values_max
        )

        add_feature(
            f"mean_{base}_{suffix}",
            values_mean
        )


# ============================================================
# DIRECTIONAL BEHAVIOR
# ============================================================

print("  Creating directional features...")


for radius in RADII:

    suffix = radius

    col = (
        f"directional_consistency_{suffix}"
    )

    if col in spatial.columns:

        values_max = (
            spatial
            .groupby(grid)[col]
            .max()
        )

        values_mean = (
            spatial
            .groupby(grid)[col]
            .mean()
        )


        add_feature(
            f"max_directional_consistency_{suffix}",
            values_max
        )

        add_feature(
            f"mean_directional_consistency_{suffix}",
            values_mean
        )


# Directional activity totals.

direction_columns = [
    "n_activity",
    "s_activity",
    "e_activity",
    "w_activity",
    "ne_activity",
    "nw_activity",
    "se_activity",
    "sw_activity",
]


for radius in RADII:

    suffix = radius

    available = [
        f"{d}_{suffix}"
        for d in direction_columns
        if f"{d}_{suffix}" in spatial.columns
    ]


    if not available:
        continue


    for col in available:

        values = (
            spatial
            .groupby(grid)[col]
            .sum()
        )

        add_feature(
            f"sum_{col}",
            values
        )


# ============================================================
# DAY-TO-DAY BEHAVIOR
# ============================================================

print("  Creating temporal-spatial behavior...")


if "is_new_today" in spatial.columns:

    values = (
        spatial
        .groupby(grid)["is_new_today"]
        .sum()
    )

    add_feature(
        "new_detection_days",
        values
    )


if "was_active_yesterday" in spatial.columns:

    values = (
        spatial
        .groupby(grid)["was_active_yesterday"]
        .sum()
    )

    add_feature(
        "continued_from_previous_day",
        values
    )


if "consecutive_day_comparison" in spatial.columns:

    values = (
        spatial
        .groupby(grid)["consecutive_day_comparison"]
        .sum()
    )

    add_feature(
        "consecutive_day_count",
        values
    )


# ============================================================
# FILL ONLY FEATURES THAT ARE MATHEMATICALLY COUNTS
# ============================================================

numeric_columns = [
    c
    for c in location.columns
    if c not in grid
]


for column in numeric_columns:

    location[column] = pd.to_numeric(
        location[column],
        errors="coerce"
    )


# ============================================================
# 5. MERGE SEED LABELS
# ============================================================

print("\n[5/8] Merging seed labels...")


data = location.merge(
    seed[
        grid + ["seed_label"]
    ],
    on=grid,
    how="left",
)


print(
    f"Location rows: "
    f"{len(data):,}"
)


print("\nSeed distribution:")

print(
    data["seed_label"]
    .value_counts(dropna=False)
)


# ============================================================
# 6. CLASS DISTRIBUTION ANALYSIS
# ============================================================

print("\n[6/8] Calculating class statistics...")


results = []


for feature in numeric_columns:

    print(
        f"\n{'-' * 75}"
    )

    print(
        f"FEATURE: {feature}"
    )

    print(
        f"{'-' * 75}"
    )


    for class_name in CLASSES:

        values = (
            data
            .loc[
                data["seed_label"] == class_name,
                feature
            ]
            .dropna()
        )


        if len(values) == 0:
            continue


        results.append({
            "feature": feature,
            "class": class_name,
            "count": len(values),
            "mean": values.mean(),
            "median": values.median(),
            "q75": values.quantile(0.75),
            "q90": values.quantile(0.90),
            "q95": values.quantile(0.95),
            "max": values.max(),
        })


        print(
            f"{class_name:28s}"
            f" n={len(values):6,}"
            f" mean={values.mean():8.3f}"
            f" median={values.median():8.3f}"
            f" q90={values.quantile(0.90):8.3f}"
            f" max={values.max():8.3f}"
        )


stats_df = pd.DataFrame(results)


# ============================================================
# 7. THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("THRESHOLD ANALYSIS")
print("=" * 75)


threshold_results = []


# Use thresholds based on the actual feature scales.
thresholds_by_feature = {}


for feature in numeric_columns:

    values = (
        data[feature]
        .dropna()
    )


    if len(values) == 0:
        continue


    unique_values = np.sort(
        values.unique()
    )


    # For binary / probability-like features.
    if values.max() <= 1:

        thresholds_by_feature[feature] = [
            0.25,
            0.50,
            0.75,
            0.90,
        ]

    else:

        # Quantile-based thresholds prevent
        # meaningless thresholds such as >=20
        # for a feature whose maximum is only 4.
        thresholds_by_feature[feature] = sorted(
            set([
                float(values.quantile(0.50)),
                float(values.quantile(0.75)),
                float(values.quantile(0.90)),
                float(values.quantile(0.95)),
            ])
        )


for feature, thresholds in thresholds_by_feature.items():

    print(
        f"\nFEATURE: {feature}"
    )


    for threshold in thresholds:

        print(
            f"\n  >= {threshold:.4f}"
        )


        for class_name in CLASSES:

            subset = data[
                data["seed_label"] == class_name
            ]


            if len(subset) == 0:
                continue


            count = (
                subset[feature] >= threshold
            ).sum()


            percentage = (
                count / len(subset) * 100
            )


            threshold_results.append({
                "feature": feature,
                "threshold": threshold,
                "class": class_name,
                "count": count,
                "percentage": percentage,
            })


            print(
                f"    {class_name:28s}"
                f" {count:6,}"
                f" ({percentage:6.2f}%)"
            )


threshold_df = pd.DataFrame(
    threshold_results
)


# ============================================================
# 8. SAVE
# ============================================================

print("\n[8/8] Saving results...")


location.to_csv(
    OUTPUT_FEATURES,
    index=False
)


stats_df.to_csv(
    OUTPUT_STATS,
    index=False
)


threshold_df.to_csv(
    OUTPUT_THRESHOLDS,
    index=False
)


print("\n" + "=" * 75)
print("FILES SAVED")
print("=" * 75)

print(
    f"\nLocation-level features:\n"
    f"{OUTPUT_FEATURES}"
)

print(
    f"\nClass statistics:\n"
    f"{OUTPUT_STATS}"
)

print(
    f"\nThreshold analysis:\n"
    f"{OUTPUT_THRESHOLDS}"
)


print("\n" + "=" * 75)
print("SPATIAL / CLASS EVIDENCE ANALYSIS COMPLETE")
print("=" * 75)