from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPATIAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spatial_fire_behavior_v2.csv"
)

SEED_FILE = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "seed_labels.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "spatial_behavior_by_class.csv"
)

CLASS_SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "spatial_class_summary.csv"
)

GRID_SIZE_DEG = 0.01


# ============================================================
# START
# ============================================================

print("=" * 70)
print("SPATIAL BEHAVIOR BY SEED CLASS")
print("=" * 70)


# ============================================================
# [1/6] LOAD SPATIAL DATA
# ============================================================

print("\n[1/6] Loading spatial behavior...")

spatial = pd.read_csv(
    SPATIAL_FILE,
    low_memory=False,
)

print(f"Spatial rows: {len(spatial):,}")


# ============================================================
# [2/6] LOAD SEED LABELS
# ============================================================

print("\n[2/6] Loading seed labels...")

seed = pd.read_csv(
    SEED_FILE,
    low_memory=False,
)

print(f"Seed rows: {len(seed):,}")

print("\nSeed classes:")
print(seed["seed_label"].value_counts())


# ============================================================
# [3/6] PREPARE EXACT GRID KEYS
# ============================================================

print("\n[3/6] Preparing exact grid keys...")


# ------------------------------------------------------------
# CHECK SPATIAL COLUMNS
# ------------------------------------------------------------

required_spatial_columns = [
    "lat_idx",
    "lon_idx",
]

missing_spatial_columns = [
    column
    for column in required_spatial_columns
    if column not in spatial.columns
]

if missing_spatial_columns:
    raise ValueError(
        "Spatial file is missing required columns: "
        + ", ".join(missing_spatial_columns)
    )


# ------------------------------------------------------------
# CHECK SEED COLUMNS
# ------------------------------------------------------------

required_seed_columns = [
    "lat_grid",
    "lon_grid",
    "seed_label",
]

missing_seed_columns = [
    column
    for column in required_seed_columns
    if column not in seed.columns
]

if missing_seed_columns:
    raise ValueError(
        "Seed file is missing required columns: "
        + ", ".join(missing_seed_columns)
    )


# ------------------------------------------------------------
# NORMALIZE DATA TYPES
# ------------------------------------------------------------

spatial["lat_idx"] = (
    pd.to_numeric(
        spatial["lat_idx"],
        errors="raise",
    )
    .astype(int)
)

spatial["lon_idx"] = (
    pd.to_numeric(
        spatial["lon_idx"],
        errors="raise",
    )
    .astype(int)
)


seed["lat_grid"] = pd.to_numeric(
    seed["lat_grid"],
    errors="raise",
)

seed["lon_grid"] = pd.to_numeric(
    seed["lon_grid"],
    errors="raise",
)


# ------------------------------------------------------------
# CONVERT SPATIAL GRID INDICES TO GRID COORDINATES
# ------------------------------------------------------------

spatial["lat_grid"] = (
    spatial["lat_idx"] * GRID_SIZE_DEG
).round(2)

spatial["lon_grid"] = (
    spatial["lon_idx"] * GRID_SIZE_DEG
).round(2)


seed["lat_grid"] = (
    seed["lat_grid"]
    .round(2)
)

seed["lon_grid"] = (
    seed["lon_grid"]
    .round(2)
)


print("Exact grid keys prepared successfully.")


# ============================================================
# CHECK UNIQUE LOCATIONS
# ============================================================

spatial_unique_locations = (
    spatial[
        [
            "lat_grid",
            "lon_grid",
        ]
    ]
    .drop_duplicates()
    .shape[0]
)

seed_unique_locations = (
    seed[
        [
            "lat_grid",
            "lon_grid",
        ]
    ]
    .drop_duplicates()
    .shape[0]
)

print(
    f"Spatial unique locations: "
    f"{spatial_unique_locations:,}"
)

print(
    f"Seed unique locations: "
    f"{seed_unique_locations:,}"
)


# ============================================================
# CHECK DUPLICATE SEED LOCATIONS
# ============================================================

duplicate_seed_rows = (
    seed.duplicated(
        subset=[
            "lat_grid",
            "lon_grid",
        ],
        keep=False,
    )
)

duplicate_count = int(
    duplicate_seed_rows.sum()
)

if duplicate_count > 0:

    print("\nWARNING:")
    print(
        f"Duplicate seed location rows: "
        f"{duplicate_count:,}"
    )

    print(
        "Keeping the first seed label for each grid cell."
    )

    seed = (
        seed
        .drop_duplicates(
            subset=[
                "lat_grid",
                "lon_grid",
            ],
            keep="first",
        )
        .copy()
    )

else:

    print("No duplicate seed locations found.")


# ============================================================
# SPATIAL FEATURES
# ============================================================

spatial_features = [

    "local_population_1.5km",
    "local_population_3km",
    "local_population_5km",

    "new_neighbor_count_1.5km",
    "new_neighbor_count_3km",
    "new_neighbor_count_5km",

    "continuing_neighbor_count_1.5km",
    "continuing_neighbor_count_3km",
    "continuing_neighbor_count_5km",

    "local_persistence_1.5km",
    "local_persistence_3km",
    "local_persistence_5km",

    "local_population_change_1.5km",
    "local_population_change_3km",
    "local_population_change_5km",

    "local_expansion_rate_1.5km",
    "local_expansion_rate_3km",
    "local_expansion_rate_5km",

    "directional_consistency_1.5km",
    "directional_consistency_3km",
    "directional_consistency_5km",

    "local_radius_1.5km",
    "local_radius_3km",
    "local_radius_5km",

    "is_new_today",
    "was_active_yesterday",
]


# ============================================================
# VERIFY FEATURES
# ============================================================

missing_features = [
    feature
    for feature in spatial_features
    if feature not in spatial.columns
]

if missing_features:

    raise ValueError(
        "Spatial file is missing these features:\n"
        + "\n".join(missing_features)
    )

print(
    f"Verified {len(spatial_features)} spatial features."
)


# ============================================================
# [4/6] MERGE SPATIAL DATA WITH SEED LABELS
# ============================================================

print(
    "\n[4/6] Joining spatial behavior with seed labels..."
)


seed_for_merge = seed[
    [
        "lat_grid",
        "lon_grid",
        "seed_label",
    ]
].copy()


merged = spatial.merge(
    seed_for_merge,
    on=[
        "lat_grid",
        "lon_grid",
    ],
    how="inner",
)


print(
    f"Matched spatial rows: "
    f"{len(merged):,}"
)


matched_unique_locations = (
    merged[
        [
            "lat_grid",
            "lon_grid",
        ]
    ]
    .drop_duplicates()
    .shape[0]
)

print(
    f"Unique labeled locations: "
    f"{matched_unique_locations:,}"
)


# ============================================================
# CHECK MATCH COVERAGE
# ============================================================

spatial_locations = (
    spatial[
        [
            "lat_grid",
            "lon_grid",
        ]
    ]
    .drop_duplicates()
)

matched_locations = (
    merged[
        [
            "lat_grid",
            "lon_grid",
        ]
    ]
    .drop_duplicates()
)


if len(spatial_locations) > 0:

    spatial_coverage = (
        len(matched_locations)
        / len(spatial_locations)
        * 100
    )

else:

    spatial_coverage = 0.0


if len(seed) > 0:

    seed_coverage = (
        len(matched_locations)
        / len(seed)
        * 100
    )

else:

    seed_coverage = 0.0


print(
    f"Spatial location coverage: "
    f"{spatial_coverage:.2f}%"
)

print(
    f"Seed location coverage: "
    f"{seed_coverage:.2f}%"
)


print("\nMatched seed classes:")

print(
    merged["seed_label"]
    .value_counts()
)


# ============================================================
# [5/6] AGGREGATE DAILY BEHAVIOR
# ============================================================

print(
    "\n[5/6] Aggregating daily spatial behavior..."
)


group_columns = [
    "lat_grid",
    "lon_grid",
    "seed_label",
]


# ------------------------------------------------------------
# STATISTICAL AGGREGATION
# ------------------------------------------------------------

agg_spec = {}

for feature in spatial_features:

    agg_spec[
        f"{feature}_mean"
    ] = (
        feature,
        "mean",
    )

    agg_spec[
        f"{feature}_median"
    ] = (
        feature,
        "median",
    )

    agg_spec[
        f"{feature}_max"
    ] = (
        feature,
        "max",
    )


summary = (
    merged
    .groupby(
        group_columns,
        as_index=False,
    )
    .agg(
        **{
            output_name: pd.NamedAgg(
                column=input_column,
                aggfunc=agg_function,
            )
            for output_name, (
                input_column,
                agg_function,
            )
            in agg_spec.items()
        }
    )
)


# ============================================================
# TEMPORAL BEHAVIOR
# ============================================================

extra = (
    merged
    .groupby(group_columns)
    .agg(

        spatial_observation_days=(
            "acq_date",
            "nunique",
        ),

        expansion_days_3km=(
            "local_population_change_3km",
            lambda x: int(
                (x > 0).sum()
            ),
        ),

        expansion_days_5km=(
            "local_population_change_5km",
            lambda x: int(
                (x > 0).sum()
            ),
        ),

        contraction_days_5km=(
            "local_population_change_5km",
            lambda x: int(
                (x < 0).sum()
            ),
        ),

        new_neighbor_days_3km=(
            "new_neighbor_count_3km",
            lambda x: int(
                (x > 0).sum()
            ),
        ),

        new_neighbor_days_5km=(
            "new_neighbor_count_5km",
            lambda x: int(
                (x > 0).sum()
            ),
        ),

        persistent_local_days_5km=(
            "local_persistence_5km",
            lambda x: int(
                (x > 0).sum()
            ),
        ),

        strong_persistence_days_5km=(
            "local_persistence_5km",
            lambda x: int(
                (x >= 0.5).sum()
            ),
        ),

        directional_days_5km=(
            "directional_consistency_5km",
            lambda x: int(
                (x >= 0.5).sum()
            ),
        ),

        new_cell_days=(
            "is_new_today",
            lambda x: int(
                (x == 1).sum()
            ),
        ),

        continuing_cell_days=(
            "was_active_yesterday",
            lambda x: int(
                (x == 1).sum()
            ),
        ),
    )
    .reset_index()
)


summary = summary.merge(
    extra,
    on=group_columns,
    how="left",
)


# ============================================================
# STABILIZE EXPANSION FEATURES
# ============================================================

print(
    "\nCreating stable spatial behavior features..."
)


for radius in [
    "1.5km",
    "3km",
    "5km",
]:

    raw_column = (
        f"local_expansion_rate_{radius}_mean"
    )

    capped_column = (
        f"local_expansion_rate_{radius}_capped_mean"
    )

    summary[capped_column] = (
        summary[raw_column]
        .clip(
            lower=-1,
            upper=5,
        )
    )


observation_days = (
    summary[
        "spatial_observation_days"
    ]
    .replace(
        0,
        np.nan,
    )
)


summary[
    "expansion_day_ratio_5km"
] = (
    summary[
        "expansion_days_5km"
    ]
    / observation_days
)


summary[
    "new_neighbor_day_ratio_5km"
] = (
    summary[
        "new_neighbor_days_5km"
    ]
    / observation_days
)


summary[
    "persistent_day_ratio_5km"
] = (
    summary[
        "persistent_local_days_5km"
    ]
    / observation_days
)


# ============================================================
# [6/6] CLASS-LEVEL SUMMARY
# ============================================================

print(
    "\n[6/6] Creating class-level summary..."
)


class_summary = (
    summary
    .groupby("seed_label")
    .agg(

        locations=(
            "lat_grid",
            "count",
        ),

        mean_observation_days=(
            "spatial_observation_days",
            "mean",
        ),

        median_observation_days=(
            "spatial_observation_days",
            "median",
        ),

        mean_local_population_3km=(
            "local_population_3km_mean",
            "mean",
        ),

        median_local_population_3km=(
            "local_population_3km_median",
            "mean",
        ),

        max_local_population_3km=(
            "local_population_3km_max",
            "mean",
        ),

        mean_local_population_5km=(
            "local_population_5km_mean",
            "mean",
        ),

        median_local_population_5km=(
            "local_population_5km_median",
            "mean",
        ),

        max_local_population_5km=(
            "local_population_5km_max",
            "mean",
        ),

        mean_new_neighbors_5km=(
            "new_neighbor_count_5km_mean",
            "mean",
        ),

        median_new_neighbors_5km=(
            "new_neighbor_count_5km_median",
            "mean",
        ),

        max_new_neighbors_5km=(
            "new_neighbor_count_5km_max",
            "mean",
        ),

        mean_local_persistence_5km=(
            "local_persistence_5km_mean",
            "mean",
        ),

        median_local_persistence_5km=(
            "local_persistence_5km_median",
            "mean",
        ),

        max_local_persistence_5km=(
            "local_persistence_5km_max",
            "mean",
        ),

        mean_population_change_5km=(
            "local_population_change_5km_mean",
            "mean",
        ),

        median_population_change_5km=(
            "local_population_change_5km_median",
            "mean",
        ),

        max_population_change_5km=(
            "local_population_change_5km_max",
            "mean",
        ),

        mean_directional_consistency_5km=(
            "directional_consistency_5km_mean",
            "mean",
        ),

        median_directional_consistency_5km=(
            "directional_consistency_5km_median",
            "mean",
        ),

        mean_expansion_days_5km=(
            "expansion_days_5km",
            "mean",
        ),

        mean_new_neighbor_days_5km=(
            "new_neighbor_days_5km",
            "mean",
        ),

        mean_persistent_days_5km=(
            "persistent_local_days_5km",
            "mean",
        ),

        mean_strong_persistence_days_5km=(
            "strong_persistence_days_5km",
            "mean",
        ),

        mean_directional_days_5km=(
            "directional_days_5km",
            "mean",
        ),

        mean_expansion_day_ratio_5km=(
            "expansion_day_ratio_5km",
            "mean",
        ),

        mean_new_neighbor_day_ratio_5km=(
            "new_neighbor_day_ratio_5km",
            "mean",
        ),

        mean_persistent_day_ratio_5km=(
            "persistent_day_ratio_5km",
            "mean",
        ),
    )
    .reset_index()
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SPATIAL BEHAVIOR BY CLASS")
print("=" * 70)

print(
    class_summary.to_string(
        index=False,
    )
)


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


summary.to_csv(
    OUTPUT_FILE,
    index=False,
)


class_summary.to_csv(
    CLASS_SUMMARY_FILE,
    index=False,
)


print("\nDetailed analysis saved:")
print(OUTPUT_FILE)

print("\nClass summary saved:")
print(CLASS_SUMMARY_FILE)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("SPATIAL CLASS ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nDetailed location rows: "
    f"{len(summary):,}"
)

print(
    f"Class summary rows: "
    f"{len(class_summary):,}"
)

print(
    "\nClasses found:"
)

for label in class_summary["seed_label"]:
    print(f"  - {label}")

print("\nNo ML training performed yet.")
print(
    "These results will be used to decide which spatial "
    "features are genuinely useful for classification."
)