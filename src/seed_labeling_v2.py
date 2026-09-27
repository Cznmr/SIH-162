from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PS 26162 — EVIDENCE-BASED SEED LABELING V2
#
# PURPOSE:
# Expand pseudo-label coverage using multiple independent
# evidence groups instead of brittle single rules.
#
# IMPORTANT:
# This creates a NEW file.
# It does NOT overwrite seed_labels.csv.
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
    / "training"
    / "spatial_location_evidence.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "seed_labels_v2.csv"
)


print("=" * 70)
print("PS 26162 — EVIDENCE-BASED SEED LABELING V2")
print("=" * 70)


# ============================================================
# LOAD
# ============================================================

print("\n[1/6] Loading master features...")

data = pd.read_csv(MASTER_FILE)

print(f"Master rows: {len(data):,}")


print("\n[2/6] Loading spatial evidence...")

spatial = pd.read_csv(SPATIAL_FILE)

print(f"Spatial rows: {len(spatial):,}")


# ============================================================
# VALIDATE GRID
# ============================================================

required_grid = [
    "lat_grid",
    "lon_grid",
]

for col in required_grid:
    if col not in data.columns:
        raise ValueError(
            f"Master missing required column: {col}"
        )

for col in required_grid:
    if col not in spatial.columns:
        raise ValueError(
            f"Spatial evidence missing required column: {col}"
        )


# ============================================================
# SELECT SPATIAL FEATURES
# ============================================================

spatial_features = [
    "lat_grid",
    "lon_grid",

    "spatial_observation_days",

    "expansion_days_1.5km",
    "expansion_days_3km",
    "expansion_days_5km",

    "new_activity_days_1.5km",
    "new_activity_days_3km",
    "new_activity_days_5km",

    "max_local_persistence_1.5km",
    "max_local_persistence_3km",
    "max_local_persistence_5km",

    "mean_local_persistence_1.5km",
    "mean_local_persistence_3km",
    "mean_local_persistence_5km",

    "persistent_local_days_5km",

    "directional_consistency_5km",
]


available_spatial = [
    c for c in spatial_features
    if c in spatial.columns
]

missing_spatial = [
    c for c in spatial_features
    if c not in spatial.columns
]

print("\nSpatial features available:")
for c in available_spatial:
    print("  ", c)

if missing_spatial:
    print("\nSpatial features not present:")
    for c in missing_spatial:
        print("  ", c)


spatial = spatial[available_spatial].copy()

spatial = spatial.drop_duplicates(
    subset=["lat_grid", "lon_grid"]
)


# ============================================================
# MERGE
# ============================================================

print("\n[3/6] Merging spatial evidence...")

data = data.merge(
    spatial,
    on=["lat_grid", "lon_grid"],
    how="left",
    validate="one_to_one",
)

print(f"Rows after merge: {len(data):,}")


# ============================================================
# HELPERS
# ============================================================

def flag(condition):
    return condition.fillna(False).astype(int)


# ============================================================
# BASIC EVIDENCE
# ============================================================

print("\n[4/6] Calculating evidence scores...")


# ------------------------------------------------------------
# Thermal evidence
# ------------------------------------------------------------

data["ev_frp5"] = flag(
    data["max_frp"] >= 5
)

data["ev_frp10"] = flag(
    data["max_frp"] >= 10
)

data["ev_frp15"] = flag(
    data["max_frp"] >= 15
)

data["ev_frp20"] = flag(
    data["max_frp"] >= 20
)

data["ev_det2"] = flag(
    data["detection_count"] >= 2
)

data["ev_det3"] = flag(
    data["detection_count"] >= 3
)

data["ev_det5"] = flag(
    data["detection_count"] >= 5
)


# ------------------------------------------------------------
# Persistence evidence
# ------------------------------------------------------------

data["ev_active3"] = flag(
    data["active_days"] >= 3
)

data["ev_active5"] = flag(
    data["active_days"] >= 5
)

data["ev_active180_3"] = flag(
    data["active_days_180d"] >= 3
)

data["ev_active180_5"] = flag(
    data["active_days_180d"] >= 5
)

data["ev_span30"] = flag(
    data["observation_span_days"] >= 30
)

data["ev_span60"] = flag(
    data["observation_span_days"] >= 60
)

data["ev_span90"] = flag(
    data["observation_span_days"] >= 90
)


# ------------------------------------------------------------
# Land-cover evidence
# ------------------------------------------------------------

data["ev_veg60"] = flag(
    data["dw_vegetation_score"] >= 0.60
)

data["ev_veg70"] = flag(
    data["dw_vegetation_score"] >= 0.70
)

data["ev_built20"] = flag(
    data["dw_built"] >= 0.20
)


# ------------------------------------------------------------
# Industrial context
# ------------------------------------------------------------

industrial_distance_columns = [
    "nearest_industrial_area_km",
    "nearest_industrial_works_km",
    "nearest_power_infrastructure_km",
    "nearest_storage_tank_km",
]

data["ev_industrial5"] = flag(
    data[industrial_distance_columns]
    .le(5)
    .any(axis=1)
)

data["ev_industrial10"] = flag(
    data[industrial_distance_columns]
    .le(10)
    .any(axis=1)
)

data["ev_no_industrial5"] = (
    1 - data["ev_industrial5"]
)


# ============================================================
# SPATIAL EVIDENCE
# ============================================================

data["ev_spatial_obs3"] = flag(
    data["spatial_observation_days"] >= 3
)

data["ev_spatial_obs5"] = flag(
    data["spatial_observation_days"] >= 5
)

data["ev_expansion3"] = flag(
    data["expansion_days_5km"] >= 2
)

data["ev_new_activity3"] = flag(
    data["new_activity_days_5km"] >= 2
)

data["ev_local_persistence75"] = flag(
    data["max_local_persistence_5km"] >= 0.75
)

data["ev_local_persistence90"] = flag(
    data["max_local_persistence_5km"] >= 0.90
)


# ============================================================
# WILDFIRE SCORE
# ============================================================

data["wildfire_score"] = (
    2 * data["ev_frp10"]
    + 1 * data["ev_frp15"]
    + 1 * data["ev_det2"]
    + 1 * data["ev_det3"]
    + 1 * data["ev_veg60"]
    + 1 * data["ev_veg70"]
    + 1 * data["ev_no_industrial5"]
    + 1 * data["ev_expansion3"]
    + 1 * data["ev_new_activity3"]
)


# ============================================================
# INDUSTRIAL FIRE SCORE
# ============================================================

data["industrial_fire_score"] = (
    2 * data["ev_frp10"]
    + 1 * data["ev_frp15"]
    + 2 * data["ev_industrial5"]
    + 1 * data["ev_industrial10"]
    + 1 * data["ev_built20"]
    + 1 * data["ev_det2"]
)


# ============================================================
# PERSISTENT THERMAL SOURCE SCORE
# ============================================================

data["persistent_score"] = (
    2 * (
        data["ev_active180_5"]
        & data["ev_span60"].astype(bool)
    ).astype(int)

    + 2 * data["ev_spatial_obs3"]

    + 2 * data["ev_local_persistence75"]

    + 1 * data["ev_active5"]

    + 1 * data["ev_active180_3"]

    + 1 * data["ev_span90"]

    + 1 * data["ev_spatial_obs5"]
)


# ============================================================
# OTHER SCORE
# ============================================================

data["other_score"] = (
    2 * data["ev_frp5"]
    + 1 * data["ev_det2"]
    + 1 * data["ev_active3"]
    + 1 * data["ev_span30"]
    + 1 * data["ev_spatial_obs3"]
)


# ============================================================
# CONFLICT / CLASSIFICATION
# ============================================================

print("\n[5/6] Assigning pseudo-labels...")


scores = data[
    [
        "wildfire_score",
        "industrial_fire_score",
        "persistent_score",
        "other_score",
    ]
]

data["highest_score"] = scores.max(axis=1)

data["second_highest_score"] = (
    scores.apply(
        lambda row: sorted(
            row.tolist(),
            reverse=True
        )[1],
        axis=1,
    )
)

data["score_margin"] = (
    data["highest_score"]
    - data["second_highest_score"]
)


data["seed_label_v2"] = "UNKNOWN"


# ------------------------------------------------------------
# Persistent thermal source
# ------------------------------------------------------------

persistent_condition = (
    (data["persistent_score"] >= 6)
    & (data["persistent_score"] > data["wildfire_score"])
    & (data["persistent_score"] > data["industrial_fire_score"])
)


data.loc[
    persistent_condition,
    "seed_label_v2"
] = "PERSISTENT_THERMAL_SOURCE"


# ------------------------------------------------------------
# Industrial fire
# ------------------------------------------------------------

industrial_condition = (
    (data["industrial_fire_score"] >= 6)
    & (
        data["industrial_fire_score"]
        > data["wildfire_score"]
    )
    & (
        data["industrial_fire_score"]
        > data["persistent_score"]
    )
)


data.loc[
    industrial_condition,
    "seed_label_v2"
] = "INDUSTRIAL_FIRE"


# ------------------------------------------------------------
# Wildfire
# ------------------------------------------------------------

wildfire_condition = (
    (data["wildfire_score"] >= 6)
    & (
        data["wildfire_score"]
        > data["industrial_fire_score"]
    )
    & (
        data["wildfire_score"]
        > data["persistent_score"]
    )
)


data.loc[
    wildfire_condition,
    "seed_label_v2"
] = "WILDFIRE"


# ------------------------------------------------------------
# OTHER
# ------------------------------------------------------------

other_condition = (
    (data["seed_label_v2"] == "UNKNOWN")
    & (data["other_score"] >= 4)
    & (data["highest_score"] < 6)
)


data.loc[
    other_condition,
    "seed_label_v2"
] = "OTHER"


# ============================================================
# RULE TRACE
# ============================================================

def build_rules(row):

    rules = []

    if row["ev_frp10"]:
        rules.append("FRP>=10")

    if row["ev_frp15"]:
        rules.append("FRP>=15")

    if row["ev_det2"]:
        rules.append("DETECTIONS>=2")

    if row["ev_det3"]:
        rules.append("DETECTIONS>=3")

    if row["ev_active5"]:
        rules.append("ACTIVE_DAYS>=5")

    if row["ev_active180_5"]:
        rules.append("ACTIVE180>=5")

    if row["ev_span60"]:
        rules.append("SPAN>=60")

    if row["ev_veg70"]:
        rules.append("VEGETATION>=0.70")

    if row["ev_built20"]:
        rules.append("BUILT>=0.20")

    if row["ev_industrial5"]:
        rules.append("INDUSTRIAL_CONTEXT<=5KM")

    if row["ev_spatial_obs3"]:
        rules.append("SPATIAL_OBS>=3")

    if row["ev_expansion3"]:
        rules.append("EXPANSION_5KM>=2")

    if row["ev_new_activity3"]:
        rules.append("NEW_ACTIVITY_5KM>=2")

    if row["ev_local_persistence75"]:
        rules.append("LOCAL_PERSISTENCE_5KM>=0.75")

    return "|".join(rules)


data["rules_triggered"] = data.apply(
    build_rules,
    axis=1
)


# ============================================================
# LABEL CONFIDENCE
# ============================================================

data["label_confidence"] = "LOW"

data.loc[
    (
        data["seed_label_v2"] != "UNKNOWN"
    )
    & (data["highest_score"] >= 7)
    & (data["score_margin"] >= 2),
    "label_confidence"
] = "HIGH"


data.loc[
    (
        data["seed_label_v2"] != "UNKNOWN"
    )
    & (data["label_confidence"] == "LOW")
    & (data["highest_score"] >= 6)
    & (data["score_margin"] >= 1),
    "label_confidence"
] = "MEDIUM"


# ============================================================
# SAVE
# ============================================================

print("\n[6/6] Saving V2 labels...")

data.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("V2 LABEL DISTRIBUTION")
print("=" * 70)

print(
    data["seed_label_v2"]
    .value_counts()
)


print("\nPercentages:")

print(
    (
        data["seed_label_v2"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


print("\nConfidence:")

print(
    pd.crosstab(
        data["seed_label_v2"],
        data["label_confidence"]
    )
)


print("\nScore ranges:")

print(
    data[
        [
            "wildfire_score",
            "industrial_fire_score",
            "persistent_score",
            "other_score",
        ]
    ].describe()
)


print(
    f"\nSaved: {OUTPUT_FILE}"
)

print("=" * 70)