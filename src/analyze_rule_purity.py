from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PS 26162 — RULE PURITY ANALYSIS
#
# PURPOSE:
# Compare candidate evidence rules against the CURRENT
# high-confidence seed labels.
#
# IMPORTANT:
# This does NOT modify seed labels.
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


print("=" * 70)
print("PS 26162 — RULE PURITY ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD
# ============================================================

print("\n[1/5] Loading data...")

master = pd.read_csv(MASTER_FILE)
seed = pd.read_csv(SEED_FILE)
spatial = pd.read_csv(SPATIAL_FILE)

print(f"Master:  {len(master):,}")
print(f"Seeds:   {len(seed):,}")
print(f"Spatial: {len(spatial):,}")


# ============================================================
# VALIDATE SEED COORDINATES
# ============================================================

required_seed = [
    "lat_grid",
    "lon_grid",
    "seed_label",
]

missing = [
    c for c in required_seed
    if c not in seed.columns
]

if missing:
    raise ValueError(
        f"Seed file missing columns: {missing}"
    )


# Seed file already contains the canonical 0.01-degree grid keys.\n# ============================================================
# MERGE CURRENT LABELS
# ============================================================

data = master.merge(
    seed[
        [
            "lat_grid",
            "lon_grid",
            "seed_label",
        ]
    ],
    on=[
        "lat_grid",
        "lon_grid",
    ],
    how="left",
)


print(
    f"\nRows after seed merge: "
    f"{len(data):,}"
)


print("\nCurrent seed distribution:")

print(
    data["seed_label"]
    .value_counts(dropna=False)
)


# ============================================================
# MERGE SPATIAL FEATURES
# ============================================================

print("\n[2/5] Adding spatial evidence...")


if (
    "lat_idx" in spatial.columns
    and "lon_idx" in spatial.columns
):

    spatial["lat_grid"] = (
        np.floor(
            spatial["lat_idx"] * 0.01
        )
        * 0.01
    ).round(2)

    spatial["lon_grid"] = (
        np.floor(
            spatial["lon_idx"] * 0.01
        )
        * 0.01
    ).round(2)

else:

    raise ValueError(
        "Spatial file does not contain "
        "lat_idx/lon_idx."
    )


spatial_columns = [
    "lat_grid",
    "lon_grid",
]

for column in [
    "persistent_local_days_5km",
    "expansion_days_5km",
    "spatial_observation_days",
]:

    if column in spatial.columns:
        spatial_columns.append(column)


spatial_small = (
    spatial[spatial_columns]
    .drop_duplicates(
        subset=[
            "lat_grid",
            "lon_grid",
        ]
    )
)


data = data.merge(
    spatial_small,
    on=[
        "lat_grid",
        "lon_grid",
    ],
    how="left",
)


# ============================================================
# CREATE EVIDENCE FLAGS
# ============================================================

print("\n[3/5] Creating candidate rules...")


# ------------------------------------------------------------
# Basic evidence
# ------------------------------------------------------------

data["frp5"] = (
    data["max_frp"] >= 5
)

data["frp7"] = (
    data["max_frp"] >= 7
)

data["frp10"] = (
    data["max_frp"] >= 10
)

data["frp15"] = (
    data["max_frp"] >= 15
)

data["frp20"] = (
    data["max_frp"] >= 20
)


data["det2"] = (
    data["detection_count"] >= 2
)

data["det3"] = (
    data["detection_count"] >= 3
)

data["det5"] = (
    data["detection_count"] >= 5
)


data["active3"] = (
    data["active_days"] >= 3
)

data["active5"] = (
    data["active_days"] >= 5
)


data["active180_3"] = (
    data["active_days_180d"] >= 3
)

data["active180_5"] = (
    data["active_days_180d"] >= 5
)


data["span30"] = (
    data["observation_span_days"] >= 30
)

data["span60"] = (
    data["observation_span_days"] >= 60
)

data["span90"] = (
    data["observation_span_days"] >= 90
)


data["veg60"] = (
    data["dw_vegetation_score"] >= 0.60
)

data["veg70"] = (
    data["dw_vegetation_score"] >= 0.70
)

data["veg80"] = (
    data["dw_vegetation_score"] >= 0.80
)


# ------------------------------------------------------------
# Industrial context
# ------------------------------------------------------------

industrial_columns = [
    "nearest_industrial_area_km",
    "nearest_industrial_works_km",
    "nearest_power_infrastructure_km",
    "nearest_storage_tank_km",
]


data["industrial5"] = (
    data[industrial_columns]
    .le(5)
    .any(axis=1)
)


data["industrial10"] = (
    data[industrial_columns]
    .le(10)
    .any(axis=1)
)


data["built20"] = (
    data["dw_built"] >= 0.20
)


# ============================================================
# DEFINE CANDIDATE RULES
# ============================================================

rules = {}


# ------------------------------------------------------------
# WILDFIRE
# ------------------------------------------------------------

rules["WF_01_FRP5_DET2_VEG60"] = (
    data["frp5"]
    & data["det2"]
    & data["veg60"]
)

rules["WF_02_FRP5_DET2_VEG70"] = (
    data["frp5"]
    & data["det2"]
    & data["veg70"]
)

rules["WF_03_FRP10_DET2_VEG60"] = (
    data["frp10"]
    & data["det2"]
    & data["veg60"]
)

rules["WF_04_FRP10_DET2_VEG70"] = (
    data["frp10"]
    & data["det2"]
    & data["veg70"]
)

rules["WF_05_FRP10_DET3_VEG70_NOIND"] = (
    data["frp10"]
    & data["det3"]
    & data["veg70"]
    & ~data["industrial5"]
)

rules["WF_06_FRP15_DET2_VEG70_NOIND"] = (
    data["frp15"]
    & data["det2"]
    & data["veg70"]
    & ~data["industrial5"]
)


# ------------------------------------------------------------
# INDUSTRIAL FIRE
# ------------------------------------------------------------

rules["IF_01_FRP5_DET2_IND"] = (
    data["frp5"]
    & data["det2"]
    & data["industrial5"]
)

rules["IF_02_FRP10_DET2_IND"] = (
    data["frp10"]
    & data["det2"]
    & data["industrial5"]
)

rules["IF_03_FRP10_DET3_IND"] = (
    data["frp10"]
    & data["det3"]
    & data["industrial5"]
)

rules["IF_04_FRP10_DET3_IND_BUILT"] = (
    data["frp10"]
    & data["det3"]
    & data["industrial5"]
    & data["built20"]
)

rules["IF_05_FRP15_DET2_IND"] = (
    data["frp15"]
    & data["det2"]
    & data["industrial5"]
)


# ------------------------------------------------------------
# PERSISTENT THERMAL SOURCE
# ------------------------------------------------------------

rules["PTS_01_ACTIVE180_3_SPAN30"] = (
    data["active180_3"]
    & data["span30"]
)

rules["PTS_02_ACTIVE180_5_SPAN60"] = (
    data["active180_5"]
    & data["span60"]
)

rules["PTS_03_ACTIVE5_SPAN60"] = (
    data["active5"]
    & data["span60"]
)

rules["PTS_04_ACTIVE180_5_SPAN60_IND"] = (
    data["active180_5"]
    & data["span60"]
    & data["industrial5"]
)


# ------------------------------------------------------------
# GENERAL RECURRING THERMAL
# ------------------------------------------------------------

rules["THERMAL_01_FRP5_DET2"] = (
    data["frp5"]
    & data["det2"]
)

rules["THERMAL_02_FRP7_DET2"] = (
    data["frp7"]
    & data["det2"]
)

rules["THERMAL_03_FRP10_DET2"] = (
    data["frp10"]
    & data["det2"]
)

rules["THERMAL_04_FRP5_DET3"] = (
    data["frp5"]
    & data["det3"]
)

rules["THERMAL_05_FRP7_DET3"] = (
    data["frp7"]
    & data["det3"]
)

rules["THERMAL_06_FRP10_DET3"] = (
    data["frp10"]
    & data["det3"]
)


# ============================================================
# ANALYZE RULES
# ============================================================

print("\n[4/5] Measuring rule purity...")


results = []


for rule_name, mask in rules.items():

    subset = data.loc[mask].copy()

    total = len(subset)

    if total == 0:
        continue


    counts = (
        subset["seed_label"]
        .value_counts()
        .to_dict()
    )


    wildfire = counts.get(
        "WILDFIRE",
        0
    )

    industrial = counts.get(
        "INDUSTRIAL_FIRE",
        0
    )

    persistent = counts.get(
        "PERSISTENT_THERMAL_SOURCE",
        0
    )

    other = counts.get(
        "OTHER",
        0
    )

    unknown = counts.get(
        "UNKNOWN",
        0
    )


    known = (
        total
        - unknown
    )


    # Purity among ALL rows
    wildfire_purity = (
        wildfire / total * 100
    )

    industrial_purity = (
        industrial / total * 100
    )

    persistent_purity = (
        persistent / total * 100
    )

    other_purity = (
        other / total * 100
    )


    # Purity among already-labelled rows
    if known > 0:

        known_wildfire = (
            wildfire / known * 100
        )

        known_industrial = (
            industrial / known * 100
        )

        known_persistent = (
            persistent / known * 100
        )

        known_other = (
            other / known * 100
        )

    else:

        known_wildfire = 0
        known_industrial = 0
        known_persistent = 0
        known_other = 0


    results.append({

        "rule": rule_name,

        "total": total,

        "wildfire": wildfire,

        "industrial_fire": industrial,

        "persistent": persistent,

        "other": other,

        "unknown": unknown,

        "known": known,

        "wildfire_pct_all": wildfire_purity,

        "industrial_pct_all": industrial_purity,

        "persistent_pct_all": persistent_purity,

        "other_pct_all": other_purity,

        "wildfire_pct_known": known_wildfire,

        "industrial_pct_known": known_industrial,

        "persistent_pct_known": known_persistent,

        "other_pct_known": known_other,

    })


results_df = pd.DataFrame(results)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 100)
print("RULE PURITY RESULTS")
print("=" * 100)

print(
    results_df[
        [
            "rule",
            "total",
            "wildfire",
            "industrial_fire",
            "persistent",
            "other",
            "unknown",
            "wildfire_pct_known",
            "industrial_pct_known",
            "persistent_pct_known",
            "other_pct_known",
        ]
    ]
    .to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "rule_purity_analysis.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


print(
    f"\nSaved:\n{output_file}"
)

print("\n" + "=" * 70)
print("RULE PURITY ANALYSIS COMPLETE")
print("=" * 70)