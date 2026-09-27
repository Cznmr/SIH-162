import pandas as pd
import numpy as np


MASTER_FILE = "data/training/master_feature_table_final.csv"
SEED_FILE = "data/training/seed_labels.csv"


print("=" * 70)
print("UNKNOWN POOL EVIDENCE COMBINATION ANALYSIS")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1/6] Loading master feature table...")
master = pd.read_csv(MASTER_FILE)
print(f"Master rows: {len(master):,}")

print("\n[2/6] Loading seed labels...")
seed = pd.read_csv(SEED_FILE)
print(f"Seed rows: {len(seed):,}")


# ============================================================
# 2. VALIDATE GRID KEYS
# ============================================================

print("\n[3/6] Checking grid keys...")

grid_columns = [
    "lat_grid",
    "lon_grid",
]

for col in grid_columns:
    if col not in master.columns:
        raise ValueError(f"Master file missing column: {col}")

    if col not in seed.columns:
        raise ValueError(f"Seed file missing column: {col}")

print("Using existing lat_grid/lon_grid keys.")


# ============================================================
# 3. ATTACH SEED LABELS
# ============================================================

print("\n[4/6] Attaching seed labels...")

seed_small = seed[
    [
        "lat_grid",
        "lon_grid",
        "seed_label",
    ]
].copy()

df = master.merge(
    seed_small,
    on=["lat_grid", "lon_grid"],
    how="left",
    validate="one_to_one",
)

if len(df) != len(master):
    raise ValueError(
        "Merge changed row count. Grid keys are not one-to-one."
    )

unknown = df[df["seed_label"] == "UNKNOWN"].copy()

print(f"Total locations: {len(df):,}")
print(f"UNKNOWN locations: {len(unknown):,}")


# ============================================================
# 4. CREATE EVIDENCE FLAGS
# ============================================================

print("\n[5/6] Creating evidence flags...")


# ------------------------------------------------------------
# FIRMS / THERMAL
# ------------------------------------------------------------

unknown["frp_5"] = unknown["max_frp"] >= 5
unknown["frp_10"] = unknown["max_frp"] >= 10
unknown["frp_15"] = unknown["max_frp"] >= 15
unknown["frp_20"] = unknown["max_frp"] >= 20
unknown["frp_30"] = unknown["max_frp"] >= 30
unknown["frp_50"] = unknown["max_frp"] >= 50

unknown["detections_2"] = unknown["detection_count"] >= 2
unknown["detections_3"] = unknown["detection_count"] >= 3
unknown["detections_5"] = unknown["detection_count"] >= 5

unknown["active_days_2"] = unknown["active_days"] >= 2
unknown["active_days_3"] = unknown["active_days"] >= 3
unknown["active_days_5"] = unknown["active_days"] >= 5

unknown["span_7"] = unknown["observation_span_days"] >= 7
unknown["span_30"] = unknown["observation_span_days"] >= 30
unknown["span_60"] = unknown["observation_span_days"] >= 60


# ------------------------------------------------------------
# PERSISTENCE
# ------------------------------------------------------------

unknown["persistent_180_5"] = (
    unknown["active_days_180d"] >= 5
)

unknown["persistent_180_10"] = (
    unknown["active_days_180d"] >= 10
)


# ------------------------------------------------------------
# DYNAMIC WORLD
# ------------------------------------------------------------

unknown["vegetation_50"] = (
    unknown["dw_vegetation_score"] >= 0.50
)

unknown["vegetation_60"] = (
    unknown["dw_vegetation_score"] >= 0.60
)

unknown["vegetation_70"] = (
    unknown["dw_vegetation_score"] >= 0.70
)

unknown["vegetation_80"] = (
    unknown["dw_vegetation_score"] >= 0.80
)

unknown["built_10"] = (
    unknown["dw_built"] >= 0.10
)

unknown["built_20"] = (
    unknown["dw_built"] >= 0.20
)


# ------------------------------------------------------------
# OSM CONTEXT
# ------------------------------------------------------------

unknown["relevant_osm_5km"] = (
    unknown["nearest_relevant_osm_km"] <= 5
)

unknown["industrial_area_5km"] = (
    unknown["nearest_industrial_area_km"] <= 5
)

unknown["industrial_works_5km"] = (
    unknown["nearest_industrial_works_km"] <= 5
)

unknown["power_5km"] = (
    unknown["nearest_power_infrastructure_km"] <= 5
)

unknown["storage_5km"] = (
    unknown["nearest_storage_tank_km"] <= 5
)


# Combined industrial/infrastructure context

unknown["industrial_context_5km"] = (
    unknown["industrial_area_5km"]
    | unknown["industrial_works_5km"]
    | unknown["power_5km"]
    | unknown["storage_5km"]
)


unknown["no_industrial_context_5km"] = (
    ~unknown["industrial_context_5km"]
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def report(name, mask):
    count = int(mask.sum())
    pct = count / len(unknown) * 100

    print(
        f"{name:<65}"
        f"{count:>7,} "
        f"({pct:>6.2f}%)"
    )


# ============================================================
# 5. COMBINATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("COMBINATION RESULTS")
print("=" * 70)


# ------------------------------------------------------------
# A. BASIC THERMAL COMBINATIONS
# ------------------------------------------------------------

print("\n--- A. THERMAL + REPEAT DETECTIONS ---")

report(
    "FRP >= 5 + >=2 detections",
    unknown.frp_5 & unknown.detections_2
)

report(
    "FRP >= 10 + >=2 detections",
    unknown.frp_10 & unknown.detections_2
)

report(
    "FRP >= 10 + >=3 detections",
    unknown.frp_10 & unknown.detections_3
)

report(
    "FRP >= 15 + >=2 detections",
    unknown.frp_15 & unknown.detections_2
)

report(
    "FRP >= 20 + >=2 detections",
    unknown.frp_20 & unknown.detections_2
)

report(
    "FRP >= 30 + >=2 detections",
    unknown.frp_30 & unknown.detections_2
)


# ------------------------------------------------------------
# B. POSSIBLE WILDFIRE CANDIDATES
# ------------------------------------------------------------

print("\n--- B. POSSIBLE WILDFIRE CANDIDATES ---")

report(
    "FRP >= 5 + >=2 detections + vegetation >= 0.70",
    unknown.frp_5
    & unknown.detections_2
    & unknown.vegetation_70
)

report(
    "FRP >= 10 + >=2 detections + vegetation >= 0.70",
    unknown.frp_10
    & unknown.detections_2
    & unknown.vegetation_70
)

report(
    "FRP >= 10 + >=3 detections + vegetation >= 0.70",
    unknown.frp_10
    & unknown.detections_3
    & unknown.vegetation_70
)

report(
    "FRP >= 15 + >=2 detections + vegetation >= 0.70",
    unknown.frp_15
    & unknown.detections_2
    & unknown.vegetation_70
)

report(
    "FRP >= 10 + >=2 detections + vegetation >= 0.70 + NO industrial context",
    unknown.frp_10
    & unknown.detections_2
    & unknown.vegetation_70
    & unknown.no_industrial_context_5km
)

report(
    "FRP >= 15 + >=2 detections + vegetation >= 0.70 + NO industrial context",
    unknown.frp_15
    & unknown.detections_2
    & unknown.vegetation_70
    & unknown.no_industrial_context_5km
)


# ------------------------------------------------------------
# C. POSSIBLE INDUSTRIAL FIRE CANDIDATES
# ------------------------------------------------------------

print("\n--- C. POSSIBLE INDUSTRIAL FIRE CANDIDATES ---")

report(
    "FRP >= 5 + >=2 detections + industrial context",
    unknown.frp_5
    & unknown.detections_2
    & unknown.industrial_context_5km
)

report(
    "FRP >= 10 + >=2 detections + industrial context",
    unknown.frp_10
    & unknown.detections_2
    & unknown.industrial_context_5km
)

report(
    "FRP >= 10 + >=3 detections + industrial context",
    unknown.frp_10
    & unknown.detections_3
    & unknown.industrial_context_5km
)

report(
    "FRP >= 15 + >=2 detections + industrial context",
    unknown.frp_15
    & unknown.detections_2
    & unknown.industrial_context_5km
)

report(
    "FRP >= 30 + >=2 detections + industrial context",
    unknown.frp_30
    & unknown.detections_2
    & unknown.industrial_context_5km
)


# ------------------------------------------------------------
# D. POSSIBLE PERSISTENT THERMAL SOURCES
# ------------------------------------------------------------

print("\n--- D. POSSIBLE PERSISTENT THERMAL SOURCES ---")

report(
    ">=5 active days + span >=30 days",
    unknown.active_days_5
    & unknown.span_30
)

report(
    ">=3 active days + span >=30 days",
    unknown.active_days_3
    & unknown.span_30
)

report(
    ">=3 active days + span >=60 days",
    unknown.active_days_3
    & unknown.span_60
)

report(
    ">=5 active days + span >=60 days",
    unknown.active_days_5
    & unknown.span_60
)

report(
    "180d >=5 active days + span >=60 days",
    unknown.persistent_180_5
    & unknown.span_60
)

report(
    "180d >=5 active days + span >=60 days + industrial context",
    unknown.persistent_180_5
    & unknown.span_60
    & unknown.industrial_context_5km
)


# ------------------------------------------------------------
# E. MODERATE EVIDENCE
# ------------------------------------------------------------

print("\n--- E. MODERATE EVIDENCE POOL ---")

report(
    ">=2 detections + FRP >=5",
    unknown.detections_2
    & unknown.frp_5
)

report(
    ">=2 detections + FRP >=5 + vegetation >=0.70",
    unknown.detections_2
    & unknown.frp_5
    & unknown.vegetation_70
)

report(
    ">=2 detections + FRP >=5 + industrial context",
    unknown.detections_2
    & unknown.frp_5
    & unknown.industrial_context_5km
)

report(
    ">=3 detections + FRP >=5 + span >=7 days",
    unknown.detections_3
    & unknown.frp_5
    & unknown.span_7
)


# ============================================================
# F. MUTUALLY EXCLUSIVE HIGH-LEVEL GROUPS
# ============================================================

print("\n" + "=" * 70)
print("HIGH-LEVEL EVIDENCE GROUPS")
print("=" * 70)

# Strong wildfire candidate
strong_wildfire = (
    unknown.frp_10
    & unknown.detections_2
    & unknown.vegetation_70
    & unknown.no_industrial_context_5km
)

# Strong industrial candidate
strong_industrial = (
    unknown.frp_10
    & unknown.detections_2
    & unknown.industrial_context_5km
)

# Persistent candidate
persistent_candidate = (
    unknown.persistent_180_5
    & unknown.span_60
)

# Moderate recurring thermal
moderate_thermal = (
    unknown.frp_5
    & unknown.detections_2
)

# Everything else
classified_evidence = (
    strong_wildfire
    | strong_industrial
    | persistent_candidate
    | moderate_thermal
)

remaining_unknown = ~classified_evidence


report(
    "Strong wildfire candidate",
    strong_wildfire
)

report(
    "Strong industrial-fire candidate",
    strong_industrial
)

report(
    "Persistent thermal candidate",
    persistent_candidate
)

report(
    "Moderate recurring thermal",
    moderate_thermal
)

report(
    "Remaining weak/insufficient evidence",
    remaining_unknown
)


# ============================================================
# 6. OVERLAP ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("OVERLAP ANALYSIS")
print("=" * 70)

report(
    "Wildfire + industrial context",
    strong_wildfire
    & unknown.industrial_context_5km
)

report(
    "Wildfire + persistence",
    strong_wildfire
    & persistent_candidate
)

report(
    "Industrial + persistence",
    strong_industrial
    & persistent_candidate
)

report(
    "Wildfire + industrial + persistence",
    strong_wildfire
    & strong_industrial
    & persistent_candidate
)


# ============================================================
# SAVE DIAGNOSTIC FILE
# ============================================================

output_file = "data/training/unknown_evidence_combinations.csv"

output_columns = [
    "lat_grid",
    "lon_grid",
    "max_frp",
    "mean_frp",
    "detection_count",
    "active_days",
    "active_days_180d",
    "observation_span_days",
    "persistence_score",
    "dw_vegetation_score",
    "dw_built",
    "nearest_industrial_area_km",
    "nearest_industrial_works_km",
    "nearest_power_infrastructure_km",
    "nearest_storage_tank_km",
    "nearest_relevant_osm_km",
    "seed_label",
]

# Keep only columns that actually exist.
output_columns = [
    col
    for col in output_columns
    if col in unknown.columns
]

unknown[output_columns].to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nSaved diagnostic data to:\n{output_file}"
)