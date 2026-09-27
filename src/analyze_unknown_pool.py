from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("UNKNOWN POOL EVIDENCE ANALYSIS")
print("=" * 70)

print("\n[1/5] Loading master feature table...")

master = pd.read_csv(
    MASTER_FILE,
    low_memory=False
)

print(f"Master rows: {len(master):,}")

print("\n[2/5] Loading seed labels...")

seed = pd.read_csv(
    SEED_FILE,
    low_memory=False
)

print(f"Seed rows: {len(seed):,}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

print("\n[3/5] Checking required columns...")

required_columns = [                    
    "max_frp",
    "mean_frp",
    "detection_count",
    "active_days",
    "active_days_30d",
    "active_days_90d",
    "active_days_180d",
    "observation_span_days",
    "persistence_score",
    "nearest_industrial_area_km",
    "nearest_industrial_works_km",
    "nearest_power_plant_km",
    "nearest_power_infrastructure_km",
    "nearest_storage_tank_km",
    "nearest_relevant_osm_km",
    "dw_vegetation_score",
    "dw_built",
    "night_ratio",
]

missing = [
    col
    for col in required_columns
    if col not in master.columns
]

if missing:
    print("\nERROR: Missing columns:")
    for col in missing:
        print(f"  - {col}")

    print("\nAvailable columns:")
    print(master.columns.tolist())

    raise SystemExit(1)


# ============================================================
# ATTACH SEED LABEL
# ============================================================

print("\n[4/5] Attaching seed labels...")

# The master and seed files already contain
# the exact project grid keys.

required_grid_columns = [
    "lat_grid",
    "lon_grid",
]

missing_master_grid = [
    col
    for col in required_grid_columns
    if col not in master.columns
]

missing_seed_grid = [
    col
    for col in required_grid_columns
    if col not in seed.columns
]

if missing_master_grid:
    raise ValueError(
        f"Master file is missing grid columns: "
        f"{missing_master_grid}"
    )

if missing_seed_grid:
    raise ValueError(
        f"Seed file is missing grid columns: "
        f"{missing_seed_grid}"
    )

print(
    "Using existing lat_grid/lon_grid keys."
)


seed_small = (
    seed[
        [
            "lat_grid",
            "lon_grid",
            "seed_label",
        ]
    ]
    .drop_duplicates(
        subset=["lat_grid", "lon_grid"]
    )
)


master = master.merge(
    seed_small,
    on=["lat_grid", "lon_grid"],
    how="left"
)

master["seed_label"] = (
    master["seed_label"]
    .fillna("UNKNOWN")
)


unknown = master[
    master["seed_label"] == "UNKNOWN"
].copy()


print(
    f"\nTotal locations: {len(master):,}"
)

print(
    f"Known seed locations: "
    f"{len(master) - len(unknown):,}"
)

print(
    f"UNKNOWN locations: "
    f"{len(unknown):,}"
)


# ============================================================
# EVIDENCE DISTRIBUTIONS
# ============================================================

print("\n[5/5] UNKNOWN evidence distributions")

print("\n" + "=" * 70)
print("THERMAL / FIRMS EVIDENCE")
print("=" * 70)

thermal_columns = [
    "max_frp",
    "mean_frp",
    "detection_count",
    "active_days",
    "active_days_30d",
    "active_days_90d",
    "active_days_180d",
    "observation_span_days",
    "persistence_score",
    "night_ratio",
]

print(
    unknown[thermal_columns]
    .describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
    .T
    .to_string()
)


# ============================================================
# OSM EVIDENCE
# ============================================================

print("\n" + "=" * 70)
print("OSM / INFRASTRUCTURE EVIDENCE")
print("=" * 70)

osm_columns = [
    "nearest_industrial_area_km",
    "nearest_industrial_works_km",
    "nearest_power_plant_km",
    "nearest_power_infrastructure_km",
    "nearest_storage_tank_km",
    "nearest_relevant_osm_km",
]

print(
    unknown[osm_columns]
    .describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
    .T
    .to_string()
)


# ============================================================
# DYNAMIC WORLD
# ============================================================

print("\n" + "=" * 70)
print("DYNAMIC WORLD EVIDENCE")
print("=" * 70)

dw_columns = [
    "dw_vegetation_score",
    "dw_built",
]

print(
    unknown[dw_columns]
    .describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
    .T
    .to_string()
)


# ============================================================
# SIMPLE EVIDENCE COUNTS
# ============================================================

print("\n" + "=" * 70)
print("UNKNOWN POOL BREAKDOWN")
print("=" * 70)


def show_count(
    description,
    condition,
):
    count = int(condition.sum())
    percentage = (
        count / len(unknown) * 100
    )

    print(
        f"{description:<55}"
        f"{count:>8,}"
        f" ({percentage:>6.2f}%)"
    )


print(
    "\nThermal evidence:"
)

show_count(
    "At least 1 detection",
    unknown["detection_count"] >= 1,
)

show_count(
    "At least 2 detections",
    unknown["detection_count"] >= 2,
)

show_count(
    "At least 3 detections",
    unknown["detection_count"] >= 3,
)

show_count(
    "Max FRP >= 10",
    unknown["max_frp"] >= 10,
)

show_count(
    "Max FRP >= 15",
    unknown["max_frp"] >= 15,
)

show_count(
    "Max FRP >= 20",
    unknown["max_frp"] >= 20,
)

show_count(
    "Max FRP >= 30",
    unknown["max_frp"] >= 30,
)

show_count(
    "Max FRP >= 50",
    unknown["max_frp"] >= 50,
)


print(
    "\nPersistence evidence:"
)

show_count(
    "Active >= 2 days",
    unknown["active_days"] >= 2,
)

show_count(
    "Active >= 3 days",
    unknown["active_days"] >= 3,
)

show_count(
    "Active >= 5 days",
    unknown["active_days"] >= 5,
)

show_count(
    "Active >= 10 days",
    unknown["active_days"] >= 10,
)

show_count(
    "Active 180d >= 5 days",
    unknown["active_days_180d"] >= 5,
)

show_count(
    "Active 180d >= 10 days",
    unknown["active_days_180d"] >= 10,
)

show_count(
    "Observation span >= 30 days",
    unknown["observation_span_days"] >= 30,
)

show_count(
    "Observation span >= 60 days",
    unknown["observation_span_days"] >= 60,
)


print(
    "\nContext evidence:"
)

show_count(
    "Vegetation score >= 0.50",
    unknown["dw_vegetation_score"] >= 0.50,
)

show_count(
    "Vegetation score >= 0.70",
    unknown["dw_vegetation_score"] >= 0.70,
)

show_count(
    "Built score >= 0.20",
    unknown["dw_built"] >= 0.20,
)

show_count(
    "Relevant OSM within 5 km",
    unknown["nearest_relevant_osm_km"] <= 5,
)

show_count(
    "Industrial area within 5 km",
    unknown["nearest_industrial_area_km"] <= 5,
)

show_count(
    "Industrial works within 5 km",
    unknown["nearest_industrial_works_km"] <= 5,
)

show_count(
    "Power infrastructure within 5 km",
    unknown["nearest_power_infrastructure_km"] <= 5,
)

show_count(
    "Storage tank within 5 km",
    unknown["nearest_storage_tank_km"] <= 5,
)


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)