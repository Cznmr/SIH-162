import os
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "data/processed/noaa20_telangana_12months.csv"

OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "noaa20_persistence_features.csv"
)

# Approximate spatial grid
# 0.01 degree is roughly ~1 km
GRID_SIZE = 0.01

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading NOAA-20 12-month dataset...")

df = pd.read_csv(INPUT_FILE)

df["acq_date"] = pd.to_datetime(df["acq_date"])

print(f"Total detections: {len(df):,}")
print(f"Date range: {df['acq_date'].min().date()} "
      f"to {df['acq_date'].max().date()}")


# ============================================================
# CREATE APPROXIMATE LOCATION GRID
# ============================================================

df["lat_grid"] = (
    np.floor(df["latitude"] / GRID_SIZE) * GRID_SIZE
).round(2)

df["lon_grid"] = (
    np.floor(df["longitude"] / GRID_SIZE) * GRID_SIZE
).round(2)


# ============================================================
# UNIQUE ACTIVE DAYS
# ============================================================

print("\nCalculating active days...")

daily = (
    df[
        [
            "lat_grid",
            "lon_grid",
            "acq_date"
        ]
    ]
    .drop_duplicates()
)


# ============================================================
# LOCATION-LEVEL PERSISTENCE
# ============================================================

print("Calculating persistence statistics...")

location_stats = (
    daily
    .groupby(["lat_grid", "lon_grid"])
    .agg(
        active_days=("acq_date", "nunique"),
        first_detection=("acq_date", "min"),
        last_detection=("acq_date", "max")
    )
    .reset_index()
)


# Total detections per location

detection_counts = (
    df
    .groupby(["lat_grid", "lon_grid"])
    .size()
    .reset_index(name="total_detections")
)

location_stats = location_stats.merge(
    detection_counts,
    on=["lat_grid", "lon_grid"],
    how="left"
)


# ============================================================
# PERSISTENCE WINDOWS
# ============================================================

max_date = df["acq_date"].max()

def active_days_since(days):

    start_date = max_date - pd.Timedelta(days=days)

    recent = daily[
        daily["acq_date"] >= start_date
    ]

    return (
        recent
        .groupby(["lat_grid", "lon_grid"])["acq_date"]
        .nunique()
        .rename(f"active_days_{days}d")
    )


for window in [30, 90, 180]:

    result = active_days_since(window)

    location_stats = location_stats.merge(
        result,
        on=["lat_grid", "lon_grid"],
        how="left"
    )


# Replace missing values with zero

for window in [30, 90, 180]:

    column = f"active_days_{window}d"

    location_stats[column] = (
        location_stats[column]
        .fillna(0)
        .astype(int)
    )


# ============================================================
# PERSISTENCE SCORE
# ============================================================

score_30 = np.minimum(
    location_stats["active_days_30d"] / 10,
    1
)

score_90 = np.minimum(
    location_stats["active_days_90d"] / 30,
    1
)

score_180 = np.minimum(
    location_stats["active_days_180d"] / 60,
    1
)

location_stats["persistence_score"] = (
    0.20 * score_30 +
    0.35 * score_90 +
    0.45 * score_180
)


# ============================================================
# PERSISTENCE LABELS
# ============================================================

location_stats["persistent_30d"] = (
    location_stats["active_days_30d"] >= 3
).astype(int)

location_stats["persistent_90d"] = (
    location_stats["active_days_90d"] >= 5
).astype(int)

location_stats["persistent_180d"] = (
    location_stats["active_days_180d"] >= 10
).astype(int)


# ============================================================
# SORT
# ============================================================

location_stats = location_stats.sort_values(
    "persistence_score",
    ascending=False
)


# ============================================================
# SAVE
# ============================================================

location_stats.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================================")
print("PERSISTENCE ANALYSIS COMPLETE")
print("==============================================")

print(f"Unique locations : {len(location_stats):,}")

print(
    f"Persistent 30d  : "
    f"{location_stats['persistent_30d'].sum():,}"
)

print(
    f"Persistent 90d  : "
    f"{location_stats['persistent_90d'].sum():,}"
)

print(
    f"Persistent 180d : "
    f"{location_stats['persistent_180d'].sum():,}"
)

print(
    f"Mean persistence score : "
    f"{location_stats['persistence_score'].mean():.4f}"
)

print(
    f"Maximum persistence score : "
    f"{location_stats['persistence_score'].max():.4f}"
)


# ============================================================
# TOP PERSISTENT LOCATIONS
# ============================================================

print("\nTop 20 persistent locations:")

print(
    location_stats[
        [
            "lat_grid",
            "lon_grid",
            "total_detections",
            "active_days",
            "active_days_30d",
            "active_days_90d",
            "active_days_180d",
            "persistence_score"
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("==============================================")