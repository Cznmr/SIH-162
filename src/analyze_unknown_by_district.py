import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

UNKNOWN_DISTRICT = ROOT / "data/training/unknown_firms_by_district.csv"
MASTER = ROOT / "data/training/master_feature_table_final.csv"

OUT = ROOT / "data/training/unknown_district_evidence.csv"


print("Loading district assignments...")
district = pd.read_csv(UNKNOWN_DISTRICT)

print("Loading master feature table...")
master = pd.read_csv(MASTER)

print("\nSchemas:")
print("District:", district.columns.tolist())
print("Master:", master.columns.tolist())


# ---------------------------------------------------------
# 1. Keep only UNKNOWN locations with valid district names
# ---------------------------------------------------------

district = district.dropna(subset=["district_name"]).copy()

district["lat_grid"] = district["lat_grid"].round(2)
district["lon_grid"] = district["lon_grid"].round(2)

print("\nDistrict-assigned UNKNOWN locations:", len(district))


# ---------------------------------------------------------
# 2. Merge with master features
# ---------------------------------------------------------

keys = ["lat_grid", "lon_grid"]

df = district.merge(
    master,
    on=keys,
    how="left",
    validate="one_to_one"
)

print("Merged rows:", len(df))


# ---------------------------------------------------------
# 3. Check whether master features were found
# ---------------------------------------------------------

if "max_frp" not in df.columns:
    raise ValueError("max_frp missing from master table.")

missing_master = df["max_frp"].isna().sum()

print("Rows without master features:", missing_master)


# ---------------------------------------------------------
# 4. Create evidence flags
# ---------------------------------------------------------

df["high_frp_10"] = df["max_frp"] >= 10
df["high_frp_30"] = df["max_frp"] >= 30

df["repeated_detection"] = df["detection_count"] >= 2
df["recurring_activity"] = df["active_days"] >= 3

df["strong_persistence"] = (
    (df["active_days_180d"] >= 5) &
    (df["observation_span_days"] >= 60)
)

df["high_vegetation"] = df["dw_vegetation_score"] >= 0.70
df["high_crop"] = df["dw_crops"] >= 0.30

df["industrial_near_5km"] = (
    df["nearest_relevant_osm_km"] <= 5
)

df["industrial_near_10km"] = (
    df["nearest_relevant_osm_km"] <= 10
)


# ---------------------------------------------------------
# 5. Aggregate by district
# ---------------------------------------------------------

summary = (
    df.groupby("district_name")
    .agg(
        unknown_locations=("lat_grid", "count"),

        high_frp_10=("high_frp_10", "sum"),
        high_frp_30=("high_frp_30", "sum"),

        repeated_detection=("repeated_detection", "sum"),
        recurring_activity=("recurring_activity", "sum"),
        strong_persistence=("strong_persistence", "sum"),

        high_vegetation=("high_vegetation", "sum"),
        high_crop=("high_crop", "sum"),

        industrial_near_5km=("industrial_near_5km", "sum"),
        industrial_near_10km=("industrial_near_10km", "sum"),

        mean_max_frp=("max_frp", "mean"),
        median_max_frp=("max_frp", "median"),

        mean_detection_count=("detection_count", "mean"),
        median_detection_count=("detection_count", "median"),

        mean_active_days=("active_days", "mean"),
        median_active_days=("active_days", "median"),

        mean_vegetation=("dw_vegetation_score", "mean"),
        mean_crops=("dw_crops", "mean"),

        median_industrial_distance_km=(
            "nearest_relevant_osm_km",
            "median"
        ),
    )
    .reset_index()
)


# ---------------------------------------------------------
# 6. Calculate percentages
# ---------------------------------------------------------

for col in [
    "high_frp_10",
    "high_frp_30",
    "repeated_detection",
    "recurring_activity",
    "strong_persistence",
    "high_vegetation",
    "high_crop",
    "industrial_near_5km",
    "industrial_near_10km",
]:
    summary[col + "_pct"] = (
        summary[col] /
        summary["unknown_locations"] * 100
    )


# ---------------------------------------------------------
# 7. Sort by useful UNKNOWN activity
# ---------------------------------------------------------

summary = summary.sort_values(
    ["high_frp_10", "repeated_detection", "unknown_locations"],
    ascending=False
)


# ---------------------------------------------------------
# 8. Save
# ---------------------------------------------------------

summary.to_csv(OUT, index=False)

print("\nSaved:")
print(OUT)

print("\nDistrict evidence:")
print(summary.to_string(index=False))

print("\nTop districts by high-FRP UNKNOWN locations:")
print(
    summary[
        [
            "district_name",
            "unknown_locations",
            "high_frp_10",
            "high_frp_30",
            "repeated_detection",
            "recurring_activity",
            "high_crop",
            "industrial_near_5km",
        ]
    ]
    .head(15)
    .to_string(index=False)
)