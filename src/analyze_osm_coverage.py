import pandas as pd

INPUT = "data/training/master_feature_table.csv"

df = pd.read_csv(INPUT)

# OSM evidence columns
osm_flags = [
    "near_mine_1km",
    "near_mine_2km",
    "near_power_1km",
    "near_power_2km",
    "near_industrial_area_1km",
    "near_industrial_area_2km",
    "near_industrial_works_1km",
    "near_industrial_works_2km",
]

# Make sure missing columns don't crash the script
available = [c for c in osm_flags if c in df.columns]

# Any nearby OSM industrial/infrastructure evidence
df["has_osm_context"] = df[available].fillna(0).max(axis=1)

osm_context = df[df["has_osm_context"] == 1]
no_osm_context = df[df["has_osm_context"] == 0]

print("=" * 60)
print("OSM COVERAGE ANALYSIS")
print("=" * 60)

print(f"\nTotal FIRMS grid locations: {len(df):,}")
print(f"Locations with OSM context: {len(osm_context):,}")
print(f"Locations without OSM context: {len(no_osm_context):,}")

print(
    f"\nOSM context coverage: "
    f"{len(osm_context) / len(df) * 100:.2f}%"
)

print(
    f"No OSM context: "
    f"{len(no_osm_context) / len(df) * 100:.2f}%"
)

print("\nOSM evidence counts:")

for col in available:
    print(f"{col:35} {int(df[col].fillna(0).sum()):,}")

# Save locations that need Dynamic World fallback
fallback = no_osm_context[
    ["lat_grid", "lon_grid"]
].copy()

fallback.to_csv(
    "data/processed/dynamic_world_targets.csv",
    index=False
)

print(
    f"\nSaved Dynamic World targets: "
    f"{len(fallback):,}"
)

print(
    "\nOutput:"
    "\ndata/processed/dynamic_world_targets.csv"
)