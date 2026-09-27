import pandas as pd
from pathlib import Path

# ============================================================
# PS 26162 - Build ML-Ready Feature Table
# ============================================================

BASE = Path(__file__).resolve().parent.parent

MASTER_FILE = BASE / "data" / "training" / "master_feature_table_final.csv"
SPATIAL_FILE = BASE / "data" / "training" / "spatial_behavior_by_class.csv"
SEED_FILE = BASE / "data" / "training" / "seed_labels.csv"

OUTPUT_FILE = BASE / "data" / "training" / "ml_feature_table.csv"


print("=" * 70)
print("BUILDING ML-READY FEATURE TABLE")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load files
# ------------------------------------------------------------

print("\n[1/6] Loading source files...")

master = pd.read_csv(MASTER_FILE)
spatial = pd.read_csv(SPATIAL_FILE)
seed = pd.read_csv(SEED_FILE)

print(f"Master rows:  {len(master):,}")
print(f"Spatial rows: {len(spatial):,}")
print(f"Seed rows:    {len(seed):,}")


# ------------------------------------------------------------
# 2. Validate grid keys
# ------------------------------------------------------------

print("\n[2/6] Validating grid keys...")

for name, df in [
    ("master", master),
    ("spatial", spatial),
    ("seed", seed),
]:
    if not {"lat_grid", "lon_grid"}.issubset(df.columns):
        raise ValueError(
            f"{name} is missing lat_grid/lon_grid columns."
        )

    duplicate_count = df.duplicated(
        subset=["lat_grid", "lon_grid"]
    ).sum()

    print(
        f"{name}: {len(df):,} rows, "
        f"{df[['lat_grid', 'lon_grid']].drop_duplicates().shape[0]:,} "
        f"unique grid locations, "
        f"{duplicate_count:,} duplicate rows"
    )


# ------------------------------------------------------------
# 3. Prepare spatial features
# ------------------------------------------------------------

print("\n[3/6] Preparing spatial features...")

# seed_label is the TARGET, not an input feature.
spatial_feature_cols = [
    c for c in spatial.columns
    if c not in ["lat_grid", "lon_grid", "seed_label"]
]

spatial_features = spatial[
    ["lat_grid", "lon_grid"] + spatial_feature_cols
].copy()

# Safety check
if spatial_features.duplicated(
    subset=["lat_grid", "lon_grid"]
).any():
    raise ValueError(
        "Spatial file contains duplicate grid locations."
    )

print(f"Spatial feature columns: {len(spatial_feature_cols)}")


# ------------------------------------------------------------
# 4. Merge spatial features into master table
# ------------------------------------------------------------

print("\n[4/6] Merging spatial features...")

ml = master.merge(
    spatial_features,
    on=["lat_grid", "lon_grid"],
    how="left",
    validate="one_to_one",
)

print(f"Rows after merge: {len(ml):,}")

if len(ml) != len(master):
    raise ValueError(
        "Row count changed during spatial merge."
    )

# Check spatial coverage
missing_spatial = ml[spatial_feature_cols].isna().all(axis=1).sum()

print(f"Locations without spatial features: {missing_spatial:,}")


# ------------------------------------------------------------
# 5. Add target labels
# ------------------------------------------------------------

print("\n[5/6] Adding target labels...")

seed_target = seed[
    ["lat_grid", "lon_grid", "seed_label"]
].copy()

if seed_target.duplicated(
    subset=["lat_grid", "lon_grid"]
).any():
    raise ValueError(
        "Seed file contains duplicate grid locations."
    )

ml = ml.merge(
    seed_target,
    on=["lat_grid", "lon_grid"],
    how="left",
    validate="one_to_one",
)

if ml["seed_label"].isna().any():
    missing_labels = ml["seed_label"].isna().sum()
    raise ValueError(
        f"{missing_labels:,} locations are missing seed labels."
    )


# ------------------------------------------------------------
# 6. Remove obsolete / leakage columns and save
# ------------------------------------------------------------

print("\n[6/6] Cleaning and validating final table...")

# These old OSM columns are obsolete and mostly empty.
obsolete_osm_columns = [
    "first_detection_osm",
    "last_detection_osm",
    "persistent_30d_osm",
    "persistent_90d_osm",
    "persistent_180d_osm",
]

removed_columns = []

for col in obsolete_osm_columns:
    if col in ml.columns:
        ml.drop(columns=col, inplace=True)
        removed_columns.append(col)

print("Removed obsolete OSM columns:")
for col in removed_columns:
    print(f"  - {col}")


# ------------------------------------------------------------
# Final validation
# ------------------------------------------------------------

expected_rows = 47677

if len(ml) != expected_rows:
    raise ValueError(
        f"Expected {expected_rows:,} rows, got {len(ml):,}."
    )

unique_locations = ml[
    ["lat_grid", "lon_grid"]
].drop_duplicates().shape[0]

if unique_locations != expected_rows:
    raise ValueError(
        "Final table does not contain exactly one row per grid location."
    )

if ml["seed_label"].isna().any():
    raise ValueError(
        "Missing seed labels detected."
    )

# Target distribution
print("\nTarget distribution:")
print(ml["seed_label"].value_counts())

# Check that spatial target did not sneak in
leakage_columns = [
    c for c in spatial.columns
    if c == "seed_label"
    and c in spatial_feature_cols
]

if leakage_columns:
    raise ValueError(
        f"Target leakage detected: {leakage_columns}"
    )

# Save
ml.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print("ML FEATURE TABLE CREATED")
print("=" * 70)

print(f"Rows:              {len(ml):,}")
print(f"Columns:           {len(ml.columns):,}")
print(f"Unique locations:  {unique_locations:,}")
print(f"Output:")
print(OUTPUT_FILE)
print("=" * 70)