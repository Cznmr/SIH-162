import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MASTER_FILE = BASE_DIR / "data" / "training" / "master_feature_table.csv"
DW_FILE = BASE_DIR / "data" / "processed" / "dynamic_world_features_clean.csv"
OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "training"
    / "master_feature_table_with_dynamic_world.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MERGING MASTER FEATURES + DYNAMIC WORLD")
print("=" * 70)

print("\nLoading master feature table...")
master = pd.read_csv(MASTER_FILE)

print(f"Master rows: {len(master):,}")
print(f"Master columns: {len(master.columns)}")

print("\nLoading cleaned Dynamic World features...")
dw = pd.read_csv(DW_FILE)

print(f"Dynamic World rows: {len(dw):,}")
print(f"Dynamic World columns: {len(dw.columns)}")


# ============================================================
# CHECK REQUIRED KEYS
# ============================================================

KEYS = ["lat_grid", "lon_grid"]

for key in KEYS:
    if key not in master.columns:
        raise ValueError(f"Missing key '{key}' in master feature table.")

    if key not in dw.columns:
        raise ValueError(f"Missing key '{key}' in Dynamic World file.")


# ============================================================
# CHECK KEY UNIQUENESS
# ============================================================

print("\nChecking key uniqueness...")

master_duplicates = master.duplicated(subset=KEYS).sum()
dw_duplicates = dw.duplicated(subset=KEYS).sum()

print(f"Master duplicate locations: {master_duplicates:,}")
print(f"Dynamic World duplicate locations: {dw_duplicates:,}")

if master_duplicates > 0:
    raise ValueError(
        "Master feature table contains duplicate lat_grid/lon_grid locations."
    )

if dw_duplicates > 0:
    raise ValueError(
        "Cleaned Dynamic World file still contains duplicate lat_grid/lon_grid locations."
    )


# ============================================================
# PREPARE DYNAMIC WORLD COLUMNS
# ============================================================

# Keep the coordinate keys unchanged.
# Prefix all other Dynamic World columns with "dw_"
# to avoid confusing them with existing master features.

dw_feature_columns = [
    column for column in dw.columns
    if column not in KEYS
]

dw_renamed = dw.rename(
    columns={
        column: f"dw_{column}"
        for column in dw_feature_columns
    }
)


# ============================================================
# CHECK COLUMN COLLISIONS
# ============================================================

master_columns = set(master.columns)

collisions = [
    column
    for column in dw_renamed.columns
    if column in master_columns and column not in KEYS
]

if collisions:
    raise ValueError(
        f"Unexpected column collisions after renaming: {collisions}"
    )


# ============================================================
# MERGE
# ============================================================

print("\nPerforming LEFT JOIN...")

merged = master.merge(
    dw_renamed,
    on=KEYS,
    how="left",
    validate="one_to_one"
)


# ============================================================
# VALIDATE MERGE
# ============================================================

print("\n" + "=" * 70)
print("MERGE VALIDATION")
print("=" * 70)

print(f"Master rows before merge : {len(master):,}")
print(f"Rows after merge         : {len(merged):,}")

if len(merged) != len(master):
    raise ValueError(
        "ERROR: Number of rows changed after merge!"
    )

duplicate_after_merge = merged.duplicated(subset=KEYS).sum()

print(f"Duplicate locations after merge: {duplicate_after_merge:,}")

if duplicate_after_merge > 0:
    raise ValueError(
        "ERROR: Duplicate locations detected after merge!"
    )


# ============================================================
# DYNAMIC WORLD COVERAGE
# ============================================================

dw_probability_columns = [
    "dw_water",
    "dw_trees",
    "dw_grass",
    "dw_flooded_vegetation",
    "dw_crops",
    "dw_shrub_and_scrub",
    "dw_built",
    "dw_bare",
    "dw_snow_and_ice",
]

available_probability_columns = [
    column
    for column in dw_probability_columns
    if column in merged.columns
]

if not available_probability_columns:
    raise ValueError(
        "No Dynamic World probability columns found after merge."
    )

has_dw = merged[available_probability_columns].notna().any(axis=1)

with_dw = has_dw.sum()
without_dw = (~has_dw).sum()

coverage = (with_dw / len(merged)) * 100
missing = (without_dw / len(merged)) * 100

print("\nDynamic World coverage:")
print(f"Locations with DW data    : {with_dw:,}")
print(f"Locations without DW data : {without_dw:,}")
print(f"DW coverage               : {coverage:.2f}%")
print(f"Missing DW                 : {missing:.2f}%")


# ============================================================
# CHECK DYNAMIC WORLD QUALITY
# ============================================================

if "dw_probability_sum" in merged.columns:

    valid_probability_sum = merged.loc[
        has_dw,
        "dw_probability_sum"
    ]

    print("\nDynamic World probability-sum diagnostic:")

    if len(valid_probability_sum) > 0:
        print(
            f"Mean   : {valid_probability_sum.mean():.6f}"
        )
        print(
            f"Median : {valid_probability_sum.median():.6f}"
        )
        print(
            f"Min    : {valid_probability_sum.min():.6f}"
        )
        print(
            f"Max    : {valid_probability_sum.max():.6f}"
        )


# ============================================================
# SAVE
# ============================================================

print("\nSaving merged feature table...")

merged.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nSaved to:")
print(OUTPUT_FILE)

print("\nFinal shape:")
print(f"Rows    : {len(merged):,}")
print(f"Columns : {len(merged.columns)}")

print("\n" + "=" * 70)
print("MERGE COMPLETED SUCCESSFULLY")
print("=" * 70)