from pathlib import Path
import pandas as pd

# ============================================================
# PS 26162 - Merge Dynamic World tiled CSV exports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "landcover"
    / "dynamic_world"
    / "PS26162_DynamicWorld"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "dynamic_world_features.csv"


# ------------------------------------------------------------
# 1. Find the 30 Dynamic World CSV files
# ------------------------------------------------------------

csv_files = sorted(INPUT_DIR.glob("*.csv"))

print("=" * 70)
print("DYNAMIC WORLD DATA MERGE - PS 26162")
print("=" * 70)

print(f"\nInput folder:")
print(INPUT_DIR)

print(f"\nCSV files found: {len(csv_files)}")

if len(csv_files) != 30:
    raise RuntimeError(
        f"Expected exactly 30 Dynamic World CSV files, "
        f"but found {len(csv_files)}."
    )


# ------------------------------------------------------------
# 2. Read and merge all files
# ------------------------------------------------------------

dataframes = []

for i, file in enumerate(csv_files, start=1):
    print(f"Reading {i:02d}/30: {file.name}")

    df = pd.read_csv(file)

    # Keep track of the source tile
    df["source_tile"] = file.stem

    dataframes.append(df)


merged = pd.concat(dataframes, ignore_index=True)


# ------------------------------------------------------------
# 3. Basic information
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MERGE COMPLETE")
print("=" * 70)

print(f"Total rows: {len(merged):,}")
print(f"Total columns: {len(merged.columns)}")

print("\nColumns:")
for column in merged.columns:
    print(f"  - {column}")


# ------------------------------------------------------------
# 4. Check required coordinate columns
# ------------------------------------------------------------

required_columns = [
    "lat_grid",
    "lon_grid",
]

missing_required = [
    col for col in required_columns
    if col not in merged.columns
]

if missing_required:
    raise RuntimeError(
        f"Missing required columns: {missing_required}"
    )


# ------------------------------------------------------------
# 5. Check duplicate grid locations
# ------------------------------------------------------------

duplicate_mask = merged.duplicated(
    subset=["lat_grid", "lon_grid"],
    keep=False
)

duplicate_rows = int(duplicate_mask.sum())

duplicate_locations = (
    merged.loc[duplicate_mask, ["lat_grid", "lon_grid"]]
    .drop_duplicates()
    .shape[0]
)

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

print(f"Duplicate rows: {duplicate_rows:,}")
print(f"Duplicate grid locations: {duplicate_locations:,}")


# ------------------------------------------------------------
# 6. Check tile coverage
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TILE COVERAGE")
print("=" * 70)

tile_counts = merged["source_tile"].value_counts().sort_index()

print(f"Unique tiles represented: {merged['source_tile'].nunique()}")

for tile, count in tile_counts.items():
    print(f"  {tile}: {count:,} rows")


# ------------------------------------------------------------
# 7. Check Dynamic World probability columns
# ------------------------------------------------------------

probability_columns = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
]

print("\n" + "=" * 70)
print("DYNAMIC WORLD PROBABILITY CHECK")
print("=" * 70)

missing_probability_columns = [
    col
    for col in probability_columns
    if col not in merged.columns
]

if missing_probability_columns:
    print(
        "WARNING: Missing probability columns:",
        missing_probability_columns
    )
else:
    print("All 9 Dynamic World probability bands are present.")

    missing_values = merged[probability_columns].isna().sum()

    print("\nMissing values per probability band:")

    for column, count in missing_values.items():
        print(f"  {column}: {count:,}")


# ------------------------------------------------------------
# 8. Check probability sum
# ------------------------------------------------------------

if not missing_probability_columns:

    merged["probability_sum"] = merged[probability_columns].sum(axis=1)

    probability_difference = (
        merged["probability_sum"] - 1.0
    ).abs()

    print("\nProbability-sum check:")
    print(
        f"  Maximum difference from 1.0: "
        f"{probability_difference.max():.8f}"
    )

    print(
        f"  Rows with difference > 0.01: "
        f"{(probability_difference > 0.01).sum():,}"
    )

    # Remove temporary validation column
    merged.drop(columns=["probability_sum"], inplace=True)


# ------------------------------------------------------------
# 9. Remove exact duplicate rows only
# ------------------------------------------------------------

exact_duplicates = int(merged.duplicated().sum())

print("\n" + "=" * 70)
print("EXACT DUPLICATE CHECK")
print("=" * 70)

print(f"Exact duplicate rows: {exact_duplicates:,}")

if exact_duplicates > 0:
    print("Removing exact duplicate rows...")
    merged = merged.drop_duplicates().reset_index(drop=True)
else:
    print("No exact duplicate rows found.")


# ------------------------------------------------------------
# 10. Save merged dataset
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

merged.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("FILE SAVED")
print("=" * 70)

print(f"Output file:")
print(OUTPUT_FILE)

print(f"Final rows: {len(merged):,}")
print(f"Final columns: {len(merged.columns)}")

print("\nDynamic World merge finished successfully.")
print("=" * 70)