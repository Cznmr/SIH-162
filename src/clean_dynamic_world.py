from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dynamic_world_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dynamic_world_features_clean.csv"
)

PROBABILITY_COLUMNS = [
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

KEY_COLUMNS = [
    "lat_grid",
    "lon_grid",
]


print("=" * 70)
print("CLEAN DYNAMIC WORLD DATA - PS 26162")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nInput rows: {len(df):,}")
print(f"Input columns: {len(df.columns)}")

# Check duplicate locations
duplicate_mask = df.duplicated(KEY_COLUMNS, keep=False)
duplicate_rows = int(duplicate_mask.sum())
duplicate_locations = (
    df.loc[duplicate_mask, KEY_COLUMNS]
    .drop_duplicates()
    .shape[0]
)

print(f"Duplicate rows: {duplicate_rows:,}")
print(f"Duplicate locations: {duplicate_locations:,}")


# ------------------------------------------------------------
# Aggregate duplicate tile-boundary locations
# ------------------------------------------------------------

# Columns that can be safely averaged
numeric_columns = [
    column
    for column in [
        *PROBABILITY_COLUMNS,
        "vegetation_score",
        "nonvegetation_score",
    ]
    if column in df.columns
]

# Keep one representative value for metadata
metadata_columns = [
    column
    for column in [
        "dw_period_start",
        "dw_period_end",
    ]
    if column in df.columns
]


def first_valid(series):
    values = series.dropna()
    return values.iloc[0] if len(values) else None


aggregation = {}

for column in numeric_columns:
    aggregation[column] = "mean"

for column in metadata_columns:
    aggregation[column] = first_valid


clean = (
    df.groupby(KEY_COLUMNS, as_index=False)
    .agg(aggregation)
)


# ------------------------------------------------------------
# Recalculate contextual scores from cleaned probabilities
# ------------------------------------------------------------

vegetation_columns = [
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
]

nonvegetation_columns = [
    "water",
    "built",
    "bare",
    "snow_and_ice",
]

clean["vegetation_score"] = clean[vegetation_columns].sum(axis=1)

clean["nonvegetation_score"] = (
    clean[nonvegetation_columns].sum(axis=1)
)


# ------------------------------------------------------------
# Add probability sum for quality control
# ------------------------------------------------------------

clean["probability_sum"] = clean[PROBABILITY_COLUMNS].sum(axis=1)


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLEANING RESULT")
print("=" * 70)

print(f"Output rows: {len(clean):,}")
print(f"Unique locations: {clean[KEY_COLUMNS].drop_duplicates().shape[0]:,}")

remaining_duplicates = int(
    clean.duplicated(KEY_COLUMNS).sum()
)

print(f"Remaining duplicate locations: {remaining_duplicates:,}")

print("\nProbability-sum statistics:")
print(f"  Mean:   {clean['probability_sum'].mean():.6f}")
print(f"  Median: {clean['probability_sum'].median():.6f}")
print(f"  Min:    {clean['probability_sum'].min():.6f}")
print(f"  Max:    {clean['probability_sum'].max():.6f}")

print(
    f"  Between 0.95 and 1.05: "
    f"{((clean['probability_sum'] >= 0.95) & (clean['probability_sum'] <= 1.05)).sum():,}"
)

print(
    f"  Below 0.95: "
    f"{(clean['probability_sum'] < 0.95).sum():,}"
)

print(
    f"  Above 1.05: "
    f"{(clean['probability_sum'] > 1.05).sum():,}"
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

clean.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("FILE SAVED")
print("=" * 70)

print(OUTPUT_FILE)
print(f"Final rows: {len(clean):,}")

print("\nDynamic World cleaning completed.")