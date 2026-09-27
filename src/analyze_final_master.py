import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(r"C:\Users\user\Desktop\ENTRO-26162")

MASTER_FILE = (
    BASE_DIR
    / "data"
    / "training"
    / "master_feature_table_final.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_feature_distribution.csv"
)


print("=" * 80)
print("FINAL MASTER FEATURE DISTRIBUTION ANALYSIS")
print("=" * 80)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    MASTER_FILE,
    low_memory=False
)

print("\nDataset shape:")
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns):,}")


# ============================================================
# DUPLICATES
# ============================================================

print("\nDuplicate location check:")

if "lat_grid" in df.columns and "lon_grid" in df.columns:

    duplicates = df.duplicated(
        subset=["lat_grid", "lon_grid"]
    ).sum()

    print(
        f"Duplicate locations: {duplicates:,}"
    )


# ============================================================
# DATA TYPES
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

print(
    f"\nNumeric features: "
    f"{len(numeric_columns):,}"
)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("MISSING VALUE ANALYSIS")
print("=" * 80)

missing_rows = []

for col in df.columns:

    missing = df[col].isna().sum()

    percentage = (
        missing
        / len(df)
        * 100
    )

    missing_rows.append({
        "feature": col,
        "missing_count": missing,
        "missing_percent": percentage
    })


missing_df = pd.DataFrame(
    missing_rows
).sort_values(
    "missing_percent",
    ascending=False
)

print(
    missing_df.head(30).to_string(
        index=False
    )
)


# ============================================================
# NUMERIC DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 80)
print("NUMERIC FEATURE DISTRIBUTIONS")
print("=" * 80)

distribution_rows = []


for col in numeric_columns:

    series = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    valid = series.dropna()

    if len(valid) == 0:
        continue

    distribution_rows.append({

        "feature": col,

        "count": len(valid),

        "missing": series.isna().sum(),

        "missing_percent":
            series.isna().mean() * 100,

        "mean":
            valid.mean(),

        "std":
            valid.std(),

        "min":
            valid.min(),

        "q01":
            valid.quantile(0.01),

        "q05":
            valid.quantile(0.05),

        "q25":
            valid.quantile(0.25),

        "median":
            valid.median(),

        "q75":
            valid.quantile(0.75),

        "q90":
            valid.quantile(0.90),

        "q95":
            valid.quantile(0.95),

        "q99":
            valid.quantile(0.99),

        "max":
            valid.max(),

        "unique":
            valid.nunique(),

        "zero_percent":
            (valid == 0).mean() * 100
    })


distribution_df = pd.DataFrame(
    distribution_rows
)


# ============================================================
# PRINT IMPORTANT FEATURES
# ============================================================

important_keywords = [
    "detection",
    "frp",
    "bright",
    "confidence",
    "night",
    "day",
    "active",
    "persistence",
    "observation",
    "dw_",
    "nearest_",
    "near_"
]


important_features = []

for col in distribution_df["feature"]:

    col_lower = col.lower()

    if any(
        keyword in col_lower
        for keyword in important_keywords
    ):

        important_features.append(col)


important_distribution = distribution_df[
    distribution_df["feature"].isin(
        important_features
    )
].copy()


print(
    "\nImportant feature distributions:"
)

print(
    important_distribution[
        [
            "feature",
            "missing_percent",
            "mean",
            "q01",
            "q05",
            "q25",
            "median",
            "q75",
            "q90",
            "q95",
            "q99",
            "max",
            "zero_percent"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# OSM-SPECIFIC ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("OSM FEATURE ANALYSIS")
print("=" * 80)


osm_distance_columns = [
    col
    for col in df.columns
    if col.startswith("nearest_")
]


for col in osm_distance_columns:

    series = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    print(f"\n{col}")

    print(
        series.describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        )
    )


# ============================================================
# OSM PROXIMITY COUNTS
# ============================================================

print("\nOSM proximity coverage:")

osm_flag_columns = [
    col
    for col in df.columns
    if col.startswith("near_")
]


for col in osm_flag_columns:

    count = (
        pd.to_numeric(
            df[col],
            errors="coerce"
        )
        .fillna(0)
        .sum()
    )

    percentage = (
        count
        / len(df)
        * 100
    )

    print(
        f"{col:50s}"
        f"{int(count):8,}"
        f" ({percentage:6.2f}%)"
    )


# ============================================================
# DYNAMIC WORLD
# ============================================================

print("\n" + "=" * 80)
print("DYNAMIC WORLD ANALYSIS")
print("=" * 80)


dw_columns = [
    col
    for col in df.columns
    if col.startswith("dw_")
]


for col in dw_columns:

    series = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    print(
        f"\n{col}"
    )

    print(
        series.describe(
            percentiles=[
                0.05,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        )
    )


# ============================================================
# PERSISTENCE
# ============================================================

print("\n" + "=" * 80)
print("PERSISTENCE ANALYSIS")
print("=" * 80)


persistence_columns = [
    col
    for col in df.columns
    if (
        "active_days" in col
        or "persistence" in col
        or "persistent_" in col
    )
]


for col in persistence_columns:

    print(
        f"\n{col}"
    )

    print(
        df[col].value_counts(
            dropna=False
        ).head(20)
    )


# ============================================================
# SAVE FULL DISTRIBUTION TABLE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

distribution_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"\nDistribution table saved to:"
)

print(
    OUTPUT_FILE
)