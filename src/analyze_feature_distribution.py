import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "training"
    / "master_feature_table_with_dynamic_world.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "feature_distribution_report.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FEATURE DISTRIBUTION ANALYSIS")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# NUMERIC FEATURES
# ============================================================

numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()

print("\nNumeric features:", len(numeric_columns))


# ============================================================
# CREATE DISTRIBUTION REPORT
# ============================================================

report = []

for column in numeric_columns:

    series = df[column]

    non_null = series.dropna()

    if len(non_null) == 0:
        continue

    report.append({
        "feature": column,
        "dtype": str(df[column].dtype),
        "missing_count": int(series.isna().sum()),
        "missing_percent": float(series.isna().mean() * 100),
        "unique_values": int(series.nunique()),
        "min": float(non_null.min()),
        "q01": float(non_null.quantile(0.01)),
        "q05": float(non_null.quantile(0.05)),
        "q25": float(non_null.quantile(0.25)),
        "median": float(non_null.median()),
        "q75": float(non_null.quantile(0.75)),
        "q95": float(non_null.quantile(0.95)),
        "q99": float(non_null.quantile(0.99)),
        "max": float(non_null.max()),
        "mean": float(non_null.mean()),
        "std": float(non_null.std())
    })


report_df = pd.DataFrame(report)


# ============================================================
# SORT BY IMPORTANCE FOR INSPECTION
# ============================================================

report_df = report_df.sort_values(
    by="feature"
).reset_index(drop=True)


# ============================================================
# SAVE REPORT
# ============================================================

report_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# PRINT REPORT
# ============================================================

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", "{:.4f}".format)

print("\n" + "=" * 70)
print("FEATURE DISTRIBUTION")
print("=" * 70)

print(
    report_df[
        [
            "feature",
            "missing_percent",
            "unique_values",
            "min",
            "q01",
            "q25",
            "median",
            "q75",
            "q95",
            "q99",
            "max"
        ]
    ].to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

print("\n" + "=" * 70)
print("REPORT SAVED")
print("=" * 70)

print(OUTPUT_FILE)