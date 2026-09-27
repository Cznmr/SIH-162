import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


DATA_PATH = "data/training/ml_feature_table.csv"


print("[1/7] Loading ML feature table...")

df = pd.read_csv(DATA_PATH)

print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns):,}")


# ---------------------------------------------------------
# 1. Keep only labeled seed examples
# ---------------------------------------------------------
print("\n[2/7] Selecting labeled data...")

df = df[df["seed_label"] != "UNKNOWN"].copy()

print(f"Labeled rows: {len(df):,}")

print("\nClass distribution:")
print(df["seed_label"].value_counts())


# ---------------------------------------------------------
# 2. Prepare target
# ---------------------------------------------------------
target = "seed_label"

drop_columns = [
    target,

    # Spatial coordinates
    "lat_grid",
    "lon_grid",

    # Date/string fields
    "first_detection",
    "last_detection",
    "first_detection_date",
    "last_detection_date",
    "dw_dw_period_start",
    "dw_dw_period_end",
]

X = df.drop(columns=drop_columns, errors="ignore")
y = df[target]

X = X.select_dtypes(include=[np.number]).copy()

print(f"\nInitial numeric features: {X.shape[1]}")


# ---------------------------------------------------------
# 3. Remove groups that directly describe the seed rules
# ---------------------------------------------------------
print("\n[3/7] Removing rule-dependent feature groups...")


REMOVE_PATTERNS = [
    # Thermal intensity
    "frp",
    "bright_ti4",
    "bright_ti5",

    # Persistence / duration
    "active_days",
    "persistent",
    "persistence",
    "observation_span",
    "detection_count",
    "total_detections",

    # OSM / industrial context
    "industrial",
    "power",
    "storage",
    "quarry",
    "relevant_osm",
    "oil_gas",
    "osm",

    # Dynamic World land-cover context
    "dw_",
    "vegetation",
    "built",
    "trees",
    "grass",
    "crops",
    "shrub",
    "bare",
    "water",
    "flooded",

    # Spatial population context
    "population",
    "neighbor",
    "expansion",
    "contraction",
    "directional",
    "local_radius",
]


features_to_remove = []

for col in X.columns:
    col_lower = col.lower()

    for pattern in REMOVE_PATTERNS:
        if pattern in col_lower:
            features_to_remove.append(col)
            break


features_to_remove = sorted(set(features_to_remove))

print("\nFeatures removed:")

for col in features_to_remove:
    print(f"  - {col}")

print(
    f"\nTotal removed: {len(features_to_remove)}"
)


X_independent = X.drop(
    columns=features_to_remove,
    errors="ignore",
)

print(
    f"Remaining independent features: "
    f"{X_independent.shape[1]}"
)


# ---------------------------------------------------------
# 4. Show remaining features
# ---------------------------------------------------------
print("\nRemaining features:")

for col in X_independent.columns:
    print(f"  - {col}")


# ---------------------------------------------------------
# 5. Prepare data
# ---------------------------------------------------------
print("\n[4/7] Preparing features...")

X_independent = X_independent.replace(
    [np.inf, -np.inf],
    np.nan,
)

missing_before = int(
    X_independent.isna().sum().sum()
)

X_independent = X_independent.fillna(
    X_independent.median(numeric_only=True)
)

missing_after = int(
    X_independent.isna().sum().sum()
)

print(
    f"Missing values before imputation: "
    f"{missing_before:,}"
)

print(
    f"Missing values after imputation: "
    f"{missing_after:,}"
)


# ---------------------------------------------------------
# 6. Train/test split
# ---------------------------------------------------------
print("\n[5/7] Creating stratified train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X_independent,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Training rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")


# ---------------------------------------------------------
# 7. Train and evaluate
# ---------------------------------------------------------
print("\n[6/7] Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2,
)

model.fit(X_train, y_train)

print("Training complete.")


print("\n[7/7] Evaluating...")

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions,
)

print("\n" + "=" * 70)
print("INDEPENDENT FEATURE TEST")
print("=" * 70)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification report:")

print(
    classification_report(
        y_test,
        predictions,
        digits=4,
        zero_division=0,
    )
)


print("\nTop feature importances:")

importance = pd.Series(
    model.feature_importances_,
    index=X_independent.columns,
).sort_values(ascending=False)

print(importance.head(25).to_string())