import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


DATA_PATH = "data/training/ml_feature_table.csv"


# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
print("[1/6] Loading ML feature table...")

df = pd.read_csv(DATA_PATH)

print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns):,}")


# ---------------------------------------------------------
# 2. Keep only labeled seed examples
# ---------------------------------------------------------
print("\n[2/6] Selecting labeled training data...")

df = df[df["seed_label"] != "UNKNOWN"].copy()

print(f"Labeled rows: {len(df):,}")

print("\nClass distribution:")
print(df["seed_label"].value_counts())


# ---------------------------------------------------------
# 3. Prepare target and features
# ---------------------------------------------------------
target = "seed_label"

drop_columns = [
    target,
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


# Keep numeric columns only
non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()

if non_numeric:
    print("\nRemoving non-numeric columns:")
    for col in non_numeric:
        print(f"  - {col}")

X = X.select_dtypes(include=[np.number]).copy()

print(f"\nNumeric features: {X.shape[1]}")


# ---------------------------------------------------------
# 4. Define rule-direct features
# ---------------------------------------------------------
rule_direct_features = [
    # FRP rules
    "max_frp",
    "mean_frp",

    # Persistence rules
    "active_days_180d",
    "persistent_180d",
    "persistence_score",
    "observation_span_days",

    # OSM infrastructure rules
    "nearest_industrial_area_km",
    "nearest_industrial_works_km",
    "nearest_power_plant_km",
    "nearest_power_infrastructure_km",
    "nearest_storage_tank_km",
    "nearest_relevant_osm_km",

    # Dynamic World rules
    "dw_built",
    "dw_vegetation_score",
]


# Add all OSM near_* features because they represent the
# same infrastructure-context signal at different radii.
for col in X.columns:
    if col.startswith("near_"):
        rule_direct_features.append(col)


# Remove duplicates and keep only features actually present
rule_direct_features = sorted(
    set(rule_direct_features).intersection(X.columns)
)


print("\nRule-direct features to remove:")
for col in rule_direct_features:
    print(f"  - {col}")

print(f"\nTotal rule-direct features removed: {len(rule_direct_features)}")


# ---------------------------------------------------------
# 5. Train/evaluate function
# ---------------------------------------------------------
def run_experiment(name, X_data):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    X_data = X_data.replace([np.inf, -np.inf], np.nan)

    missing_before = int(X_data.isna().sum().sum())

    # Median imputation
    X_data = X_data.fillna(X_data.median(numeric_only=True))

    missing_after = int(X_data.isna().sum().sum())

    print(f"Features: {X_data.shape[1]}")
    print(f"Missing values before imputation: {missing_before:,}")
    print(f"Missing values after imputation: {missing_after:,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_data,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    print("Training Random Forest...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

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

    return accuracy


# ---------------------------------------------------------
# 6. Run both experiments
# ---------------------------------------------------------

# Experiment A: full feature set
full_accuracy = run_experiment(
    "EXPERIMENT A — FULL FEATURES",
    X.copy(),
)


# Experiment B: remove rule-direct features
X_ablation = X.drop(
    columns=rule_direct_features,
    errors="ignore",
)

ablation_accuracy = run_experiment(
    "EXPERIMENT B — RULE-DIRECT FEATURES REMOVED",
    X_ablation,
)


# ---------------------------------------------------------
# Final comparison
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("ABLATION TEST SUMMARY")
print("=" * 70)

print(f"Full-feature accuracy:       {full_accuracy:.4f}")
print(f"Rule-ablated accuracy:       {ablation_accuracy:.4f}")
print(
    f"Accuracy difference:         "
    f"{full_accuracy - ablation_accuracy:+.4f}"
)

print("\nInterpretation:")
if ablation_accuracy >= full_accuracy - 0.02:
    print(
        "Performance remained similar after removing rule-direct "
        "features. This suggests the model may be learning additional "
        "patterns beyond the seed-label rules."
    )
else:
    print(
        "Performance dropped noticeably after removing rule-direct "
        "features. The original high accuracy is likely strongly "
        "dependent on reproducing the seed-label rules."
    )