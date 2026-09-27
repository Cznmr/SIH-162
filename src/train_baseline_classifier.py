import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import joblib


# ============================================================
# PS 26162 - Baseline Random Forest Classifier
# ============================================================

BASE = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE / "data" / "training" / "ml_feature_table.csv"
MODEL_FILE = BASE / "models" / "fire_classifier.pkl"


print("=" * 70)
print("PS 26162 - BASELINE ML CLASSIFIER")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

print("\n[1/7] Loading ML feature table...")

df = pd.read_csv(INPUT_FILE)

print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns):,}")


# ------------------------------------------------------------
# 2. Keep only known seed labels
# ------------------------------------------------------------

print("\n[2/7] Selecting labeled training data...")

UNKNOWN_LABEL = "UNKNOWN"

labeled = df[df["seed_label"] != UNKNOWN_LABEL].copy()

print(f"Labeled rows: {len(labeled):,}")
print(f"UNKNOWN rows excluded: {len(df) - len(labeled):,}")

print("\nTraining class distribution:")
print(labeled["seed_label"].value_counts())


# ------------------------------------------------------------
# 3. Prepare X and y
# ------------------------------------------------------------

print("\n[3/7] Preparing features and target...")

TARGET = "seed_label"

# Coordinate columns are identifiers, not predictive features.
# They are excluded to prevent the model from simply learning
# geographic location.
EXCLUDE_COLUMNS = [
    TARGET,
    "lat_grid",
    "lon_grid",
]

X = labeled.drop(columns=EXCLUDE_COLUMNS)
y = labeled[TARGET]

print(f"Initial feature count: {X.shape[1]}")


# ------------------------------------------------------------
# 4. Remove non-numeric columns
# ------------------------------------------------------------

print("\n[4/7] Preparing numeric features...")

non_numeric = X.select_dtypes(
    exclude=[np.number]
).columns.tolist()

if non_numeric:
    print("Removing non-numeric columns:")
    for col in non_numeric:
        print(f"  - {col}")

    X = X.drop(columns=non_numeric)

# Replace infinite values
X = X.replace([np.inf, -np.inf], np.nan)

# Median imputation for numeric missing values
missing_before = X.isna().sum().sum()

X = X.fillna(X.median(numeric_only=True))

missing_after = X.isna().sum().sum()

print(f"Numeric features: {X.shape[1]:,}")
print(f"Missing values before imputation: {missing_before:,}")
print(f"Missing values after imputation:  {missing_after:,}")


# ------------------------------------------------------------
# 5. Train/test split
# ------------------------------------------------------------

print("\n[5/7] Creating stratified train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Training rows: {len(X_train):,}")
print(f"Testing rows:  {len(X_test):,}")


# ------------------------------------------------------------
# 6. Train Random Forest
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 7. Evaluate
# ------------------------------------------------------------

print("\n[7/7] Evaluating classifier...")

y_pred = model.predict(X_test)

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        digits=4,
        zero_division=0,
    )
)

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

classes = model.classes_

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=classes,
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual_{c}" for c in classes],
    columns=[f"Predicted_{c}" for c in classes],
)

print(cm_df)


# ------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_,
}).sort_values(
    "importance",
    ascending=False,
)

print("\n" + "=" * 70)
print("TOP 25 FEATURE IMPORTANCES")
print("=" * 70)

print(importance.head(25).to_string(index=False))


# ------------------------------------------------------------
# Save model
# ------------------------------------------------------------

MODEL_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

joblib.dump(
    {
        "model": model,
        "features": X.columns.tolist(),
        "classes": model.classes_.tolist(),
        "imputation_medians": X.median().to_dict(),
    },
    MODEL_FILE,
)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(MODEL_FILE)
print("=" * 70)