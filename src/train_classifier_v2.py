from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

TRAINING_DATA = ROOT / "data" / "training" / "classifier_training_data.csv"
MODEL_PATH = ROOT / "models" / "fire_classifier_v2.pkl"


# ---------------------------------------------------------
# Features
# ---------------------------------------------------------
FEATURES = [
    # FIRMS thermal / temporal
    "total_detections",
    "active_days",
    "active_days_30d",
    "active_days_90d",
    "active_days_180d",
    "persistence_score",
    "mean_bright_ti4",
    "max_bright_ti4",
    "mean_bright_ti5",
    "max_bright_ti5",
    "mean_frp",
    "max_frp",
    "day_detections",
    "night_detections",
    "mean_confidence",
    "max_confidence",
    "night_ratio",
    "observation_span_days",

    # Dynamic World
    "dw_water",
    "dw_trees",
    "dw_grass",
    "dw_flooded_vegetation",
    "dw_crops",
    "dw_shrub_and_scrub",
    "dw_built",
    "dw_bare",
    "dw_vegetation_score",
    "dw_nonvegetation_score",

    # OSM
    "nearest_industrial_area_km",
    "nearest_quarry_km",
    "nearest_power_plant_km",
    "nearest_power_infrastructure_km",
    "nearest_industrial_works_km",
    "nearest_storage_tank_km",
    "nearest_oil_gas_km",
    "nearest_relevant_osm_km",

    # Forest
    "inside_forest",
    "nearest_forest_km",

    # Spatial behaviour
    "spatial_observation_days",
    "expansion_days_1.5km",
    "expansion_days_3km",
    "expansion_days_5km",
    "new_activity_days_1.5km",
    "new_activity_days_3km",
    "new_activity_days_5km",
    "max_local_persistence_1.5km",
    "max_local_persistence_3km",
    "max_local_persistence_5km",
    "new_detection_days",
    "consecutive_day_count",
]


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
df = pd.read_csv(TRAINING_DATA, low_memory=False)

print("Dataset shape:", df.shape)

missing_features = [f for f in FEATURES if f not in df.columns]

if missing_features:
    raise ValueError(
        "Missing required features:\n"
        + "\n".join(missing_features)
    )

X = df[FEATURES].copy()
y = df["training_label"].copy()


# ---------------------------------------------------------
# Spatial groups
#
# 0.1 degree blocks prevent nearby locations from appearing
# in both training and validation sets.
# ---------------------------------------------------------
groups = (
    (df["lat_grid"] // 0.1).astype(int).astype(str)
    + "_"
    + (df["lon_grid"] // 0.1).astype(int).astype(str)
)


# ---------------------------------------------------------
# Spatial train/validation split
# ---------------------------------------------------------
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, val_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]

train_groups = set(groups.iloc[train_idx])
val_groups = set(groups.iloc[val_idx])

print("\nSpatial split")
print("-------------------------")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Training blocks:", len(train_groups))
print("Validation blocks:", len(val_groups))
print("Overlapping blocks:", len(train_groups & val_groups))


# ---------------------------------------------------------
# Class distributions
# ---------------------------------------------------------
print("\nTraining class distribution")
print("-------------------------")
print(y_train.value_counts())

print("\nValidation class distribution")
print("-------------------------")
print(y_val.value_counts())


# ---------------------------------------------------------
# Pipeline
#
# Median imputation handles the 24 missing Dynamic World
# records without leaking validation information.
#
# class_weight='balanced' compensates for the rare
# industrial/persistent classes.
# ---------------------------------------------------------
pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


# ---------------------------------------------------------
# Train
# ---------------------------------------------------------
print("\nTraining Random Forest...")
pipeline.fit(X_train, y_train)

print("Training complete.")


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------
pred = pipeline.predict(X_val)

labels = [
    "BACKGROUND",
    "WILDFIRE",
    "INDUSTRIAL_FIRE",
    "PERSISTENT_THERMAL_SOURCE",
]

print("\nClassification report")
print("=========================")

print(
    classification_report(
        y_val,
        pred,
        labels=labels,
        zero_division=0,
    )
)


print("\nConfusion matrix")
print("=========================")

cm = confusion_matrix(
    y_val,
    pred,
    labels=labels,
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels,
)

print(cm_df)


# ---------------------------------------------------------
# Feature importance
# ---------------------------------------------------------
model = pipeline.named_steps["model"]

importance = pd.Series(
    model.feature_importances_,
    index=FEATURES,
).sort_values(ascending=False)

print("\nTop 20 feature importances")
print("=========================")
print(importance.head(20).to_string())


# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------
bundle = {
    "model": pipeline,
    "features": FEATURES,
    "labels": labels,
    "training_rows": len(X_train),
    "validation_rows": len(X_val),
    "random_state": 42,
    "spatial_block_size": 0.1,
}

joblib.dump(bundle, MODEL_PATH)

print("\nModel saved:")
print(MODEL_PATH)