from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MASTER_PATH = (
    ROOT / "data" / "training"
    / "master_feature_table_with_forest_spatial.csv"
)

LABEL_PATH = (
    ROOT / "data" / "training"
    / "seed_labels.csv"
)

MODEL_PATH = ROOT / "models" / "fire_classifier_v2.pkl"

OUTPUT_PATH = (
    ROOT / "outputs"
    / "classified_unknown_v2.csv"
)


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
features = bundle["features"]
labels = bundle["labels"]

print("Model loaded.")
print("Features:", len(features))
print("Classes:", labels)


# ---------------------------------------------------------
# Load master data + original seed labels
# ---------------------------------------------------------
master = pd.read_csv(
    MASTER_PATH,
    low_memory=False,
)

seed = pd.read_csv(
    LABEL_PATH,
    low_memory=False,
    usecols=["lat_grid", "lon_grid", "seed_label"],
)


# ---------------------------------------------------------
# Identify original UNKNOWN locations
# ---------------------------------------------------------
data = master.merge(
    seed,
    on=["lat_grid", "lon_grid"],
    how="inner",
    validate="one_to_one",
)

unknown = data[
    data["seed_label"].eq("UNKNOWN")
].copy()

print("\nUNKNOWN locations:", len(unknown))


# ---------------------------------------------------------
# Prepare features
# ---------------------------------------------------------
missing = [
    f for f in features
    if f not in unknown.columns
]

if missing:
    raise ValueError(
        "Missing model features:\n"
        + "\n".join(missing)
    )

X = unknown[features]


# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------
print("\nRunning predictions...")

predicted = model.predict(X)
probabilities = model.predict_proba(X)

unknown["predicted_class"] = predicted

for i, label in enumerate(model.classes_):
    unknown[f"prob_{label}"] = probabilities[:, i]

unknown["prediction_confidence"] = probabilities.max(axis=1)


# ---------------------------------------------------------
# Sort by confidence
# ---------------------------------------------------------
unknown = unknown.sort_values(
    "prediction_confidence",
    ascending=False,
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

unknown.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
print("\nPrediction distribution")
print("========================")

print(
    unknown["predicted_class"]
    .value_counts()
)


print("\nPrediction percentages")
print("========================")

print(
    (
        unknown["predicted_class"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


print("\nConfidence distribution")
print("========================")

print(
    unknown["prediction_confidence"]
    .describe()
)


print("\nHigh-confidence predictions")
print("========================")

for threshold in [0.90, 0.95, 0.99]:
    count = (
        unknown["prediction_confidence"] >= threshold
    ).sum()

    print(
        f">= {threshold:.2f}: {count}"
    )


print("\nSaved:")
print(OUTPUT_PATH)