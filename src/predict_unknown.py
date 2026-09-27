import os
import joblib
import pandas as pd
import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(ROOT, "models", "fire_classifier.pkl")
DATA_PATH = os.path.join(ROOT, "data", "training", "ml_feature_table.csv")
OUTPUT_PATH = os.path.join(
    ROOT, "data", "training", "unknown_predictions.csv"
)


print("=" * 70)
print("UNKNOWN POOL PREDICTION")
print("=" * 70)


# ---------------------------------------------------------
# 1. Load model bundle
# ---------------------------------------------------------
print("\n[1/5] Loading trained model...")

model_bundle = joblib.load(MODEL_PATH)

model = model_bundle["model"]
trained_features = model_bundle["features"]
imputation_medians = model_bundle["imputation_medians"]

print("Model type:", type(model).__name__)
print("Saved features:", len(trained_features))
print("Model classes:", list(model_bundle["classes"]))


# ---------------------------------------------------------
# 2. Load feature table
# ---------------------------------------------------------
print("\n[2/5] Loading ML feature table...")

df = pd.read_csv(DATA_PATH)

print("Total rows:", len(df))
print("Total columns:", len(df.columns))


# ---------------------------------------------------------
# 3. Select UNKNOWN rows
# ---------------------------------------------------------
print("\n[3/5] Selecting UNKNOWN locations...")

if "seed_label" not in df.columns:
    raise ValueError("seed_label column not found.")

unknown = df[df["seed_label"] == "UNKNOWN"].copy()

print("UNKNOWN rows:", len(unknown))


# ---------------------------------------------------------
# 4. Prepare EXACT training features
# ---------------------------------------------------------
print("\n[4/5] Preparing features...")

# Check that every feature used during training exists
missing_features = [
    feature
    for feature in trained_features
    if feature not in unknown.columns
]

if missing_features:
    print("\nERROR: Missing trained features:")
    for feature in missing_features:
        print(" -", feature)

    raise ValueError(
        f"{len(missing_features)} trained features are missing."
    )


# Use EXACTLY the same feature columns as training
X = unknown[trained_features].copy()

print("Features used:", X.shape[1])


# Replace infinite values
X = X.replace([np.inf, -np.inf], np.nan)


# Use EXACT medians saved during training
for feature in trained_features:

    if X[feature].isna().any():

        if feature not in imputation_medians:
            raise ValueError(
                f"No saved imputation median found for: {feature}"
            )

        X[feature] = X[feature].fillna(
            imputation_medians[feature]
        )


remaining_missing = X.isna().sum().sum()

print(
    "Missing values after imputation:",
    remaining_missing
)

if remaining_missing > 0:
    raise ValueError(
        "Missing values remain after applying saved medians."
    )


# ---------------------------------------------------------
# 5. Predict UNKNOWN locations
# ---------------------------------------------------------
print("\n[5/5] Predicting UNKNOWN locations...")

predictions = model.predict(X)
probabilities = model.predict_proba(X)

classes = model.classes_

max_probability = probabilities.max(axis=1)


# ---------------------------------------------------------
# Build result table
# ---------------------------------------------------------
result = unknown[
    ["lat_grid", "lon_grid", "seed_label"]
].copy()

result["predicted_class"] = predictions
result["confidence"] = max_probability


# Add probability for every class
for i, class_name in enumerate(classes):

    result[f"prob_{class_name}"] = probabilities[:, i]


# Sort highest confidence first
result = result.sort_values(
    "confidence",
    ascending=False
).reset_index(drop=True)


# ---------------------------------------------------------
# Save predictions
# ---------------------------------------------------------
result.to_csv(
    OUTPUT_PATH,
    index=False
)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print("UNKNOWN locations:", len(result))

print("\nPredicted classes:")
print(
    result["predicted_class"].value_counts()
)

print("\nConfidence distribution:")
print(
    result["confidence"].describe()
)

print(
    "\nConfidence >= 0.90:",
    (result["confidence"] >= 0.90).sum()
)

print(
    "Confidence >= 0.95:",
    (result["confidence"] >= 0.95).sum()
)

print(
    "Confidence >= 0.99:",
    (result["confidence"] >= 0.99).sum()
)

print("\nSaved:")
print(OUTPUT_PATH)

print("=" * 70)