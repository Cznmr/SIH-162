import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


DATA_PATH = "data/training/ml_feature_table.csv"

BLOCK_SIZE = 0.25


print("[1/8] Loading ML feature table...")

df = pd.read_csv(DATA_PATH)

print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns):,}")


# ---------------------------------------------------------
# 1. Keep labeled seed examples
# ---------------------------------------------------------
print("\n[2/8] Selecting labeled data...")

df = df[df["seed_label"] != "UNKNOWN"].copy()

print(f"Labeled rows: {len(df):,}")

print("\nClass distribution:")
print(df["seed_label"].value_counts())


# ---------------------------------------------------------
# 2. Create geographic blocks
# ---------------------------------------------------------
print("\n[3/8] Creating geographic blocks...")

df["lat_block"] = np.floor(df["lat_grid"] / BLOCK_SIZE)
df["lon_block"] = np.floor(df["lon_grid"] / BLOCK_SIZE)

df["spatial_block"] = (
    df["lat_block"].astype(str)
    + "_"
    + df["lon_block"].astype(str)
)

print(f"Unique spatial blocks: {df['spatial_block'].nunique():,}")


# ---------------------------------------------------------
# 3. Show class distribution by block
# ---------------------------------------------------------
block_summary = (
    df.groupby("spatial_block")["seed_label"]
    .agg(["count", "nunique"])
)

print("\nBlock statistics:")
print(f"Minimum labeled rows in block: {block_summary['count'].min()}")
print(f"Maximum labeled rows in block: {block_summary['count'].max()}")
print(f"Median labeled rows in block: {block_summary['count'].median():.0f}")


# ---------------------------------------------------------
# 4. Select test blocks
# ---------------------------------------------------------
print("\n[4/8] Creating spatial train/test split...")

# Deterministic selection of approximately 20% of blocks.
# We select blocks using a fixed random seed so the result
# is reproducible.

rng = np.random.RandomState(42)

all_blocks = np.array(sorted(df["spatial_block"].unique()))

rng.shuffle(all_blocks)

n_test_blocks = max(1, int(round(len(all_blocks) * 0.20)))

test_blocks = set(all_blocks[:n_test_blocks])

df["is_test"] = df["spatial_block"].isin(test_blocks)

train_df = df[~df["is_test"]].copy()
test_df = df[df["is_test"]].copy()

print(f"Training blocks: {len(all_blocks) - n_test_blocks:,}")
print(f"Testing blocks: {n_test_blocks:,}")

print(f"Training rows: {len(train_df):,}")
print(f"Testing rows: {len(test_df):,}")


# ---------------------------------------------------------
# 5. Check class coverage
# ---------------------------------------------------------
print("\n[5/8] Checking class coverage...")

print("\nTraining classes:")
print(train_df["seed_label"].value_counts())

print("\nTesting classes:")
print(test_df["seed_label"].value_counts())


missing_train = set(df["seed_label"].unique()) - set(
    train_df["seed_label"].unique()
)

missing_test = set(df["seed_label"].unique()) - set(
    test_df["seed_label"].unique()
)

if missing_train:
    print("\nWARNING: Classes missing from training:")
    print(missing_train)

if missing_test:
    print("\nWARNING: Classes missing from testing:")
    print(missing_test)


# ---------------------------------------------------------
# 6. Prepare features
# ---------------------------------------------------------
print("\n[6/8] Preparing features...")

drop_columns = [
    "seed_label",
    "lat_grid",
    "lon_grid",
    "lat_block",
    "lon_block",
    "spatial_block",
    "is_test",

    # Date/string fields
    "first_detection",
    "last_detection",
    "first_detection_date",
    "last_detection_date",
    "dw_dw_period_start",
    "dw_dw_period_end",
]

X_train = train_df.drop(
    columns=drop_columns,
    errors="ignore",
)

X_test = test_df.drop(
    columns=drop_columns,
    errors="ignore",
)

y_train = train_df["seed_label"]
y_test = test_df["seed_label"]


# Numeric only
X_train = X_train.select_dtypes(include=[np.number]).copy()
X_test = X_test.select_dtypes(include=[np.number]).copy()

print(f"Numeric features: {X_train.shape[1]}")


# ---------------------------------------------------------
# 7. Align columns
# ---------------------------------------------------------
X_test = X_test.reindex(columns=X_train.columns)


# Replace infinite values
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)


# Median imputation based ONLY on training data
train_medians = X_train.median()

X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)


print(
    f"Training missing values after imputation: "
    f"{X_train.isna().sum().sum():,}"
)

print(
    f"Testing missing values after imputation: "
    f"{X_test.isna().sum().sum():,}"
)


# ---------------------------------------------------------
# 8. Train and evaluate
# ---------------------------------------------------------
print("\n[7/8] Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2,
)

model.fit(X_train, y_train)

print("Training complete.")


print("\n[8/8] Evaluating on geographically unseen blocks...")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 70)
print("SPATIAL HOLDOUT RESULTS")
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

print("\nSpatial holdout completed.")