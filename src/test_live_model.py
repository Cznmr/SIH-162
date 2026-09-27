import os
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "fire_classifier_v2.pkl"

FEATURE_TABLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "master_feature_table_with_forest_spatial.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "live_model_predictions.csv"
)

WEST = 77.0
SOUTH = 15.5
EAST = 81.6
NORTH = 20.2

GRID_SIZE_DEG = 0.01

FIRMS_AREA = f"{WEST},{SOUTH},{EAST},{NORTH}"

FIRMS_API_BASE = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
)

FIRMS_SOURCES = {
    "NOAA-20": "VIIRS_NOAA20_NRT",
    "NOAA-21": "VIIRS_NOAA21_NRT",
}

FIRMS_DAYS = 2


# ============================================================
# HELPERS
# ============================================================

def lat_to_idx(lat):
    return int(np.floor(lat / GRID_SIZE_DEG))


def lon_to_idx(lon):
    return int(np.floor(lon / GRID_SIZE_DEG))


def get_firms_key():
    key = os.getenv("FIRMS_MAP_KEY")

    if not key:
        raise RuntimeError(
            "FIRMS_MAP_KEY is not configured in this terminal."
        )

    return key.strip()


# ============================================================
# FETCH LIVE FIRMS
# ============================================================

def fetch_live_firms():

    key = get_firms_key()

    frames = []

    for satellite_name, source_code in FIRMS_SOURCES.items():

        url = (
            f"{FIRMS_API_BASE}/"
            f"{key}/"
            f"{source_code}/"
            f"{FIRMS_AREA}/"
            f"{FIRMS_DAYS}"
        )

        print(f"Fetching {satellite_name}...")

        response = requests.get(
            url,
            timeout=45,
        )

        response.raise_for_status()

        if not response.text.strip():
            print(f"{satellite_name}: empty response")
            continue

        df = pd.read_csv(
            StringIO(response.text)
        )

        if df.empty:
            print(f"{satellite_name}: no detections")
            continue

        df["source_name"] = satellite_name
        df["source_dataset"] = source_code

        frames.append(df)

        print(
            f"{satellite_name}: "
            f"{len(df):,} detections"
        )

    if not frames:
        return pd.DataFrame()

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for column in [
        "latitude",
        "longitude",
        "frp",
        "bright_ti4",
        "bright_ti5",
    ]:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df = df.dropna(
        subset=[
            "latitude",
            "longitude",
        ]
    )

    # --------------------------------------------------------
    # Telangana bounding box
    # --------------------------------------------------------

    df = df[
        (df["latitude"] >= SOUTH)
        & (df["latitude"] <= NORTH)
        & (df["longitude"] >= WEST)
        & (df["longitude"] <= EAST)
    ].copy()

    # --------------------------------------------------------
    # Acquisition datetime
    # --------------------------------------------------------

    if "acq_date" in df.columns:

        df["acq_date"] = (
            df["acq_date"]
            .astype(str)
        )

    if "acq_time" in df.columns:

        df["acq_time"] = (
            df["acq_time"]
            .astype(str)
            .str.replace(
                ".0",
                "",
                regex=False,
            )
            .str.zfill(4)
        )

    df["acq_datetime"] = pd.to_datetime(
        df["acq_date"]
        + " "
        + df["acq_time"].str[:2]
        + ":"
        + df["acq_time"].str[2:4],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Remove duplicate satellite observations
    # --------------------------------------------------------

    dedupe_columns = [
        column
        for column in [
            "latitude",
            "longitude",
            "acq_date",
            "acq_time",
            "satellite",
            "instrument",
        ]
        if column in df.columns
    ]

    if dedupe_columns:

        df = df.drop_duplicates(
            subset=dedupe_columns
        )

    return (
        df.sort_values(
            "acq_datetime",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# LOAD FEATURE TABLE
# ============================================================

def load_feature_table(
    trained_features,
):

    print()
    print("Loading feature table...")

    if not FEATURE_TABLE_PATH.exists():

        raise FileNotFoundError(
            f"Feature table not found:\n"
            f"{FEATURE_TABLE_PATH}"
        )

    df = pd.read_csv(
        FEATURE_TABLE_PATH,
        low_memory=False,
    )

    print(
        f"Feature table rows: {len(df):,}"
    )

    required_columns = [
        "lat_grid",
        "lon_grid",
        *trained_features,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Feature table is missing these "
            "model features:\n"
            + "\n".join(missing)
        )

    return df


# ============================================================
# MATCH LIVE DETECTIONS TO TRAINING GRID
# ============================================================

def attach_grid_features(
    live_df,
    feature_df,
    trained_features,
):

    print()
    print(
        "Matching live detections "
        "to model feature grid..."
    )

    live_df = live_df.copy()

    # --------------------------------------------------------
    # Calculate the same 0.01 degree grid
    # --------------------------------------------------------

    live_df["lat_grid"] = (
        live_df["latitude"]
        .apply(lat_to_idx)
        * GRID_SIZE_DEG
    )

    live_df["lon_grid"] = (
        live_df["longitude"]
        .apply(lon_to_idx)
        * GRID_SIZE_DEG
    )

    live_df["lat_grid"] = (
        live_df["lat_grid"]
        .round(2)
    )

    live_df["lon_grid"] = (
        live_df["lon_grid"]
        .round(2)
    )

    # --------------------------------------------------------
    # Build feature lookup
    # --------------------------------------------------------

    feature_lookup = feature_df[
        [
            "lat_grid",
            "lon_grid",
            *trained_features,
        ]
    ].copy()

    # --------------------------------------------------------
    # Match
    # --------------------------------------------------------

    merged = live_df.merge(
        feature_lookup,
        on=[
            "lat_grid",
            "lon_grid",
        ],
        how="left",
        indicator=True,
    )

    matched = (
        merged["_merge"] == "both"
    ).sum()

    unmatched = (
        merged["_merge"] != "both"
    ).sum()

    print(
        f"Matched live detections: "
        f"{matched:,}"
    )

    print(
        f"Unmatched live detections: "
        f"{unmatched:,}"
    )

    merged = merged.drop(
        columns=["_merge"]
    )

    return merged


# ============================================================
# RUN MODEL
# ============================================================

def classify_live_detections(
    live_with_features,
):

    print()
    print("Loading ENTRO classifier...")

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load model bundle
    # --------------------------------------------------------

    model_bundle = joblib.load(
        MODEL_PATH
    )

    if not isinstance(
        model_bundle,
        dict,
    ):

        raise RuntimeError(
            "The saved classifier is not "
            "in the expected dictionary format."
        )

    # --------------------------------------------------------
    # Extract actual trained model
    # --------------------------------------------------------

    model = model_bundle["model"]

    # --------------------------------------------------------
    # Extract exact feature list used during training
    # --------------------------------------------------------

    trained_features = model_bundle[
        "features"
    ]

    print(
        f"Model loaded: {MODEL_PATH.name}"
    )

    print(
        f"Training features expected by model: "
        f"{len(trained_features)}"
    )

    # --------------------------------------------------------
    # Verify features exist
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in trained_features
        if feature not in live_with_features.columns
    ]

    if missing_features:

        raise RuntimeError(
            "Live data is missing these "
            "model features:\n"
            + "\n".join(missing_features)
        )

    # --------------------------------------------------------
    # Create model input
    # --------------------------------------------------------

    model_input = live_with_features[
        trained_features
    ].copy()

    # The model pipeline contains its own
    # median imputer, so rows do not need
    # to be completely non-null.
    valid_mask = model_input.notna().any(
        axis=1
    )

    valid_input = model_input.loc[
        valid_mask
    ]

    print(
        f"Rows sent to model: "
        f"{len(valid_input):,}"
    )

    if valid_input.empty:

        raise RuntimeError(
            "No live detections have usable "
            "model features."
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = model.predict(
        valid_input
    )

    probabilities = model.predict_proba(
        valid_input
    )

    # --------------------------------------------------------
    # Classes
    # --------------------------------------------------------

    if hasattr(
        model,
        "classes_",
    ):

        classes = list(
            model.classes_
        )

    else:

        classes = list(
            model_bundle["labels"]
        )

    # --------------------------------------------------------
    # Build results
    # --------------------------------------------------------

    result = live_with_features.loc[
        valid_mask
    ].copy()

    result["prediction"] = predictions

    result["model_confidence"] = (
        probabilities.max(
            axis=1
        )
    )

    for index, class_name in enumerate(
        classes
    ):

        result[
            f"prob_{class_name}"
        ] = probabilities[
            :,
            index,
        ]

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "ENTRO-26162 LIVE MODEL TEST"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load model first
    # --------------------------------------------------------

    print()
    print(
        "Loading model metadata..."
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL_PATH}"
        )

    model_bundle = joblib.load(
        MODEL_PATH
    )

    if not isinstance(
        model_bundle,
        dict,
    ):

        raise RuntimeError(
            "Unexpected model file format."
        )

    trained_features = (
        model_bundle["features"]
    )

    print(
        f"Model expects "
        f"{len(trained_features)} features."
    )

    # --------------------------------------------------------
    # Fetch live NASA FIRMS
    # --------------------------------------------------------

    live_df = fetch_live_firms()

    if live_df.empty:

        print()
        print(
            "No live FIRMS detections "
            "were returned."
        )

        return

    print()
    print(
        f"Total live detections: "
        f"{len(live_df):,}"
    )

    # --------------------------------------------------------
    # Load feature table
    # --------------------------------------------------------

    feature_df = load_feature_table(
        trained_features
    )

    # --------------------------------------------------------
    # Match detections to grid
    # --------------------------------------------------------

    live_with_features = (
        attach_grid_features(
            live_df,
            feature_df,
            trained_features,
        )
    )

    # --------------------------------------------------------
    # Run ENTRO model
    # --------------------------------------------------------

    predictions = (
        classify_live_detections(
            live_with_features
        )
    )

    # --------------------------------------------------------
    # Sort by confidence
    # --------------------------------------------------------

    predictions = (
        predictions
        .sort_values(
            "model_confidence",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL RESULTS")
    print("=" * 70)

    print()

    print(
        predictions[
            "prediction"
        ].value_counts()
    )

    print()

    print(
        "Average model confidence:",
        f"{predictions['model_confidence'].mean():.3f}",
    )

    print()

    print("Top predictions:")

    display_columns = [
        "latitude",
        "longitude",
        "frp",
        "prediction",
        "model_confidence",
    ]

    available = [
        column
        for column in display_columns
        if column in predictions.columns
    ]

    print(
        predictions[
            available
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 70)

    print(
        f"Saved results to:\n"
        f"{OUTPUT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()