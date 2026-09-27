import os
from io import StringIO

import joblib
import numpy as np
import pydeck as pdk

import folium
import pandas as pd
import requests
import streamlit as st
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

def load_local_env():
    """Load simple KEY=VALUE pairs from the project .env file without requiring activation."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"").strip("'")
                if key:
                    os.environ.setdefault(key, value)
    except OSError:
        pass


load_local_env()


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ENTRO-26162 | Thermal Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

WEST = 77.0
SOUTH = 15.5
EAST = 81.6
NORTH = 20.2

FIRMS_AREA = f"{WEST},{SOUTH},{EAST},{NORTH}"

FIRMS_API_BASE = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
)

FIRMS_SOURCES = {
    "NOAA-20": "VIIRS_NOAA20_NRT",
    "NOAA-21": "VIIRS_NOAA21_NRT",
}

FIRMS_DAYS = 2

# Maximum number of points shown on the browser map.
# FIRMS data itself is not deleted.
MAP_POINT_LIMIT = 2000

# ENTRO model/grid configuration
GRID_SIZE_DEG = 0.01
MODEL_PATH = "models/fire_classifier_v2.pkl"
FEATURE_TABLE_PATH = "data/training/master_feature_table_with_forest_spatial.csv"

CLASSIFICATION_LABELS = {
    "WILDFIRE": "Wildfire candidate",
    "INDUSTRIAL_FIRE": "Industrial fire candidate",
    "PERSISTENT_THERMAL_SOURCE": "Persistent thermal source",
    "BACKGROUND": "Background / low-priority",
}


# ============================================================
# COLOR PALETTE
# ============================================================

COLORS = {
    "background": "#FFFBF4",
    "card": "#FFFFFF",
    "border": "#EEE5D8",
    "primary": "#1F4E5F",
    "secondary": "#5B7C8D",
    "map": "#2A9D8F",
    "thermal": "#E9A23B",
    "industrial": "#D95D5D",
    "persistent": "#7C6FC4",
    "text": "#263746",
    "muted": "#71808D",
    "soft_teal": "#EAF6F3",
    "soft_amber": "#FFF5DF",
    "soft_red": "#FCECEC",
    "soft_violet": "#F2EFFB",
}

CLASSIFICATION_COLORS = {
    "WILDFIRE": COLORS["thermal"],
    "INDUSTRIAL_FIRE": COLORS["industrial"],
    "PERSISTENT_THERMAL_SOURCE": COLORS["persistent"],
    "BACKGROUND": COLORS["map"],
}


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    f"""
<style>
[data-testid="stHeader"] {{
    display: none;
}}

[data-testid="stToolbar"] {{
    display: none;
}}

[data-testid="stDecoration"] {{
    display: none;
}}
.stApp {{
    background: {COLORS["background"]};
    color: {COLORS["text"]};
}}

.block-container {{
    max-width: 1500px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}}

[data-testid="stSidebar"] {{
    background: #FFFDF9;
    border-right: 1px solid {COLORS["border"]};
}}

[data-testid="stSidebar"] * {{
    color: {COLORS["text"]};
}}

h1, h2, h3 {{
    color: {COLORS["primary"]};
}}

.header-box {{
    background: {COLORS["card"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 14px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
}}

.header-title {{
    font-size: 1.45rem;
    font-weight: 700;
    color: {COLORS["primary"]};
    margin-bottom: 0.15rem;
}}

.header-subtitle {{
    font-size: 0.82rem;
    color: {COLORS["muted"]};
}}

.status-online {{
    color: #2A806C;
    font-weight: 700;
}}

.status-offline {{
    color: {COLORS["industrial"]};
    font-weight: 700;
}}

.section-title {{
    font-size: 0.76rem;
    font-weight: 700;
    color: {COLORS["primary"]};
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 0.8rem;
    margin-bottom: 0.5rem;
}}

.info-box {{
    background: {COLORS["card"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 12px;
    padding: 1rem;
}}

.small-label {{
    color: {COLORS["muted"]};
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

.big-number {{
    color: {COLORS["primary"]};
    font-size: 1.55rem;
    font-weight: 750;
    margin-top: 0.15rem;
}}

.small-text {{
    color: {COLORS["muted"]};
    font-size: 0.76rem;
}}

.footer {{
    color: {COLORS["muted"]};
    font-size: 0.72rem;
    text-align: center;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid {COLORS["border"]};
}}

div[data-testid="stMetric"] {{
    background: {COLORS["card"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 12px;
    padding: 0.8rem;
}}

div[data-testid="stMetricLabel"] {{
    color: {COLORS["muted"]};
}}

div[data-testid="stMetricValue"] {{
    color: {COLORS["primary"]};
}}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# FIRMS API KEY
# ============================================================

def get_firms_map_key():
    """
    Read FIRMS_MAP_KEY from:
    1. Environment variable
    2. Streamlit secrets
    """

    key = os.getenv("FIRMS_MAP_KEY")

    if key:
        return key.strip()

    try:
        key = st.secrets.get("FIRMS_MAP_KEY")

        if key:
            return str(key).strip()

    except Exception:
        pass

    return None


# ============================================================
# FETCH NASA FIRMS DATA
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_firms_data():

    map_key = get_firms_map_key()

    if not map_key:
        return pd.DataFrame(), {
            "NOAA-20": "FIRMS_MAP_KEY is not configured.",
            "NOAA-21": "FIRMS_MAP_KEY is not configured.",
        }

    all_frames = []
    errors = {}

    for satellite_name, source_code in FIRMS_SOURCES.items():

        url = (
            f"{FIRMS_API_BASE}/"
            f"{map_key}/"
            f"{source_code}/"
            f"{FIRMS_AREA}/"
            f"{FIRMS_DAYS}"
        )

        try:

            response = requests.get(
                url,
                timeout=45,
            )

            response.raise_for_status()

            if not response.text.strip():
                errors[satellite_name] = "Empty response from NASA FIRMS."
                continue

            df = pd.read_csv(
                StringIO(response.text)
            )

            if df.empty:
                errors[satellite_name] = "No detections returned."
                continue

            df["source_name"] = satellite_name
            df["source_dataset"] = source_code

            all_frames.append(df)

        except Exception as exc:

            errors[satellite_name] = str(exc)

    if not all_frames:
        return pd.DataFrame(), errors

    df = pd.concat(
        all_frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_columns = [
        "latitude",
        "longitude",
        "frp",
        "bright_ti4",
        "bright_ti5",
        "scan",
        "track",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Telangana bounding box
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "latitude",
            "longitude",
        ]
    )

    df = df[
        (df["latitude"] >= SOUTH)
        & (df["latitude"] <= NORTH)
        & (df["longitude"] >= WEST)
        & (df["longitude"] <= EAST)
    ].copy()

    # --------------------------------------------------------
    # Acquisition date/time
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

    if (
        "acq_date" in df.columns
        and "acq_time" in df.columns
    ):

        df["acq_datetime"] = pd.to_datetime(
            df["acq_date"]
            + " "
            + df["acq_time"].str[:2]
            + ":"
            + df["acq_time"].str[2:4],
            errors="coerce",
        )

    else:

        df["acq_datetime"] = pd.NaT

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

    # --------------------------------------------------------
    # Sort newest first
    # --------------------------------------------------------

    df = (
        df.sort_values(
            "acq_datetime",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return df, errors


# ============================================================
# ENTRO MODEL INTEGRATION
# ============================================================

@st.cache_resource(show_spinner=False)
def load_entro_model():
    """Load the trained ENTRO model bundle once per Streamlit process."""
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found: {MODEL_PATH}"

    try:
        bundle = joblib.load(MODEL_PATH)
        if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
            return None, "Invalid ENTRO model bundle."
        return bundle, None
    except Exception as exc:
        return None, f"Could not load ENTRO model: {exc}"


@st.cache_data(show_spinner=False)
def load_entro_feature_table():
    """Load the historical engineered feature table used by the trained model."""
    if not os.path.exists(FEATURE_TABLE_PATH):
        return pd.DataFrame(), f"Feature table not found: {FEATURE_TABLE_PATH}"

    try:
        df = pd.read_csv(FEATURE_TABLE_PATH, low_memory=False)
        required_grid = {"lat_grid", "lon_grid"}
        if not required_grid.issubset(df.columns):
            return pd.DataFrame(), "Feature table is missing lat_grid/lon_grid."
        return df, None
    except Exception as exc:
        return pd.DataFrame(), f"Could not load feature table: {exc}"


def _live_grid(df):
    """Apply the exact floor-based 0.01 degree grid used during training."""
    out = df.copy()
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out["lat_grid"] = np.floor(out["latitude"] / GRID_SIZE_DEG) * GRID_SIZE_DEG
    out["lon_grid"] = np.floor(out["longitude"] / GRID_SIZE_DEG) * GRID_SIZE_DEG
    out["lat_grid"] = out["lat_grid"].round(2)
    out["lon_grid"] = out["lon_grid"].round(2)
    return out


@st.cache_data(ttl=900, show_spinner=False)
def classify_live_firms(live_df):
    """Classify live FIRMS detections using the existing trained ENTRO model."""
    empty = pd.DataFrame()

    if live_df is None or live_df.empty:
        return empty, {"error": "No live FIRMS detections available."}

    bundle, model_error = load_entro_model()
    feature_df, feature_error = load_entro_feature_table()

    if model_error:
        return empty, {"error": model_error}
    if feature_error:
        return empty, {"error": feature_error}

    model = bundle["model"]
    trained_features = list(bundle["features"])

    live = _live_grid(live_df)
    feature_df = feature_df.copy()
    feature_df["lat_grid"] = pd.to_numeric(feature_df["lat_grid"], errors="coerce").round(2)
    feature_df["lon_grid"] = pd.to_numeric(feature_df["lon_grid"], errors="coerce").round(2)

    # Keep one engineered row per model grid cell.
    feature_lookup = feature_df.drop_duplicates(
        subset=["lat_grid", "lon_grid"], keep="first"
    )

    merged = live.merge(
        feature_lookup[["lat_grid", "lon_grid"] + trained_features],
        on=["lat_grid", "lon_grid"],
        how="left",
        suffixes=("", "_feature"),
        indicator="_feature_match",
    )

    # A matched row may contain NaNs because the trained pipeline has an imputer.
    matched_mask = merged["_feature_match"].eq("both")

    matched = merged.loc[matched_mask].copy()
    unmatched_count = int((~matched_mask).sum())

    if matched.empty:
        return empty, {
            "error": "No live FIRMS detections matched the trained feature grid.",
            "unmatched": unmatched_count,
        }

    X = matched[trained_features].copy()
    predictions = model.predict(X)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        confidences = probabilities.max(axis=1)
        classes = list(getattr(model, "classes_", bundle.get("labels", [])))
    else:
        probabilities = None
        confidences = np.full(len(matched), np.nan)
        classes = list(bundle.get("labels", []))

    matched["prediction"] = predictions
    matched["model_confidence"] = confidences

    if probabilities is not None and classes:
        for index, class_name in enumerate(classes):
            safe_name = str(class_name).lower().replace(" ", "_")
            matched[f"prob_{safe_name}"] = probabilities[:, index]

    return matched, {
        "unmatched": unmatched_count,
        "matched": int(len(matched)),
        "error": None,
    }


# ============================================================
# LOAD DATA
# ============================================================

firms_df, firms_errors = fetch_firms_data()

firms_available = not firms_df.empty

# Run the existing ENTRO classifier against the live FIRMS observations.
classified_df, classification_status = classify_live_firms(firms_df)
classification_available = not classified_df.empty


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        f"""
<div style="
    padding:0.4rem 0 1rem 0;
    border-bottom:1px solid {COLORS["border"]};
">
    <div style="
        font-size:0.72rem;
        color:{COLORS["muted"]};
        letter-spacing:0.08em;
        font-weight:700;
    ">
        ENTRO-26162
    </div>

    <div style="
        font-size:1.05rem;
        color:{COLORS["primary"]};
        font-weight:750;
        margin-top:3px;
    ">
        Thermal Intelligence
    </div>

    <div style="
        font-size:0.72rem;
        color:{COLORS["muted"]};
        margin-top:3px;
    ">
        Telangana monitoring system
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("### Monitoring")

    show_detections = st.checkbox(
        "Live FIRMS detections",
        value=True,
    )

    show_map = st.checkbox(
        "Thermal activity map",
        value=True,
    )

    st.markdown("### Classification")

    show_wildfire = st.checkbox(
        "Wildfire candidates",
        value=True,
        disabled=not classification_available,
    )

    show_industrial = st.checkbox(
        "Industrial fire candidates",
        value=True,
        disabled=not classification_available,
    )

    show_persistent = st.checkbox(
        "Persistent thermal sources",
        value=True,
        disabled=not classification_available,
    )

    st.caption(
        "AI outputs are candidate classifications from the ENTRO prototype model and are not field-validated confirmations."
    )

    st.markdown("### Data sources")

    st.checkbox(
        "NASA FIRMS NOAA-20",
        value=True,
        disabled=True,
    )

    st.checkbox(
        "NASA FIRMS NOAA-21",
        value=True,
        disabled=True,
    )

    st.checkbox(
        "OpenStreetMap",
        value=True,
        disabled=True,
    )

    st.checkbox(
        "Dynamic World",
        value=True,
        disabled=True,
    )

    st.checkbox(
        "Forest boundary",
        value=True,
        disabled=True,
    )

    st.divider()

    if st.button(
        "Refresh FIRMS data",
        use_container_width=True,
    ):

        st.cache_data.clear()
        st.rerun()

    st.caption(
        "FIRMS refresh interval: 15 minutes"
    )

    if firms_available:

        st.caption(
            f"Current live detections: {len(firms_df):,}"
        )

    else:

        st.caption(
            "Current live detections: unavailable"
        )


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns(
    [3.5, 1.5],
    vertical_alignment="center",
)

with header_left:

    st.markdown(
        f"""
<div class="header-box">

<div class="header-title">
Thermal Intelligence
</div>

<div class="header-subtitle">
Industrial Fire & Persistent Thermal Source Detection
</div>

</div>
""",
        unsafe_allow_html=True,
    )


with header_right:

    status_text = (
        "FIRMS Connected"
        if firms_available
        else "FIRMS Unavailable"
    )

    status_class = (
        "status-online"
        if firms_available
        else "status-offline"
    )

    st.markdown(
        f"""
<div class="header-box">

<div style="
    font-size:0.72rem;
    color:{COLORS["muted"]};
    text-transform:uppercase;
    letter-spacing:0.05em;
">
Monitoring region
</div>

<div style="
    color:{COLORS["primary"]};
    font-size:0.95rem;
    font-weight:700;
    margin-top:2px;
">
Telangana, India
</div>

<div class="{status_class}" style="
    font-size:0.75rem;
    margin-top:6px;
">
● {status_text}
</div>

</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# FIRMS ERROR INFORMATION
# ============================================================

if firms_errors:

    with st.expander(
        "FIRMS connection details",
        expanded=not firms_available,
    ):

        for satellite_name, error_message in firms_errors.items():

            st.warning(
                f"{satellite_name}: {error_message}"
            )


# ============================================================
# KPI SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Live situation overview</div>',
    unsafe_allow_html=True,
)

if firms_available:

    total_detections = len(firms_df)

    unique_days = (
        firms_df["acq_date"]
        .nunique()
        if "acq_date" in firms_df.columns
        else 0
    )

    max_frp = (
        float(firms_df["frp"].max())
        if "frp" in firms_df.columns
        and firms_df["frp"].notna().any()
        else 0
    )

    latest_time = firms_df["acq_datetime"].max()

else:

    total_detections = 0
    unique_days = 0
    max_frp = 0
    latest_time = None


k1, k2, k3, k4 = st.columns(4)

with k1:

    st.metric(
        "Live thermal detections",
        f"{total_detections:,}",
    )

with k2:

    st.metric(
        "Observation days",
        f"{unique_days}",
    )

with k3:

    st.metric(
        "Maximum FRP",
        f"{max_frp:.1f} MW",
    )

with k4:

    if latest_time is not None:
        latest_display = latest_time.strftime(
            "%d %b %H:%M"
        )
    else:
        latest_display = "Unavailable"

    st.metric(
        "Latest observation",
        latest_display,
    )


# ============================================================
# MAIN MAP + LIVE INTELLIGENCE
# ============================================================

st.markdown(
    '<div class="section-title">Live thermal monitoring</div>',
    unsafe_allow_html=True,
)

map_column, intelligence_column = st.columns(
    [3.4, 1.2],
    gap="medium",
)


# ============================================================
# MAP
# ============================================================

with map_column:

    if show_map:

        # ----------------------------------------------------
        # Create Folium map
        # ----------------------------------------------------

        m = folium.Map(
            location=[
                17.95,
                79.20,
            ],
            zoom_start=6.4,
            tiles="OpenStreetMap",
            control_scale=True,
            prefer_canvas=True,
        )

        # Telangana monitoring extent
        folium.Rectangle(
            bounds=[
                [SOUTH, WEST],
                [NORTH, EAST],
            ],
            color=COLORS["map"],
            weight=2,
            fill=False,
            opacity=0.7,
            tooltip="ENTRO-26162 monitoring extent",
        ).add_to(m)

        # ----------------------------------------------------
        # Detection layer
        # ----------------------------------------------------

        if (
            firms_available
            and show_detections
        ):

            marker_cluster = MarkerCluster(
                name="NASA FIRMS detections",
                options={
                    "maxClusterRadius": 35,
                    "disableClusteringAtZoom": 10,
                },
            )

            marker_cluster.add_to(m)

            map_df = firms_df.head(
                MAP_POINT_LIMIT
            )

            for _, row in map_df.iterrows():

                latitude = row.get(
                    "latitude"
                )

                longitude = row.get(
                    "longitude"
                )

                if pd.isna(latitude) or pd.isna(longitude):
                    continue

                # ------------------------------------------------
                # FRP
                # ------------------------------------------------

                try:
                    frp = float(
                        row.get(
                            "frp",
                            0,
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    frp = 0.0

                radius = max(
                    4,
                    min(
                        10,
                        4 + frp / 20,
                    ),
                )

                # ------------------------------------------------
                # Values
                # ------------------------------------------------

                satellite = row.get(
                    "satellite",
                    row.get(
                        "source_name",
                        "Unknown",
                    ),
                )

                date_value = row.get(
                    "acq_date",
                    "Unknown",
                )

                time_value = row.get(
                    "acq_time",
                    "Unknown",
                )

                confidence = row.get(
                    "confidence",
                    "Unknown",
                )

                bright_ti4 = row.get(
                    "bright_ti4",
                    None,
                )

                bright_ti5 = row.get(
                    "bright_ti5",
                    None,
                )

                daynight = row.get(
                    "daynight",
                    "Unknown",
                )

                popup_html = f"""
                <div style="
                    font-family:Arial,sans-serif;
                    width:250px;
                ">

                    <div style="
                        font-size:15px;
                        font-weight:700;
                        color:#1F4E5F;
                        margin-bottom:8px;
                    ">
                        Thermal Detection
                    </div>

                    <table style="
                        width:100%;
                        font-size:12px;
                    ">

                        <tr>
                            <td><b>Satellite</b></td>
                            <td>{satellite}</td>
                        </tr>

                        <tr>
                            <td><b>Date</b></td>
                            <td>{date_value}</td>
                        </tr>

                        <tr>
                            <td><b>Time</b></td>
                            <td>{time_value}</td>
                        </tr>

                        <tr>
                            <td><b>FRP</b></td>
                            <td>{frp:.2f} MW</td>
                        </tr>

                        <tr>
                            <td><b>Brightness</b></td>
                            <td>{bright_ti4}</td>
                        </tr>

                        <tr>
                            <td><b>Confidence</b></td>
                            <td>{confidence}</td>
                        </tr>

                        <tr>
                            <td><b>Day/Night</b></td>
                            <td>{daynight}</td>
                        </tr>

                        <tr>
                            <td><b>Latitude</b></td>
                            <td>{float(latitude):.5f}</td>
                        </tr>

                        <tr>
                            <td><b>Longitude</b></td>
                            <td>{float(longitude):.5f}</td>
                        </tr>

                    </table>

                </div>
                """

                folium.CircleMarker(
                    location=[
                        float(latitude),
                        float(longitude),
                    ],
                    radius=radius,
                    color="#FFFFFF",
                    weight=1,
                    fill=True,
                    fill_color=COLORS["thermal"],
                    fill_opacity=0.82,
                    popup=folium.Popup(
                        popup_html,
                        max_width=300,
                    ),
                    tooltip=(
                        f"Thermal detection | "
                        f"FRP {frp:.1f} MW"
                    ),
                ).add_to(marker_cluster)

        # ----------------------------------------------------
        # AI classification layer
        # ----------------------------------------------------

        if classification_available:

            selected_classes = []
            if show_wildfire:
                selected_classes.append("WILDFIRE")
            if show_industrial:
                selected_classes.append("INDUSTRIAL_FIRE")
            if show_persistent:
                selected_classes.append("PERSISTENT_THERMAL_SOURCE")

            ai_df = classified_df[
                classified_df["prediction"].isin(selected_classes)
            ].head(MAP_POINT_LIMIT).copy()

            for _, row in ai_df.iterrows():

                latitude = row.get("latitude")
                longitude = row.get("longitude")
                prediction = row.get("prediction")

                if pd.isna(latitude) or pd.isna(longitude):
                    continue

                confidence_value = row.get("model_confidence", 0)
                try:
                    confidence_value = float(confidence_value)
                except (TypeError, ValueError):
                    confidence_value = 0.0

                label = CLASSIFICATION_LABELS.get(
                    str(prediction), str(prediction)
                )
                marker_color = CLASSIFICATION_COLORS.get(
                    str(prediction), COLORS["map"]
                )

                frp_value = row.get("frp", 0)
                try:
                    frp_value = float(frp_value)
                except (TypeError, ValueError):
                    frp_value = 0.0

                popup_html = f"""
                <div style="font-family:Arial,sans-serif;width:255px;">
                    <div style="font-size:15px;font-weight:700;color:#1F4E5F;margin-bottom:8px;">
                        AI Classification Candidate
                    </div>
                    <div style="padding:7px 9px;background:{marker_color};color:white;border-radius:6px;font-weight:700;margin-bottom:8px;">
                        {label}
                    </div>
                    <table style="width:100%;font-size:12px;">
                        <tr><td><b>Model confidence</b></td><td>{confidence_value * 100:.1f}%</td></tr>
                        <tr><td><b>FRP</b></td><td>{frp_value:.2f} MW</td></tr>
                        <tr><td><b>Satellite</b></td><td>{row.get('satellite', row.get('source_name', 'Unknown'))}</td></tr>
                        <tr><td><b>Date</b></td><td>{row.get('acq_date', 'Unknown')}</td></tr>
                        <tr><td><b>Time</b></td><td>{row.get('acq_time', 'Unknown')}</td></tr>
                        <tr><td><b>Location</b></td><td>{float(latitude):.5f}, {float(longitude):.5f}</td></tr>
                    </table>
                    <div style="margin-top:8px;font-size:10px;color:#71808D;">
                        Prototype candidate output; not an independently verified fire event.
                    </div>
                </div>
                """

                folium.CircleMarker(
                    location=[float(latitude), float(longitude)],
                    radius=7,
                    color="#FFFFFF",
                    weight=2,
                    fill=True,
                    fill_color=marker_color,
                    fill_opacity=0.95,
                    popup=folium.Popup(popup_html, max_width=320),
                    tooltip=f"{label} | {confidence_value * 100:.1f}%",
                ).add_to(m)

        # ----------------------------------------------------
        # Map legend
        # ----------------------------------------------------

        legend_html = f"""
        <div style="
            position:fixed;
            bottom:20px;
            left:20px;
            z-index:9999;
            background:white;
            border:1px solid #EEE5D8;
            border-radius:8px;
            padding:8px 12px;
            font-size:11px;
            color:#263746;
            box-shadow:0 1px 5px rgba(0,0,0,0.12);
        ">
            <div style="margin-bottom:5px;font-weight:700;">Map layers</div>
            <div><span style="display:inline-block;width:9px;height:9px;background:{COLORS["thermal"]};border-radius:50%;margin-right:5px;"></span>NASA FIRMS detection</div>
            <div><span style="display:inline-block;width:9px;height:9px;background:{COLORS["thermal"]};border-radius:50%;margin-right:5px;"></span>Wildfire candidate</div>
            <div><span style="display:inline-block;width:9px;height:9px;background:{COLORS["industrial"]};border-radius:50%;margin-right:5px;"></span>Industrial fire candidate</div>
            <div><span style="display:inline-block;width:9px;height:9px;background:{COLORS["persistent"]};border-radius:50%;margin-right:5px;"></span>Persistent thermal source</div>
        </div>
        """

        m.get_root().html.add_child(
            folium.Element(
                legend_html
            )
        )

        # ----------------------------------------------------
        # Render map
        # ----------------------------------------------------

        st_folium(
            m,
            width=None,
            height=590,
            returned_objects=[],
            key="entro_live_telangan_map",
        )

        if firms_available and len(firms_df) > MAP_POINT_LIMIT:

            st.caption(
                f"Showing the latest {MAP_POINT_LIMIT:,} detections "
                f"on the map for browser performance. "
                f"Total retrieved: {len(firms_df):,}."
            )

    else:

        st.info(
            "Enable 'Thermal activity map' from the sidebar."
        )


# ============================================================
# LIVE INTELLIGENCE PANEL
# ============================================================

with intelligence_column:

    st.markdown(
        f"""
<div class="info-box">

<div class="small-label">
Live intelligence
</div>

<div style="
    color:{COLORS["primary"]};
    font-size:1.15rem;
    font-weight:750;
    margin-top:5px;
">
FIRMS Monitoring
</div>

<div class="small-text" style="
    margin-top:5px;
">
NASA VIIRS near-real-time observations
for the Telangana monitoring extent.
</div>

</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    # --------------------------------------------------------
    # Current status
    # --------------------------------------------------------

    status_col = (
        "Connected"
        if firms_available
        else "Unavailable"
    )

    st.metric(
        "NASA FIRMS",
        status_col,
    )

    st.metric(
        "Detections",
        f"{total_detections:,}",
    )

    # --------------------------------------------------------
    # Latest event
    # --------------------------------------------------------

    st.markdown(
        "#### Latest event"
    )

    if firms_available:

        latest = firms_df.iloc[0]

        latest_lat = latest.get(
            "latitude",
            None,
        )

        latest_lon = latest.get(
            "longitude",
            None,
        )

        latest_frp = latest.get(
            "frp",
            None,
        )

        latest_satellite = latest.get(
            "satellite",
            latest.get(
                "source_name",
                "Unknown",
            ),
        )

        st.write(
            f"**Satellite:** {latest_satellite}"
        )

        st.write(
            f"**Date:** {latest.get('acq_date', 'Unknown')}"
        )

        st.write(
            f"**Time:** {latest.get('acq_time', 'Unknown')}"
        )

        if pd.notna(latest_frp):

            st.write(
                f"**FRP:** {float(latest_frp):.2f} MW"
            )

        if (
            pd.notna(latest_lat)
            and pd.notna(latest_lon)
        ):

            st.write(
                f"**Location:** "
                f"{float(latest_lat):.4f}, "
                f"{float(latest_lon):.4f}"
            )

    else:

        st.info(
            "Live FIRMS data is currently unavailable."
        )

    # --------------------------------------------------------
    # Model status
    # --------------------------------------------------------

    st.markdown(
        "#### AI classification"
    )

    if classification_available:

        prediction_counts = classified_df["prediction"].value_counts()

        st.metric(
            "Classified live detections",
            f"{len(classified_df):,}",
        )

        st.metric(
            "Wildfire candidates",
            f"{int(prediction_counts.get('WILDFIRE', 0)):,}",
        )

        st.metric(
            "Industrial fire candidates",
            f"{int(prediction_counts.get('INDUSTRIAL_FIRE', 0)):,}",
        )

        st.metric(
            "Persistent thermal sources",
            f"{int(prediction_counts.get('PERSISTENT_THERMAL_SOURCE', 0)):,}",
        )

        avg_confidence = pd.to_numeric(
            classified_df["model_confidence"], errors="coerce"
        ).mean()

        st.write(
            f"**Average model confidence:** {avg_confidence * 100:.1f}%"
        )

        unmatched = int(classification_status.get("unmatched", 0))
        if unmatched:
            st.caption(
                f"Unmatched live detections: {unmatched:,}"
            )

    else:

        st.warning(
            classification_status.get(
                "error",
                "AI classification is unavailable.",
            )
        )


# ============================================================
# THERMAL ACTIVITY TREND
# ============================================================

st.markdown(
    '<div class="section-title">Thermal activity trend</div>',
    unsafe_allow_html=True,
)

if firms_available:

    trend_df = (
        firms_df
        .dropna(
            subset=["acq_date"]
        )
        .groupby(
            "acq_date"
        )
        .size()
        .reset_index(
            name="detections"
        )
    )

    trend_df["acq_date"] = pd.to_datetime(
        trend_df["acq_date"],
        errors="coerce",
    )

    trend_df = trend_df.dropna(
        subset=["acq_date"]
    )

    trend_df = trend_df.sort_values(
        "acq_date"
    )

    if not trend_df.empty:

        st.line_chart(
            trend_df.set_index(
                "acq_date"
            )["detections"],
            height=260,
        )

    else:

        st.info(
            "No trend data available."
        )

else:

    st.info(
        "Thermal activity trend will appear when NASA FIRMS data is available."
    )


# ============================================================
# AI CLASSIFICATION RESULTS
# ============================================================

st.markdown(
    '<div class="section-title">AI classification results</div>',
    unsafe_allow_html=True,
)

if classification_available:

    classification_table = classified_df.copy()
    classification_table["Classification"] = classification_table["prediction"].map(
        CLASSIFICATION_LABELS
    ).fillna(classification_table["prediction"])
    classification_table["Confidence"] = (
        pd.to_numeric(classification_table["model_confidence"], errors="coerce") * 100
    ).round(1)

    classification_table = classification_table.sort_values(
        "model_confidence", ascending=False
    )

    display_class_columns = [
        "Classification",
        "Confidence",
        "frp",
        "latitude",
        "longitude",
        "source_name",
        "acq_date",
        "acq_time",
    ]

    classification_table = classification_table[
        [c for c in display_class_columns if c in classification_table.columns]
    ].head(15).copy()

    classification_table = classification_table.rename(
        columns={
            "Confidence": "Confidence (%)",
            "frp": "FRP (MW)",
            "latitude": "Latitude",
            "longitude": "Longitude",
            "source_name": "Satellite",
            "acq_date": "Date",
            "acq_time": "Time",
        }
    )

    if "FRP (MW)" in classification_table:
        classification_table["FRP (MW)"] = pd.to_numeric(
            classification_table["FRP (MW)"], errors="coerce"
        ).round(2)
    if "Latitude" in classification_table:
        classification_table["Latitude"] = classification_table["Latitude"].round(4)
    if "Longitude" in classification_table:
        classification_table["Longitude"] = classification_table["Longitude"].round(4)

    st.dataframe(
        classification_table,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Prototype note: these are candidate classifications generated from the trained ENTRO model using live FIRMS detections and previously engineered geospatial context. They are not independently field-validated confirmations."
    )

else:

    st.info(
        "AI classification results are unavailable until the trained model and feature table are available."
    )


# ============================================================
# LATEST DETECTIONS
# ============================================================

st.markdown(
    '<div class="section-title">Latest detections</div>',
    unsafe_allow_html=True,
)

if firms_available:

    display_columns = [
        "source_name",
        "acq_date",
        "acq_time",
        "latitude",
        "longitude",
        "frp",
        "bright_ti4",
        "confidence",
        "daynight",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in firms_df.columns
    ]

    latest_table = (
        firms_df[
            available_columns
        ]
        .head(15)
        .copy()
    )

    rename_map = {
        "source_name": "Satellite",
        "acq_date": "Date",
        "acq_time": "Time",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "frp": "FRP (MW)",
        "bright_ti4": "Brightness",
        "confidence": "Confidence",
        "daynight": "Day/Night",
    }

    latest_table = latest_table.rename(
        columns=rename_map
    )

    if "Latitude" in latest_table.columns:

        latest_table["Latitude"] = (
            latest_table["Latitude"]
            .round(4)
        )

    if "Longitude" in latest_table.columns:

        latest_table["Longitude"] = (
            latest_table["Longitude"]
            .round(4)
        )

    if "FRP (MW)" in latest_table.columns:

        latest_table["FRP (MW)"] = (
            pd.to_numeric(
                latest_table["FRP (MW)"],
                errors="coerce",
            )
            .round(2)
        )

    st.dataframe(
        latest_table,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No live FIRMS detections are currently available."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
<div class="footer">

ENTRO-26162 · AI-Based Detection and Classification of
Industrial Fires and Persistent Thermal Sources

<br>

NASA FIRMS · OpenStreetMap · Dynamic World · Telangana
geospatial context

</div>
""",
    unsafe_allow_html=True,
)