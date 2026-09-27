"""
PS 26162
UNKNOWN FIRMS BY DISTRICT

Purpose:
    Assign UNKNOWN FIRMS grid locations to Telangana districts.

Inputs:
    data/training/master_feature_table_final.csv
    data/training/seed_labels.csv

Official boundary source:
    TGRAC Master Administrative Boundary
    District Boundary layer

Outputs:
    data/raw/telangana_district_boundaries.geojson
    data/training/unknown_firms_by_district.csv
    data/training/unknown_firms_district_summary.csv
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import json
import sys

import pandas as pd
import geopandas as gpd
import requests

from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\ENTRO-26162"
)

MASTER_FILE = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "master_feature_table_final.csv"
)

SEED_FILE = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "seed_labels.csv"
)

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

TRAINING_DIR = (
    PROJECT_ROOT
    / "data"
    / "training"
)

DISTRICT_BOUNDARY_FILE = (
    RAW_DIR
    / "telangana_district_boundaries.geojson"
)

OUTPUT_LOCATION_FILE = (
    TRAINING_DIR
    / "unknown_firms_by_district.csv"
)

OUTPUT_SUMMARY_FILE = (
    TRAINING_DIR
    / "unknown_firms_district_summary.csv"
)


# ============================================================
# TGRAC DISTRICT BOUNDARY SERVICE
# ============================================================

DISTRICT_LAYER_URL = (
    "https://tgrac.telangana.gov.in/arcgis/rest/services/"
    "Master_Administrative_Folder/"
    "Master_Administrative_Boundary/"
    "MapServer/2"
)


# ============================================================
# BASIC CHECKS
# ============================================================

def check_project_files():

    print("\n[1/7] Checking project files...")

    required_files = [
        MASTER_FILE,
        SEED_FILE,
    ]

    for file in required_files:

        print(f"Checking: {file}")

        if not file.exists():
            raise FileNotFoundError(
                f"\nRequired file not found:\n{file}"
            )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    TRAINING_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Project files OK.")


# ============================================================
# LOAD MASTER
# ============================================================

def load_master():

    print("\n[2/7] Loading master feature table...")

    master = pd.read_csv(
        MASTER_FILE,
        low_memory=False
    )

    print("Master rows:", len(master))
    print("Master columns:", len(master.columns))

    required_columns = {
        "lat_grid",
        "lon_grid",
    }

    missing = required_columns - set(master.columns)

    if missing:

        raise ValueError(
            "Master file is missing required columns: "
            + ", ".join(sorted(missing))
        )

    # Ensure coordinates are numeric
    master["lat_grid"] = pd.to_numeric(
        master["lat_grid"],
        errors="coerce"
    )

    master["lon_grid"] = pd.to_numeric(
        master["lon_grid"],
        errors="coerce"
    )

    invalid_coordinates = (
        master["lat_grid"].isna()
        | master["lon_grid"].isna()
        | ~master["lat_grid"].between(-90, 90)
        | ~master["lon_grid"].between(-180, 180)
    )

    invalid_count = invalid_coordinates.sum()

    print(
        "Invalid coordinate rows:",
        invalid_count
    )

    if invalid_count > 0:

        raise ValueError(
            "Master file contains invalid FIRMS grid coordinates."
        )

    return master


# ============================================================
# LOAD SEED LABELS
# ============================================================

def load_seed_labels():

    print("\n[3/7] Loading seed labels...")

    seed = pd.read_csv(
        SEED_FILE,
        low_memory=False
    )

    print("Seed rows:", len(seed))

    required_columns = {
        "lat_grid",
        "lon_grid",
        "seed_label",
    }

    missing = required_columns - set(seed.columns)

    if missing:

        raise ValueError(
            "Seed file is missing required columns: "
            + ", ".join(sorted(missing))
        )

    seed["lat_grid"] = pd.to_numeric(
        seed["lat_grid"],
        errors="coerce"
    )

    seed["lon_grid"] = pd.to_numeric(
        seed["lon_grid"],
        errors="coerce"
    )

    duplicate_mask = seed.duplicated(
        subset=["lat_grid", "lon_grid"],
        keep=False
    )

    duplicate_count = duplicate_mask.sum()

    print(
        "Duplicate seed grid rows:",
        duplicate_count
    )

    if duplicate_count > 0:

        duplicate_cells = (
            seed.loc[
                duplicate_mask,
                ["lat_grid", "lon_grid"]
            ]
            .drop_duplicates()
        )

        print(
            duplicate_cells.head(20).to_string(
                index=False
            )
        )

        raise ValueError(
            "Seed labels contain duplicate grid cells. "
            "Fix this before performing the district join."
        )

    return seed


# ============================================================
# MERGE MASTER + SEED
# ============================================================

def create_unknown_dataframe(
    master,
    seed
):

    print("\n[4/7] Merging seed labels with master...")

    merged = master.merge(
        seed[
            [
                "lat_grid",
                "lon_grid",
                "seed_label",
            ]
        ],
        on=[
            "lat_grid",
            "lon_grid",
        ],
        how="left",
        validate="one_to_one",
    )

    print(
        "Rows after seed merge:",
        len(merged)
    )

    missing_labels = merged["seed_label"].isna().sum()

    print(
        "Rows without seed label:",
        missing_labels
    )

    if missing_labels > 0:

        raise ValueError(
            "Some master grid cells do not have seed labels."
        )

    unknown = merged[
        merged["seed_label"].astype(str).str.upper()
        == "UNKNOWN"
    ].copy()

    print(
        "UNKNOWN locations:",
        len(unknown)
    )

    if len(unknown) == 0:

        raise ValueError(
            "No UNKNOWN FIRMS locations were found."
        )

    return unknown


# ============================================================
# CREATE FIRMS POINTS
# ============================================================

def create_unknown_points(
    unknown
):

    print("\n[5/7] Creating FIRMS grid points...")

    geometry = [
        Point(
            float(lon),
            float(lat)
        )
        for lon, lat in zip(
            unknown["lon_grid"],
            unknown["lat_grid"]
        )
    ]

    unknown_points = gpd.GeoDataFrame(
        unknown.copy(),
        geometry=geometry,
        crs="EPSG:4326",
    )

    print(
        "Point CRS:",
        unknown_points.crs
    )

    print(
        "UNKNOWN points:",
        len(unknown_points)
    )

    return unknown_points


# ============================================================
# CONVERT ESRI JSON RINGS TO SHAPELY GEOMETRY
# ============================================================

def esri_geometry_to_shapely(
    geometry
):

    if not geometry:
        return None

    rings = geometry.get(
        "rings",
        []
    )

    if not rings:
        return None

    polygons = []

    for ring in rings:

        if not ring:
            continue

        if len(ring) < 4:
            continue

        try:

            polygon = Polygon(
                ring
            )

            if not polygon.is_empty:

                polygons.append(
                    polygon
                )

        except Exception:

            continue

    if not polygons:

        return None

    if len(polygons) == 1:

        result = polygons[0]

    else:

        result = MultiPolygon(
            polygons
        )

    # Repair small geometry problems
    if not result.is_valid:

        result = result.buffer(0)

    if result.is_empty:

        return None

    return result


# ============================================================
# DOWNLOAD DISTRICT BOUNDARIES
# ============================================================

def download_district_boundaries():

    print("\n[6/7] Downloading TGRAC district boundaries...")

    print(
        "Requesting:"
    )

    print(
        DISTRICT_LAYER_URL
    )

    query_url = (
        DISTRICT_LAYER_URL
        + "/query"
    )

    params = {
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "json",
    }

    try:

        response = requests.get(
            query_url,
            params=params,
            timeout=120,
        )

    except requests.RequestException as exc:

        raise RuntimeError(
            f"TGRAC request failed:\n{exc}"
        )

    print(
        "HTTP status:",
        response.status_code
    )

    if response.status_code != 200:

        print(
            "Response:"
        )

        print(
            response.text[:2000]
        )

        raise RuntimeError(
            "TGRAC returned an HTTP error."
        )

    try:

        data = response.json()

    except ValueError:

        print(
            response.text[:2000]
        )

        raise RuntimeError(
            "TGRAC did not return valid JSON."
        )

    if "error" in data:

        print(
            "TGRAC server error:"
        )

        print(
            json.dumps(
                data["error"],
                indent=2
            )
        )

        raise RuntimeError(
            "TGRAC district query failed."
        )

    features = data.get(
        "features",
        []
    )

    print(
        "Features received:",
        len(features)
    )

    if len(features) == 0:

        raise RuntimeError(
            "TGRAC returned zero district features."
        )

    records = []

    for feature in features:

        attributes = feature.get(
            "attributes",
            {}
        )

        geometry = feature.get(
            "geometry"
        )

        shapely_geometry = (
            esri_geometry_to_shapely(
                geometry
            )
        )

        if shapely_geometry is None:

            continue

        record = dict(
            attributes
        )

        record["geometry"] = (
            shapely_geometry
        )

        records.append(
            record
        )

    if not records:

        raise RuntimeError(
            "No usable district geometries were created."
        )

    districts = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs="EPSG:4326",
    )

    print(
        "Usable district polygons:",
        len(districts)
    )

    # --------------------------------------------------------
    # Identify district name field
    # --------------------------------------------------------

    possible_name_fields = [
        "name",
        "NAME",
        "district",
        "District",
        "DISTRICT",
        "district_name",
        "District_Name",
        "DIST_NAME",
        "DIST_NAME_1",
        "DistrictName",
    ]

    district_name_field = None

    for field in possible_name_fields:

        if field in districts.columns:

            district_name_field = field
            break

    if district_name_field is None:

        print(
            "\nAvailable district fields:"
        )

        print(
            list(districts.columns)
        )

        raise RuntimeError(
            "Could not identify the district name field."
        )

    print(
        "District name field:",
        district_name_field
    )

    district_names = (
        districts[
            district_name_field
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
    )

    print(
        "Unique district names:",
        len(district_names)
    )

    print(
        "\nDistrict names:"
    )

    for name in sorted(
        district_names
    ):

        print(
            " -",
            name
        )

    # Telangana currently has 33 districts.
    # We don't hard-fail if the service returns a different
    # number because boundary services can change.

    if len(district_names) < 30:

        print(
            "\nWARNING:"
        )

        print(
            "The boundary service returned fewer than "
            "30 unique district names."
        )

        print(
            "Please inspect the names before using the result."
        )

    # --------------------------------------------------------
    # Save local GeoJSON copy
    # --------------------------------------------------------

    try:

        districts.to_file(
            DISTRICT_BOUNDARY_FILE,
            driver="GeoJSON",
        )

        print(
            "\nSaved district boundaries:"
        )

        print(
            DISTRICT_BOUNDARY_FILE
        )

    except Exception as exc:

        print(
            "\nWARNING: Could not save GeoJSON copy:"
        )

        print(exc)

    return (
        districts,
        district_name_field,
    )


# ============================================================
# SPATIAL JOIN
# ============================================================

def assign_districts(
    unknown_points,
    districts,
    district_name_field,
):

    print(
        "\nAssigning UNKNOWN FIRMS to districts..."
    )

    district_subset = districts[
        [
            district_name_field,
            "geometry",
        ]
    ].copy()

    district_subset = (
        district_subset
        .rename(
            columns={
                district_name_field:
                "district_name"
            }
        )
    )

    # --------------------------------------------------------
    # Spatial join
    # --------------------------------------------------------

    joined = gpd.sjoin(
        unknown_points,
        district_subset,
        how="left",
        predicate="within",
    )

    # --------------------------------------------------------
    # Check duplicate spatial matches
    # --------------------------------------------------------

    duplicate_matches = (
        joined.duplicated(
            subset=[
                "lat_grid",
                "lon_grid",
            ],
            keep=False
        )
    )

    duplicate_match_count = (
        duplicate_matches.sum()
    )

    if duplicate_match_count > 0:

        print(
            "\nWARNING:"
        )

        print(
            "Some FIRMS grid cells matched multiple "
            "district polygons."
        )

        print(
            "Duplicate matched rows:",
            duplicate_match_count
        )

        # Keep first match only.
        joined = (
            joined
            .drop_duplicates(
                subset=[
                    "lat_grid",
                    "lon_grid",
                ],
                keep="first"
            )
        )

    # --------------------------------------------------------
    # Match statistics
    # --------------------------------------------------------

    matched_count = (
        joined["district_name"]
        .notna()
        .sum()
    )

    unmatched_count = (
        joined["district_name"]
        .isna()
        .sum()
    )

    print(
        "\nDistrict matching:"
    )

    print(
        "UNKNOWN locations:",
        len(unknown_points)
    )

    print(
        "Matched:",
        matched_count
    )

    print(
        "Unmatched:",
        unmatched_count
    )

    match_percentage = (
        matched_count
        / len(unknown_points)
        * 100
    )

    print(
        f"Match percentage: "
        f"{match_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # Save location-level result
    # --------------------------------------------------------

    output_columns = [
        "lat_grid",
        "lon_grid",
        "district_name",
    ]

    location_output = (
        joined[
            output_columns
        ]
        .copy()
    )

    location_output.to_csv(
        OUTPUT_LOCATION_FILE,
        index=False,
    )

    print(
        "\nSaved location-level file:"
    )

    print(
        OUTPUT_LOCATION_FILE
    )

    # --------------------------------------------------------
    # District summary
    # --------------------------------------------------------

    district_summary = (
        location_output
        .dropna(
            subset=[
                "district_name"
            ]
        )
        .groupby(
            "district_name"
        )
        .size()
        .reset_index(
            name="unknown_locations"
        )
    )

    total_unknown = (
        len(unknown_points)
    )

    district_summary[
        "unknown_percentage"
    ] = (
        district_summary[
            "unknown_locations"
        ]
        / total_unknown
        * 100
    )

    district_summary = (
        district_summary
        .sort_values(
            "unknown_locations",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    district_summary.to_csv(
        OUTPUT_SUMMARY_FILE,
        index=False,
    )

    print(
        "\nSaved district summary:"
    )

    print(
        OUTPUT_SUMMARY_FILE
    )

    return (
        location_output,
        district_summary,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PS 26162 — UNKNOWN FIRMS BY DISTRICT"
    )

    print(
        "=" * 70
    )

    try:

        # ----------------------------------------------------
        # 1. Check files
        # ----------------------------------------------------

        check_project_files()

        # ----------------------------------------------------
        # 2. Load master
        # ----------------------------------------------------

        master = load_master()

        # ----------------------------------------------------
        # 3. Load seed labels
        # ----------------------------------------------------

        seed = load_seed_labels()

        # ----------------------------------------------------
        # 4. Create UNKNOWN dataset
        # ----------------------------------------------------

        unknown = create_unknown_dataframe(
            master,
            seed,
        )

        # ----------------------------------------------------
        # 5. Create FIRMS points
        # ----------------------------------------------------

        unknown_points = create_unknown_points(
            unknown
        )

        # ----------------------------------------------------
        # 6. Download districts
        # ----------------------------------------------------

        (
            districts,
            district_name_field,
        ) = download_district_boundaries()

        # ----------------------------------------------------
        # 7. Spatial join + summary
        # ----------------------------------------------------

        (
            location_output,
            district_summary,
        ) = assign_districts(
            unknown_points,
            districts,
            district_name_field,
        )

        # ----------------------------------------------------
        # Final output
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "UNKNOWN FIRMS BY DISTRICT — COMPLETE"
        )

        print(
            "=" * 70
        )

        print(
            "\nTop districts by UNKNOWN FIRMS:"
        )

        print(
            district_summary
            .head(15)
            .to_string(
                index=False
            )
        )

        print(
            "\nOutput files:"
        )

        print(
            "1.",
            OUTPUT_LOCATION_FILE
        )

        print(
            "2.",
            OUTPUT_SUMMARY_FILE
        )

        print(
            "\nDone."
        )

    except Exception as exc:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "SCRIPT FAILED"
        )

        print(
            "=" * 70
        )

        print(
            "\nError:"
        )

        print(
            str(exc)
        )

        print(
            "\nThe script stopped safely."
        )

        sys.exit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()