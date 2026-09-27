import json
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


AGRICULTURE_FILE = "data/processed/agriculture_features_raw.csv"
MASTER_FILE = "data/training/master_feature_table_final.csv"
DISTRICT_FILE = "data/raw/district_boundaries_telangana.json"


def esri_geometry_to_shape(geometry):
    """Convert ArcGIS Esri polygon rings to a Shapely geometry."""

    rings = geometry.get("rings", [])

    if not rings:
        return None

    polygons = []

    for ring in rings:
        if len(ring) >= 4:
            polygons.append(Polygon(ring))

    if not polygons:
        return None

    return unary_union(polygons)


def main():

    print("Loading agriculture data...")
    agriculture = pd.read_csv(
        AGRICULTURE_FILE,
        usecols=["lat_grid", "lon_grid"]
    )

    print("Loading master feature table...")
    master = pd.read_csv(
        MASTER_FILE,
        usecols=["lat_grid", "lon_grid"]
    )

    # Keep only agriculture locations that exist in the master table
    matched = agriculture.merge(
        master,
        on=["lat_grid", "lon_grid"],
        how="inner"
    ).drop_duplicates(
        subset=["lat_grid", "lon_grid"]
    )

    print(f"Matched locations: {len(matched):,}")

    print("Loading district boundaries...")
    with open(DISTRICT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    districts = []

    for feature in data["features"]:

        name = feature["attributes"]["dist_name"]
        geometry = feature.get("geometry")

        if not geometry:
            continue

        district_shape = esri_geometry_to_shape(geometry)

        if district_shape is not None:
            districts.append(
                (name, district_shape)
            )

    print(f"District polygons loaded: {len(districts)}")

    # Assign district
    district_names = []

    for lat, lon in zip(
        matched["lat_grid"],
        matched["lon_grid"]
    ):

        point = Point(lon, lat)

        assigned = None

        for name, district_shape in districts:

            if (
                district_shape.contains(point)
                or district_shape.touches(point)
            ):
                assigned = name
                break

        district_names.append(assigned)

    matched["district_name"] = district_names

    assigned_count = matched["district_name"].notna().sum()
    unassigned_count = matched["district_name"].isna().sum()

    print()
    print("--- DISTRICT ASSIGNMENT ---")
    print(f"Matched locations:  {len(matched):,}")
    print(f"Assigned locations: {assigned_count:,}")
    print(f"Unassigned:         {unassigned_count:,}")

    print()
    print("--- DISTRICT COUNTS ---")
    print(
        matched["district_name"]
        .value_counts()
        .to_string()
    )

    # Save result for the next step
    output_file = (
        "data/processed/"
        "agriculture_master_district_mapping.csv"
    )

    matched.to_csv(
        output_file,
        index=False
    )

    print()
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()