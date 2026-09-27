# src/diagnose_agriculture_district_coverage.py

import json
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


AGRI_FILE = "data/processed/agriculture_features_raw.csv"
MASTER_FILE = "data/training/master_feature_table_final.csv"
DISTRICT_FILE = "data/raw/district_boundaries_telangana.json"


def esri_to_shape(geometry):
    rings = geometry.get("rings", [])

    polygons = [
        Polygon(ring)
        for ring in rings
        if len(ring) >= 4
    ]

    if not polygons:
        return None

    return unary_union(polygons)


def main():

    agriculture = pd.read_csv(AGRI_FILE)
    master = pd.read_csv(
        MASTER_FILE,
        usecols=["lat_grid", "lon_grid"]
    )

    with open(DISTRICT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_districts = {
        "Nizamabad",
        "Kamareddy",
        "Mahabubabad",
        "Khammam",
    }

    districts = {}

    for feature in data["features"]:

        name = feature["attributes"]["dist_name"]

        if name in target_districts:
            districts[name] = esri_to_shape(
                feature["geometry"]
            )

    print("Target districts found:")
    print(list(districts.keys()))

    print("\n--- AGRICULTURE COVERAGE ---")

    for district_name, district_shape in districts.items():

        agriculture_inside = []

        for lat, lon in zip(
            agriculture["lat_grid"],
            agriculture["lon_grid"]
        ):

            point = Point(lon, lat)

            if (
                district_shape.contains(point)
                or district_shape.touches(point)
            ):
                agriculture_inside.append(True)
            else:
                agriculture_inside.append(False)

        agriculture_district = agriculture[
            agriculture_inside
        ]

        master_keys = set(
            zip(
                master["lat_grid"],
                master["lon_grid"]
            )
        )

        matched = agriculture_district[
            agriculture_district.apply(
                lambda r:
                (r["lat_grid"], r["lon_grid"])
                in master_keys,
                axis=1
            )
        ]

        valid_agriculture = (
            agriculture_district["crop_probability"]
            .notna()
            .sum()
        )

        print(f"\n{district_name}")
        print(
            f"  Agriculture grid cells: "
            f"{len(agriculture_district):,}"
        )
        print(
            f"  Also present in master: "
            f"{len(matched):,}"
        )
        print(
            f"  Valid agriculture features: "
            f"{valid_agriculture:,}"
        )
        print(
            f"  Missing agriculture features: "
            f"{len(agriculture_district) - valid_agriculture:,}"
        )


if __name__ == "__main__":
    main()