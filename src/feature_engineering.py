import os
import pandas as pd
import numpy as np


# ============================================================
# INPUT FILES
# ============================================================

FIRMS_FILE = (
    "data/raw/firms/"
    "firms_telangana_5days.csv"
)

OSM_FILE = (
    "data/raw/osm/"
    "telangana_industrial_facilities.csv"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = "data/processed"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "firms_osm_matched.csv"
)


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Calculate distance between latitude/longitude
    points in kilometers.
    """

    earth_radius = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(
        np.sqrt(a)
    )

    return earth_radius * c


# ============================================================
# OSM CATEGORY CLASSIFICATION
# ============================================================

def classify_osm_facility(row):

    industrial = str(
        row.get("industrial", "")
    ).lower()

    landuse = str(
        row.get("landuse", "")
    ).lower()

    power = str(
        row.get("power", "")
    ).lower()

    man_made = str(
        row.get("man_made", "")
    ).lower()

    name = str(
        row.get("name", "")
    ).lower()

    # --------------------------------------------------------
    # 1. POWER PLANT
    # --------------------------------------------------------

    if power == "plant":

        return "POWER_PLANT"

    if "power plant" in name:

        return "POWER_PLANT"

    # --------------------------------------------------------
    # 2. MINE / QUARRY
    # --------------------------------------------------------

    if landuse in [
        "mine",
        "quarry"
    ]:

        return "MINE_QUARRY"

    if any(
        word in name
        for word in [
            "mine",
            "mining",
            "quarry",
            "colliery"
        ]
    ):

        return "MINE_QUARRY"

    # --------------------------------------------------------
    # 3. OIL / GAS
    # --------------------------------------------------------

    if industrial in [
        "oil",
        "gas",
        "petroleum"
    ]:

        return "OIL_GAS"

    if any(
        word in name
        for word in [
            "oil",
            "petroleum",
            "refinery",
            "gas plant",
            "gas terminal",
            "lng",
            "lpg"
        ]
    ):

        return "OIL_GAS"

    # --------------------------------------------------------
    # 4. INDUSTRIAL AREA
    # --------------------------------------------------------

    if landuse == "industrial":

        return "INDUSTRIAL_AREA"

    # --------------------------------------------------------
    # 5. INDUSTRIAL WORKS
    # --------------------------------------------------------

    if man_made == "works":

        return "INDUSTRIAL_WORKS"

    if industrial not in [
        "",
        "none",
        "nan"
    ]:

        return "INDUSTRIAL_WORKS"

    # --------------------------------------------------------
    # 6. STORAGE TANK
    # --------------------------------------------------------

    if man_made == "storage_tank":

        return "STORAGE_TANK"

    # --------------------------------------------------------
    # 7. OTHER
    # --------------------------------------------------------

    return "OTHER"


# ============================================================
# FIND NEAREST FACILITY OF A CATEGORY
# ============================================================

def find_nearest(
    fire_lat,
    fire_lon,
    osm_df
):

    if osm_df.empty:

        return np.nan, None

    distances = haversine_distance(
        fire_lat,
        fire_lon,
        osm_df["latitude"].values,
        osm_df["longitude"].values
    )

    index = np.argmin(distances)

    return (
        distances[index],
        osm_df.iloc[index]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("================================")
    print("FIRMS + CATEGORIZED OSM MATCHING")
    print("================================")

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\nLoading FIRMS...")

    firms = pd.read_csv(
        FIRMS_FILE
    )

    print(
        f"FIRMS detections: {len(firms)}"
    )

    print("\nLoading OSM...")

    osm = pd.read_csv(
        OSM_FILE
    )

    print(
        f"OSM objects: {len(osm)}"
    )

    # ========================================================
    # CLEAN COORDINATES
    # ========================================================

    firms = firms.dropna(
        subset=[
            "latitude",
            "longitude"
        ]
    ).copy()

    osm = osm.dropna(
        subset=[
            "latitude",
            "longitude"
        ]
    ).copy()

    # ========================================================
    # CLASSIFY OSM OBJECTS
    # ========================================================

    print(
        "\nClassifying OSM facilities..."
    )

    osm["facility_category"] = (
        osm.apply(
            classify_osm_facility,
            axis=1
        )
    )

    print("\nOSM category distribution:")

    print(
        osm["facility_category"]
        .value_counts()
    )

    # ========================================================
    # CREATE CATEGORY DATASETS
    # ========================================================

    categories = {

        "industry": [
            "INDUSTRIAL_AREA",
            "INDUSTRIAL_WORKS",
            "POWER_PLANT",
            "MINE_QUARRY",
            "OIL_GAS"
        ],

        "power": [
            "POWER_PLANT"
        ],

        "mine": [
            "MINE_QUARRY"
        ],

        "oil_gas": [
            "OIL_GAS"
        ],

        "industrial_area": [
            "INDUSTRIAL_AREA"
        ]
    }

    category_data = {}

    for category_name, category_values in categories.items():

        category_data[category_name] = osm[
            osm["facility_category"].isin(
                category_values
            )
        ].copy()

    # ========================================================
    # MATCH FIRMS
    # ========================================================

    print(
        "\nMatching FIRMS detections..."
    )

    results = []

    total = len(firms)

    for counter, (_, fire) in enumerate(
        firms.iterrows(),
        start=1
    ):

        fire_lat = fire["latitude"]
        fire_lon = fire["longitude"]

        result = fire.to_dict()

        # ----------------------------------------------------
        # Find nearest overall industrial facility
        # ----------------------------------------------------

        distance, facility = find_nearest(
            fire_lat,
            fire_lon,
            category_data["industry"]
        )

        result[
            "distance_to_industry_km"
        ] = distance

        if facility is not None:

            result[
                "nearest_industry_name"
            ] = facility["name"]

            result[
                "nearest_industry_category"
            ] = facility[
                "facility_category"
            ]

            result[
                "nearest_industry_osm_id"
            ] = facility["osm_id"]

        else:

            result[
                "nearest_industry_name"
            ] = np.nan

            result[
                "nearest_industry_category"
            ] = np.nan

            result[
                "nearest_industry_osm_id"
            ] = np.nan

        # ----------------------------------------------------
        # Power plant
        # ----------------------------------------------------

        distance, facility = find_nearest(
            fire_lat,
            fire_lon,
            category_data["power"]
        )

        result[
            "distance_to_powerplant_km"
        ] = distance

        # ----------------------------------------------------
        # Mine / quarry
        # ----------------------------------------------------

        distance, facility = find_nearest(
            fire_lat,
            fire_lon,
            category_data["mine"]
        )

        result[
            "distance_to_mine_quarry_km"
        ] = distance

        # ----------------------------------------------------
        # Oil / gas
        # ----------------------------------------------------

        distance, facility = find_nearest(
            fire_lat,
            fire_lon,
            category_data["oil_gas"]
        )

        result[
            "distance_to_oil_gas_km"
        ] = distance

        # ----------------------------------------------------
        # Industrial area
        # ----------------------------------------------------

        distance, facility = find_nearest(
            fire_lat,
            fire_lon,
            category_data["industrial_area"]
        )

        result[
            "distance_to_industrial_area_km"
        ] = distance

        results.append(result)

        if (
            counter % 100 == 0
            or counter == total
        ):

            print(
                f"  Processed "
                f"{counter}/{total}"
            )

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    matched = pd.DataFrame(
        results
    )

    # ========================================================
    # PROXIMITY FLAGS
    # ========================================================

    # These are useful ML features.
    #
    # 1 km is a starting prototype threshold.
    # We can tune these later.

    matched[
        "near_industry_1km"
    ] = (
        matched[
            "distance_to_industry_km"
        ] <= 1.0
    ).astype(int)

    matched[
        "near_powerplant_1km"
    ] = (
        matched[
            "distance_to_powerplant_km"
        ] <= 1.0
    ).astype(int)

    matched[
        "near_mine_quarry_1km"
    ] = (
        matched[
            "distance_to_mine_quarry_km"
        ] <= 1.0
    ).astype(int)

    matched[
        "near_oil_gas_1km"
    ] = (
        matched[
            "distance_to_oil_gas_km"
        ] <= 1.0
    ).astype(int)

    matched[
        "near_industrial_area_1km"
    ] = (
        matched[
            "distance_to_industrial_area_km"
        ] <= 1.0
    ).astype(int)

    # ========================================================
    # SAVE
    # ========================================================

    matched.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("================================")
    print("CATEGORIZED MATCHING COMPLETE")
    print("================================")

    print(
        f"Total FIRMS records: "
        f"{len(matched)}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print()

    print("Proximity summary:")

    print(
        f"Near industry (<1 km): "
        f"{matched['near_industry_1km'].sum()}"
    )

    print(
        f"Near power plant (<1 km): "
        f"{matched['near_powerplant_1km'].sum()}"
    )

    print(
        f"Near mine/quarry (<1 km): "
        f"{matched['near_mine_quarry_1km'].sum()}"
    )

    print(
        f"Near oil/gas (<1 km): "
        f"{matched['near_oil_gas_1km'].sum()}"
    )

    print(
        f"Near industrial area (<1 km): "
        f"{matched['near_industrial_area_1km'].sum()}"
    )

    # ========================================================
    # SHOW RESULTS
    # ========================================================

    print()
    print("Closest industrial detections:")

    display_columns = [

        "latitude",
        "longitude",

        "distance_to_industry_km",

        "nearest_industry_name",

        "nearest_industry_category",

        "distance_to_powerplant_km",

        "distance_to_mine_quarry_km",

        "distance_to_oil_gas_km",

        "distance_to_industrial_area_km"

    ]

    print(
        matched[
            display_columns
        ]
        .sort_values(
            "distance_to_industry_km"
        )
        .head(15)
        .to_string(index=False)
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()