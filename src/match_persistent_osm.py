import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2


# ============================================================
# PATHS
# ============================================================

PERSISTENCE_FILE = (
    "data/processed/noaa20_persistence_features.csv"
)

OSM_FILE = (
    "data/raw/osm/telangana_industrial_facilities.csv"
)

OUTPUT_FILE = (
    "data/processed/persistent_locations_osm_distance_features.csv"
)


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two latitude/longitude points.
    """

    R = 6371.0

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

    return 2 * R * np.arcsin(np.sqrt(a))


# ============================================================
# OSM CATEGORY
# ============================================================

def classify_osm_category(row):

    industrial = str(row.get("industrial", "")).lower().strip()
    landuse = str(row.get("landuse", "")).lower().strip()
    power = str(row.get("power", "")).lower().strip()
    man_made = str(row.get("man_made", "")).lower().strip()
    name = str(row.get("name", "")).lower().strip()
    description = str(row.get("description", "")).lower().strip()

    # ========================================================
    # 1. POWER PLANT
    # ========================================================
    # Only classify as a power plant when OSM explicitly says
    # power=plant.
    if power == "plant":
        return "POWER_PLANT"

    # ========================================================
    # 2. OIL / GAS
    # ========================================================
    # Use explicit/strong oil & gas evidence.
    oil_gas_keywords = [
        "oil refinery",
        "oilfield",
        "oil field",
        "petroleum",
        "lng",
        "lpg",
        "natural gas",
        "gas field",
        "gas plant",
        "gas terminal",
        "refinery"
    ]

    text = " ".join([
        industrial,
        landuse,
        man_made,
        name,
        description
    ])

    if any(keyword in text for keyword in oil_gas_keywords):
        return "OIL_GAS"

    # ========================================================
    # 3. MINE / QUARRY
    # ========================================================
    # Prefer explicit OSM tags rather than loose name matching.
    if landuse in ["mine", "quarry"]:
        return "MINE_QUARRY"

    if man_made in ["mineshaft", "adit"]:
        return "MINE_QUARRY"

    mine_keywords = [
        "coal mine",
        "coalfield",
        "open cast mine",
        "open pit mine",
        "quarry"
    ]

    if any(keyword in text for keyword in mine_keywords):
        return "MINE_QUARRY"

    # ========================================================
    # 4. INDUSTRIAL AREA
    # ========================================================
    # Explicit OSM landuse tag.
    if landuse == "industrial":
        return "INDUSTRIAL_AREA"

    # ========================================================
    # 5. INDUSTRIAL WORKS
    # ========================================================
    # Require explicit industrial/works/factory tagging.
    if industrial not in ["", "nan", "none"]:
        return "INDUSTRIAL_WORKS"

    if man_made in ["works", "factory"]:
        return "INDUSTRIAL_WORKS"

    # Strong facility terms in name/description.
    industrial_keywords = [
        "factory",
        "steel works",
        "cement works",
        "industrial works",
        "manufacturing plant",
        "cement factory",
        "steel factory"
    ]

    if any(keyword in text for keyword in industrial_keywords):
        return "INDUSTRIAL_WORKS"

    # ========================================================
    # 6. OTHER
    # ========================================================
    return "OTHER"

    industrial = str(row.get("industrial", "")).lower()
    landuse = str(row.get("landuse", "")).lower()
    power = str(row.get("power", "")).lower()
    man_made = str(row.get("man_made", "")).lower()
    name = str(row.get("name", "")).lower()
    description = str(row.get("description", "")).lower()

    text = " ".join([
        industrial,
        landuse,
        power,
        man_made,
        name,
        description
    ])

    # Power
    if power == "plant":
        return "POWER_PLANT"

    # Oil / gas
    oil_gas_keywords = [
        "oil",
        "gas",
        "petroleum",
        "refinery",
        "fuel",
        "lng",
        "lpg"
    ]

    if any(word in text for word in oil_gas_keywords):
        return "OIL_GAS"

    # Mines / quarries
    mine_keywords = [
        "mine",
        "mineshaft",
        "quarry",
        "coal",
        "open pit"
    ]

    if (
        landuse in ["mine", "quarry"]
        or man_made in ["mineshaft", "adit"]
        or any(word in text for word in mine_keywords)
    ):
        return "MINE_QUARRY"

    # Industrial area
    if landuse == "industrial":
        return "INDUSTRIAL_AREA"

    # Industrial works
    industrial_keywords = [
        "factory",
        "works",
        "industrial",
        "plant"
    ]

    if (
        industrial
        or man_made in ["works", "factory"]
        or any(word in text for word in industrial_keywords)
    ):
        return "INDUSTRIAL_WORKS"

    return "OTHER"


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading persistence data...")

persistent = pd.read_csv(PERSISTENCE_FILE)

# Only locations classified as persistent over 180 days
persistent = persistent[
    persistent["persistent_180d"] == 1
].copy()

print(f"Persistent locations: {len(persistent)}")


print("\nLoading OSM data...")

osm = pd.read_csv(OSM_FILE)

osm["category"] = osm.apply(
    classify_osm_category,
    axis=1
)

print(f"OSM facilities: {len(osm)}")

print("\nOSM categories:")
print(osm["category"].value_counts())


# ============================================================
# PREPARE OSM ARRAYS
# ============================================================

osm_lat = osm["latitude"].to_numpy()
osm_lon = osm["longitude"].to_numpy()


# ============================================================
# DISTANCE FEATURES
# ============================================================

categories = [
    "INDUSTRIAL_AREA",
    "INDUSTRIAL_WORKS",
    "POWER_PLANT",
    "MINE_QUARRY",
    "OIL_GAS"
]

radii = [0.5, 1, 2, 5]


results = []


for _, row in persistent.iterrows():

    fire_lat = row["lat_grid"]
    fire_lon = row["lon_grid"]

    distances = haversine_km(
        fire_lat,
        fire_lon,
        osm_lat,
        osm_lon
    )

    result = row.to_dict()

    # --------------------------------------------------------
    # Overall nearest OSM facility
    # --------------------------------------------------------

    nearest_idx = np.argmin(distances)

    result["nearest_osm_distance_km"] = distances[nearest_idx]

    result["nearest_osm_category"] = (
        osm.iloc[nearest_idx]["category"]
    )

    result["nearest_osm_name"] = (
        osm.iloc[nearest_idx]["name"]
    )

    # --------------------------------------------------------
    # Category-specific distances
    # --------------------------------------------------------

    for category in categories:

        mask = (
            osm["category"].to_numpy() == category
        )

        if mask.any():

            category_distances = distances[mask]

            result[
                f"nearest_{category.lower()}_km"
            ] = category_distances.min()

        else:

            result[
                f"nearest_{category.lower()}_km"
            ] = np.nan

    # --------------------------------------------------------
    # Facility counts within each radius
    # --------------------------------------------------------

    for category in categories:

        category_mask = (
            osm["category"].to_numpy() == category
        )

        category_distances = distances[category_mask]

        for radius in radii:

            count = (
                category_distances <= radius
            ).sum()

            column_name = (
                f"{category.lower()}_within_{radius}km"
            )

            result[column_name] = int(count)

    results.append(result)


# ============================================================
# SAVE
# ============================================================

output = pd.DataFrame(results)

output.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("OSM DISTANCE ANALYSIS COMPLETE")
print("========================================")

print(f"\nOutput:")
print(OUTPUT_FILE)

print(f"\nLocations analysed: {len(output)}")


print("\n----------------------------------------")
print("DISTANCE SUMMARY")
print("----------------------------------------")

distance_columns = [
    "nearest_industrial_area_km",
    "nearest_industrial_works_km",
    "nearest_power_plant_km",
    "nearest_mine_quarry_km",
    "nearest_oil_gas_km"
]

print(
    output[distance_columns].round(3).to_string(
        index=False
    )
)


print("\n----------------------------------------")
print("FACILITIES WITHIN RADIUS")
print("----------------------------------------")

for category in categories:

    print(f"\n{category}")

    for radius in radii:

        column = (
            f"{category.lower()}_within_{radius}km"
        )

        locations = (
            output[column] > 0
        ).sum()

        print(
            f"  within {radius} km: "
            f"{locations} / {len(output)} locations"
        )


print("\nDone.")