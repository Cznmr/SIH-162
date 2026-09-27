import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"C:\Users\user\Desktop\ENTRO-26162")

MASTER_FILE = (
    BASE_DIR
    / "data"
    / "training"
    / "master_feature_table_with_dynamic_world.csv"
)

OSM_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "osm"
    / "telangana_industrial_facilities.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "osm_all_location_features.csv"
)


EARTH_RADIUS_KM = 6371.0088

DISTANCE_THRESHOLDS = [0.5, 1, 2, 5]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("BUILDING OSM FEATURES FOR ALL MASTER LOCATIONS")
print("=" * 70)

print("\nLoading master feature table...")
master = pd.read_csv(MASTER_FILE, low_memory=False)

print(f"Master rows: {len(master):,}")

print("\nLoading OSM data...")
osm = pd.read_csv(OSM_FILE)

print(f"OSM rows: {len(osm):,}")


# ============================================================
# CLEAN TEXT COLUMNS
# ============================================================

text_columns = [
    "industrial",
    "landuse",
    "power",
    "man_made",
    "name",
    "operator",
    "description",
]

for col in text_columns:
    if col in osm.columns:
        osm[col] = (
            osm[col]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )


# ============================================================
# REMOVE INVALID COORDINATES
# ============================================================

master = master.dropna(
    subset=["lat_grid", "lon_grid"]
).copy()

osm = osm.dropna(
    subset=["latitude", "longitude"]
).copy()


print(f"\nValid master locations: {len(master):,}")
print(f"Valid OSM locations: {len(osm):,}")


# ============================================================
# MASTER COORDINATES
# ============================================================

master_coords_deg = master[
    ["lat_grid", "lon_grid"]
].to_numpy()

master_coords_rad = np.radians(
    master_coords_deg
)


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def nearest_distance(
    master_coords_rad,
    osm_subset
):
    """
    Find nearest OSM feature distance
    for every master location.
    """

    if len(osm_subset) == 0:
        return np.full(
            len(master_coords_rad),
            np.nan
        )

    osm_coords_deg = osm_subset[
        ["latitude", "longitude"]
    ].to_numpy()

    osm_coords_rad = np.radians(
        osm_coords_deg
    )

    tree = BallTree(
        osm_coords_rad,
        metric="haversine"
    )

    distances_rad, _ = tree.query(
        master_coords_rad,
        k=1
    )

    return (
        distances_rad[:, 0]
        * EARTH_RADIUS_KM
    )


# ============================================================
# BUILD OSM CATEGORIES
# ============================================================

print("\nBuilding OSM categories...")


# ------------------------------------------------------------
# 1. INDUSTRIAL AREA
# ------------------------------------------------------------

industrial_area = osm[
    osm["landuse"] == "industrial"
].copy()


# ------------------------------------------------------------
# 2. QUARRY / MINING
# ------------------------------------------------------------

quarry = osm[
    osm["landuse"] == "quarry"
].copy()


# ------------------------------------------------------------
# 3. POWER PLANT
# ------------------------------------------------------------

power_plant = osm[
    osm["power"] == "plant"
].copy()


# ------------------------------------------------------------
# 4. POWER INFRASTRUCTURE
# ------------------------------------------------------------

power_infrastructure = osm[
    osm["power"].isin([
        "plant",
        "substation"
    ])
].copy()


# ------------------------------------------------------------
# 5. INDUSTRIAL WORKS
# ------------------------------------------------------------

industrial_works = osm[
    (osm["man_made"] == "works")
    |
    (
        osm["industrial"].isin([
            "factory",
            "pharmaceutical company",
            "pharmaceutical company",
            "chemical",
            "food_industry",
            "automotive_industry",
            "aerospace",
            "biotechnology company",
            "biotechnology_and_pharmaceutical",
            "agrochemical company",
            "concrete_plant",
            "rice_mill",
            "grinding_mill",
            "mineral_processing",
            "captive_workshop",
            "wood",
            "scrap_yard",
            "slaughterhouse",
            "brickyard"
        ])
    )
].copy()


# ------------------------------------------------------------
# 6. STORAGE TANK
# ------------------------------------------------------------

storage_tank = osm[
    osm["man_made"] == "storage_tank"
].copy()


# ------------------------------------------------------------
# 7. OIL / GAS
# ------------------------------------------------------------

oil_gas = osm[
    osm["industrial"].isin([
        "oil",
        "oil_gas",
        "gas"
    ])
].copy()


# ------------------------------------------------------------
# 8. RELEVANT INDUSTRIAL CONTEXT
# ------------------------------------------------------------

relevant_osm = osm[
    (
        osm["landuse"].isin([
            "industrial",
            "quarry"
        ])
    )
    |
    (
        osm["power"].isin([
            "plant",
            "substation"
        ])
    )
    |
    (
        osm["man_made"].isin([
            "works",
            "storage_tank"
        ])
    )
    |
    (
        osm["industrial"] != ""
    )
].copy()


# ============================================================
# CATEGORY SUMMARY
# ============================================================

categories = {
    "industrial_area": industrial_area,
    "quarry": quarry,
    "power_plant": power_plant,
    "power_infrastructure": power_infrastructure,
    "industrial_works": industrial_works,
    "storage_tank": storage_tank,
    "oil_gas": oil_gas,
    "relevant_osm": relevant_osm,
}


print("\nOSM category sizes:")

for name, subset in categories.items():
    print(
        f"{name:30s} "
        f"{len(subset):,}"
    )


# ============================================================
# OUTPUT TABLE
# ============================================================

osm_features = master[
    [
        "lat_grid",
        "lon_grid"
    ]
].copy()


# ============================================================
# CALCULATE NEAREST DISTANCES
# ============================================================

print("\nCalculating nearest distances...")


for name, subset in categories.items():

    print(
        f"  Processing {name}..."
    )

    column_name = (
        f"nearest_{name}_km"
    )

    osm_features[column_name] = (
        nearest_distance(
            master_coords_rad,
            subset
        )
    )


# ============================================================
# DISTANCE FLAGS
# ============================================================

print("\nCreating distance flags...")


distance_columns = [
    col
    for col in osm_features.columns
    if col.startswith("nearest_")
]


for distance_col in distance_columns:

    base_name = (
        distance_col
        .replace("nearest_", "")
        .replace("_km", "")
    )

    for threshold in DISTANCE_THRESHOLDS:

        threshold_name = str(
            threshold
        ).replace(".", "_")

        flag_name = (
            f"near_{base_name}_"
            f"{threshold_name}km"
        )

        osm_features[flag_name] = (
            osm_features[distance_col]
            <= threshold
        ).astype(int)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

osm_features.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("OSM FEATURE EXTRACTION COMPLETE")
print("=" * 70)

print(
    f"\nOutput file:\n{OUTPUT_FILE}"
)

print(
    f"\nRows: {len(osm_features):,}"
)

print(
    f"Columns: {len(osm_features.columns):,}"
)


# ============================================================
# MISSING VALUES
# ============================================================

print(
    "\nMissing nearest-distance values:"
)

for col in distance_columns:

    missing = (
        osm_features[col]
        .isna()
        .sum()
    )

    percentage = (
        missing
        / len(osm_features)
        * 100
    )

    print(
        f"{col:40s} "
        f"{missing:8,} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# DISTANCE STATISTICS
# ============================================================

print(
    "\nNearest-distance statistics:"
)

for col in distance_columns:

    print(
        f"\n{col}"
    )

    print(
        osm_features[col].describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        )
    )


# ============================================================
# FLAG COVERAGE
# ============================================================

print(
    "\nDistance flag coverage:"
)

flag_columns = [
    col
    for col in osm_features.columns
    if col.startswith("near_")
]

for col in flag_columns:

    count = (
        osm_features[col]
        .sum()
    )

    percentage = (
        count
        / len(osm_features)
        * 100
    )

    print(
        f"{col:50s} "
        f"{count:8,} "
        f"({percentage:.2f}%)"
    )


print("\nDone.")