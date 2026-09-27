import pandas as pd
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
    / "processed"
    / "osm_all_location_features.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "training"
    / "master_feature_table_final.csv"
)


print("=" * 70)
print("MERGING CLEAN OSM FEATURES INTO MASTER TABLE")
print("=" * 70)


# ============================================================
# LOAD
# ============================================================

print("\nLoading master table...")

master = pd.read_csv(
    MASTER_FILE,
    low_memory=False
)

print(
    f"Master rows: {len(master):,}"
)

print(
    f"Master columns: {len(master.columns):,}"
)


print("\nLoading new OSM features...")

osm = pd.read_csv(
    OSM_FILE
)

print(
    f"OSM rows: {len(osm):,}"
)

print(
    f"OSM columns: {len(osm.columns):,}"
)


# ============================================================
# CHECK KEYS
# ============================================================

KEYS = [
    "lat_grid",
    "lon_grid"
]

for col in KEYS:

    if col not in master.columns:
        raise ValueError(
            f"Missing {col} from master table"
        )

    if col not in osm.columns:
        raise ValueError(
            f"Missing {col} from OSM table"
        )


# ============================================================
# CHECK DUPLICATES
# ============================================================

master_duplicates = master.duplicated(
    subset=KEYS
).sum()

osm_duplicates = osm.duplicated(
    subset=KEYS
).sum()


print(
    f"\nMaster duplicate locations: "
    f"{master_duplicates:,}"
)

print(
    f"OSM duplicate locations: "
    f"{osm_duplicates:,}"
)


if master_duplicates > 0:
    raise ValueError(
        "Master table contains duplicate locations."
    )

if osm_duplicates > 0:
    raise ValueError(
        "OSM feature table contains duplicate locations."
    )


# ============================================================
# IDENTIFY OLD OSM COLUMNS
# ============================================================

old_osm_keywords = [
    "industrial_area_within",
    "industrial_works_within",
    "mine_quarry_within",
    "power_plant_within",
    "oil_gas_within",
    "nearest_",
    "near_mine_",
    "near_power_",
    "near_industrial_"
]


old_osm_columns = []

for col in master.columns:

    if any(
        keyword in col
        for keyword in old_osm_keywords
    ):
        old_osm_columns.append(col)


print(
    "\nOld OSM columns detected:"
)

for col in old_osm_columns:
    print(
        f"  - {col}"
    )

print(
    f"\nTotal old OSM columns: "
    f"{len(old_osm_columns)}"
)


# ============================================================
# REMOVE OLD OSM FEATURES
# ============================================================

master_clean = master.drop(
    columns=old_osm_columns,
    errors="ignore"
).copy()


print(
    "\nMaster after removing old OSM features:"
)

print(
    f"Rows: {len(master_clean):,}"
)

print(
    f"Columns: {len(master_clean.columns):,}"
)


# ============================================================
# IDENTIFY NEW OSM FEATURE COLUMNS
# ============================================================

osm_feature_columns = [
    col
    for col in osm.columns
    if col not in KEYS
]


# ============================================================
# MAKE SURE THERE ARE NO COLUMN CONFLICTS
# ============================================================

conflicts = [
    col
    for col in osm_feature_columns
    if col in master_clean.columns
]


if conflicts:

    print(
        "\nWARNING: Existing columns will be replaced:"
    )

    for col in conflicts:
        print(
            f"  - {col}"
        )

    master_clean = master_clean.drop(
        columns=conflicts
    )


# ============================================================
# MERGE
# ============================================================

print(
    "\nMerging new OSM features..."
)

final = master_clean.merge(
    osm,
    on=KEYS,
    how="left",
    validate="one_to_one"
)


# ============================================================
# CHECK ROW COUNT
# ============================================================

print(
    "\nAfter merge:"
)

print(
    f"Rows: {len(final):,}"
)

print(
    f"Columns: {len(final.columns):,}"
)


if len(final) != len(master):
    raise ValueError(
        "Row count changed after OSM merge!"
    )


# ============================================================
# CHECK OSM COVERAGE
# ============================================================

distance_columns = [
    col
    for col in osm_feature_columns
    if col.startswith("nearest_")
]


print(
    "\nOSM distance coverage:"
)

for col in distance_columns:

    if col not in final.columns:
        continue

    missing = final[col].isna().sum()

    percentage = (
        missing
        / len(final)
        * 100
    )

    print(
        f"{col:45s}"
        f"{missing:8,} missing "
        f"({percentage:.2f}%)"
    )


# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicates_after = final.duplicated(
    subset=KEYS
).sum()


print(
    f"\nDuplicate locations after merge: "
    f"{duplicates_after:,}"
)


if duplicates_after > 0:

    raise ValueError(
        "Duplicate locations detected after merge!"
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

final.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLEAN MASTER TABLE CREATED")
print("=" * 70)

print(
    f"\nOutput:\n{OUTPUT_FILE}"
)

print(
    f"\nRows: {len(final):,}"
)

print(
    f"Columns: {len(final.columns):,}"
)

print(
    "\nExpected rows: 47,677"
)

print(
    "\nDone."
)