from pathlib import Path
import sys

import numpy as np
import pandas as pd
import geopandas as gpd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\user\Desktop\ENTRO-26162")

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

FOREST_FILE = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "forest_location_evidence.csv"
)

SPATIAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "spatial_location_evidence.csv"
)

DISTRICT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "telangana_district_boundaries.geojson"
)

PREVIOUS_DISTRICT_FILE = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "unknown_firms_by_district.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "district_agriculture_fire_evidence.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "district_agriculture_fire_summary.csv"
)


# ============================================================
# HELPERS
# ============================================================

def check_file(path):
    if not path.exists():
        print("\nERROR: File not found:")
        print(path)
        sys.exit(1)


def safe_mean(series):
    values = pd.to_numeric(series, errors="coerce")

    if values.notna().sum() == 0:
        return np.nan

    return values.mean()


def safe_median(series):
    values = pd.to_numeric(series, errors="coerce")

    if values.notna().sum() == 0:
        return np.nan

    return values.median()


def percentage(numerator, denominator):
    if denominator == 0:
        return 0.0

    return (numerator / denominator) * 100.0


# ============================================================
# START
# ============================================================

print("=" * 75)
print("TELANGANA DISTRICT AGRICULTURE + FIRMS EVIDENCE ANALYSIS")
print("=" * 75)


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking input files...")

required_files = [
    MASTER_FILE,
    SEED_FILE,
    FOREST_FILE,
    SPATIAL_FILE,
    DISTRICT_FILE,
    PREVIOUS_DISTRICT_FILE,
]

for path in required_files:
    check_file(path)

print("All required files found.")


# ============================================================
# 1. LOAD MASTER
# ============================================================

print("\n[1/9] Loading master feature table...")

master = pd.read_csv(
    MASTER_FILE,
    low_memory=False
)

required_master_columns = [
    "lat_grid",
    "lon_grid",
    "max_frp",
    "detection_count",
    "active_days",
    "active_days_180d",
    "dw_crops",
    "dw_vegetation_score",
]

missing = [
    col
    for col in required_master_columns
    if col not in master.columns
]

if missing:
    print("\nERROR: Missing master columns:")
    print(missing)
    sys.exit(1)

print(f"Master rows: {len(master):,}")


# ============================================================
# 2. LOAD SEED LABELS
# ============================================================

print("\n[2/9] Loading seed labels...")

seed = pd.read_csv(
    SEED_FILE,
    low_memory=False
)

required_seed_columns = [
    "lat_grid",
    "lon_grid",
    "seed_label",
]

missing = [
    col
    for col in required_seed_columns
    if col not in seed.columns
]

if missing:
    print("\nERROR: Missing seed columns:")
    print(missing)
    sys.exit(1)

seed_small = seed[
    [
        "lat_grid",
        "lon_grid",
        "seed_label",
    ]
].copy()

if seed_small.duplicated(
    ["lat_grid", "lon_grid"]
).any():
    print("\nERROR: Duplicate grid cells in seed labels.")
    sys.exit(1)


# ============================================================
# 3. MERGE MASTER + SEED
# ============================================================

print("\n[3/9] Merging master + seed labels...")

df = master.merge(
    seed_small,
    on=["lat_grid", "lon_grid"],
    how="left",
    validate="one_to_one",
)

if df["seed_label"].isna().any():
    print(
        "WARNING:",
        df["seed_label"].isna().sum(),
        "master locations have no seed label."
    )

print(f"Total locations: {len(df):,}")

unknown_total_before_geo = (
    df["seed_label"].eq("UNKNOWN").sum()
)

print(
    f"UNKNOWN locations before geographic filtering: "
    f"{unknown_total_before_geo:,}"
)


# ============================================================
# 4. LOAD FOREST EVIDENCE
# ============================================================

print("\n[4/9] Loading forest evidence...")

forest = pd.read_csv(
    FOREST_FILE,
    low_memory=False
)

forest_required = [
    "lat_grid",
    "lon_grid",
]

missing = [
    col
    for col in forest_required
    if col not in forest.columns
]

if missing:
    print("\nERROR: Missing forest columns:")
    print(missing)
    sys.exit(1)

if forest.duplicated(
    ["lat_grid", "lon_grid"]
).any():
    print(
        "\nERROR: Duplicate grid cells in forest evidence."
    )
    sys.exit(1)

forest_columns_to_add = [
    col
    for col in forest.columns
    if col not in ["lat_grid", "lon_grid"]
]

df = df.merge(
    forest,
    on=["lat_grid", "lon_grid"],
    how="left",
    validate="one_to_one",
)

print(
    f"Forest evidence columns added: "
    f"{len(forest_columns_to_add)}"
)


# ============================================================
# 5. LOAD SPATIAL EVIDENCE
# ============================================================

print("\n[5/9] Loading spatial evidence...")

spatial = pd.read_csv(
    SPATIAL_FILE,
    low_memory=False
)

spatial_required = [
    "lat_grid",
    "lon_grid",
]

missing = [
    col
    for col in spatial_required
    if col not in spatial.columns
]

if missing:
    print("\nERROR: Missing spatial columns:")
    print(missing)
    sys.exit(1)

if spatial.duplicated(
    ["lat_grid", "lon_grid"]
).any():
    print(
        "\nERROR: Duplicate grid cells in spatial evidence."
    )
    sys.exit(1)

spatial_columns_to_add = [
    col
    for col in spatial.columns
    if col not in ["lat_grid", "lon_grid"]
]

df = df.merge(
    spatial,
    on=["lat_grid", "lon_grid"],
    how="left",
    validate="one_to_one",
)

print(
    f"Spatial evidence columns added: "
    f"{len(spatial_columns_to_add)}"
)


# ============================================================
# 6. LOAD TELANGANA DISTRICTS
# ============================================================

print("\n[6/9] Loading Telangana district boundaries...")

districts = gpd.read_file(
    DISTRICT_FILE
)

if len(districts) != 33:
    print(
        f"WARNING: Expected 33 districts, "
        f"found {len(districts)}."
    )

if districts.empty:
    print("\nERROR: District layer is empty.")
    sys.exit(1)

if districts.crs is None:
    districts = districts.set_crs("EPSG:4326")

districts = districts.to_crs("EPSG:4326")

# Find district-name column
candidate_name_fields = [
    "district_name",
    "District",
    "DISTRICT",
    "district",
    "District_Name",
    "DIST_NAME",
    "DIST_NAME_1",
    "name",
    "NAME",
    "DistrictName",
]

district_name_field = None

for col in candidate_name_fields:
    if col in districts.columns:
        district_name_field = col
        break

if district_name_field is None:
    print("\nERROR: Could not identify district-name field.")
    print("Available columns:")
    print(list(districts.columns))
    sys.exit(1)

districts = districts[
    [district_name_field, "geometry"]
].copy()

districts = districts.rename(
    columns={
        district_name_field: "district_name"
    }
)

districts = districts[
    districts.geometry.notna()
].copy()

districts = districts[
    ~districts.geometry.is_empty
].copy()

# Repair invalid geometry
districts["geometry"] = districts.geometry.buffer(0)

print(
    f"District polygons loaded: {len(districts)}"
)

print(
    f"Unique district names: "
    f"{districts['district_name'].nunique()}"
)


# ============================================================
# DISTRICT AREA
# ============================================================

districts_utm = districts.to_crs("EPSG:32644")

districts["district_area_km2"] = (
    districts_utm.geometry.area / 1_000_000
)

district_area = districts[
    [
        "district_name",
        "district_area_km2",
    ]
].drop_duplicates(
    subset=["district_name"]
)


# ============================================================
# ASSIGN EVERY MASTER GRID CELL TO TELANGANA DISTRICT
# ============================================================

print("\nAssigning master grid cells to districts...")

# IMPORTANT:
# Master coordinates were generated with floor() at 0.01 degrees.
# Use cell centers for geographic assignment.

points = gpd.GeoDataFrame(
    df[
        [
            "lat_grid",
            "lon_grid",
        ]
    ].copy(),
    geometry=gpd.points_from_xy(
        df["lon_grid"] + 0.005,
        df["lat_grid"] + 0.005,
    ),
    crs="EPSG:4326",
)

points["row_id"] = np.arange(len(points))

joined = gpd.sjoin(
    points,
    districts,
    how="left",
    predicate="within",
)

joined = joined.sort_values(
    "row_id"
)

joined = joined.drop_duplicates(
    subset=["row_id"],
    keep="first"
)

assignment = joined[
    [
        "row_id",
        "district_name",
    ]
].copy()

df["row_id"] = np.arange(len(df))

df = df.merge(
    assignment,
    on="row_id",
    how="left",
    validate="one_to_one",
)

matched = df["district_name"].notna().sum()
outside_tel = df["district_name"].isna().sum()

print(
    f"Inside Telangana district polygons: "
    f"{matched:,}"
)

print(
    f"Outside Telangana district polygons: "
    f"{outside_tel:,}"
)

print(
    f"Telangana coverage of master grid: "
    f"{percentage(matched, len(df)):.2f}%"
)


# ============================================================
# CRITICAL GEOGRAPHIC FILTER
# ============================================================

print("\nFiltering to actual Telangana locations...")

df_tel = df[
    df["district_name"].notna()
].copy()

print(
    f"Locations retained inside Telangana: "
    f"{len(df_tel):,}"
)

print(
    f"Locations discarded outside district boundaries: "
    f"{len(df) - len(df_tel):,}"
)


# ============================================================
# UNKNOWN CROSS-CHECK
# ============================================================

unknown_tel = df_tel[
    df_tel["seed_label"].eq("UNKNOWN")
].copy()

print(
    f"\nUNKNOWN locations inside Telangana: "
    f"{len(unknown_tel):,}"
)

print(
    f"Expected validated UNKNOWN total: "
    f"{unknown_total_before_geo:,}"
)

# We expect some UNKNOWN locations to have been outside
# the Telangana district boundaries, because the original
# FIRMS processing used a larger bounding box.

if len(unknown_tel) < 1:
    print("\nERROR: No UNKNOWN locations inside Telangana.")
    sys.exit(1)


# ============================================================
# 7. BUILD DISTRICT EVIDENCE
# ============================================================

print("\n[7/9] Building district-level evidence...")

records = []

for district_name, group in df_tel.groupby(
    "district_name"
):

    unknown = group[
        group["seed_label"].eq("UNKNOWN")
    ].copy()

    if unknown.empty:
        continue

    unknown_count = len(unknown)

    # --------------------------------------------------------
    # FIRMS / FIRE ACTIVITY
    # --------------------------------------------------------

    max_frp = pd.to_numeric(
        unknown["max_frp"],
        errors="coerce"
    )

    detection_count = pd.to_numeric(
        unknown["detection_count"],
        errors="coerce"
    )

    active_days = pd.to_numeric(
        unknown["active_days"],
        errors="coerce"
    )

    active_days_180 = pd.to_numeric(
        unknown["active_days_180d"],
        errors="coerce"
    )

    frp10 = (max_frp >= 10).sum()
    frp15 = (max_frp >= 15).sum()
    frp30 = (max_frp >= 30).sum()

    repeated2 = (
        detection_count >= 2
    ).sum()

    repeated3 = (
        detection_count >= 3
    ).sum()

    active2 = (
        active_days >= 2
    ).sum()

    active3 = (
        active_days >= 3
    ).sum()

    recurring_high_frp = (
        (max_frp >= 10)
        &
        (detection_count >= 2)
    ).sum()

    persistent5 = (
        active_days_180 >= 5
    ).sum()

    # --------------------------------------------------------
    # DYNAMIC WORLD
    # --------------------------------------------------------

    crops = pd.to_numeric(
        unknown["dw_crops"],
        errors="coerce"
    )

    vegetation = pd.to_numeric(
        unknown["dw_vegetation_score"],
        errors="coerce"
    )

    crop_mean = safe_mean(crops)
    crop_median = safe_median(crops)

    crop20 = (crops >= 0.20).sum()
    crop40 = (crops >= 0.40).sum()
    crop50 = (crops >= 0.50).sum()
    crop60 = (crops >= 0.60).sum()

    vegetation50 = (
        vegetation >= 0.50
    ).sum()

    vegetation70 = (
        vegetation >= 0.70
    ).sum()

    # --------------------------------------------------------
    # FOREST
    # --------------------------------------------------------

    inside_forest = 0

    if "inside_forest" in unknown.columns:

        forest_value = unknown[
            "inside_forest"
        ]

        if forest_value.dtype == object:

            inside_forest = (
                forest_value
                .astype(str)
                .str.lower()
                .isin(
                    [
                        "true",
                        "1",
                        "yes",
                    ]
                )
                .sum()
            )

        else:

            inside_forest = (
                forest_value
                .fillna(False)
                .astype(bool)
                .sum()
            )

    forest_context = (
        unknown["forest_context"]
        .astype(str)
        .str.upper()
        if "forest_context" in unknown.columns
        else pd.Series(
            "",
            index=unknown.index
        )
    )

    very_near_forest = (
        forest_context == "VERY_NEAR_FOREST"
    ).sum()

    near_forest = (
        forest_context == "NEAR_FOREST"
    ).sum()

    within5_forest = (
        forest_context == "WITHIN_5KM_FOREST"
    ).sum()

    within10_forest = (
        forest_context == "WITHIN_10KM_FOREST"
    ).sum()

    # --------------------------------------------------------
    # SPATIAL BEHAVIOUR
    # --------------------------------------------------------

    expansion3 = 0
    expansion5 = 0

    new_activity3 = 0
    new_activity5 = 0

    local_persistence75 = 0

    consecutive2 = 0
    consecutive3 = 0

    if "expansion_days_3km" in unknown.columns:

        expansion3 = (
            pd.to_numeric(
                unknown["expansion_days_3km"],
                errors="coerce"
            ) >= 3
        ).sum()

    if "expansion_days_5km" in unknown.columns:

        expansion5 = (
            pd.to_numeric(
                unknown["expansion_days_5km"],
                errors="coerce"
            ) >= 3
        ).sum()

    if "new_activity_days_3km" in unknown.columns:

        new_activity3 = (
            pd.to_numeric(
                unknown["new_activity_days_3km"],
                errors="coerce"
            ) >= 3
        ).sum()

    if "new_activity_days_5km" in unknown.columns:

        new_activity5 = (
            pd.to_numeric(
                unknown["new_activity_days_5km"],
                errors="coerce"
            ) >= 3
        ).sum()

    if "max_local_persistence_5km" in unknown.columns:

        local_persistence75 = (
            pd.to_numeric(
                unknown[
                    "max_local_persistence_5km"
                ],
                errors="coerce"
            ) >= 0.75
        ).sum()

    if "consecutive_day_count" in unknown.columns:

        consecutive = pd.to_numeric(
            unknown["consecutive_day_count"],
            errors="coerce"
        )

        consecutive2 = (
            consecutive >= 2
        ).sum()

        consecutive3 = (
            consecutive >= 3
        ).sum()

    # --------------------------------------------------------
    # DISTRICT AREA
    # --------------------------------------------------------

    area_row = district_area[
        district_area["district_name"]
        == district_name
    ]

    if len(area_row) == 1:

        area_km2 = float(
            area_row.iloc[0][
                "district_area_km2"
            ]
        )

    else:

        area_km2 = np.nan

    unknown_density = (
        unknown_count / area_km2 * 100
        if area_km2 > 0
        else np.nan
    )

    # --------------------------------------------------------
    # PERCENTAGES
    # --------------------------------------------------------

    frp10_pct = percentage(
        frp10,
        unknown_count
    )

    repeated2_pct = percentage(
        repeated2,
        unknown_count
    )

    recurring_high_frp_pct = percentage(
        recurring_high_frp,
        unknown_count
    )

    crop40_pct = percentage(
        crop40,
        unknown_count
    )

    crop50_pct = percentage(
        crop50,
        unknown_count
    )

    crop60_pct = percentage(
        crop60,
        unknown_count
    )

    forest_pct = percentage(
        inside_forest,
        unknown_count
    )

    # --------------------------------------------------------
    # AGRICULTURE CONTEXT INDICATORS
    #
    # These are NOT labels.
    # They are only indicators for selecting districts
    # for deeper agriculture analysis.
    # --------------------------------------------------------

    agriculture_context_signal = (
        0.30 * crop50_pct
        + 0.20 * crop40_pct
        + 0.20 * frp10_pct
        + 0.15 * repeated2_pct
        + 0.15 * (100.0 - forest_pct)
    )

    forest_context_signal = (
        0.55 * forest_pct
        + 0.25 * frp10_pct
        + 0.20 * repeated2_pct
    )

    records.append(
        {
            "district_name":
                district_name,

            "district_area_km2":
                area_km2,

            "unknown_locations":
                unknown_count,

            "unknown_density_per_100km2":
                unknown_density,

            # FIRMS
            "unknown_frp_ge_10":
                frp10,

            "unknown_frp_ge_15":
                frp15,

            "unknown_frp_ge_30":
                frp30,

            "unknown_repeated_ge_2":
                repeated2,

            "unknown_repeated_ge_3":
                repeated3,

            "unknown_active_days_ge_2":
                active2,

            "unknown_active_days_ge_3":
                active3,

            "unknown_frp10_repeated":
                recurring_high_frp,

            "unknown_persistent_180d_ge_5":
                persistent5,

            "frp10_pct":
                frp10_pct,

            "repeated2_pct":
                repeated2_pct,

            "frp10_repeated_pct":
                recurring_high_frp_pct,

            # Dynamic World
            "dw_crops_mean":
                crop_mean,

            "dw_crops_median":
                crop_median,

            "dw_crops_ge_20":
                crop20,

            "dw_crops_ge_40":
                crop40,

            "dw_crops_ge_50":
                crop50,

            "dw_crops_ge_60":
                crop60,

            "dw_crops_ge_40_pct":
                crop40_pct,

            "dw_crops_ge_50_pct":
                crop50_pct,

            "dw_crops_ge_60_pct":
                crop60_pct,

            "dw_vegetation_ge_50":
                vegetation50,

            "dw_vegetation_ge_70":
                vegetation70,

            # Forest
            "inside_forest":
                inside_forest,

            "inside_forest_pct":
                forest_pct,

            "very_near_forest":
                very_near_forest,

            "near_forest":
                near_forest,

            "within5km_forest":
                within5_forest,

            "within10km_forest":
                within10_forest,

            # Spatial
            "expansion_days3_ge_3":
                expansion3,

            "expansion_days5_ge_3":
                expansion5,

            "new_activity_days3_ge_3":
                new_activity3,

            "new_activity_days5_ge_3":
                new_activity5,

            "local_persistence5_ge_75":
                local_persistence75,

            "consecutive_days_ge_2":
                consecutive2,

            "consecutive_days_ge_3":
                consecutive3,

            # Research indicators
            "agriculture_context_signal":
                agriculture_context_signal,

            "forest_context_signal":
                forest_context_signal,
        }
    )


# ============================================================
# 8. CREATE SUMMARY
# ============================================================

print("\n[8/9] Creating final district summary...")

summary = pd.DataFrame(records)

if summary.empty:
    print("\nERROR: No district records generated.")
    sys.exit(1)


# ============================================================
# CROSS-CHECK WITH VALIDATED UNKNOWN DISTRICT FILE
# ============================================================

print("\nCross-checking against validated district UNKNOWN file...")

previous = pd.read_csv(
    PREVIOUS_DISTRICT_FILE,
    low_memory=False
)

if (
    "district_name" not in previous.columns
    or "unknown_locations" not in previous.columns
):

    print(
        "\nERROR: Validated district file does not contain:"
    )
    print(
        "district_name and unknown_locations"
    )
    print("\nAvailable columns:")
    print(list(previous.columns))

    sys.exit(1)


previous_check = previous[
    [
        "district_name",
        "unknown_locations",
    ]
].copy()

previous_check = previous_check.rename(
    columns={
        "unknown_locations":
            "validated_unknown_locations"
    }
)

summary = summary.merge(
    previous_check,
    on="district_name",
    how="outer",
)


# ============================================================
# CROSS-CHECK DIFFERENCES
# ============================================================

summary["unknown_count_difference"] = (
    summary["unknown_locations"]
    - summary["validated_unknown_locations"]
)

crosscheck = summary[
    [
        "district_name",
        "unknown_locations",
        "validated_unknown_locations",
        "unknown_count_difference",
    ]
].copy()

# Fill NaN for districts absent from one side
crosscheck[
    [
        "unknown_locations",
        "validated_unknown_locations",
        "unknown_count_difference",
    ]
] = crosscheck[
    [
        "unknown_locations",
        "validated_unknown_locations",
        "unknown_count_difference",
    ]
].fillna(0)


print("\nDistrict UNKNOWN cross-check:")

print(
    crosscheck[
        crosscheck["unknown_count_difference"] != 0
    ]
    .to_string(index=False)
)


# ============================================================
# SORT
# ============================================================

summary = summary.sort_values(
    [
        "agriculture_context_signal",
        "unknown_locations",
    ],
    ascending=False,
).reset_index(drop=True)


# ============================================================
# SAVE FULL OUTPUT
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

summary.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SAVE COMPACT OUTPUT
# ============================================================

compact_columns = [
    "district_name",
    "district_area_km2",

    "unknown_locations",
    "validated_unknown_locations",
    "unknown_count_difference",

    "unknown_density_per_100km2",

    "unknown_frp_ge_10",
    "unknown_frp_ge_15",
    "unknown_frp_ge_30",

    "unknown_repeated_ge_2",
    "unknown_active_days_ge_2",

    "unknown_frp10_repeated",
    "frp10_repeated_pct",

    "dw_crops_mean",
    "dw_crops_median",
    "dw_crops_ge_40_pct",
    "dw_crops_ge_50_pct",
    "dw_crops_ge_60_pct",

    "inside_forest",
    "inside_forest_pct",

    "within5km_forest",

    "expansion_days3_ge_3",
    "new_activity_days3_ge_3",

    "local_persistence5_ge_75",

    "agriculture_context_signal",
    "forest_context_signal",
]

compact_columns = [
    col
    for col in compact_columns
    if col in summary.columns
]

compact = summary[
    compact_columns
].copy()

compact.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 75)
print("DISTRICT AGRICULTURE + FIRMS EVIDENCE — COMPLETE")
print("=" * 75)

print(
    f"\nTelangana districts analysed: "
    f"{summary['district_name'].notna().sum()}"
)

print(
    f"UNKNOWN locations inside Telangana: "
    f"{int(summary['unknown_locations'].sum()):,}"
)

print(
    f"Validated UNKNOWN locations: "
    f"{int(summary['validated_unknown_locations'].sum()):,}"
)

difference = (
    summary["unknown_locations"]
    - summary["validated_unknown_locations"]
).abs().sum()

print(
    f"Total district-count difference: "
    f"{int(difference):,}"
)


# ============================================================
# TOP AGRICULTURE-CONTEXT DISTRICTS
# ============================================================

print("\n")
print("-" * 75)
print("TOP DISTRICTS BY AGRICULTURE CONTEXT")
print("-" * 75)

print(
    summary[
        [
            "district_name",
            "unknown_locations",
            "unknown_density_per_100km2",
            "dw_crops_mean",
            "dw_crops_ge_50_pct",
            "inside_forest_pct",
            "unknown_frp10_repeated",
            "agriculture_context_signal",
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# TOP FOREST-CONTEXT DISTRICTS
# ============================================================

print("\n")
print("-" * 75)
print("TOP DISTRICTS BY FOREST CONTEXT")
print("-" * 75)

print(
    summary[
        [
            "district_name",
            "unknown_locations",
            "unknown_density_per_100km2",
            "dw_crops_mean",
            "dw_crops_ge_50_pct",
            "inside_forest_pct",
            "unknown_frp10_repeated",
            "forest_context_signal",
        ]
    ]
    .sort_values(
        "forest_context_signal",
        ascending=False
    )
    .head(15)
    .to_string(index=False)
)


# ============================================================
# TOP UNKNOWN DENSITY
# ============================================================

print("\n")
print("-" * 75)
print("TOP DISTRICTS BY UNKNOWN FIRMS DENSITY")
print("-" * 75)

print(
    summary[
        [
            "district_name",
            "unknown_locations",
            "unknown_density_per_100km2",
            "dw_crops_mean",
            "dw_crops_ge_50_pct",
            "inside_forest_pct",
        ]
    ]
    .sort_values(
        "unknown_density_per_100km2",
        ascending=False
    )
    .head(15)
    .to_string(index=False)
)


# ============================================================
# OUTPUT FILES
# ============================================================

print("\n")
print("Output files:")
print(OUTPUT_FILE)
print(SUMMARY_FILE)

print("\n" + "=" * 75)