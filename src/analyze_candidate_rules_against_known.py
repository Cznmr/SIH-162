import pandas as pd
import numpy as np


MASTER_FILE = "data/training/master_feature_table_final.csv"
SEED_FILE = "data/training/seed_labels.csv"


print("=" * 70)
print("CANDIDATE RULE VALIDATION AGAINST KNOWN SEEDS")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1/7] Loading master feature table...")
master = pd.read_csv(MASTER_FILE, low_memory=False)
print(f"Master rows: {len(master):,}")

print("\n[2/7] Loading seed labels...")
seed = pd.read_csv(SEED_FILE)
print(f"Seed rows: {len(seed):,}")


# ============================================================
# 2. VALIDATE GRID KEYS
# ============================================================

print("\n[3/7] Checking grid keys...")

grid_columns = [
    "lat_grid",
    "lon_grid",
]

for col in grid_columns:
    if col not in master.columns:
        raise ValueError(
            f"Master file missing column: {col}"
        )

    if col not in seed.columns:
        raise ValueError(
            f"Seed file missing column: {col}"
        )

print("Using existing lat_grid/lon_grid keys.")


# ============================================================
# 3. MERGE MASTER FEATURES WITH LABELS
# ============================================================

print("\n[4/7] Attaching seed labels...")

seed_small = seed[
    [
        "lat_grid",
        "lon_grid",
        "seed_label",
    ]
].copy()

df = master.merge(
    seed_small,
    on=["lat_grid", "lon_grid"],
    how="inner",
    validate="one_to_one",
)

print(f"Matched labeled locations: {len(df):,}")

if len(df) != 2182:
    print(
        f"WARNING: Expected approximately 2,182 known seeds "
        f"but found {len(df):,}."
    )


# ============================================================
# 4. SHOW EXISTING LABEL DISTRIBUTION
# ============================================================

print("\n[5/7] Existing known-label distribution...")

print()

label_counts = (
    df["seed_label"]
    .value_counts()
    .sort_index()
)

for label, count in label_counts.items():
    pct = count / len(df) * 100

    print(
        f"{label:<30}"
        f"{count:>7,} "
        f"({pct:>6.2f}%)"
    )


# ============================================================
# 5. CREATE PROPOSED CANDIDATE RULES
# ============================================================

print("\n[6/7] Testing proposed candidate rules...")


# ------------------------------------------------------------
# BASIC EVIDENCE FLAGS
# ------------------------------------------------------------

df["frp_5"] = df["max_frp"] >= 5
df["frp_10"] = df["max_frp"] >= 10
df["frp_15"] = df["max_frp"] >= 15
df["frp_30"] = df["max_frp"] >= 30

df["detections_2"] = df["detection_count"] >= 2
df["detections_3"] = df["detection_count"] >= 3

df["active_days_3"] = df["active_days"] >= 3
df["active_days_5"] = df["active_days"] >= 5

df["span_30"] = df["observation_span_days"] >= 30
df["span_60"] = df["observation_span_days"] >= 60

df["persistent_180_5"] = (
    df["active_days_180d"] >= 5
)


# ------------------------------------------------------------
# DYNAMIC WORLD
# ------------------------------------------------------------

df["vegetation_70"] = (
    df["dw_vegetation_score"] >= 0.70
)


# ------------------------------------------------------------
# OSM CONTEXT
# ------------------------------------------------------------

df["industrial_area_5km"] = (
    df["nearest_industrial_area_km"] <= 5
)

df["industrial_works_5km"] = (
    df["nearest_industrial_works_km"] <= 5
)

df["power_5km"] = (
    df["nearest_power_infrastructure_km"] <= 5
)

df["storage_5km"] = (
    df["nearest_storage_tank_km"] <= 5
)


df["industrial_context_5km"] = (
    df["industrial_area_5km"]
    | df["industrial_works_5km"]
    | df["power_5km"]
    | df["storage_5km"]
)


df["no_industrial_context_5km"] = (
    ~df["industrial_context_5km"]
)


# ============================================================
# PROPOSED RULES
# ============================================================

rules = {

    # Proposed wildfire candidate
    "WILDFIRE_V1": (
        df["frp_10"]
        & df["detections_2"]
        & df["vegetation_70"]
        & df["no_industrial_context_5km"]
    ),

    # More conservative wildfire
    "WILDFIRE_V2": (
        df["frp_10"]
        & df["detections_3"]
        & df["vegetation_70"]
        & df["no_industrial_context_5km"]
    ),

    # Proposed industrial fire
    "INDUSTRIAL_V1": (
        df["frp_10"]
        & df["detections_2"]
        & df["industrial_context_5km"]
    ),

    # More conservative industrial fire
    "INDUSTRIAL_V2": (
        df["frp_10"]
        & df["detections_3"]
        & df["industrial_context_5km"]
    ),

    # Proposed persistent thermal source
    "PERSISTENT_V1": (
        df["persistent_180_5"]
        & df["span_60"]
    ),

    # More conservative persistent source
    "PERSISTENT_V2": (
        df["active_days_5"]
        & df["span_60"]
        & df["industrial_context_5km"]
    ),

    # General recurring thermal activity
    "MODERATE_THERMAL_V1": (
        df["frp_5"]
        & df["detections_2"]
    ),

    # More conservative recurring thermal
    "MODERATE_THERMAL_V2": (
        df["frp_5"]
        & df["detections_3"]
        & df["span_30"]
    ),
}


# ============================================================
# HELPER
# ============================================================

def print_rule_results(rule_name, mask):

    subset = df[mask].copy()

    total = len(subset)

    print("\n" + "-" * 70)
    print(rule_name)
    print("-" * 70)

    print(
        f"Locations matched: {total:,} "
        f"({total / len(df) * 100:.2f}% of known seeds)"
    )

    if total == 0:
        print("No known seeds matched this rule.")
        return

    counts = (
        subset["seed_label"]
        .value_counts()
        .sort_index()
    )

    print("\nExisting labels among matched locations:")

    for label, count in counts.items():

        pct = count / total * 100

        print(
            f"  {label:<28}"
            f"{count:>7,} "
            f"({pct:>6.2f}%)"
        )

    print("\nPurity relative to proposed class:")

    proposed_class = None

    if rule_name.startswith("WILDFIRE"):
        proposed_class = "WILDFIRE"

    elif rule_name.startswith("INDUSTRIAL"):
        proposed_class = "INDUSTRIAL_FIRE"

    elif rule_name.startswith("PERSISTENT"):
        proposed_class = "PERSISTENT_THERMAL_SOURCE"

    elif rule_name.startswith("MODERATE"):
        proposed_class = "OTHER"

    if proposed_class in counts.index:

        purity = (
            counts[proposed_class]
            / total
            * 100
        )

        print(
            f"  {proposed_class}: "
            f"{purity:.2f}%"
        )

    else:

        print(
            f"  {proposed_class}: 0.00%"
        )


# ============================================================
# RUN EACH RULE
# ============================================================

for rule_name, mask in rules.items():

    print_rule_results(
        rule_name,
        mask
    )


# ============================================================
# 6. CONFUSION-STYLE ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("CANDIDATE RULE OVERLAP WITH EXISTING CLASSES")
print("=" * 70)


# ------------------------------------------------------------
# Wildfire V1
# ------------------------------------------------------------

wildfire_v1 = rules["WILDFIRE_V1"]

print("\nWILDFIRE_V1 by existing class:")

print(
    pd.crosstab(
        df.loc[wildfire_v1, "seed_label"],
        columns="count"
    )
)


# ------------------------------------------------------------
# Industrial V1
# ------------------------------------------------------------

industrial_v1 = rules["INDUSTRIAL_V1"]

print("\nINDUSTRIAL_V1 by existing class:")

print(
    pd.crosstab(
        df.loc[industrial_v1, "seed_label"],
        columns="count"
    )
)


# ------------------------------------------------------------
# Persistent V1
# ------------------------------------------------------------

persistent_v1 = rules["PERSISTENT_V1"]

print("\nPERSISTENT_V1 by existing class:")

print(
    pd.crosstab(
        df.loc[persistent_v1, "seed_label"],
        columns="count"
    )
)


# ------------------------------------------------------------
# Moderate thermal V1
# ------------------------------------------------------------

moderate_v1 = rules["MODERATE_THERMAL_V1"]

print("\nMODERATE_THERMAL_V1 by existing class:")

print(
    pd.crosstab(
        df.loc[moderate_v1, "seed_label"],
        columns="count"
    )
)


# ============================================================
# 7. OVERLAP BETWEEN PROPOSED RULES
# ============================================================

print("\n")
print("=" * 70)
print("RULE OVERLAPS")
print("=" * 70)

report_pairs = [
    (
        "Wildfire V1 + Industrial V1",
        wildfire_v1 & industrial_v1
    ),
    (
        "Wildfire V1 + Persistent V1",
        wildfire_v1 & persistent_v1
    ),
    (
        "Industrial V1 + Persistent V1",
        industrial_v1 & persistent_v1
    ),
    (
        "Wildfire V1 + Industrial V1 + Persistent V1",
        wildfire_v1 & industrial_v1 & persistent_v1
    ),
]


for name, mask in report_pairs:

    count = int(mask.sum())

    print(
        f"{name:<55}"
        f"{count:>7,}"
    )


# ============================================================
# SAVE VALIDATION RESULTS
# ============================================================

output_file = (
    "data/training/"
    "candidate_rule_validation_known.csv"
)

output_columns = [
    "lat_grid",
    "lon_grid",
    "seed_label",
    "max_frp",
    "mean_frp",
    "detection_count",
    "active_days",
    "active_days_180d",
    "observation_span_days",
    "persistence_score",
    "dw_vegetation_score",
    "dw_built",
    "nearest_industrial_area_km",
    "nearest_industrial_works_km",
    "nearest_power_infrastructure_km",
    "nearest_storage_tank_km",
    "nearest_relevant_osm_km",
]


# Add rule flags
for rule_name in rules:

    output_columns.append(
        rule_name
    )

    df[rule_name] = (
        rules[rule_name]
        .astype(int)
    )


output_columns = [
    col
    for col in output_columns
    if col in df.columns
]


df[output_columns].to_csv(
    output_file,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

print(
    f"\nSaved validation results to:\n"
    f"{output_file}"
)

print(
    "\nIMPORTANT:"
    "\nThese rules have NOT changed any training labels."
    "\nThis script only measures how the proposed rules"
    "\nbehave against the existing known seed labels."
)