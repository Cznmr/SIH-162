import math
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "noaa20_telangana_12months.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spatial_fire_behavior_v2.csv"
)

GRID_SIZE_DEG = 0.01

# Improved spatial scales
RADII_KM = [1.5, 3.0, 5.0]


# ============================================================
# DISTANCE
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    dlat = lat2 - lat1
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 6371.0088 * 2 * math.asin(
        math.sqrt(a)
    )


# ============================================================
# GRID FUNCTIONS
# ============================================================
def lat_to_idx(lat):
    return int(np.floor(lat / GRID_SIZE_DEG))

def lon_to_idx(lon):
    return int(np.floor(lon / GRID_SIZE_DEG))


def idx_to_lat(idx):
    return idx * GRID_SIZE_DEG


def idx_to_lon(idx):
    return idx * GRID_SIZE_DEG


# ============================================================
# BUILD NEIGHBOR OFFSETS
# ============================================================

def build_offsets(radius_km):

    # 0.01 degree is roughly 1.1 km
    max_cells = int(
        math.ceil(radius_km / 1.0)
    ) + 2

    offsets = []

    for di in range(
        -max_cells,
        max_cells + 1
    ):

        for dj in range(
            -max_cells,
            max_cells + 1
        ):

            if di == 0 and dj == 0:
                continue

            # Conservative approximate distance
            approx_distance = math.sqrt(
                (di * 1.1132) ** 2
                + (dj * 1.10) ** 2
            )

            if approx_distance <= (
                radius_km + 1.5
            ):

                offsets.append(
                    (di, dj)
                )

    return offsets


OFFSETS = {
    radius: build_offsets(radius)
    for radius in RADII_KM
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SPATIAL FIRE BEHAVIOR - V2")
print("Grid hashing + local persistence + expansion")
print("=" * 70)

print("\n[1/8] Loading FIRMS data...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

df["acq_date"] = pd.to_datetime(
    df["acq_date"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "latitude",
        "longitude",
        "acq_date"
    ]
).copy()

print(
    f"Loaded detections: {len(df):,}"
)


# ============================================================
# CREATE GRID
# ============================================================

print("\n[2/8] Creating 0.01 degree grid...")

df["lat_idx"] = (
    df["latitude"]
    .apply(lat_to_idx)
)

df["lon_idx"] = (
    df["longitude"]
    .apply(lon_to_idx)
)

daily_cells = (
    df[
        [
            "acq_date",
            "lat_idx",
            "lon_idx"
        ]
    ]
    .drop_duplicates()
    .sort_values("acq_date")
)

print(
    f"Unique daily cells: "
    f"{len(daily_cells):,}"
)


# ============================================================
# DAILY SPATIAL HASH
# ============================================================

print("\n[3/8] Building daily spatial hash...")

daily_sets = {}

for date, group in daily_cells.groupby(
    "acq_date"
):

    daily_sets[date] = set(
        zip(
            group["lat_idx"].astype(int),
            group["lon_idx"].astype(int)
        )
    )

dates = sorted(
    daily_sets.keys()
)

print(
    f"Observation days: {len(dates)}"
)


# ============================================================
# HELPER: FIND LOCAL CELLS
# ============================================================

def get_local_cells(
    center,
    active_set,
    radius,
    lat,
    lon
):

    lat_idx, lon_idx = center

    local_cells = []
    distances = []

    for di, dj in OFFSETS[radius]:

        candidate = (
            lat_idx + di,
            lon_idx + dj
        )

        if candidate not in active_set:
            continue

        candidate_lat = (
            idx_to_lat(candidate[0])
        )

        candidate_lon = (
            idx_to_lon(candidate[1])
        )

        distance = haversine_km(
            lat,
            lon,
            candidate_lat,
            candidate_lon
        )

        if distance <= radius:

            local_cells.append(
                candidate
            )

            distances.append(
                distance
            )

    return local_cells, distances


# ============================================================
# MAIN PROCESSING
# ============================================================

print(
    "\n[4/8] Calculating local spatial behavior..."
)

results = []

total_cells = len(daily_cells)
processed = 0


for date in dates:

    current_set = daily_sets[date]

    previous_date = (
        date - pd.Timedelta(days=1)
    )

    previous_set = daily_sets.get(
        previous_date,
        set()
    )

    # New cells appearing today
    new_today = (
        current_set - previous_set
    )

    # Cells present both days
    continuing_today = (
        current_set & previous_set
    )

    for lat_idx, lon_idx in current_set:

        center = (
            lat_idx,
            lon_idx
        )

        lat = idx_to_lat(lat_idx)
        lon = idx_to_lon(lon_idx)

        row = {
            "acq_date": date,
            "latitude": lat,
            "longitude": lon,
            "lat_idx": lat_idx,
            "lon_idx": lon_idx,
        }

        # ====================================================
        # MULTIPLE SPATIAL SCALES
        # ====================================================

        for radius in RADII_KM:

            current_neighbors, current_distances = (
                get_local_cells(
                    center,
                    current_set,
                    radius,
                    lat,
                    lon
                )
            )

            previous_neighbors, previous_distances = (
                get_local_cells(
                    center,
                    previous_set,
                    radius,
                    lat,
                    lon
                )
            )

            current_neighbors = set(
                current_neighbors
            )

            previous_neighbors = set(
                previous_neighbors
            )

            # -----------------------------------------------
            # Current population
            # -----------------------------------------------

            current_population = len(
                current_neighbors
            )

            previous_population = len(
                previous_neighbors
            )

            row[
                f"local_population_{radius:g}km"
            ] = current_population

            row[
                f"previous_local_population_{radius:g}km"
            ] = previous_population

            # -----------------------------------------------
            # Population change
            # -----------------------------------------------

            population_change = (
                current_population
                - previous_population
            )

            row[
                f"local_population_change_{radius:g}km"
            ] = population_change

            # -----------------------------------------------
            # Expansion rate
            # -----------------------------------------------

            if previous_population > 0:

                expansion_rate = (
                    population_change
                    / previous_population
                )

            else:

                expansion_rate = float(
                    current_population
                )

            row[
                f"local_expansion_rate_{radius:g}km"
            ] = expansion_rate

            # -----------------------------------------------
            # New nearby cells
            # -----------------------------------------------

            new_neighbors = (
                current_neighbors
                & new_today
            )

            row[
                f"new_neighbor_count_{radius:g}km"
            ] = len(
                new_neighbors
            )

            # -----------------------------------------------
            # Continuing nearby cells
            # -----------------------------------------------

            continuing_neighbors = (
                current_neighbors
                & previous_set
            )

            row[
                f"continuing_neighbor_count_{radius:g}km"
            ] = len(
                continuing_neighbors
            )

            # -----------------------------------------------
            # Local persistence
            # -----------------------------------------------

            if current_population > 0:

                local_persistence = (
                    len(
                        current_neighbors
                        & previous_neighbors
                    )
                    / current_population
                )

            else:

                local_persistence = 0.0

            row[
                f"local_persistence_{radius:g}km"
            ] = local_persistence

            # -----------------------------------------------
            # Local radius
            # -----------------------------------------------

            if current_distances:

                row[
                    f"local_radius_{radius:g}km"
                ] = max(
                    current_distances
                )

            else:

                row[
                    f"local_radius_{radius:g}km"
                ] = 0.0

            # =================================================
            # NEW-CELL DIRECTION
            # =================================================

            directions = {
                "N": 0,
                "S": 0,
                "E": 0,
                "W": 0,
                "NE": 0,
                "NW": 0,
                "SE": 0,
                "SW": 0,
            }

            new_distances = []

            for candidate in new_neighbors:

                c_lat_idx, c_lon_idx = (
                    candidate
                )

                dlat = (
                    c_lat_idx
                    - lat_idx
                )

                dlon = (
                    c_lon_idx
                    - lon_idx
                )

                candidate_lat = (
                    idx_to_lat(c_lat_idx)
                )

                candidate_lon = (
                    idx_to_lon(c_lon_idx)
                )

                distance = haversine_km(
                    lat,
                    lon,
                    candidate_lat,
                    candidate_lon
                )

                new_distances.append(
                    distance
                )

                if dlat > 0 and dlon > 0:
                    directions["NE"] += 1

                elif dlat > 0 and dlon < 0:
                    directions["NW"] += 1

                elif dlat < 0 and dlon > 0:
                    directions["SE"] += 1

                elif dlat < 0 and dlon < 0:
                    directions["SW"] += 1

                elif dlat > 0:
                    directions["N"] += 1

                elif dlat < 0:
                    directions["S"] += 1

                elif dlon > 0:
                    directions["E"] += 1

                elif dlon < 0:
                    directions["W"] += 1

            for direction, count in (
                directions.items()
            ):

                row[
                    f"{direction.lower()}_activity_{radius:g}km"
                ] = count

            total_directional_activity = sum(
                directions.values()
            )

            if total_directional_activity > 0:

                dominant_direction = max(
                    directions,
                    key=directions.get
                )

                directional_consistency = (
                    max(directions.values())
                    / total_directional_activity
                )

            else:

                dominant_direction = "NONE"
                directional_consistency = 0.0

            row[
                f"dominant_direction_{radius:g}km"
            ] = dominant_direction

            row[
                f"directional_consistency_{radius:g}km"
            ] = directional_consistency

            # -----------------------------------------------
            # New-cell movement
            # -----------------------------------------------

            if new_distances:

                row[
                    f"mean_new_distance_{radius:g}km"
                ] = np.mean(
                    new_distances
                )

                row[
                    f"max_new_distance_{radius:g}km"
                ] = np.max(
                    new_distances
                )

            else:

                row[
                    f"mean_new_distance_{radius:g}km"
                ] = 0.0

                row[
                    f"max_new_distance_{radius:g}km"
                ] = 0.0

        # ====================================================
        # EVENT STATE
        # ====================================================

        row["is_new_today"] = int(
            center in new_today
        )

        row["was_active_yesterday"] = int(
            center in continuing_today
        )

        row["previous_date"] = (
            previous_date.date()
        )

        row[
            "consecutive_day_comparison"
        ] = int(
            previous_date in daily_sets
        )

        results.append(row)

    processed += len(current_set)

    if (
        processed % 5000 < len(current_set)
        or processed == total_cells
    ):

        print(
            f"    Processed "
            f"{processed:,}/{total_cells:,} cells"
        )


# ============================================================
# DATAFRAME
# ============================================================

print(
    "\n[5/8] Creating output dataframe..."
)

result_df = pd.DataFrame(results)

result_df = result_df.sort_values(
    [
        "acq_date",
        "lat_idx",
        "lon_idx"
    ]
).reset_index(drop=True)

print(
    f"Output rows: {len(result_df):,}"
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n[6/8] Spatial behavior summary..."
)

summary_columns = [
    "local_population_1.5km",
    "local_population_3km",
    "local_population_5km",
    "new_neighbor_count_1.5km",
    "new_neighbor_count_3km",
    "new_neighbor_count_5km",
    "local_persistence_1.5km",
    "local_persistence_3km",
    "local_persistence_5km",
    "local_population_change_5km",
    "local_expansion_rate_5km",
]

print(
    result_df[
        summary_columns
    ].describe()
)


# ============================================================
# SAVE
# ============================================================

print(
    "\n[7/8] Saving output..."
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nSaved:\n{OUTPUT_FILE}"
)


# ============================================================
# FINAL CHECK
# ============================================================

print(
    "\n[8/8] Final checks..."
)

print(
    f"Rows: {len(result_df):,}"
)

print(
    f"Unique dates: "
    f"{result_df['acq_date'].nunique()}"
)

print(
    f"Unique grid cells: "
    f"{result_df[['lat_idx', 'lon_idx']].drop_duplicates().shape[0]:,}"
)

print(
    "\n" + "=" * 70
)

print(
    "SPATIAL FIRE BEHAVIOR V2 COMPLETE"
)

print(
    "=" * 70
)