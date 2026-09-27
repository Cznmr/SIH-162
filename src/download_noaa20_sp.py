import os
import requests
import pandas as pd
from datetime import date, timedelta

# ============================================================
# CONFIG
# ============================================================

MAP_KEY = os.getenv("FIRMS_MAP_KEY")

WEST = 77.0
SOUTH = 15.5
EAST = 81.6
NORTH = 20.2

AREA = f"{WEST},{SOUTH},{EAST},{NORTH}"

OUTPUT_DIR = "data/raw/firms/check_gap"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FIRMS CHECK FUNCTION
# ============================================================

def check_date(check_date, source):

    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{MAP_KEY}/{source}/{AREA}/1/{check_date}"
    )

    response = requests.get(
        url,
        timeout=120
    )

    response.raise_for_status()

    # NASA returns CSV
    from io import StringIO

    df = pd.read_csv(StringIO(response.text))

    return df

# ============================================================
# CHECK JUNE 30 - JULY 5
# ============================================================

dates_sources = [
    (date(2026, 6, 30), "VIIRS_NOAA20_SP"),
    (date(2026, 7, 1), "VIIRS_NOAA20_NRT"),
    (date(2026, 7, 2), "VIIRS_NOAA20_NRT"),
    (date(2026, 7, 3), "VIIRS_NOAA20_NRT"),
    (date(2026, 7, 4), "VIIRS_NOAA20_NRT"),
    (date(2026, 7, 5), "VIIRS_NOAA20_NRT"),
]


all_detections = []

print("\n==============================================")
print("CHECKING JUNE 30 - JULY 5, 2026")
print("==============================================\n")

for check_date_value, source in dates_sources:

    print(f"Checking {check_date_value} | {source} ...")

    try:
        df = check_date(check_date_value, source)

        count = len(df)

        print(f"  Detections: {count}")

        if count > 0:
            df["source_dataset"] = source
            all_detections.append(df)

    except Exception as e:
        print(f"  ERROR: {e}")


# ============================================================
# SAVE ONLY IF DETECTIONS EXIST
# ============================================================

print("\n==============================================")

if all_detections:

    gap_df = pd.concat(all_detections, ignore_index=True)

    output_file = os.path.join(
        OUTPUT_DIR,
        "noaa20_gap_check_2026-06-30_to_2026-07-05.csv"
    )

    gap_df.to_csv(output_file, index=False)

    print("DETECTIONS FOUND")
    print(f"Total detections : {len(gap_df)}")
    print(f"Saved to         : {output_file}")

    if "acq_date" in gap_df.columns:
        print("\nDates with detections:")
        print(
            gap_df["acq_date"]
            .value_counts()
            .sort_index()
        )

else:

    print("NO DETECTIONS FOUND")
    print("No CSV file was created.")

print("==============================================")