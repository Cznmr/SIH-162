import os
import requests
import pandas as pd
from datetime import date, timedelta
import time

MAP_KEY = os.getenv("FIRMS_MAP_KEY")

if not MAP_KEY:
    raise ValueError(
        "FIRMS_MAP_KEY environment variable is not set."
    )

# Telangana bounding box
WEST = 77.0
SOUTH = 15.5
EAST = 81.6
NORTH = 20.2

AREA = f"{WEST},{SOUTH},{EAST},{NORTH}"

# 6 months of historical data
END_DATE = date.today() - timedelta(days=7)
START_DATE = END_DATE - timedelta(days=180)

SOURCE = "VIIRS_NOAA20_SP"

OUTPUT_DIR = "data/raw/firms"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "viirs_noaa20_sp_telangana_6months.csv"
)


def download_period(start_date, days=5):

    url = (
        f"https://firms.modaps.eosdis.nasa.gov/"
        f"api/area/csv/"
        f"{MAP_KEY}/"
        f"{SOURCE}/"
        f"{AREA}/"
        f"{days}/"
        f"{start_date}"
    )

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    from io import StringIO
    return pd.read_csv(StringIO(response.text))


if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_data = []

    current_date = START_DATE

    print("================================")
    print("TELANGANA HISTORICAL FIRMS DATA")
    print("================================")
    print(f"Source     : {SOURCE}")
    print(f"Start date : {START_DATE}")
    print(f"End date   : {END_DATE}")
    print(f"Area       : {AREA}")
    print("================================")

    while current_date <= END_DATE:

        remaining_days = (END_DATE - current_date).days + 1
        request_days = min(5, remaining_days)

        print(
            f"Downloading {current_date} "
            f"({request_days} days)..."
        )

        try:

            df = download_period(
                current_date,
                request_days
            )

            if not df.empty:

                all_data.append(df)

                print(
                    f"  ✓ {len(df)} detections"
                )

            else:

                print("  - No detections")

        except Exception as error:

            print(
                f"  ❌ Error: {error}"
            )

        current_date += timedelta(
            days=request_days
        )

        # Small pause between requests
        time.sleep(1)


    if all_data:

        combined = pd.concat(
            all_data,
            ignore_index=True
        )

        # Remove duplicate detections
        combined = combined.drop_duplicates()

        combined.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print("\n================================")
        print("DOWNLOAD COMPLETE")
        print("================================")
        print(
            f"Total detections: {len(combined)}"
        )
        print(
            f"Saved to: {OUTPUT_FILE}"
        )

    else:

        print("\nNo historical data found.")