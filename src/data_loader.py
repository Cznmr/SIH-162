import os
import requests
import pandas as pd

MAP_KEY = os.getenv("FIRMS_MAP_KEY")

if not MAP_KEY:
    raise ValueError(
        "FIRMS_MAP_KEY environment variable is not set."
    )

# Telangana region
WEST = 77.0
SOUTH = 15.5
EAST = 81.6
NORTH = 20.2

AREA = f"{WEST},{SOUTH},{EAST},{NORTH}"
DAYS = 5

SOURCES = [
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT"
]

OUTPUT_DIR = "data/raw/firms"


def download_firms(source):

    url = (
        f"https://firms.modaps.eosdis.nasa.gov/"
        f"api/area/csv/"
        f"{MAP_KEY}/"
        f"{source}/"
        f"{AREA}/"
        f"{DAYS}"
    )

    print("\n================================")
    print(f"Downloading {source}")
    print("================================")

    print(f"Region : Telangana")
    print(f"Area   : {AREA}")
    print(f"Days   : {DAYS}")

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{source.lower()}_telangana_5days.csv"
    )

    with open(output_file, "wb") as file:
        file.write(response.content)

    print(f"✓ Saved: {output_file}")

    df = pd.read_csv(output_file)

    print(f"✓ Records: {len(df)}")

    return df


if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_data = []

    for source in SOURCES:

        try:

            df = download_firms(source)

            df["source_dataset"] = source

            all_data.append(df)

        except Exception as error:

            print(f"❌ Error downloading {source}")
            print(error)

    if all_data:

        combined = pd.concat(
            all_data,
            ignore_index=True
        )

        combined_file = os.path.join(
            OUTPUT_DIR,
            "firms_telangana_5days.csv"
        )

        combined.to_csv(
            combined_file,
            index=False
        )

        print("\n================================")
        print("TELANGANA FIRMS DATA COMPLETE")
        print("================================")

        print(f"Total records: {len(combined)}")

        print(
            f"Saved to: {combined_file}"
        )

    else:

        print("No data was downloaded.")