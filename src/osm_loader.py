import os
import time
import requests
import pandas as pd


# ============================================================
# TELANGANA BOUNDING BOX
# ============================================================

SOUTH = 15.5
WEST = 77.0
NORTH = 20.2
EAST = 81.6


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = "data/raw/osm"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "telangana_industrial_facilities.csv"
)


# ============================================================
# OVERPASS QUERY
# ============================================================

QUERY = f"""
[out:json][timeout:300];

(
  node["landuse"="industrial"]({SOUTH},{WEST},{NORTH},{EAST});
  way["landuse"="industrial"]({SOUTH},{WEST},{NORTH},{EAST});
  relation["landuse"="industrial"]({SOUTH},{WEST},{NORTH},{EAST});

  node["industrial"]({SOUTH},{WEST},{NORTH},{EAST});
  way["industrial"]({SOUTH},{WEST},{NORTH},{EAST});
  relation["industrial"]({SOUTH},{WEST},{NORTH},{EAST});

  node["power"="plant"]({SOUTH},{WEST},{NORTH},{EAST});
  way["power"="plant"]({SOUTH},{WEST},{NORTH},{EAST});
  relation["power"="plant"]({SOUTH},{WEST},{NORTH},{EAST});

  node["man_made"="works"]({SOUTH},{WEST},{NORTH},{EAST});
  way["man_made"="works"]({SOUTH},{WEST},{NORTH},{EAST});
  relation["man_made"="works"]({SOUTH},{WEST},{NORTH},{EAST});

  node["man_made"="storage_tank"]({SOUTH},{WEST},{NORTH},{EAST});
  way["man_made"="storage_tank"]({SOUTH},{WEST},{NORTH},{EAST});

  node["industrial"="oil"]({SOUTH},{WEST},{NORTH},{EAST});
  way["industrial"="oil"]({SOUTH},{WEST},{NORTH},{EAST});

  node["industrial"="gas"]({SOUTH},{WEST},{NORTH},{EAST});
  way["industrial"="gas"]({SOUTH},{WEST},{NORTH},{EAST});

  node["landuse"="quarry"]({SOUTH},{WEST},{NORTH},{EAST});
  way["landuse"="quarry"]({SOUTH},{WEST},{NORTH},{EAST});

  node["landuse"="mine"]({SOUTH},{WEST},{NORTH},{EAST});
  way["landuse"="mine"]({SOUTH},{WEST},{NORTH},{EAST});
);

out center tags;
"""


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    url = "https://overpass-api.de/api/interpreter"

    # Identify our application properly.
    headers = {
        "User-Agent": (
            "ENTRO-26162-SIH-Prototype/1.0 "
            "(OpenStreetMap Overpass data collection)"
        ),
        "Referer": "https://www.openstreetmap.org/"
    }

    print("================================")
    print("DOWNLOADING OSM INDUSTRIAL DATA")
    print("================================")
    print(f"Region : Telangana")
    print(f"BBox   : {SOUTH},{WEST},{NORTH},{EAST}")
    print()

    # --------------------------------------------------------
    # Try request
    # --------------------------------------------------------

    try:

        response = requests.post(
            url,
            data={"data": QUERY},
            headers=headers,
            timeout=600
        )

        print(f"HTTP Status: {response.status_code}")

        # Show server message if request failed
        if response.status_code != 200:

            print("\nOverpass server response:")
            print(response.text[:2000])

        response.raise_for_status()

    except requests.exceptions.RequestException as error:

        print("\n❌ Overpass request failed.")
        print(error)

        # Overpass recommends waiting after 406/429.
        if response is not None and response.status_code in [406, 429]:

            print()
            print(
                "The Overpass server requested a pause."
            )
            print(
                "Wait about 30 seconds before trying again."
            )

        return

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        data = response.json()

    except ValueError:

        print("\n❌ Server did not return valid JSON.")
        print(response.text[:2000])

        return

    elements = data.get("elements", [])

    print()
    print(
        f"OSM elements received: {len(elements)}"
    )

    if not elements:

        print("\nNo OSM facilities found.")

        return

    # --------------------------------------------------------
    # Convert OSM elements into records
    # --------------------------------------------------------

    records = []

    for element in elements:

        tags = element.get("tags", {})

        latitude = None
        longitude = None

        # -------------------------------
        # Node
        # -------------------------------

        if "lat" in element and "lon" in element:

            latitude = element["lat"]
            longitude = element["lon"]

        # -------------------------------
        # Way / Relation center
        # -------------------------------

        elif "center" in element:

            center = element["center"]

            latitude = center.get("lat")
            longitude = center.get("lon")

        # Skip elements without coordinates
        if latitude is None or longitude is None:

            continue

        records.append({

            "osm_type":
                element.get("type"),

            "osm_id":
                element.get("id"),

            "latitude":
                latitude,

            "longitude":
                longitude,

            "name":
                tags.get("name"),

            "industrial":
                tags.get("industrial"),

            "landuse":
                tags.get("landuse"),

            "power":
                tags.get("power"),

            "man_made":
                tags.get("man_made"),

            "operator":
                tags.get("operator"),

            "description":
                tags.get("description")

        })

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(records)

    if df.empty:

        print(
            "\n❌ OSM elements were received, "
            "but none contained usable coordinates."
        )

        return

    # --------------------------------------------------------
    # Remove duplicate OSM objects
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=[
            "osm_type",
            "osm_id"
        ]
    )

    duplicates_removed = before - len(df)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("================================")
    print("OSM DOWNLOAD COMPLETE")
    print("================================")

    print(
        f"Facilities : {len(df)}"
    )

    print(
        f"Duplicates removed : "
        f"{duplicates_removed}"
    )

    print(
        f"Saved to : {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Category statistics
    # --------------------------------------------------------

    print()
    print("================================")
    print("FACILITY CATEGORIES")
    print("================================")

    print(
        f"Industrial tagged : "
        f"{df['industrial'].notna().sum()}"
    )

    print(
        f"Industrial landuse : "
        f"{df['landuse'].notna().sum()}"
    )

    print(
        f"Power plants : "
        f"{df['power'].notna().sum()}"
    )

    print(
        f"Man-made works : "
        f"{df['man_made'].notna().sum()}"
    )

    print()

    # --------------------------------------------------------
    # Show some sample facilities
    # --------------------------------------------------------

    print("Sample facilities:")

    columns_to_show = [
        "name",
        "latitude",
        "longitude",
        "industrial",
        "landuse",
        "power",
        "man_made"
    ]

    print(
        df[columns_to_show]
        .head(10)
        .to_string(index=False)
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()