import json
import os
import time
import requests

# ============================================================
# Telangana Forest Boundary downloader
# Uses spatial tiling instead of resultOffset pagination.
# ============================================================

QUERY_URL = (
    "https://tgrac.telangana.gov.in/arcgis/rest/services/"
    "AdministrativeInfoSystem_Folder/"
    "Administrative_Information_System/MapServer/31/query"
)

OUTPUT = r"data\raw\forest_boundary_telangana.json"
CHECKPOINT = r"data\raw\forest_boundary_checkpoint.json"

# Layer extent from the official ArcGIS metadata.
# Spatial reference: EPSG:32644
XMIN = 115299.24932408438
YMIN = 1769995.3953576554
XMAX = 533852.4415871539
YMAX = 2201389.140331445

# Start with a 4 x 4 grid.
# Tiles that are too large are automatically subdivided.
INITIAL_DIVISIONS = 4

session = requests.Session()


def query_tile(xmin, ymin, xmax, ymax):
    """
    Query one spatial tile.
    Returns the ArcGIS JSON response.
    """

    geometry = {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "spatialReference": {"wkid": 32644}
    }

    params = {
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "true",
        "geometry": json.dumps(geometry, separators=(",", ":")),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "32644",
        "spatialRel": "esriSpatialRelIntersects",
        "outSR": "32644",
        "f": "json"
    }

    for attempt in range(5):
        try:
            response = session.get(
                QUERY_URL,
                params=params,
                timeout=120
            )

            response.raise_for_status()
            data = response.json()

            if "error" in data:
                print("ArcGIS error:", data["error"])
                time.sleep(3)
                continue

            return data

        except Exception as e:
            print(f"Request failed (attempt {attempt + 1}/5): {e}")
            time.sleep(3)

    raise RuntimeError("Tile request failed after 5 attempts.")


def split_tile(xmin, ymin, xmax, ymax):
    """Split one tile into four smaller tiles."""

    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2

    return [
        (xmin, ymin, xmid, ymid),
        (xmid, ymin, xmax, ymid),
        (xmin, ymid, xmid, ymax),
        (xmid, ymid, xmax, ymax),
    ]


def save_checkpoint(features):
    """Save current unique features."""

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)

    data = {
        "geometryType": "esriGeometryPolygon",
        "spatialReference": {
            "wkid": 32644
        },
        "features": list(features.values())
    }

    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"Checkpoint saved: {len(features):,} unique features")


def main():

    print("=" * 70)
    print("TELANGANA FOREST BOUNDARY DOWNLOADER")
    print("=" * 70)

    # --------------------------------------------------------
    # Load checkpoint if it exists
    # --------------------------------------------------------

    features = {}

    if os.path.exists(CHECKPOINT):

        print("\nExisting checkpoint found.")
        print("Loading checkpoint...")

        with open(CHECKPOINT, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)

        for feature in checkpoint.get("features", []):

            attrs = feature.get("attributes", {})

            # FID is the actual OID field in this layer.
            fid = attrs.get("FID")

            if fid is not None:
                features[str(fid)] = feature

        print(f"Loaded {len(features):,} unique features.")

    # --------------------------------------------------------
    # Create initial spatial tiles
    # --------------------------------------------------------

    tiles = []

    dx = (XMAX - XMIN) / INITIAL_DIVISIONS
    dy = (YMAX - YMIN) / INITIAL_DIVISIONS

    for ix in range(INITIAL_DIVISIONS):

        for iy in range(INITIAL_DIVISIONS):

            xmin = XMIN + ix * dx
            xmax = XMIN + (ix + 1) * dx

            ymin = YMIN + iy * dy
            ymax = YMIN + (iy + 1) * dy

            tiles.append((xmin, ymin, xmax, ymax))

    print(f"\nInitial tiles: {len(tiles)}")

    processed_tiles = 0

    # --------------------------------------------------------
    # Process tiles recursively
    # --------------------------------------------------------

    while tiles:

        xmin, ymin, xmax, ymax = tiles.pop()

        processed_tiles += 1

        print("\n" + "-" * 70)
        print(f"Tile #{processed_tiles}")
        print(
            f"X: {xmin:.1f} -> {xmax:.1f} | "
            f"Y: {ymin:.1f} -> {ymax:.1f}"
        )

        data = query_tile(xmin, ymin, xmax, ymax)

        tile_features = data.get("features", [])

        exceeded = data.get("exceededTransferLimit", False)

        print(f"Returned: {len(tile_features):,}")
        print(f"Transfer limit exceeded: {exceeded}")

        # ----------------------------------------------------
        # If the server indicates that this tile is too large,
        # subdivide it.
        #
        # Also subdivide whenever exactly 1000 records are
        # returned because 1000 is the server's maximum.
        # ----------------------------------------------------

        if exceeded or len(tile_features) >= 1000:

            print("Tile may contain more than 1000 features.")
            print("Subdividing tile...")

            smaller_tiles = split_tile(
                xmin,
                ymin,
                xmax,
                ymax
            )

            tiles.extend(smaller_tiles)

            continue

        # ----------------------------------------------------
        # Add features using FID as unique key.
        # This removes duplicates caused by polygons touching
        # multiple spatial tiles.
        # ----------------------------------------------------

        added = 0
        duplicate = 0

        for feature in tile_features:

            attrs = feature.get("attributes", {})

            fid = attrs.get("FID")

            if fid is None:
                print("WARNING: feature without FID skipped.")
                continue

            key = str(fid)

            if key in features:
                duplicate += 1
            else:
                features[key] = feature
                added += 1

        print(f"New unique features: {added:,}")
        print(f"Duplicate features ignored: {duplicate:,}")
        print(f"Total unique features: {len(features):,}")

        # Save checkpoint after every successful tile.
        save_checkpoint(features)

        time.sleep(0.3)

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    final_data = {
        "displayFieldName": "FB_Type",
        "geometryType": "esriGeometryPolygon",
        "spatialReference": {
            "wkid": 32644
        },
        "features": list(features.values())
    }

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(
            final_data,
            f,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    print(f"Unique forest polygons: {len(features):,}")
    print(f"Final file: {OUTPUT}")

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    fids = [
        feature["attributes"]["FID"]
        for feature in features.values()
        if feature.get("attributes", {}).get("FID") is not None
    ]

    print(f"FID count: {len(fids):,}")
    print(f"Unique FID count: {len(set(fids)):,}")

    if len(fids) == len(set(fids)):
        print("✓ No duplicate FIDs.")
    else:
        print("WARNING: Duplicate FIDs detected.")

    print("\nThe checkpoint can now be deleted.")
    print(f"Checkpoint: {CHECKPOINT}")


if __name__ == "__main__":
    main()