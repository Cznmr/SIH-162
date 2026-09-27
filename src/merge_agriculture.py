from pathlib import Path
import pandas as pd


INPUT_DIR = Path("data/raw/landcover/agriculture")
OUTPUT_FILE = Path("data/processed/agriculture_features_raw.csv")


def main():
    files = sorted(INPUT_DIR.glob("PS26162_Agriculture_*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No agriculture CSV files found in {INPUT_DIR}"
        )

    print(f"Found {len(files)} agriculture CSV files")

    frames = []

    for file in files:
        print(f"Reading: {file.name}")

        df = pd.read_csv(file)

        print(
            f"  Rows: {len(df):,} | "
            f"Columns: {len(df.columns)}"
        )

        df["source_file"] = file.name

        frames.append(df)

    merged = pd.concat(frames, ignore_index=True, sort=False)

    # Check coordinate uniqueness
    duplicate_coords = merged.duplicated(
        subset=["lat_grid", "lon_grid"],
        keep=False
    ).sum()

    print("\n--- MERGE SUMMARY ---")
    print(f"Total rows: {len(merged):,}")
    print(
        "Unique coordinates:",
        merged[["lat_grid", "lon_grid"]].drop_duplicates().shape[0]
    )
    print(f"Duplicate coordinates: {duplicate_coords:,}")
    print(f"Total columns: {len(merged.columns)}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()