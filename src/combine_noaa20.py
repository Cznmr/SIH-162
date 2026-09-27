import os
import pandas as pd

SP_FILE = "data/raw/firms/sp/noaa20_sp_telangana_2025-09-24_to_2026-06-30.csv"
NRT_FILE = "data/raw/firms/nrt/noaa20_nrt_telangana_2026-07-01_to_2026-09-23.csv"

OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "noaa20_telangana_12months.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading SP...")
sp = pd.read_csv(SP_FILE)

print("Loading NRT...")
nrt = pd.read_csv(NRT_FILE)

# Mark the source
sp["source_dataset"] = "NOAA20_SP"
nrt["source_dataset"] = "NOAA20_NRT"

# Combine
df = pd.concat([sp, nrt], ignore_index=True)

# Convert date
df["acq_date"] = pd.to_datetime(df["acq_date"])

# Remove exact duplicate rows
before = len(df)

df = df.drop_duplicates()

duplicates_removed = before - len(df)

# Sort
df = df.sort_values(
    ["acq_date", "latitude", "longitude"]
).reset_index(drop=True)

# Save
df.to_csv(OUTPUT_FILE, index=False)

print("\n==============================================")
print("NOAA-20 12-MONTH DATASET")
print("==============================================")

print(f"SP detections       : {len(sp):,}")
print(f"NRT detections      : {len(nrt):,}")
print(f"Before deduplication: {before:,}")
print(f"Duplicates removed  : {duplicates_removed:,}")
print(f"Final detections    : {len(df):,}")

print(f"\nFirst date: {df['acq_date'].min().date()}")
print(f"Last date : {df['acq_date'].max().date()}")

print("\nDetections by source:")
print(df["source_dataset"].value_counts())

print("\nDetections by month:")
print(
    df.groupby(df["acq_date"].dt.to_period("M"))
      .size()
)

print(f"\nSaved to:")
print(OUTPUT_FILE)

print("==============================================")