import os
import pandas as pd
import folium

INPUT_FILE = "data/raw/firms/firms_combined_5days.csv"
OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "fire_map.html")

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# Remove records with missing coordinates
df = df.dropna(subset=["latitude", "longitude"])

# Center map around Telangana
fire_map = folium.Map(
    location=[17.5, 79.0],
    zoom_start=7,
    tiles="OpenStreetMap"
)

for _, row in df.iterrows():
    popup_text = f"""
    <b>Satellite:</b> {row.get('satellite', 'N/A')}<br>
    <b>Date:</b> {row.get('acq_date', 'N/A')}<br>
    <b>Time:</b> {row.get('acq_time', 'N/A')}<br>
    <b>Brightness:</b> {row.get('bright_ti4', row.get('brightness', 'N/A'))}<br>
    <b>FRP:</b> {row.get('frp', 'N/A')}<br>
    <b>Confidence:</b> {row.get('confidence', 'N/A')}
    """

    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=5,
        popup=folium.Popup(popup_text, max_width=300),
        fill=True
    ).add_to(fire_map)

fire_map.save(OUTPUT_FILE)

print(f"Map created successfully: {OUTPUT_FILE}")
print(f"Total plotted detections: {len(df)}")