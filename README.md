# ENTRO-26162
## AI-Based Detection and Classification of Industrial Fires and Persistent Thermal Sources

### Smart India Hackathon 2026 – Problem Statement 26162

ENTRO-26162 is an AI-assisted geospatial intelligence prototype for detecting, classifying, and prioritizing thermal events across Telangana using NASA FIRMS satellite fire detections combined with OpenStreetMap infrastructure, Google Dynamic World land-cover context, Telangana forest boundaries, and spatial-temporal fire behavior.

The system is designed to help distinguish potential industrial fires, wildfire candidates, persistent thermal sources, and lower-priority background detections.

---

## 1. Problem Statement

Satellite-based fire products can detect thermal anomalies over large geographic areas, but a thermal detection alone does not explain what caused it.

A detected hotspot may correspond to:

- An industrial fire
- A wildfire or vegetation fire
- A persistent industrial thermal source
- Agricultural or other thermal activity
- Other background thermal events

The objective of ENTRO-26162 is to combine satellite thermal detections with geographic, land-cover, infrastructure, persistence, and spatial-behavior information to produce a more useful classification and prioritization layer.

---

## 2. What ENTRO-26162 Builds

The prototype provides an end-to-end pipeline that:

1. Collects satellite thermal detections from NASA FIRMS.
2. Processes historical FIRMS observations for Telangana.
3. Builds temporal persistence features.
4. Builds spatial fire-behavior features.
5. Extracts nearby industrial and infrastructure context from OpenStreetMap.
6. Adds land-cover context using Google Dynamic World.
7. Adds Telangana forest-boundary context.
8. Creates a combined machine-learning feature table.
9. Generates conservative seed labels for model development.
10. Trains a Random Forest classifier.
11. Applies the trained classifier to live FIRMS detections.
12. Displays candidate classifications and confidence values on an interactive dashboard.

---

## 3. System Architecture

```text
                 NASA FIRMS
                     |
                     v
             Thermal Detections
                     |
                     v
        Temporal Feature Engineering
                     |
        +------------+------------+
        |            |            |
        v            v            v
      OSM       Dynamic World    Forest
 Infrastructure   Land Cover    Boundaries
        |            |            |
        +------------+------------+
                     |
                     v
          Spatial Behavior Features
                     |
                     v
          Master Feature Table
                     |
                     v
            Conservative Seed Labels
                     |
                     v
          Random Forest Classifier
                     |
                     v
            fire_classifier_v2.pkl
                     |
                     v
            Live NASA FIRMS Data
                     |
                     v
        Candidate Classification
                     |
          +----------+----------+
          |          |          |
          v          v          v
       Wildfire   Industrial   Persistent
       Candidate     Fire       Thermal
                    Candidate    Source
                     |
                     v
             Streamlit Dashboard



4. Data Sources
NASA FIRMS

NASA FIRMS provides satellite-based active fire and thermal anomaly observations.

The prototype uses:

VIIRS NOAA-20
VIIRS NOAA-21
Historical FIRMS data
Near-real-time FIRMS data

The geographical area used by the prototype is Telangana.

OpenStreetMap

OpenStreetMap is used to provide infrastructure context around detected thermal events.

Relevant infrastructure includes:

Industrial areas
Industrial works
Power plants
Power infrastructure
Storage tanks
Quarries
Oil and gas related infrastructure

OSM information is used as contextual evidence and is not treated as ground truth.

Google Dynamic World

Google Dynamic World provides land-cover context at approximately 10 m resolution.

The prototype uses classes including:

Water
Trees
Grass
Flooded vegetation
Crops
Shrub and scrub
Built
Bare
Snow and ice

Dynamic World is used to provide environmental context around thermal detections.

Dataset attribution:

This dataset is produced for the Dynamic World Project by Google in partnership with National Geographic Society and the World Resources Institute.

Telangana Forest Boundary

Official Telangana government GIS forest-boundary data is used to determine:

Whether a location lies inside a forest boundary
Distance to the nearest forest
Proximity to forest areas

Forest proximity is treated as contextual evidence and does not automatically classify an event as wildfire.

5. Geographic Coverage

The prototype currently operates over Telangana using the following bounding box:

West:  77.0
South: 15.5
East:  81.6
North: 20.2
6. Feature Engineering

The model uses a combination of thermal, temporal, environmental, infrastructure, forest, and spatial features.

FIRMS Features

Examples include:

Total detections
Active days
Active days over 30 days
Active days over 90 days
Active days over 180 days
Persistence score
Mean and maximum brightness temperature
Mean and maximum FRP
Day/night detections
Mean and maximum confidence
Night detection ratio
Observation span
Dynamic World Features

Examples include:

Water probability
Trees probability
Grass probability
Flooded vegetation probability
Crops probability
Shrub and scrub probability
Built probability
Bare probability
Vegetation score
Non-vegetation score
OSM Features

Examples include:

Distance to nearest industrial area
Distance to nearest quarry
Distance to nearest power plant
Distance to nearest power infrastructure
Distance to nearest industrial works
Distance to nearest storage tank
Distance to nearest oil/gas infrastructure
Distance to nearest relevant OSM feature
Forest Features

Examples include:

Inside forest
Distance to nearest forest
Spatial Behavior Features

Examples include:

Spatial observation days
Expansion activity within 1.5 km
Expansion activity within 3 km
Expansion activity within 5 km
New activity within 1.5 km
New activity within 3 km
New activity within 5 km
Maximum local persistence within 1.5 km
Maximum local persistence within 3 km
Maximum local persistence within 5 km
New detection days
Consecutive detection days
7. Persistence Analysis

Persistent thermal activity is analyzed using a 0.01° spatial grid.

The prototype calculates persistence across multiple temporal windows:

30 days
90 days
180 days

The persistence score combines these windows with greater weight given to longer-term persistence.

Persistent thermal activity is treated separately from a single high-intensity thermal event because recurring activity can indicate a continuously active thermal source.

8. Seed Label Generation

Because independently verified ground-truth labels were not available for the entire study area, the prototype uses conservative rule-based seed labels for model development.

The seed-label categories are:

BACKGROUND
WILDFIRE
INDUSTRIAL_FIRE
PERSISTENT_THERMAL_SOURCE
UNKNOWN

The rules use combinations of:

FRP
Persistence
Observation span
Infrastructure proximity
Land-cover context
Built-up context
Vegetation context

The seed-label process is intentionally conservative.

The model should therefore be interpreted as learning patterns from these development labels rather than learning from a fully field-verified fire database.

9. Machine Learning Model

The final prototype uses a Random Forest classifier.

Model pipeline
Input Features
      |
      v
Median Imputation
      |
      v
Random Forest Classifier
      |
      v
Class Prediction
      |
      v
Prediction Confidence

The classifier uses:

500 decision trees
min_samples_leaf = 2
max_features = sqrt
Balanced class weights
Random state = 42

The trained model is stored at:

models/fire_classifier_v2.pkl

The model bundle contains:

Trained model
Exact feature list
Class labels
Training-row information
Validation-row information
Random state
Spatial block information
10. Model Validation

The model was evaluated using a spatially separated validation strategy.

The geographic space was divided into 0.1° spatial blocks so that training and validation locations were separated geographically.

This reduces the risk of simply evaluating the model on nearby locations that are highly similar to training observations.

The development validation results were approximately:

Class	Precision	Recall	F1
Background	1.00	1.00	1.00
Wildfire	0.99	1.00	0.99
Industrial Fire	1.00	0.90	0.95
Persistent Thermal Source	1.00	1.00	1.00

Overall validation accuracy was approximately 1.00 on the development validation set.

Important limitation

These metrics measure agreement with the project's conservative seed labels.

They should not be interpreted as independently verified real-world fire-detection accuracy.

Field-validated or independently annotated fire-event data would be required for a stronger evaluation of real-world performance.

11. Live FIRMS Integration

The Streamlit application connects directly to the NASA FIRMS API.

The application retrieves near-real-time observations from:

VIIRS_NOAA20_NRT
VIIRS_NOAA21_NRT

The live observations are converted to the same spatial grid used during feature engineering.

The application then matches live detections against the trained feature table and applies the trained classifier.

The dashboard displays:

Live thermal detections
Candidate classification
Model confidence
FRP
Satellite source
Detection date
Detection time
Geographic coordinates
Relevant spatial context
12. Classification Categories
Wildfire Candidate

A live thermal detection that the model identifies as having characteristics associated with wildfire-type activity.

Industrial Fire Candidate

A thermal detection that the model identifies as having characteristics associated with industrial or infrastructure-related activity.

Persistent Thermal Source

A location showing recurring or persistent thermal activity that may represent a continuously active thermal source.

Background / Low-Priority

A detection that does not strongly match the positive prototype categories.

13. Streamlit Dashboard

The dashboard provides an interactive interface for exploring live detections.

Main components include:

Telangana-focused interactive map
Live NASA FIRMS detections
AI candidate classification layers
Model confidence
Intelligence summary
Classification results table
Latest detection information
Classification legend

The dashboard is designed as a decision-support prototype rather than an automated authority for confirming incidents.

14. Project Structure
ENTRO-26162/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .env
│
├── models/
│   └── fire_classifier_v2.pkl
│
├── data/
│   ├── raw/
│   │   ├── firms/
│   │   ├── osm/
│   │   ├── industrial/
│   │   ├── satellite/
│   │   └── landcover/
│   │       └── dynamic_world/
│   │
│   ├── processed/
│   │
│   └── training/
│
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── classifier.py
│   └── map_generator.py
│
├── notebooks/
│   └── prototype.ipynb
│
└── outputs/
    ├── classified_events.csv
    └── fire_map.html
15. Environment Setup

Create a .env file in the project root:

FIRMS_MAP_KEY=YOUR_NASA_FIRMS_MAP_KEY

The .env file must not be committed to GitHub.

Add this to .gitignore:

.env
16. Installation

Create and activate the Python virtual environment:

python -m venv venv

Activate it on Windows:

.\venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt
17. Run the Dashboard

From the project root:

.\venv\Scripts\python.exe -m streamlit run app.py

The application will start locally and provide a Streamlit URL.

18. Security

The NASA FIRMS MAP KEY is stored locally in .env.

Never:

Commit .env to GitHub
Put the API key directly in source code
Share the API key publicly
Include the API key in screenshots or presentations
19. Current Prototype Status

The following components have been implemented:

 NASA FIRMS historical data processing
 NASA FIRMS live API integration
 Telangana geographic filtering
 Temporal persistence analysis
 Spatial fire-behavior analysis
 OpenStreetMap infrastructure context
 Google Dynamic World land-cover context
 Telangana forest-boundary context
 Master feature table
 Conservative seed-label generation
 Random Forest classifier
 Spatial validation
 Saved trained model
 Live model inference
 Candidate confidence scores
 Interactive Streamlit dashboard
 Map-based visualization
 AI classification results table
20. Limitations

The current prototype has several important limitations:

FIRMS detections represent satellite-observed thermal anomalies and do not by themselves establish the cause of an event.
The training labels are conservative development labels rather than a complete independently verified ground-truth dataset.
Model confidence represents the classifier's confidence, not the probability that a real-world event has been independently confirmed.
OSM infrastructure data may be incomplete or outdated.
Dynamic World provides land-cover context and should not be treated as fire-event ground truth.
Forest proximity alone cannot determine whether a thermal event is a wildfire.
Cloud cover, satellite revisit timing, sensor characteristics, and detection limitations can affect satellite observations.
Industrial-fire predictions should be interpreted as candidate events requiring further verification.
The current system is a prototype and has not been deployed as an operational emergency-response system.