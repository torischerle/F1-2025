import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import fastf1

# Enable FastF1 caching
# fastf1.Cache.enable_cache("f1_cache")

# Load FastF1 2024 Monaco GP race session
session_2024 = fastf1.get_session(2024, 8, "R")
session_2024.load()

# Extract lap times times for all drivers
laps_2024 = session_2024.laps[["Driver", "LapTime"]].copy()
laps_2024.dropna(subset=["LapTime"], inplace=True)
laps_2024["LapTime (s)"] = laps_2024["LapTime"].dt.total_seconds()

# 2025 Qualifying session - data from the official F1 App
# Ordered in starting gride - despite quali times
# Stroll penalized 4 grid places
# Hamilton penalized 3 grid places
# Bearman penalizes 10 grid places
# Russell DNF in Q2, so Q1 lap time used for predictions
qualifying_2025 = pd.DataFrame({
    "Driver": ["Lando Norris", "Charles Leclerc", "Oscar Piastri", "Max Verstappen",
                "Isack Hadjar", "Fernando Alonso", "Lewis Hamilton", "Esteban Ocon",
                "Liam Lawson", "Alexander Albon", "Carlos Sainz", "Yuki Tsunoda", 
                "Nico Hulkenberg", "George Russell", "Kimi Antonelli", "Gabriel Bortoleto",
                "Pierre Gasly", "Franco Colapinto", "Lance Stroll", "Oliver Bearman"],
    "QualifyingTime (s)": [69.954, 70.063, 70.129, 70.669,
                           70.924, 70.924, 70.382, 70.942,
                           71.129, 71.213, 71.362, 71.415 , 
                           71.596, 71.507, 71.880, 71.902,
                           71.994, 72.597, 72.563, 71.979]
})

# FastF1 uses 3-letter driver codes, we need to map them to full driver names
driver_mapping = {
    "Lando Norris": "NOR", "Charles Leclerc": "LEC", "Oscar Piastri": "PIA", "Max Verstappen": "VER",
    "Isack Hadjar": "HAD", "Fernando Alonso": "ALO", "Lewis Hamilton": "HAM", "Esteban Ocon": "OCO",
    "Liam Lawson": "LAW", "Alexander Albon": "ALB", "Carlos Sainz": "SAI", "Yuki Tsunoda": "TSU", 
    "Nico Hulkenberg": "HUL", "George Russell": "RUS", "Kimi Antonelli": "ANT", "Gabriel Bortoleto": "BOR",
    "Pierre Gasly": "GAS", "Franco Colapinto": "COL", "Lance Stroll": "STR", "Oliver Bearman": "BEA"
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

# Merge 2025 Qualifying Data with 2024 Race Data
merged_data = qualifying_2025.merge(laps_2024, left_on="DriverCode", right_on="Driver").fillna(0)
print("Merged Data:\n", merged_data)

# Use only "QualifyingTime (s)" as a feature
X = merged_data[["QualifyingTime (s)"]]
y = merged_data["LapTime (s)"]

# Check if the dataset is empty
if X.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing. Check data sources!")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting Regressor
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predict using 2025 qualifying times
predicted_lap_times = model.predict(qualifying_2025[["QualifyingTime (s)"]])
qualifying_2025["PredictedRaceTime (s)"] = predicted_lap_times

# Rank drivers by predicted race time
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)")

# Print final predictions
print("\n🏁 Predicted 2025 Monaco GP Winner 🏁\n")
print(qualifying_2025[["Driver", "PredictedRaceTime (s)"]])

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")