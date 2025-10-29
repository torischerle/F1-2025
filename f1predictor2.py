import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import fastf1
from fastf1.ergast import Ergast

# Enable cache for fastf1
# fastf1.Cache.enable_cache("f1_cache")  # replace with your cache directory

# Load FastF1 2024 Monaco GP race session
session_2024 = fastf1.get_session(2024, 8, "R")
session_2024.load()

# Extract lap times & sector times for all drivers
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(subset=["LapTime"], inplace=True)

# Convert times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Group by driver to get average sector times per driver
sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()

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

# Merge 2025 Qualifying Data with 2024 Race Data including sector data
merged_data = qualifying_2025.merge(sector_times_2024, left_on="DriverCode", right_on="Driver", how="left")
print("Merged Data:\n", merged_data)

# Use only "QualifyingTime (s)" AND Sector Times as features
X = merged_data[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].fillna(0)
y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reset_index()["LapTime (s)"].fillna(0)

# Check unique drivers in laps_2024
print("Drivers in laps_2024:", laps_2024["Driver"].unique())

# Check unique driver codes in merged_data
print("Driver Codes in merged_data:", merged_data["DriverCode"].unique())

# Ensure y is defined correctly
y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reset_index().fillna(0)
# Print before filtering
print("Before filtering y:", y)

# Filter y to match drivers in merged_data
y = y[y["Driver"].isin(merged_data["DriverCode"])]  # This line filters out drivers not in merged_data
# Print after filtering
print("After filtering y:", y)

# Set Driver as index
y = y.set_index("Driver")["LapTime (s)"]  # Set Driver as index

# Ensure y is defined
# y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reset_index()
# y = y[y["Driver"].isin(merged_data["DriverCode"])]  # Filter y to match drivers in X
# y = y.set_index("Driver")["LapTime (s)"]  # Set Driver as index
# print(f"X : {X}, y : {y}")
# Check if X and y have the same number of samples
if X.shape[0] != y.shape[0]:
    raise ValueError(f"Mismatch in number of samples: X has {X.shape[0]} samples, y has {y.shape[0]} samples.")

# Train Gradient Boosting Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
model.fit(X_train, y_train)

# Predict using 2025 qualifying and sector data
predicted_lap_times = model.predict(X)
qualifying_2025["PredictedRaceTime (s)"] = predicted_lap_times

# Rank drivers by predicted race time
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)")

# Print final predictions
print("\n🏁 Predicted 2025 Monaco GP Winner 🏁\n")
print(qualifying_2025[["Driver", "PredictedRaceTime (s)"]])

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")