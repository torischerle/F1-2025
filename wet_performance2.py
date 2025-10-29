import fastf1
import pandas as pd

# Enable cache for fastf1
# I used my default cache directory

# Updated the wet driver performance analysis to use 2024 and 2025 data (proxy for wet and dry conditions)
# Load the 2025 Canadian GP race data (dry race)
session_2025 = fastf1.get_session(2025, "Canada", "R")
session_2025.load()

# Load the 2024 Canadian GP race data (wet race)
session_2024 = fastf1.get_session(2024, "Canada", "R")
session_2024.load()

# extract lap times and driver codes
laps_2025 = session_2025.laps[["Driver", "LapTime"]].copy()
laps_2024 = session_2024.laps[["Driver", "LapTime"]].copy()

# In case of missing laps, drop NaN values
laps_2025.dropna(subset=["LapTime"], inplace=True)
laps_2024.dropna(subset=["LapTime"], inplace=True)

# Convert lap times to total seconds for easier comparison
laps_2025["LapTime (s)"] = laps_2025["LapTime"].dt.total_seconds()
laps_2024["LapTime (s)"] = laps_2024["LapTime"].dt.total_seconds()

# Calculate average lap times per driver for both years
avg_laps_2025 = laps_2025.groupby("Driver")["LapTime (s)"].mean().reset_index()
avg_laps_2024 = laps_2024.groupby("Driver")["LapTime (s)"].mean().reset_index()

# Merge the two datasets on Driver code to compare lap times
merged_laps = pd.merge(avg_laps_2025, avg_laps_2024, on="Driver", suffixes=("_2025", "_2024"))

# Calculate the performance difference in lap times between wet and dry conditions (dry-wet)
merged_laps["LapTimeDifference (s)"] = merged_laps["LapTime (s)_2025"] - merged_laps["LapTime (s)_2024"]

# Calculate percentage change in in lap times between wet and dry conditions (diff/dry)
merged_laps["PerformanceChange (%)"] = (merged_laps["LapTimeDifference (s)"] / merged_laps["LapTime (s)_2025"]) * 100

# Create wet performance score
merged_laps["WetPerformanceScore"] = 1 + merged_laps["PerformanceChange (%)"] / 100

# Print the wet performance results for each driver
print("\n🌧️ Driver Wet Performance scores (Canada GP 2024 vs 2025):")
print(merged_laps[["Driver", "LapTime (s)_2025", "LapTime (s)_2024", "LapTimeDifference (s)", "PerformanceChange (%)", "WetPerformanceScore"]])