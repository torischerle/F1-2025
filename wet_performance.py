import fastf1
import pandas as pd

# Enable cache for fastf1
# I used my default cache directory

# Using the Canadian GP for 2022 and 2023 as a proxy for wet and dry conditions
# Load the 2022 Canadian GP race data (wet race)
session_2022 = fastf1.get_session(2022, "Canada", "R")
session_2022.load()

# Load the 2023 Canadian GP race data (dry race)
session_2023 = fastf1.get_session(2023, "Canada", "R")
session_2023.load()

# extract lap times and driver codes
laps_2022 = session_2022.laps[["Driver", "LapTime"]].copy()
laps_2023 = session_2023.laps[["Driver", "LapTime"]].copy()

# In case of missing laps, drop NaN values
laps_2022.dropna(subset=["LapTime"], inplace=True)
laps_2023.dropna(subset=["LapTime"], inplace=True)

# Convert lap times to total seconds for easier comparison
laps_2022["LapTime (s)"] = laps_2022["LapTime"].dt.total_seconds()
laps_2023["LapTime (s)"] = laps_2023["LapTime"].dt.total_seconds()

# Calculate average lap times per driver for both years
avg_laps_2022 = laps_2022.groupby("Driver")["LapTime (s)"].mean().reset_index()
avg_laps_2023 = laps_2023.groupby("Driver")["LapTime (s)"].mean().reset_index()

# Merge the two datasets on Driver code to compare lap times
merged_laps = pd.merge(avg_laps_2022, avg_laps_2023, on="Driver", suffixes=("_2022", "_2023"))

# Calculate the performance difference in lap times between wet and dry conditions (dry-wet)
merged_laps["LapTimeDifference (s)"] = merged_laps["LapTime (s)_2023"] - merged_laps["LapTime (s)_2022"]

# Calculate percentage change in in lap times between wet and dry conditions (diff/dry) where dry is race pace
merged_laps["PerformanceChange (%)"] = (merged_laps["LapTimeDifference (s)"] / merged_laps["LapTime (s)_2023"]) * 100

# Create wet performance score
merged_laps["WetPerformanceScore"] = 1 + merged_laps["PerformanceChange (%)"] / 100

# Print the wet performance results for each driver
print("\n🌧️ Driver Wet Performance scores (Canada GP 2022 vs 2023):")
print(merged_laps[["Driver", "LapTime (s)_2022", "LapTime (s)_2023", "LapTimeDifference (s)", "PerformanceChange (%)", "WetPerformanceScore"]])