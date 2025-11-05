# 🏎️ F1 Project: Predictions & Visualization
Welcome to my F1 project repository, Merging two of my favourite things! These projects use the FastF1 API for live and historical race data and machine learning for predictions. 

## Data sources
* FastF1 API
* Realtime 2025 Qualifying data from the official F1 App (When the API hasn't been updated)
* Driver wet performance (pre- Canadian GP)

## Dependencies
* pandas
* numpy
* scikit-learn
* matplotlib
* fastf1

## How The Predictor Works
* Historical data from the FastF1 API
* Live data from qualifying 2025

## Usage
Run the prediction script:

Expected outcome:


## Model Performance
Model performance is evaluated using the Mean Absolute Error (MAE). 

## File Structure, Features added & Effect on MAE
f1predictor1 = Using 2024 race and 2025 quali data = MAE 49.50 secs
f1predictor2 = Add sector times = MAE 
f1predictor3 = Only 2024 race data = MAE 
f1predictor4 = Add wet performance data & weather API = MAE
f1predictor5 = Add clean air race pace = MAE 


## Next Steps


## License
This repository is licensed under the MIT License.

