
# Energy Forecasting Hackathon

## Overview
This repository contains a data processing workflow for the Energy Forecasting Hackathon. The goal is to create a model capable of predicting the country (from a list of nine) that will have the most surplus of green energy in the next hour. For this task, both the energy generation from renewable sources (wind, solar, geothermic, etc.), and the load (energy consumption) need to be considered. The surplus of green energy is considered to be the difference between the generated green energy and the consumed energy.

The countries to focus on are: Spain, UK, Germany, Denmark, Sweden, Hungary, Italy, Poland, and the Netherlands.

## Data Processing Workflow

### Imputing Missing Values
Missing values in the dataset are imputed as the mean between the preceding and following values.

### Resampling
Data with a resolution finer than 1 hour is resampled to an hourly level. For example, data at a 15-minute resolution is aggregated into 1-hour intervals by summing every 4 consecutive rows.

### Identifying Energy Types
Green energy types for each column are identified as per [Transparency Platform RESTful API - user guide](https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html)
- B01 Biomass
- B09 Geothermal
- B10 Hydro Pumped Storage
- B11 Hydro Run-of-river and poundage
- B12 Hydro Water Reservoir
- B13 Marine
- B15 Other renewable
- B16 Solar
- B18 Wind Offshore
- B19 Wind Onshore
- Green energy 

Non-green energy sources are discarded.

### Aggregating Data
In the end, the data processing workflow generates a single CSV file with columns per country, representing the following values:
- Generated green energy per energy type (one column per wind, solar, etc.)
- Load
All values are in the same units (MAW).

### Labeling
An additional column is added as the label: the ID of the country with the biggest surplus of green energy for the next hour.

## Model Selection
For this forecasting task, the ARIMA (AutoRegressive Integrated Moving Average) model was selected. A traditional time series model known for its effectiveness in capturing temporal patterns. To ensure the suitability of the model, a stationarity check using the Augmented Dickey-Fuller Test was performed. This test helps assess whether the time series data is stationary, a crucial assumption for ARIMA models.

After confirming stationarity, training and evaluating the ARIMA model on the provided dataset was carried out. The choice of ARIMA was motivated by its simplicity and ability to capture linear trends in time series data. 

Performing a grid search was done to find the best parameters for tuning an ARIMA model which involves systematically testing different combinations of hyperparameters and selecting the set that results in the best model performance.

### Prediction Format
The model predictions file containing prediction of a country ID with the biggest surplus for the next hour was stored as predictions.json.

## Results
Following the completion of the prediction process, the model's performance was evaluated using various metrics. The Mean Absolute Error (MAE) was found to be 0.1678, indicating the average absolute difference between the predicted and actual values. Additionally, the Mean Squared Error (MSE) was calculated as 0.2603, representing the average of squared differences, and the Root Mean Squared Error (RMSE) was 0.5102, providing a measure of the model's prediction accuracy in the original unit of the target variable. 

