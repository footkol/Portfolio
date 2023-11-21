import pandas as pd
import argparse
import numpy as np
import json
from matplotlib import pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
import itertools
import warnings
warnings.filterwarnings("ignore")

def load_data(file_path):
    
    df = pd.read_csv(file_path, parse_dates=['StartTime'], index_col='StartTime')
    
    return df

def split_data(df):
    
    train_ratio = 0.6  
    val_ratio = 0.2    
    test_ratio = 0.2
    
    train_split = int(train_ratio * len(df))
    val_split = int((train_ratio + val_ratio) * len(df))
    
    train_set = df['label'].iloc[:train_split]
    val_set = df['label'].iloc[train_split:val_split]
    test_set = df['label'].iloc[val_split:]

    test_set.to_csv('../data/test_set.csv', index=False)
    
return train_set, val_set, test_set

def train_model(train_set, val_set):
    
    # Performing a grid search to find the best parameters for tuning an ARIMA model 
    p_values = range(0, 3)  
    d_values = range(0, 2)  
    q_values = range(0, 3) 

    # Creating a list of all possible combinations of p, d, and q
    parameter_grid = list(itertools.product(p_values, d_values, q_values))

    best_model_params = None
    best_mae = float('inf')

    for params in parameter_grid:
        p, d, q = params
        model = ARIMA(train_set, order=(p, d, q))
        results = model.fit()
    
        # Making predictions on the test set
        forecast = results.get_forecast(steps=len(val_set))
        predicted_values = forecast.predicted_mean
    
        # Evaluating the model using Mean Absolute Error
        mae = mean_absolute_error(test_set, predicted_values)
    
        # Updating the best model if the current one is better
        if mae < best_mae:
            best_mae = mae
            best_model_params = (p, d, q)

    best_model = ARIMA(train_set, order=best_model_params)
    best_results = best_model.fit()
    
    return best_model

def save_model(best_model, model_path):
    
    best_results.save(model_path)
    
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model training script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/final_data.csv', 
        help='Path to the processed data file to train the model'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model.pkl', 
        help='Path to save the trained model'
    )
    return parser.parse_args()

def main(input_file, model_file):
    df = load_data(input_file)
    train_set, val_set, test_set = split_data(df)
    best_model = train_model(train_set)
    save_model(best_model, model_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file)