import pandas as pd
import argparse
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import pickle

def load_data(file_path):
    
    df = pd.read_csv(file_path)
    
    return df

def load_model(model_path):
    
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    return model

def make_predictions(df, model):
    
    forecast_steps = len(df)
    forecast = model.get_forecast(steps=forecast_steps)
    predictions = forecast.predicted_mean
    predictions.index = df.index
    predictions = predictions.round()
    
    return predictions

def save_predictions(predictions, predictions_file):
    
    predictions.to_json(predictions_file, orient='index')
    
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Prediction script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='../data/test_data.csv', 
        help='Path to the test data file to make predictions'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='../models/model.pkl',
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='predictions/predictions.json', 
        help='Path to save the predictions'
    )
    return parser.parse_args()

def main(input_file, model_file, output_file):
    df = load_data(input_file)
    model = load_model(model_file)
    predictions = make_predictions(df, model)
    save_predictions(predictions, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file, args.output_file)
