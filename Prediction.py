# prediction.py

"""
Prediction Module for Multivariate Extreme Value Analysis

This module contains functions for making predictions using trained models.

Techniques Used:
- Regression Models
- Clustering Models

Libraries/Tools:
- pandas
- numpy
- scikit-learn
- joblib

"""

import pandas as pd
import numpy as np
from sklearn.externals import joblib
import os

class PredictionModel:
    def __init__(self, model_dir):
        """
        Initialize the PredictionModel class.
        
        :param model_dir: str, directory containing trained models
        """
        self.model_dir = model_dir
        self.regression_model = self.load_model('regression_model.pkl')
        self.clustering_model = self.load_model('clustering_model.pkl')

    def load_model(self, model_name):
        """
        Load a trained model from a file.
        
        :param model_name: str, name of the model file
        :return: loaded model
        """
        model_path = os.path.join(self.model_dir, model_name)
        model = joblib.load(model_path)
        return model

    def make_regression_predictions(self, data):
        """
        Make predictions using the regression model.
        
        :param data: DataFrame, input data
        :return: array, predicted values
        """
        predictions = self.regression_model.predict(data)
        return predictions

    def make_clustering_predictions(self, data):
        """
        Make predictions using the clustering model.
        
        :param data: DataFrame, input data
        :return: array, cluster labels
        """
        labels = self.clustering_model.predict(data)
        return labels

    def save_predictions(self, predictions, filepath):
        """
        Save predictions to a CSV file.
        
        :param predictions: array, predictions to save
        :param filepath: str, path to the output file
        """
        pd.DataFrame(predictions).to_csv(filepath, index=False)
        print(f"Predictions saved to {filepath}")

if __name__ == "__main__":
    data_filepath = 'data/processed/processed_data.csv'
    model_dir = 'models/'
    regression_predictions_filepath = 'data/processed/y_pred.csv'
    clustering_predictions_filepath = 'data/processed/clustering_labels.csv'

    # Load input data
    data = pd.read_csv(data_filepath)

    # Initialize prediction model
    predictor = PredictionModel(model_dir)

    # Make regression predictions
    regression_predictions = predictor.make_regression_predictions(data)
    predictor.save_predictions(regression_predictions, regression_predictions_filepath)

    # Make clustering predictions
    clustering_predictions = predictor.make_clustering_predictions(data)
    predictor.save_predictions(clustering_predictions, clustering_predictions_filepath)

    print("Predictions completed and saved.")
