# evaluation.py

"""
Evaluation Module for Multivariate Extreme Value Analysis

This module contains functions for evaluating the performance and accuracy of the analysis and prediction models.

Metrics Used:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Explained Variance
- Silhouette Score (for clustering)

Libraries/Tools:
- pandas
- numpy
- scikit-learn
- matplotlib
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, silhouette_score
import matplotlib.pyplot as plt
import os

class ModelEvaluation:
    def __init__(self):
        """
        Initialize the ModelEvaluation class.
        """
        pass

    def load_data(self, filepath):
        """
        Load data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        data = pd.read_csv(filepath)
        return data

    def evaluate_regression(self, y_true, y_pred):
        """
        Evaluate regression metrics.
        
        :param y_true: array, true values
        :param y_pred: array, predicted values
        :return: dict, regression metrics
        """
        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'Explained Variance': explained_variance_score(y_true, y_pred)
        }
        return metrics

    def evaluate_clustering(self, data, labels):
        """
        Evaluate clustering metrics.
        
        :param data: DataFrame, input data
        :param labels: array, cluster labels
        :return: dict, clustering metrics
        """
        metrics = {
            'Silhouette Score': silhouette_score(data, labels)
        }
        return metrics

    def plot_regression_results(self, y_true, y_pred, output_dir):
        """
        Plot regression results.
        
        :param y_true: array, true values
        :param y_pred: array, predicted values
        :param output_dir: str, directory to save the plot
        """
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Regression Results')
        plt.savefig(os.path.join(output_dir, 'regression_results.png'))
        plt.show()

    def plot_clustering_results(self, data, labels, output_dir):
        """
        Plot clustering results.
        
        :param data: DataFrame, input data
        :param labels: array, cluster labels
        :param output_dir: str, directory to save the plot
        """
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis', alpha=0.5)
        plt.xlabel(data.columns[0])
        plt.ylabel(data.columns[1])
        plt.title('Clustering Results')
        plt.savefig(os.path.join(output_dir, 'clustering_results.png'))
        plt.show()

    def evaluate(self, data_filepath, y_true_filepath, y_pred_filepath, clustering_labels_filepath, output_dir):
        """
        Execute the full evaluation pipeline.
        
        :param data_filepath: str, path to the input data file
        :param y_true_filepath: str, path to the true values file
        :param y_pred_filepath: str, path to the predicted values file
        :param clustering_labels_filepath: str, path to the clustering labels file
        :param output_dir: str, directory to save the plots
        """
        # Load data
        data = self.load_data(data_filepath)
        y_true = pd.read_csv(y_true_filepath).values.ravel()
        y_pred = pd.read_csv(y_pred_filepath).values.ravel()
        clustering_labels = pd.read_csv(clustering_labels_filepath).values.ravel()

        # Evaluate regression metrics
        regression_metrics = self.evaluate_regression(y_true, y_pred)
        print("Regression Metrics:")
        for metric, value in regression_metrics.items():
            print(f"{metric}: {value}")

        # Plot regression results
        self.plot_regression_results(y_true, y_pred, output_dir)

        # Evaluate clustering metrics
        clustering_metrics = self.evaluate_clustering(data, clustering_labels)
        print("\nClustering Metrics:")
        for metric, value in clustering_metrics.items():
            print(f"{metric}: {value}")

        # Plot clustering results
        self.plot_clustering_results(data, clustering_labels, output_dir)

if __name__ == "__main__":
    data_filepath = 'data/processed/processed_data.csv'
    y_true_filepath = 'data/processed/y_true.csv'
    y_pred_filepath = 'data/processed/y_pred.csv'
    clustering_labels_filepath = 'data/processed/clustering_labels.csv'
    output_dir = 'results/evaluation/'
    os.makedirs(output_dir, exist_ok=True)

    evaluator = ModelEvaluation()

    # Evaluate the models
    evaluator.evaluate(data_filepath, y_true_filepath, y_pred_filepath, clustering_labels_filepath, output_dir)
    print("Model evaluation completed and results saved.")
