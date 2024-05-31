# extreme_value_analysis.py

"""
Extreme Value Analysis Module for Multivariate Extreme Value Analysis

This module contains functions for applying extreme value theory techniques to identify and analyze extreme events.

Techniques Used:
- Generalized Extreme Value (GEV) distribution
- Peaks Over Threshold (POT)
- Block Maxima

Libraries/Tools:
- pandas
- numpy
- scipy
- matplotlib

"""

import pandas as pd
import numpy as np
from scipy.stats import genextreme, genpareto
import matplotlib.pyplot as plt
import os

class ExtremeValueAnalysis:
    def __init__(self):
        """
        Initialize the ExtremeValueAnalysis class.
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

    def block_maxima(self, data, column, block_size):
        """
        Apply the Block Maxima method to identify extreme values.
        
        :param data: DataFrame, input data
        :param column: str, column name to apply Block Maxima method
        :param block_size: int, size of each block
        :return: array, block maxima
        """
        block_maxima = data[column].groupby(data.index // block_size).max().values
        return block_maxima

    def fit_gev(self, block_maxima):
        """
        Fit the Generalized Extreme Value (GEV) distribution to block maxima.
        
        :param block_maxima: array, block maxima
        :return: tuple, GEV parameters (shape, location, scale)
        """
        params = genextreme.fit(block_maxima)
        return params

    def plot_gev(self, block_maxima, params, column, output_dir):
        """
        Plot the fitted GEV distribution against the block maxima.
        
        :param block_maxima: array, block maxima
        :param params: tuple, GEV parameters
        :param column: str, column name for the plot title
        :param output_dir: str, directory to save the plot
        """
        fig, ax = plt.subplots()
        x = np.linspace(min(block_maxima), max(block_maxima), 100)
        y = genextreme.pdf(x, *params)
        ax.hist(block_maxima, bins=20, density=True, alpha=0.6, color='g')
        ax.plot(x, y, 'r-', lw=2, label='GEV fit')
        ax.set_title(f'Block Maxima and Fitted GEV Distribution ({column})')
        ax.set_xlabel(column)
        ax.set_ylabel('Density')
        ax.legend()
        plt.savefig(os.path.join(output_dir, f'gev_fit_{column}.png'))
        plt.show()

    def peaks_over_threshold(self, data, column, threshold):
        """
        Apply the Peaks Over Threshold (POT) method to identify extreme values.
        
        :param data: DataFrame, input data
        :param column: str, column name to apply POT method
        :param threshold: float, threshold value to identify peaks
        :return: array, excesses over threshold
        """
        excesses = data[column][data[column] > threshold] - threshold
        return excesses

    def fit_gpd(self, excesses):
        """
        Fit the Generalized Pareto Distribution (GPD) to excesses over threshold.
        
        :param excesses: array, excesses over threshold
        :return: tuple, GPD parameters (shape, location, scale)
        """
        params = genpareto.fit(excesses)
        return params

    def plot_gpd(self, excesses, params, column, output_dir):
        """
        Plot the fitted GPD distribution against the excesses over threshold.
        
        :param excesses: array, excesses over threshold
        :param params: tuple, GPD parameters
        :param column: str, column name for the plot title
        :param output_dir: str, directory to save the plot
        """
        fig, ax = plt.subplots()
        x = np.linspace(min(excesses), max(excesses), 100)
        y = genpareto.pdf(x, *params)
        ax.hist(excesses, bins=20, density=True, alpha=0.6, color='g')
        ax.plot(x, y, 'r-', lw=2, label='GPD fit')
        ax.set_title(f'Excesses Over Threshold and Fitted GPD Distribution ({column})')
        ax.set_xlabel(column)
        ax.set_ylabel('Density')
        ax.legend()
        plt.savefig(os.path.join(output_dir, f'gpd_fit_{column}.png'))
        plt.show()

    def analyze(self, data_filepath, column, block_size, threshold, output_dir):
        """
        Execute the full extreme value analysis pipeline.
        
        :param data_filepath: str, path to the input data file
        :param column: str, column name to analyze
        :param block_size: int, size of each block for Block Maxima method
        :param threshold: float, threshold value for Peaks Over Threshold method
        :param output_dir: str, directory to save the plots
        """
        data = self.load_data(data_filepath)

        # Block Maxima Method
        block_maxima = self.block_maxima(data, column, block_size)
        gev_params = self.fit_gev(block_maxima)
        self.plot_gev(block_maxima, gev_params, column, output_dir)

        # Peaks Over Threshold Method
        excesses = self.peaks_over_threshold(data, column, threshold)
        gpd_params = self.fit_gpd(excesses)
        self.plot_gpd(excesses, gpd_params, column, output_dir)

if __name__ == "__main__":
    data_filepath = 'data/processed/processed_data.csv'
    output_dir = 'results/extreme_value_analysis/'
    os.makedirs(output_dir, exist_ok=True)

    column = 'feature1'  # Example column name
    block_size = 30  # Example block size
    threshold = 1.5  # Example threshold value

    eva = ExtremeValueAnalysis()

    # Analyze the data
    eva.analyze(data_filepath, column, block_size, threshold, output_dir)
    print("Extreme value analysis completed and results saved.")
