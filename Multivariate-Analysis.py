# multivariate_analysis.py

"""
Multivariate Analysis Module for Multivariate Extreme Value Analysis

This module contains functions for performing multivariate analysis to understand the relationships
between multiple variables involved in extreme events.

Techniques Used:
- Principal Component Analysis (PCA)
- Canonical Correlation Analysis (CCA)
- Copulas

Libraries/Tools:
- pandas
- numpy
- scikit-learn
- statsmodels
- matplotlib

"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import norm, gaussian_kde

class MultivariateAnalysis:
    def __init__(self):
        """
        Initialize the MultivariateAnalysis class.
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

    def perform_pca(self, data, n_components):
        """
        Perform Principal Component Analysis (PCA) on the data.
        
        :param data: DataFrame, input data
        :param n_components: int, number of principal components to retain
        :return: PCA object, transformed data
        """
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(data)
        return pca, transformed_data

    def plot_pca(self, pca, transformed_data, output_dir):
        """
        Plot the explained variance and the first two principal components.
        
        :param pca: PCA object
        :param transformed_data: array, transformed data
        :param output_dir: str, directory to save the plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Explained variance plot
        plt.figure(figsize=(8, 6))
        plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        plt.title('Explained Variance by Principal Components')
        plt.savefig(os.path.join(output_dir, 'pca_explained_variance.png'))
        plt.show()

        # First two principal components plot
        plt.figure(figsize=(8, 6))
        plt.scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.5)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('First Two Principal Components')
        plt.savefig(os.path.join(output_dir, 'pca_first_two_components.png'))
        plt.show()

    def perform_cca(self, data_x, data_y, n_components):
        """
        Perform Canonical Correlation Analysis (CCA) on the data.
        
        :param data_x: DataFrame, first set of variables
        :param data_y: DataFrame, second set of variables
        :param n_components: int, number of canonical components to retain
        :return: CCA object, transformed data
        """
        cca = CCA(n_components=n_components)
        x_c, y_c = cca.fit_transform(data_x, data_y)
        return cca, x_c, y_c

    def plot_cca(self, x_c, y_c, output_dir):
        """
        Plot the first two canonical variables.
        
        :param x_c: array, first set of canonical variables
        :param y_c: array, second set of canonical variables
        :param output_dir: str, directory to save the plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(x_c[:, 0], y_c[:, 0], alpha=0.5)
        plt.xlabel('First Canonical Variable (X)')
        plt.ylabel('First Canonical Variable (Y)')
        plt.title('First Canonical Variables')
        plt.savefig(os.path.join(output_dir, 'cca_first_canonical_variables.png'))
        plt.show()

    def fit_copula(self, data):
        """
        Fit a Gaussian copula to the data.
        
        :param data: DataFrame, input data
        :return: array, copula parameters
        """
        # Transform data to uniform margins
        data_u = data.apply(lambda x: norm.cdf((x - x.mean()) / x.std()), axis=0)
        
        # Fit Gaussian copula
        kde = gaussian_kde(data_u.T)
        return kde

    def plot_copula(self, kde, data, output_dir):
        """
        Plot the fitted copula.
        
        :param kde: gaussian_kde object, fitted copula
        :param data: DataFrame, input data
        :param output_dir: str, directory to save the plot
        """
        os.makedirs(output_dir, exist_ok=True)
        
        data_u = data.apply(lambda x: norm.cdf((x - x.mean()) / x.std()), axis=0)
        
        # Plot the copula density
        plt.figure(figsize=(8, 6))
        sns.kdeplot(x=data_u.iloc[:, 0], y=data_u.iloc[:, 1], fill=True, cmap="Blues")
        plt.xlabel(data.columns[0])
        plt.ylabel(data.columns[1])
        plt.title('Gaussian Copula Density')
        plt.savefig(os.path.join(output_dir, 'copula_density.png'))
        plt.show()

    def analyze(self, data_filepath, output_dir, pca_components=2, cca_components=2):
        """
        Execute the full multivariate analysis pipeline.
        
        :param data_filepath: str, path to the input data file
        :param output_dir: str, directory to save the plots
        :param pca_components: int, number of principal components to retain
        :param cca_components: int, number of canonical components to retain
        """
        data = self.load_data(data_filepath)

        # Perform PCA
        pca, transformed_data = self.perform_pca(data, pca_components)
        self.plot_pca(pca, transformed_data, output_dir)

        # Perform CCA (using two halves of the data as an example)
        mid_point = len(data.columns) // 2
        data_x = data.iloc[:, :mid_point]
        data_y = data.iloc[:, mid_point:]
        cca, x_c, y_c = self.perform_cca(data_x, data_y, cca_components)
        self.plot_cca(x_c, y_c, output_dir)

        # Fit and plot Copula
        kde = self.fit_copula(data)
        self.plot_copula(kde, data, output_dir)

if __name__ == "__main__":
    data_filepath = 'data/processed/processed_data.csv'
    output_dir = 'results/multivariate_analysis/'
    os.makedirs(output_dir, exist_ok=True)

    pca_components = 2  # Example number of PCA components
    cca_components = 2  # Example number of CCA components

    ma = MultivariateAnalysis()

    # Analyze the data
    ma.analyze(data_filepath, output_dir, pca_components, cca_components)
    print("Multivariate analysis completed and results saved.")
