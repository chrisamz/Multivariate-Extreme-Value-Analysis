# spectral_learning.py

"""
Spectral Learning Module for Multivariate Extreme Value Analysis

This module contains functions for implementing spectral learning techniques to predict and analyze
multivariate extreme events.

Techniques Used:
- Spectral Clustering
- Singular Value Decomposition (SVD)
- Matrix Factorization

Libraries/Tools:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

"""

import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import seaborn as sns
import os

class SpectralLearning:
    def __init__(self):
        """
        Initialize the SpectralLearning class.
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

    def perform_spectral_clustering(self, data, n_clusters):
        """
        Perform spectral clustering on the data.
        
        :param data: DataFrame, input data
        :param n_clusters: int, number of clusters
        :return: array, cluster labels
        """
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', assign_labels='kmeans')
        labels = clustering.fit_predict(data)
        return labels

    def plot_spectral_clustering(self, data, labels, output_dir):
        """
        Plot the results of spectral clustering.
        
        :param data: DataFrame, input data
        :param labels: array, cluster labels
        :param output_dir: str, directory to save the plot
        """
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], hue=labels, palette='viridis')
        plt.title('Spectral Clustering Results')
        plt.xlabel(data.columns[0])
        plt.ylabel(data.columns[1])
        plt.legend(title='Cluster')
        plt.savefig(os.path.join(output_dir, 'spectral_clustering.png'))
        plt.show()

    def perform_svd(self, data, n_components):
        """
        Perform Singular Value Decomposition (SVD) on the data.
        
        :param data: DataFrame, input data
        :param n_components: int, number of components to retain
        :return: array, transformed data
        """
        svd = TruncatedSVD(n_components=n_components)
        transformed_data = svd.fit_transform(data)
        return svd, transformed_data

    def plot_svd(self, transformed_data, output_dir):
        """
        Plot the results of SVD.
        
        :param transformed_data: array, transformed data
        :param output_dir: str, directory to save the plot
        """
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.5)
        plt.title('Singular Value Decomposition (SVD) Results')
        plt.xlabel('First Component')
        plt.ylabel('Second Component')
        plt.savefig(os.path.join(output_dir, 'svd_results.png'))
        plt.show()

    def perform_matrix_factorization(self, data, n_components):
        """
        Perform Non-negative Matrix Factorization (NMF) on the data.
        
        :param data: DataFrame, input data
        :param n_components: int, number of components to retain
        :return: tuple, basis matrix and coefficient matrix
        """
        nmf = NMF(n_components=n_components, init='random', random_state=0)
        W = nmf.fit_transform(data)
        H = nmf.components_
        return W, H

    def plot_matrix_factorization(self, W, H, output_dir):
        """
        Plot the basis matrix and coefficient matrix from NMF.
        
        :param W: array, basis matrix
        :param H: array, coefficient matrix
        :param output_dir: str, directory to save the plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(W, cmap='viridis')
        plt.title('Basis Matrix (W) from NMF')
        plt.xlabel('Components')
        plt.ylabel('Samples')
        plt.savefig(os.path.join(output_dir, 'nmf_basis_matrix.png'))
        plt.show()

        plt.figure(figsize=(8, 6))
        sns.heatmap(H, cmap='viridis')
        plt.title('Coefficient Matrix (H) from NMF')
        plt.xlabel('Features')
        plt.ylabel('Components')
        plt.savefig(os.path.join(output_dir, 'nmf_coefficient_matrix.png'))
        plt.show()

    def analyze(self, data_filepath, output_dir, n_clusters=3, n_components=2):
        """
        Execute the full spectral learning analysis pipeline.
        
        :param data_filepath: str, path to the input data file
        :param output_dir: str, directory to save the plots
        :param n_clusters: int, number of clusters for spectral clustering
        :param n_components: int, number of components for SVD and NMF
        """
        data = self.load_data(data_filepath)

        # Perform Spectral Clustering
        labels = self.perform_spectral_clustering(data, n_clusters)
        self.plot_spectral_clustering(data, labels, output_dir)

        # Perform SVD
        svd, transformed_data = self.perform_svd(data, n_components)
        self.plot_svd(transformed_data, output_dir)

        # Perform Matrix Factorization
        W, H = self.perform_matrix_factorization(data, n_components)
        self.plot_matrix_factorization(W, H, output_dir)

if __name__ == "__main__":
    data_filepath = 'data/processed/processed_data.csv'
    output_dir = 'results/spectral_learning/'
    os.makedirs(output_dir, exist_ok=True)

    n_clusters = 3  # Example number of clusters for spectral clustering
    n_components = 2  # Example number of components for SVD and NMF

    sl = SpectralLearning()

    # Analyze the data
    sl.analyze(data_filepath, output_dir, n_clusters, n_components)
    print("Spectral learning analysis completed and results saved.")
