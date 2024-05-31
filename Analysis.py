# analysis.py

"""
Comprehensive Analysis Module for Multivariate Extreme Value Analysis

This module integrates various components of the project to perform a comprehensive analysis,
including data preprocessing, extreme value analysis, multivariate analysis, spectral learning, and evaluation.

Libraries/Tools:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- scipy
"""

import os
from data_preprocessing import DataPreprocessing
from extreme_value_analysis import ExtremeValueAnalysis
from multivariate_analysis import MultivariateAnalysis
from spectral_learning import SpectralLearning
from evaluation import ModelEvaluation

# Paths to directories
RAW_DATA_DIR = 'data/raw/'
PROCESSED_DATA_DIR = 'data/processed/'
RESULTS_DIR = 'results/'

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    # Data Preprocessing
    raw_data_filepath = os.path.join(RAW_DATA_DIR, 'extreme_events_data.csv')
    processed_data_filepath = os.path.join(PROCESSED_DATA_DIR, 'processed_data.csv')
    features = ['feature1', 'feature2', 'feature3', 'feature4']  # Example feature names

    preprocessing = DataPreprocessing()
    processed_data = preprocessing.preprocess(raw_data_filepath, PROCESSED_DATA_DIR, features)
    print("Data preprocessing completed and data saved.")

    # Extreme Value Analysis
    eva = ExtremeValueAnalysis()
    eva_output_dir = os.path.join(RESULTS_DIR, 'extreme_value_analysis/')
    os.makedirs(eva_output_dir, exist_ok=True)

    column = 'feature1'  # Example column name
    block_size = 30  # Example block size
    threshold = 1.5  # Example threshold value

    eva.analyze(processed_data_filepath, column, block_size, threshold, eva_output_dir)
    print("Extreme value analysis completed and results saved.")

    # Multivariate Analysis
    ma = MultivariateAnalysis()
    ma_output_dir = os.path.join(RESULTS_DIR, 'multivariate_analysis/')
    os.makedirs(ma_output_dir, exist_ok=True)

    pca_components = 2  # Example number of PCA components
    cca_components = 2  # Example number of CCA components

    ma.analyze(processed_data_filepath, ma_output_dir, pca_components, cca_components)
    print("Multivariate analysis completed and results saved.")

    # Spectral Learning
    sl = SpectralLearning()
    sl_output_dir = os.path.join(RESULTS_DIR, 'spectral_learning/')
    os.makedirs(sl_output_dir, exist_ok=True)

    n_clusters = 3  # Example number of clusters for spectral clustering
    n_components = 2  # Example number of components for SVD and NMF

    sl.analyze(processed_data_filepath, sl_output_dir, n_clusters, n_components)
    print("Spectral learning analysis completed and results saved.")

    # Evaluation
    y_true_filepath = os.path.join(PROCESSED_DATA_DIR, 'y_true.csv')
    y_pred_filepath = os.path.join(PROCESSED_DATA_DIR, 'y_pred.csv')
    clustering_labels_filepath = os.path.join(PROCESSED_DATA_DIR, 'clustering_labels.csv')
    eval_output_dir = os.path.join(RESULTS_DIR, 'evaluation/')
    os.makedirs(eval_output_dir, exist_ok=True)

    evaluator = ModelEvaluation()
    evaluator.evaluate(processed_data_filepath, y_true_filepath, y_pred_filepath, clustering_labels_filepath, eval_output_dir)
    print("Model evaluation completed and results saved.")

if __name__ == "__main__":
    main()
