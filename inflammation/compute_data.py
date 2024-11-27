"""Module containing mechanism for calculating standard deviation between datasets.
"""

import glob
import os
import numpy as np

from inflammation import models, views


def load_inflammation_data(data_dir):
    """Loads data csvs and returns them. """
    data_file_paths = glob.glob(os.path.join(data_dir, 'inflammation*.csv'))

    if len(data_file_paths) == 0:
        raise ValueError(f"No inflammation data CSV files found in path {data_dir}")
    data = map(models.load_csv, data_file_paths)
    return list(data)


def analyse_data(data):
    """Calculates the standard deviation by day between datasets.
    Works out the mean inflammation value for each day across all datasets"""
    means_by_day = map(models.daily_mean, data)
    means_by_day_matrix = np.stack(list(means_by_day))

    daily_standard_deviation = np.std(means_by_day_matrix, axis=0)
    return daily_standard_deviation


def plot_data(daily_standard_deviation):
    """Plots the standard deviation by day between datasets."""
    graph_data = {
        'standard deviation by day': daily_standard_deviation,
    }
    views.visualize(graph_data)


def main(data_dir):
    data = load_inflammation_data(data_dir)

    daily_standard_deviation = analyse_data(data)

    plot_data(daily_standard_deviation)