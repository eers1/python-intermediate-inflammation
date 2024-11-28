"""Module containing mechanism for calculating standard deviation between datasets.
"""

import glob
import os
import numpy as np

from inflammation import models


class CSVDataSource:
    """Loads all the CSV data from a specified directory."""
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_inflammation_data(self):
        """Loads data csvs and returns a list."""
        data_file_paths = glob.glob(os.path.join(self.data_dir, 'inflammation*.csv'))

        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation data CSV files found in path {self.data_dir}")
        data = map(models.load_csv, data_file_paths)
        return list(data)


class JSONDataSource:
    """Loads JSON data from a specified directory."""
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_inflammation_data(self):
        """Loads data csvs and returns a list."""
        data_file_paths = glob.glob(os.path.join(self.data_dir, 'inflammation*.json'))

        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation data JSON files found in path {self.data_dir}")
        data = map(models.load_json, data_file_paths)
        return list(data)


def load_data(infiles):
    _, extension = os.path.splitext(infiles[0])
    if extension == '.csv':
        data_source = CSVDataSource(os.path.dirname(infiles[0]))
    elif extension == '.json':
        data_source = JSONDataSource(os.path.dirname(infiles[0]))
    else:
        raise ValueError(f"Unsupported file extension {extension}")

    data = data_source.load_inflammation_data()

    return data


def analyse_data(data):
    """Calculates the standard deviation by day between datasets.
    Works out the mean inflammation value for each day across all datasets"""

    means_by_day = map(models.daily_mean, data)
    means_by_day_matrix = np.stack(list(means_by_day))

    daily_standard_deviation = np.std(means_by_day_matrix, axis=0)
    return daily_standard_deviation