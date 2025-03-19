import os

import numpy as np
import polars as pl

from data.process_manager.processor import Processor
from data.settings.datasets_config import DatasetConfig


class Extractor:
    
    def __init__(self, df: pl.DataFrame, cfg: DatasetConfig = DatasetConfig()):
        self.df = self.proc.process()
        self.cfg = cfg
        
    def _extract_features2np(self):
        """Extracts numpy array as data for Neural Network from polars.DataFrame
        Defaults feauter are: ['PulsesAmpl', 'PulsesTime', 'Xrel', 'Yrel', 'Zrel'].

        Returns:
            np.ndarray[float]: array of shape (batch_size, max_num_of_pulses, num_of_features)
        """
        # Step 0: Df converts into array, containing arrays as objects
        arr = self.df[self.cfg.features].to_numpy()
        
        # Step 1: Find the maximum length of the arrays
        max_length = np.vectorize(len)(arr[:,0]).max()

        # Step 2: Create an empty array filled with nans, with the desired shape (batch_size, max_num_of_pulses, num_of_features)
        result = np.full((arr.shape[0], max_length, arr.shape[1]), np.nan, dtype=np.float64)

        # Step 3: Vectorized operation to copy the contents of each 1D array into the 3D array
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                sublist = arr[i, j]
                result[i, :len(sublist), j] = sublist  # Fill the result array with the sublist contents
        return result
    
    def _extract_labels(self):
        """Extracts numpy array as labels for Neural Network from polars.DataFrame
        Defaults labels are: ['nu_induced'].

        Returns:
            np.ndarray[float]: array of shape (batch_size, num_of_labels)
        """
        arr = self.df[self.cfg.labels].to_numpy()
        return arr
        
    
    def extract(self):
        return self._extract_features2np(), self._extract_labels()