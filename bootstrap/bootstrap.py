"""
Library for boostraping statistics

Citation:
Efron, Bradley, and Robert J. Tibshirani.
An introduction to the bootstrap. CRC press, 1994.
"""

import numpy as np
from scipy import stats

class Bootstrap:
    
    def __init__(self):
        return
    
    def bootstrap_sample(self, data, parametric=False):
        """
        Resamples data by random sample with replacement
        
        Args:
            data (np.array): array of data to resample
            parametric (str) in ['normal', 'uniform']: parametric distribution to resample from
                if False, use nonparametric bootstrap sampling
        
        Returns:
            bootstrap resampled data (array)
        """
        sample_size = len(data)
        if parametric == 'normal':
            mean_estimate = np.mean(data)
            std_estimate = np.std(data)
            return np.random.normal(mean_estimate, std_estimate, size=sample_size)
        elif parametric == 'uniform':
            min_estimate, max_estimate = np.min(data), np.max(data)
            return np.random.uniform(min_estimate, max_estimate, size=sample_size)
        else:
            indices = [np.random.randint(0, sample_size) for i in range(sample_size)]
            return data[indices]
            
    def bootstrap_matrixsample(self, data, axis=0):
        """
        Resamples a matrix by rows or columns

        Args:
         data (np.matrix): matrix of data to resample 
         axis (int) in [0, 1]: axis to resample by
             if 0: resample rows
             if 1: resample columns

        Returns:
            bootstrap resampled data (matrix)        
        """
        if axis == 0:
            n_rows = np.shape(data)[0]
            samples = np.random.randint(n_rows, size=n_rows)
            bootstrap_matrix = data[samples, :]
        elif axis ==1:
            n_cols = np.shape(data)[1]
            samples = np.random.randint(n_cols, size=n_cols)
            bootstrap_matrix = data[:, samples]
        return bootstrap_matrix

    def jackknife_sample(self, data, index):
        """
        Single jackknife sample of data
        
        Args:
            data (np.array): array of data to resample
            index (int): Index of array to leave out in jackknife sample
            
        Returns:
            jackknife resampled data (array)
        """
        jackknife = np.delete(data, index)
        return jackknife
        
    def bootstrap_statistic(self, data, func=np.mean, n_samples=50, 
                            parametric=False, bias_correction=False,
                            axis=0):
        """
        Bootstraps a statistic and calculates the standard error of the statistic
        
        Args:
            data (array or matrix): array or matrix of data to calculate statistic and SE of statistic
            func (function): statistical function to calculate on data
                examples: np.mean, np.median
            n_samples (int): number of bootstrap samples to calculate statistic for
            parametric (str) in ['normal', 'uniform']: parametric distribution to resample from
                if False, use nonparametric bootstrap sampling
            bias_correction (bool): if True, bias correct bootstrap statistic 
            axis (int) in [0, 1]: if type(data) == np.matrix, axis to resample by
                if 0: resample rows
                if 1: resample columns
            
        Returns:
            tuple: (statistic (float), bias (float), sem (float))
            Returns the bootstrapped statistic, its bias and SEM.
        """
        plugin_estimate = func(data)
        statistics = []
        for sample in range(n_samples):
            if type(data) == np.matrix:
                resample = self.bootstrap_matrixsample(data, axis=axis)
            else: 
                resample = self.bootstrap_sample(data, parametric=parametric)
            statistic = func(resample)
            statistics.append(statistic)
        statistic = np.mean(statistics)
        bias = statistic - plugin_estimate
        if bias_correction:
            statistic = statistic - bias
        sem = stats.sem(statistics)
        return (statistic, bias, sem)
        
    def jackknife_statistic(self, data, func=np.mean):
        """
        Jackknifes a statistic and calculates the standard error of the statistic
        
        Args:
            data (array): array of data to calculate statistic and SE of statistic
            func (function): statistical function to calculate on data
                examples: np.mean, np.median
            
        Returns:
            statistic (tuple), sem (tuple)
            Returns the jackknifed statistic and the SEM of the statistic
        """
        n_samples = len(data)
        statistics = []
        for sample in range(n_samples):
            jack_sample = self.jackknife_sample(data, sample)
            statistic = func(jack_sample)
            statistics.append(statistic)
        return (np.mean(statistics), stats.sem(statistics))
