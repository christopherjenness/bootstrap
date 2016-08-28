"""
Library for boostraping statistics

Citation:
Efron, Bradley, and Robert J. Tibshirani.
An introduction to the bootstrap. CRC press, 1994.
"""

import numpy as np
from scipy.stats import sem

class Bootstrap:
    
    def __init__(self, data):
        self.data = data
        return
    
    
    def resample(self, data):
        """
        Resamples data by random sample with replacement
        
        Args:
            data (np.array): array of data to resample
        
        Returns:
            array: array of resampled data
        """
        sample_size = len(data)
        return [np.random.randint(0, sample_size) for i in range(sample_size)]

    def estimate_standard_error(self, data, func=np.mean, n_samples=50):
        """
        Args:
            data (array): array of data to calculate statistic and SE of statistic
            func (function): statistical function to calculate on data
                examples: np.mean, np.median
            n_samples (int): number of bootstrap samples to 
        """
        statistics = []
        for sample in range(n_samples):
            indices = self.resample(data)
            statistic = func(data[indices])
            statistics.append(statistic)
            print (data[indices])
        print (statistics)
        return np.mean(statistics), sem(statistics)

a = Bootstrap(np.random.uniform(0, 1, 100))
a.resample(a.data)
b = a.data[a.resample(a.data)]
print(b)
c = a.estimate_standard_error(a.data)





















