"""
Library for boostrap statistics

Citation:
Efron, Bradley, and Robert J. Tibshirani.
An introduction to the bootstrap. CRC press, 1994.
"""

import numpy as np

class Bootstrap:
    
    ___init___(self, data):
        self.data = data
        return
    
    
    def resample(self, data):
        for i in range(len(data)):
            yield np.random.randint(0, len(data)

    def estimate_standard_error(self, data, func=np.mean, n_samples=50):
        """
        Args:
            data (array): array of data to calculate statistic and SE of statistic
            func (function): statistical function to calculate on data
                examples: np.mean, np.median
            n_samples (int): number of bootstrap samples to 
        """